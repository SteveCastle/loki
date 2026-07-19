package tasks

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"log"
	"os"
	"path"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
)

// Site-specific extractors are deliberately not compiled in: they load at
// runtime from a JSON definitions file the user supplies, so the binary
// carries no knowledge of any particular site. The path comes from the
// "extractorsPath" config key (or the LOWKEY_EXTRACTORS env var), defaulting
// to "extractors.json" in the server's working directory. The file is
// re-read automatically when it changes — no restart needed.
//
// Definition format — a JSON object with an "extractors" array; regexes are
// Go regexp syntax, capture groups from itemUrl are available to subPath as
// $1..$9:
//
//	{
//	  "extractors": [
//	    {
//	      "name": "example",                      // log label + download folder
//	      "itemUrl": "^https://example\\.com/u/(\\w+)/(\\w+)$",
//	      "listingUrl": "^https://example\\.com/u/(\\w+)$",   // optional
//	      "listingLinks": "href=\"(https://[^\"]+)\"",        // optional, capture 1 = link
//	      "mediaUrl": "file\\s*:\\s*\"([^\"]+)\"",            // capture 1 = file URL
//	      "title": "<h1>([^<]+)</h1>",                        // optional, capture 1 = title
//	      "subPath": "$1/$2",                                 // path under the name folder
//	      "ext": ".m4a"                                       // fallback when the URL has none
//	    }
//	  ]
//	}
//
// A listing page is any URL matching listingUrl: it is fetched, every
// listingLinks capture is collected, and the ones matching itemUrl are
// downloaded as items.

type extractorSpec struct {
	Name         string `json:"name"`
	ItemURL      string `json:"itemUrl"`
	ListingURL   string `json:"listingUrl"`
	ListingLinks string `json:"listingLinks"`
	MediaURL     string `json:"mediaUrl"`
	Title        string `json:"title"`
	SubPath      string `json:"subPath"`
	Ext          string `json:"ext"`
}

type extractorsFile struct {
	Extractors []extractorSpec `json:"extractors"`
}

// dslExtractor implements mediaExtractor from a compiled extractorSpec.
type dslExtractor struct {
	name      string
	itemRE    *regexp.Regexp
	listingRE *regexp.Regexp // nil when the site has no listing pages
	linksRE   *regexp.Regexp
	mediaRE   *regexp.Regexp
	titleRE   *regexp.Regexp // nil when no title pattern
	subPath   string
	ext       string
}

func compileExtractorSpec(s extractorSpec) (*dslExtractor, error) {
	if s.Name == "" || strings.ContainsAny(s.Name, `/\`) {
		return nil, fmt.Errorf("name %q must be a non-empty folder name", s.Name)
	}
	if s.ItemURL == "" || s.MediaURL == "" || s.SubPath == "" {
		return nil, fmt.Errorf("%s: itemUrl, mediaUrl, and subPath are required", s.Name)
	}
	if (s.ListingURL == "") != (s.ListingLinks == "") {
		return nil, fmt.Errorf("%s: listingUrl and listingLinks must be set together", s.Name)
	}
	e := &dslExtractor{name: s.Name, subPath: s.SubPath, ext: s.Ext}
	var err error
	if e.itemRE, err = regexp.Compile(s.ItemURL); err != nil {
		return nil, fmt.Errorf("%s: itemUrl: %w", s.Name, err)
	}
	if e.mediaRE, err = regexp.Compile(s.MediaURL); err != nil {
		return nil, fmt.Errorf("%s: mediaUrl: %w", s.Name, err)
	}
	if s.ListingURL != "" {
		if e.listingRE, err = regexp.Compile(s.ListingURL); err != nil {
			return nil, fmt.Errorf("%s: listingUrl: %w", s.Name, err)
		}
		if e.linksRE, err = regexp.Compile(s.ListingLinks); err != nil {
			return nil, fmt.Errorf("%s: listingLinks: %w", s.Name, err)
		}
	}
	if s.Title != "" {
		if e.titleRE, err = regexp.Compile(s.Title); err != nil {
			return nil, fmt.Errorf("%s: title: %w", s.Name, err)
		}
	}
	return e, nil
}

func (e *dslExtractor) Name() string { return e.name }

func (e *dslExtractor) Match(u string) bool {
	return e.itemRE.MatchString(u) || (e.listingRE != nil && e.listingRE.MatchString(u))
}

func (e *dslExtractor) Resolve(ctx context.Context, input string) ([]string, error) {
	if e.itemRE.MatchString(input) {
		return []string{input}, nil
	}
	if e.listingRE == nil || !e.listingRE.MatchString(input) {
		return nil, fmt.Errorf("unrecognized URL: %s", input)
	}

	page, err := fetchExtractorPage(ctx, input)
	if err != nil {
		return nil, err
	}

	seen := make(map[string]struct{})
	var items []string
	for _, lm := range e.linksRE.FindAllStringSubmatch(page, -1) {
		link := lm[1]
		if !e.itemRE.MatchString(link) {
			continue
		}
		if _, ok := seen[link]; ok {
			continue
		}
		seen[link] = struct{}{}
		items = append(items, link)
	}
	return items, nil
}

func (e *dslExtractor) Extract(ctx context.Context, itemURL string) (mediaItem, error) {
	m := e.itemRE.FindStringSubmatch(itemURL)
	if m == nil {
		return mediaItem{}, fmt.Errorf("unrecognized item URL: %s", itemURL)
	}

	page, err := fetchExtractorPage(ctx, itemURL)
	if err != nil {
		return mediaItem{}, err
	}

	mu := e.mediaRE.FindStringSubmatch(page)
	if len(mu) < 2 {
		return mediaItem{}, fmt.Errorf("no media URL found in page")
	}
	mediaURL := mu[1]
	if strings.HasPrefix(mediaURL, "//") {
		mediaURL = "https:" + mediaURL
	}

	title := ""
	if e.titleRE != nil {
		if t := e.titleRE.FindStringSubmatch(page); len(t) > 1 {
			title = html.UnescapeString(strings.TrimSpace(t[1]))
		}
	}

	sub := path.Clean(expandCaptures(e.subPath, m))
	if sub == "." || sub == ".." || strings.HasPrefix(sub, "../") || path.IsAbs(sub) || strings.Contains(sub, `\`) {
		return mediaItem{}, fmt.Errorf("subPath %q resolves outside the download folder", sub)
	}

	return mediaItem{MediaURL: mediaURL, Title: title, SubPath: sub, Ext: e.ext}, nil
}

// expandCaptures substitutes $1..$9 in tpl with the regex submatches.
func expandCaptures(tpl string, m []string) string {
	var b strings.Builder
	for i := 0; i < len(tpl); i++ {
		if tpl[i] == '$' && i+1 < len(tpl) && tpl[i+1] >= '1' && tpl[i+1] <= '9' {
			if n := int(tpl[i+1] - '0'); n < len(m) {
				b.WriteString(m[n])
			}
			i++
			continue
		}
		b.WriteByte(tpl[i])
	}
	return b.String()
}

/* ---- loading + hot reload ------------------------------------------- */

var extractorsCache struct {
	mu     sync.Mutex
	path   string
	mtime  time.Time
	size   int64
	exts   []mediaExtractor
	logged string // last logged problem, to avoid repeating it every call
}

func extractorsFilePath() string {
	if p := strings.TrimSpace(appconfig.Get().ExtractorsPath); p != "" {
		return p
	}
	return "extractors.json"
}

// loadExtractors returns the extractors defined in the definitions file,
// re-reading it when the file changes. A missing file simply means no native
// extractors; a broken file or spec is logged once and skipped.
func loadExtractors() []mediaExtractor {
	p := extractorsFilePath()

	extractorsCache.mu.Lock()
	defer extractorsCache.mu.Unlock()

	fi, err := os.Stat(p)
	if err != nil {
		extractorsCache.path = ""
		extractorsCache.exts = nil
		return nil
	}
	if p == extractorsCache.path && fi.ModTime().Equal(extractorsCache.mtime) && fi.Size() == extractorsCache.size {
		return extractorsCache.exts
	}

	extractorsCache.path = p
	extractorsCache.mtime = fi.ModTime()
	extractorsCache.size = fi.Size()
	extractorsCache.exts = nil

	logOnce := func(msg string) {
		if msg != extractorsCache.logged {
			extractorsCache.logged = msg
			log.Print(msg)
		}
	}

	raw, err := os.ReadFile(p)
	if err != nil {
		logOnce(fmt.Sprintf("extractors: cannot read %s: %v", p, err))
		return nil
	}
	var f extractorsFile
	if err := json.Unmarshal(raw, &f); err != nil {
		logOnce(fmt.Sprintf("extractors: cannot parse %s: %v", p, err))
		return nil
	}

	var problems []string
	for _, spec := range f.Extractors {
		e, err := compileExtractorSpec(spec)
		if err != nil {
			problems = append(problems, err.Error())
			continue
		}
		extractorsCache.exts = append(extractorsCache.exts, e)
	}
	if len(problems) > 0 {
		logOnce(fmt.Sprintf("extractors: skipped %d definition(s) in %s: %s", len(problems), p, strings.Join(problems, "; ")))
	}
	return extractorsCache.exts
}

// findMediaExtractor returns the first defined extractor matching the URL,
// or nil when no native extractor handles it.
func findMediaExtractor(u string) mediaExtractor {
	u = strings.TrimSpace(u)
	for _, e := range loadExtractors() {
		if e.Match(u) {
			return e
		}
	}
	return nil
}
