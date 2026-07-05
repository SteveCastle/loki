package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// predicate mirrors the server's media_query.go Predicate wire shape.
type predicate struct {
	Type    string `json:"type"`
	Value   string `json:"value"`
	Exclude bool   `json:"exclude,omitempty"`
	Join    string `json:"join,omitempty"`
}

// stringList collects repeatable flags (--tag a --tag b).
type stringList []string

func (s *stringList) String() string     { return strings.Join(*s, ",") }
func (s *stringList) Set(v string) error { *s = append(*s, v); return nil }

func init() {
	register(command{group: "media", name: "query",
		args:    "[--tag T]... [--exclude-tag T]... [--path P] [--description D] [--hash H] [--similar PATH] [--visual TEXT] [--mode AND|OR] [--predicates FILE|-]",
		summary: "Structured library query (POST /api/media/query)", run: cmdMediaQuery})
	register(command{group: "media", name: "search", args: "<text>",
		summary: "Search descriptions (POST /api/media/search)", run: cmdMediaSearch})
	register(command{group: "media", name: "similar", args: "<path> [--limit N]",
		summary: "Visually similar media (GET /api/media/similar)", run: cmdMediaSimilar})
	register(command{group: "media", name: "visual", args: "<text> [--limit N]",
		summary: "Text-to-image search (GET /api/media/search/visual)", run: cmdMediaVisual})
	register(command{group: "media", name: "metadata", args: "<path>",
		summary: "File metadata + description + hash (POST /api/media/metadata)",
		run:     mediaPathCommand("/api/media/metadata")})
	register(command{group: "media", name: "tags", args: "<path>",
		summary: "Tags on one media item (POST /api/media/tags)",
		run:     mediaPathCommand("/api/media/tags")})
	register(command{group: "media", name: "describe", args: "<path> (--text D | --clear)",
		summary: "Set or clear a media item's description (POST /api/media/description)", run: cmdMediaDescribe})
	register(command{group: "media", name: "transcript", args: "<path> [--text T | --clear]",
		summary: "Read, set, or clear a media item's transcript (POST /api/media/transcript)", run: cmdMediaTranscript})
	register(command{group: "media", name: "rate", args: "<path> [--elo E] [--views N] [--wins N] [--losses N]",
		summary: "Read or set rating fields; no flags reads (POST /api/media/rating)", run: cmdMediaRate})
	register(command{group: "media", name: "thumbs", args: "<path> [--regenerate] [--cache C] [--timestamp S]",
		summary: "List thumbnails (POST /api/thumbnails) or regenerate one (POST /api/thumbnails/regenerate)", run: cmdMediaThumbs})
	register(command{group: "media", name: "image-search", args: "<image-file>",
		summary: "Reverse image search with a local file (POST /api/media/search/image)", run: cmdMediaImageSearch})
	register(command{group: "media", name: "generate", args: "<path> --type T [--field k=v]... [--wait] [--follow] [--timeout D]",
		summary: `AI metadata generation — runs the "metadata" job (type: description, transcript, hash, ...)`, run: cmdMediaGenerate})
	register(command{group: "media", name: "delete", args: "<path> --yes",
		summary: "Delete a media item (POST /api/media/delete)", run: cmdMediaDelete})
}

// buildPredicates converts the query flags into the predicate array.
func buildPredicates(tags, exTags stringList, path, description, hash, similar, visual, mode string) []predicate {
	var preds []predicate
	add := func(ptype, value string, exclude bool) {
		if value != "" {
			preds = append(preds, predicate{Type: ptype, Value: value, Exclude: exclude, Join: mode})
		}
	}
	for _, t := range tags {
		add("tag", t, false)
	}
	for _, t := range exTags {
		add("tag", t, true)
	}
	add("path", path, false)
	add("description", description, false)
	add("hash", hash, false)
	add("similar", similar, false)
	add("visual", visual, false)
	return preds
}

func cmdMediaQuery(a *App, args []string) int {
	fs := flag.NewFlagSet("media query", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	var tags, exTags stringList
	fs.Var(&tags, "tag", "require this tag (repeatable)")
	fs.Var(&exTags, "exclude-tag", "exclude this tag (repeatable)")
	path := fs.String("path", "", "path substring/glob filter")
	description := fs.String("description", "", "description substring filter")
	hash := fs.String("hash", "", "exact content-hash filter")
	similar := fs.String("similar", "", "visually similar to this media path")
	visual := fs.String("visual", "", "text-to-image visual search")
	mode := fs.String("mode", "AND", "combine predicates with AND or OR")
	predFile := fs.String("predicates", "", "raw predicate JSON (file or - for stdin); overrides the flags")
	if err := fs.Parse(args); err != nil {
		return a.Usage(fs, err.Error())
	}
	*mode = strings.ToUpper(*mode)
	if *mode != "AND" && *mode != "OR" {
		return a.Usage(fs, "--mode must be AND or OR")
	}

	var preds []predicate
	if *predFile != "" {
		b, err := readBodyArg(nullSafeAt(*predFile), os.Stdin)
		if err != nil {
			return a.Fail(err)
		}
		if err := json.Unmarshal(b, &preds); err != nil {
			return a.Fail(fmt.Errorf("--predicates must be a JSON array of {type,value,exclude,join}: %w", err))
		}
	} else {
		preds = buildPredicates(tags, exTags, *path, *description, *hash, *similar, *visual, *mode)
	}
	if len(preds) == 0 {
		return a.Usage(fs, "no predicates — pass at least one filter flag or --predicates")
	}

	var out any
	if err := a.Client.DoJSON("POST", "/api/media/query", map[string]any{"predicates": preds, "mode": *mode}, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

// nullSafeAt lets --predicates accept both bare paths and @path / - like --body.
func nullSafeAt(v string) string {
	if v == "-" || strings.HasPrefix(v, "@") {
		return v
	}
	return "@" + v
}

func cmdMediaSearch(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl media search <text>")
	}
	var out any
	body := map[string]any{"description": args[0], "tags": []string{}, "filteringMode": ""}
	if err := a.Client.DoJSON("POST", "/api/media/search", body, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func limitFlagCommand(usage, urlPath, queryKey string) func(a *App, args []string) int {
	return func(a *App, args []string) int {
		fs := flag.NewFlagSet(usage, flag.ContinueOnError)
		fs.SetOutput(io.Discard)
		limit := fs.Int("limit", 50, "max results")
		if len(args) == 0 || strings.HasPrefix(args[0], "-") {
			return a.Usage(fs, "usage: lokictl "+usage)
		}
		value := args[0]
		if err := fs.Parse(args[1:]); err != nil {
			return a.Usage(fs, err.Error())
		}
		q := url.Values{queryKey: {value}, "limit": {strconv.Itoa(*limit)}}
		var out any
		if err := a.Client.DoJSON("GET", urlPath+"?"+q.Encode(), nil, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}
}

func cmdMediaSimilar(a *App, args []string) int {
	return limitFlagCommand("media similar <path> [--limit N]", "/api/media/similar", "path")(a, args)
}

func cmdMediaVisual(a *App, args []string) int {
	return limitFlagCommand("media visual <text> [--limit N]", "/api/media/search/visual", "q")(a, args)
}

func mediaPathCommand(urlPath string) func(a *App, args []string) int {
	return func(a *App, args []string) int {
		if len(args) != 1 {
			return a.Usage(nil, fmt.Sprintf("usage: lokictl media %s <path>", strings.TrimPrefix(urlPath, "/api/media/")))
		}
		var out any
		if err := a.Client.DoJSON("POST", urlPath, map[string]string{"path": args[0]}, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}
}

func cmdMediaDescribe(a *App, args []string) int {
	fs := flag.NewFlagSet("media describe", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	text := fs.String("text", "", "the description to set")
	clear := fs.Bool("clear", false, "clear the description")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl media describe <path> (--text D | --clear)")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *text == "" && !*clear {
		return a.Usage(fs, "pass --text D to set or --clear to clear")
	}
	if *text != "" && *clear {
		return a.Usage(fs, "--text and --clear are mutually exclusive")
	}
	if err := a.Client.DoJSON("POST", "/api/media/description", map[string]string{"path": path, "description": *text}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "ok", "path": path})
}

func cmdMediaTranscript(a *App, args []string) int {
	fs := flag.NewFlagSet("media transcript", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	text := fs.String("text", "", "the transcript to set")
	clear := fs.Bool("clear", false, "clear the transcript")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl media transcript <path> [--text T | --clear]")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}
	if *text != "" && *clear {
		return a.Usage(fs, "--text and --clear are mutually exclusive")
	}

	// No flags — read the transcript via the metadata endpoint.
	if *text == "" && !*clear {
		var meta struct {
			Transcript *string `json:"transcript"`
		}
		if err := a.Client.DoJSON("POST", "/api/media/metadata", map[string]string{"path": path}, &meta); err != nil {
			return a.Fail(err)
		}
		out := map[string]any{"path": path, "transcript": nil}
		if meta.Transcript != nil {
			out["transcript"] = *meta.Transcript
		}
		return a.PrintJSON(out)
	}

	if err := a.Client.DoJSON("POST", "/api/media/transcript", map[string]string{"path": path, "transcript": *text}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "ok", "path": path})
}

func cmdMediaRate(a *App, args []string) int {
	fs := flag.NewFlagSet("media rate", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	elo := fs.Float64("elo", 0, "set the elo rating")
	views := fs.Int64("views", 0, "set the view count")
	wins := fs.Int64("wins", 0, "set the win count")
	losses := fs.Int64("losses", 0, "set the loss count")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl media rate <path> [--elo E] [--views N] [--wins N] [--losses N]")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}

	// Only send fields the user explicitly set — the endpoint reads otherwise.
	body := map[string]any{"path": path}
	fs.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "elo":
			body["elo"] = *elo
		case "views":
			body["views"] = *views
		case "wins":
			body["wins"] = *wins
		case "losses":
			body["losses"] = *losses
		}
	})
	var out any
	if err := a.Client.DoJSON("POST", "/api/media/rating", body, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func cmdMediaThumbs(a *App, args []string) int {
	fs := flag.NewFlagSet("media thumbs", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	regenerate := fs.Bool("regenerate", false, "regenerate instead of list")
	cache := fs.String("cache", "thumbnail_path_600", "which thumbnail: thumbnail_path_100, _600, or _1200")
	timestamp := fs.Float64("timestamp", 0, "video frame time in seconds")
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(fs, "usage: lokictl media thumbs <path> [--regenerate] [--cache C] [--timestamp S]")
	}
	path := args[0]
	if err := fs.Parse(args[1:]); err != nil {
		return a.Usage(fs, err.Error())
	}

	if !*regenerate {
		var out any
		if err := a.Client.DoJSON("POST", "/api/thumbnails", map[string]string{"path": path}, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}
	var generated any
	body := map[string]any{"path": path, "cache": *cache, "timeStamp": *timestamp}
	if err := a.Client.DoJSON("POST", "/api/thumbnails/regenerate", body, &generated); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]any{"status": "ok", "path": path, "thumbnail": generated})
}

func cmdMediaImageSearch(a *App, args []string) int {
	if len(args) != 1 {
		return a.Usage(nil, "usage: lokictl media image-search <image-file>")
	}
	f, err := os.Open(args[0])
	if err != nil {
		return a.Fail(err)
	}
	defer f.Close()
	resp, err := a.Client.DoRaw("POST", "/api/media/search/image", "application/octet-stream", f)
	if err != nil {
		return a.Fail(err)
	}
	defer resp.Body.Close()
	var out any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return a.Fail(fmt.Errorf("server sent invalid JSON: %w", err))
	}
	return a.PrintJSON(out)
}

// cmdMediaGenerate is sugar over "job run metadata": it queues AI metadata
// generation for one media path.
func cmdMediaGenerate(a *App, args []string) int {
	if len(args) == 0 || strings.HasPrefix(args[0], "-") {
		return a.Usage(nil, "usage: lokictl media generate <path> --type T [--field k=v]... [--wait] [--follow] [--timeout D]")
	}
	path := args[0]
	rest := args[1:]

	// Lift --type into the job's --field type=... argument; everything else
	// (--field/--wait/--follow/--timeout) passes through to job run.
	var jobArgs []string
	genType := ""
	for i := 0; i < len(rest); i++ {
		name, val, hasEq := strings.Cut(rest[i], "=")
		if name == "--type" {
			if hasEq {
				genType = val
			} else if i+1 < len(rest) {
				i++
				genType = rest[i]
			}
			continue
		}
		jobArgs = append(jobArgs, rest[i])
	}
	if genType == "" {
		return a.Usage(nil, "--type is required (e.g. description, transcript, hash)")
	}
	jobArgs = append([]string{"metadata", path, "--field", "type=" + genType}, jobArgs...)
	return cmdJobRun(a, jobArgs)
}

func cmdMediaDelete(a *App, args []string) int {
	var path string
	for _, arg := range args {
		if arg != "--yes" && arg != "-y" {
			path = arg
		}
	}
	if path == "" {
		return a.Usage(nil, "usage: lokictl media delete <path> --yes")
	}
	if !hasYesFlag(args) {
		return a.Usage(nil, fmt.Sprintf("this permanently deletes %q from the library — re-run with --yes to confirm", path))
	}
	if err := a.Client.DoJSON("POST", "/api/media/delete", map[string]string{"path": path}, nil); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(map[string]string{"status": "deleted", "path": path})
}
