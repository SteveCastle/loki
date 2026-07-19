package tasks

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

// useExtractorsFile points the loader at a definitions file containing the
// given JSON, restoring the previous config when the test ends.
func useExtractorsFile(t *testing.T, jsonBody string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "extractors.json")
	if err := os.WriteFile(p, []byte(jsonBody), 0644); err != nil {
		t.Fatal(err)
	}
	prev := appconfig.Get()
	t.Cleanup(func() { appconfig.Set(prev) })
	cfg := prev
	cfg.ExtractorsPath = p
	appconfig.Set(cfg)
	return p
}

func demoSpec(host string) string {
	return fmt.Sprintf(`{
	  "extractors": [
	    {
	      "name": "demo",
	      "itemUrl": "^%s/u/([0-9a-zA-Z_-]+)/([0-9a-zA-Z_-]+)$",
	      "listingUrl": "^%s/u/([0-9a-zA-Z_-]+)$",
	      "listingLinks": "href=\"([^\"]+)\"",
	      "mediaUrl": "file\\s*:\\s*\"([^\"]+)\"",
	      "title": "<h1>([^<]+)</h1>",
	      "subPath": "$1/$2",
	      "ext": ".m4a"
	    }
	  ]
	}`, host, host)
}

func TestDSLExtractorMatchAndRouting(t *testing.T) {
	useExtractorsFile(t, demoSpec(`https://demo\\.example`))

	if ext := findMediaExtractor("https://demo.example/u/alice/first-track"); ext == nil || ext.Name() != "demo" {
		t.Errorf("item URL did not route to the demo extractor (got %v)", ext)
	}
	if ext := findMediaExtractor("  https://demo.example/u/alice \n"); ext == nil {
		t.Error("listing URL with surrounding whitespace did not match")
	}
	if ext := findMediaExtractor("https://other.example/u/alice/first-track"); ext != nil {
		t.Errorf("unrelated host matched %q", ext.Name())
	}
	if ext := findMediaExtractor(""); ext != nil {
		t.Error("empty input matched an extractor")
	}
}

func TestDSLExtractorResolveAndExtract(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()

	mux.HandleFunc("/u/alice", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<ul>
		  <li><a href="%s/u/alice/first-track">one</a></li>
		  <li><a href="%s/u/alice/second-track">two</a></li>
		  <li><a href="%s/u/alice/first-track">dupe</a></li>
		  <li><a href="%s/about">not an item</a></li>
		</ul>`, srv.URL, srv.URL, srv.URL, srv.URL)
	})
	mux.HandleFunc("/u/alice/first-track", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `<h1>First Track &amp; Friends</h1>
		<script>player({ file : "%s/media/abc123.m4a" })</script>`, srv.URL)
	})

	useExtractorsFile(t, demoSpec(regexpQuoteURL(srv.URL)))
	ext := findMediaExtractor(srv.URL + "/u/alice")
	if ext == nil {
		t.Fatal("listing URL did not match the demo extractor")
	}

	items, err := ext.Resolve(context.Background(), srv.URL+"/u/alice")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	want := []string{srv.URL + "/u/alice/first-track", srv.URL + "/u/alice/second-track"}
	if len(items) != 2 || items[0] != want[0] || items[1] != want[1] {
		t.Fatalf("Resolve = %v, want %v", items, want)
	}

	item, err := ext.Extract(context.Background(), items[0])
	if err != nil {
		t.Fatalf("Extract: %v", err)
	}
	if wantURL := srv.URL + "/media/abc123.m4a"; item.MediaURL != wantURL {
		t.Errorf("MediaURL = %q, want %q", item.MediaURL, wantURL)
	}
	if item.Title != "First Track & Friends" {
		t.Errorf("Title = %q, want %q", item.Title, "First Track & Friends")
	}
	if item.SubPath != "alice/first-track" {
		t.Errorf("SubPath = %q, want %q", item.SubPath, "alice/first-track")
	}
	if item.Ext != ".m4a" {
		t.Errorf("Ext = %q, want %q", item.Ext, ".m4a")
	}
}

// regexpQuoteURL escapes a URL's regex metacharacters for embedding in the
// JSON spec's anchored patterns (JSON needs the backslash itself escaped).
func regexpQuoteURL(u string) string {
	var b strings.Builder
	for _, r := range u {
		switch r {
		case '.', '?', '+':
			b.WriteString(`\\`)
		}
		b.WriteRune(r)
	}
	return b.String()
}

func TestDSLExtractorReloadOnChange(t *testing.T) {
	p := useExtractorsFile(t, demoSpec(`https://demo\\.example`))

	if findMediaExtractor("https://demo.example/u/alice/track") == nil {
		t.Fatal("initial definitions did not load")
	}

	if err := os.WriteFile(p, []byte(demoSpec(`https://renamed\\.example`)), 0644); err != nil {
		t.Fatal(err)
	}
	if findMediaExtractor("https://renamed.example/u/alice/track") == nil {
		t.Error("changed definitions were not reloaded")
	}
	if findMediaExtractor("https://demo.example/u/alice/track") != nil {
		t.Error("stale definitions still active after reload")
	}
}

func TestDSLExtractorBadDefinitionsSkipped(t *testing.T) {
	useExtractorsFile(t, `{
	  "extractors": [
	    {"name": "broken", "itemUrl": "([", "mediaUrl": "x", "subPath": "$1"},
	    {"name": "half-listing", "itemUrl": "^https://a\\.example/(\\w+)$", "listingUrl": "^https://a\\.example/$", "mediaUrl": "x", "subPath": "$1"},
	    {"name": "ok", "itemUrl": "^https://ok\\.example/(\\w+)$", "mediaUrl": "file=\"([^\"]+)\"", "subPath": "$1"}
	  ]
	}`)

	if findMediaExtractor("https://ok.example/track") == nil {
		t.Error("valid definition was not loaded alongside broken ones")
	}
	if findMediaExtractor("https://a.example/track") != nil {
		t.Error("definition with listingUrl but no listingLinks should be skipped")
	}
}

func TestExpandCaptures(t *testing.T) {
	m := []string{"whole", "alice", "track-1"}
	tests := []struct{ tpl, want string }{
		{"$1/$2", "alice/track-1"},
		{"static", "static"},
		{"$2", "track-1"},
		{"$9", ""},        // out of range → empty
		{"a$", "a$"},      // trailing dollar is literal
		{"$0x", "$0x"},    // $0 is not a capture reference
	}
	for _, tc := range tests {
		if got := expandCaptures(tc.tpl, m); got != tc.want {
			t.Errorf("expandCaptures(%q) = %q, want %q", tc.tpl, got, tc.want)
		}
	}
}

func TestDSLExtractorSubPathTraversalRejected(t *testing.T) {
	mux := http.NewServeMux()
	srv := httptest.NewServer(mux)
	defer srv.Close()
	mux.HandleFunc("/item/x", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `file : "https://cdn.example/a.m4a"`)
	})

	useExtractorsFile(t, fmt.Sprintf(`{
	  "extractors": [
	    {
	      "name": "demo",
	      "itemUrl": "^%s/item/(\\w+)$",
	      "mediaUrl": "file\\s*:\\s*\"([^\"]+)\"",
	      "subPath": "../../$1"
	    }
	  ]
	}`, regexpQuoteURL(srv.URL)))

	ext := findMediaExtractor(srv.URL + "/item/x")
	if ext == nil {
		t.Fatal("item URL did not match")
	}
	if _, err := ext.Extract(context.Background(), srv.URL+"/item/x"); err == nil {
		t.Error("expected traversal subPath to be rejected")
	}
}
