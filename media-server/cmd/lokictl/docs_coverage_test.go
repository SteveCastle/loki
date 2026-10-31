package main

import (
	"html"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// docsPage is the published lokictl reference (GitHub Pages, docs/server/).
var docsPage = filepath.Join("..", "..", "..", "docs", "server", "cli.html")

// TestDocsCoverEveryCommand keeps the published reference from drifting
// behind the CLI: every registered command, and every --flag its usage line
// names, must appear in docs/server/cli.html. Adding a command or flag means
// documenting it there.
func TestDocsCoverEveryCommand(t *testing.T) {
	raw, err := os.ReadFile(docsPage)
	if err != nil {
		t.Skipf("docs page not available (%v)", err)
	}
	// Compare against the page's text: tags stripped, entities decoded.
	text := html.UnescapeString(regexp.MustCompile(`<[^>]+>`).ReplaceAllString(string(raw), ""))

	flagRE := regexp.MustCompile(`--[a-z][a-z0-9-]*`)
	for _, c := range commands {
		name := c.group
		if c.name != "" {
			name += " " + c.name
		}
		if !strings.Contains(text, name) {
			t.Errorf("command %q is not documented in %s", name, docsPage)
			continue
		}
		for _, f := range flagRE.FindAllString(c.args, -1) {
			if !strings.Contains(text, f) {
				t.Errorf("flag %s of %q is not documented in %s", f, name, docsPage)
			}
		}
	}
}
