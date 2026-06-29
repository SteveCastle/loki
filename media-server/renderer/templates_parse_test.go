package renderer

import "testing"

// TestTemplatesParse forces ParseFS over all embedded templates; a syntax error
// in any *.go.html (e.g. an unclosed action) makes initTemplates log.Fatalf,
// failing this test. Guards hand-edited config.go.html changes.
func TestTemplatesParse(t *testing.T) {
	tmpls := Templates()
	if tmpls == nil {
		t.Fatal("Templates() returned nil")
	}
	for _, name := range []string{"config", "jobs"} {
		if tmpls.Lookup(name) == nil {
			t.Errorf("expected template %q to be defined", name)
		}
	}
}
