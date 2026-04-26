package tasks

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func writeSidecar(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0644); err != nil {
		t.Fatalf("write sidecar: %v", err)
	}
	return p
}

func TestExtractSidecarTags(t *testing.T) {
	dir := t.TempDir()

	cases := []struct {
		name string
		body string
		want []string
	}{
		{
			name: "tags field present (gelbooru/rule34 style)",
			body: `{"tags": "1girl blonde_hair smile"}`,
			want: []string{"1girl", "blonde_hair", "smile"},
		},
		{
			name: "tag_string fallback (danbooru style)",
			body: `{"tag_string": "solo cat ears"}`,
			want: []string{"solo", "cat", "ears"},
		},
		{
			name: "tags wins when both present",
			body: `{"tags": "alpha beta", "tag_string": "gamma delta"}`,
			want: []string{"alpha", "beta"},
		},
		{
			name: "neither field returns nil",
			body: `{"id": 42}`,
			want: nil,
		},
		{
			name: "empty tags string returns nil",
			body: `{"tags": "   "}`,
			want: nil,
		},
		{
			name: "whitespace splitting and dedupe",
			body: `{"tags": "  foo\tbar\n foo  baz "}`,
			want: []string{"foo", "bar", "baz"},
		},
		{
			name: "non-string tags field falls back to tag_string",
			body: `{"tags": ["foo", "bar"], "tag_string": "real tags here"}`,
			want: []string{"real", "tags", "here"},
		},
	}

	for i, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeSidecar(t, dir, filepath.Base(t.Name())+".json", tc.body)
			_ = i
			got, err := extractSidecarTags(path)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("got %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestExtractSidecarTags_MissingFile(t *testing.T) {
	if _, err := extractSidecarTags(filepath.Join(t.TempDir(), "nope.json")); err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestExtractSidecarTags_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := writeSidecar(t, dir, "bad.json", `{"tags": "foo"`)
	if _, err := extractSidecarTags(path); err == nil {
		t.Fatal("expected error for malformed JSON, got nil")
	}
}
