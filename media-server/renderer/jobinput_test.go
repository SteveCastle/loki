package renderer

import (
	"encoding/base64"
	"reflect"
	"testing"
)

func b64(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }

func TestJobInputView_Query(t *testing.T) {
	in := "metadata --type description --apply all --query64=" + b64("tag:foo category:bar")
	v := jobInputView(in)
	if v.Kind != "query" {
		t.Fatalf("Kind = %q, want query", v.Kind)
	}
	if v.Query != "tag:foo category:bar" {
		t.Errorf("Query = %q, want decoded text", v.Query)
	}
	if v.Prefix != "metadata --type description --apply all" {
		t.Errorf("Prefix = %q, want command + flags without query64", v.Prefix)
	}
}

func TestJobInputView_EmbedQuery(t *testing.T) {
	v := jobInputView("embed --query64=" + b64("path:\"/a/b.jpg\""))
	if v.Kind != "query" || v.Query != `path:"/a/b.jpg"` || v.Prefix != "embed" {
		t.Errorf("got %+v", v)
	}
}

func TestJobInputView_InvalidBase64FallsBack(t *testing.T) {
	// "--query64=" + not-valid-base64 → not a query; and it has flags so not paths → raw.
	v := jobInputView("embed --query64=@@@notbase64@@@")
	if v.Kind != "raw" {
		t.Errorf("Kind = %q, want raw for undecodable query64", v.Kind)
	}
}

func TestJobInputView_Paths(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{`C:\Users\me\pic.jpg`, []string{`C:\Users\me\pic.jpg`}},
		{"/home/me/a.png", []string{"/home/me/a.png"}},
		{"/a/b.jpg\n/c/d.png", []string{"/a/b.jpg", "/c/d.png"}},
		{"/a/b.jpg, /c/d.png", []string{"/a/b.jpg", "/c/d.png"}},
		{`"C:/My Folder/clip.mp4"`, []string{`C:/My Folder/clip.mp4`}},
		{"holiday.jpeg", []string{"holiday.jpeg"}}, // bare filename w/ media ext
	}
	for _, c := range cases {
		v := jobInputView(c.in)
		if v.Kind != "paths" || !reflect.DeepEqual(v.Paths, c.want) {
			t.Errorf("jobInputView(%q) = %+v, want paths %v", c.in, v, c.want)
		}
	}
}

func TestJobInputView_Raw(t *testing.T) {
	for _, in := range []string{
		"cleanup",
		"metadata --type hash", // flags but no query64, not paths
		"",
	} {
		v := jobInputView(in)
		if v.Kind != "raw" {
			t.Errorf("jobInputView(%q).Kind = %q, want raw", in, v.Kind)
		}
	}
}
