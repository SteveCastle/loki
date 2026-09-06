package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// createCapture fakes /create and records the job input it received.
func createCapture(t *testing.T) (*httptest.Server, *string) {
	t.Helper()
	var input string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/create" {
			t.Errorf("unexpected path %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		input, _ = req["input"].(string)
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte(`{"id":"job-1"}`))
	}))
	t.Cleanup(srv.Close)
	return srv, &input
}

func TestThumbnailCleanupBuildsJobInput(t *testing.T) {
	cases := []struct {
		args []string
		want string
	}{
		{[]string{"--detach"}, "thumbnail-cleanup"},
		{[]string{"--dry-run", "--detach"}, "thumbnail-cleanup --dry-run"},
		// The directory goes last: the server treats the final token as the
		// job Input, and a path with spaces is quoted for its parser.
		{[]string{"--dir", "D:/my pics", "--dry-run", "--detach"}, `thumbnail-cleanup --dry-run --dir "D:/my pics"`},
		{[]string{"--dir=D:/pics", "--detach"}, "thumbnail-cleanup --dir D:/pics"},
		{[]string{"D:/pics", "--detach"}, "thumbnail-cleanup --dir D:/pics"},
		{[]string{"--db", "E:/a.sqlite", "--db=F:/b b.sqlite", "--no-discover", "--detach"},
			`thumbnail-cleanup --discover=false --db "E:/a.sqlite;F:/b b.sqlite"`},
		{[]string{"--manifest", "C:/tmp/t.tsv", "--dry-run", "--dir", "D:/pics", "--detach"},
			"thumbnail-cleanup --dry-run --manifest C:/tmp/t.tsv --dir D:/pics"},
	}
	for _, c := range cases {
		srv, input := createCapture(t)
		a, out, errOut := appForServer(srv.URL)
		if code := cmdMediaThumbnailCleanup(a, c.args); code != 0 {
			t.Errorf("%v: exit %d, stderr %s", c.args, code, errOut.String())
			continue
		}
		if *input != c.want {
			t.Errorf("%v: input = %q, want %q", c.args, *input, c.want)
		}
		if !strings.Contains(out.String(), "job-1") {
			t.Errorf("%v: stdout = %s", c.args, out.String())
		}
	}
}

func TestThumbnailCleanupRejectsBadArgs(t *testing.T) {
	for _, args := range [][]string{
		{"--bogus"},
		{"--dir"},
		{"D:/a", "D:/b"},
		{"--dir", `D:/has"quote`, "--detach"},
		{"--db"},
		{"--db", "E:/a.sqlite;F:/b.sqlite", "--detach"},
		{"--manifest", "", "--detach"},
	} {
		srv, _ := createCapture(t)
		a, _, _ := appForServer(srv.URL)
		if code := cmdMediaThumbnailCleanup(a, args); code != 2 {
			t.Errorf("%v: exit %d, want 2 (usage)", args, code)
		}
	}
}
