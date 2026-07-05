package main

import (
	"encoding/json"
	"io"
	"mime"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestConfigSetMergesAndStripsRedacted(t *testing.T) {
	var posted map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/config" && r.Method == http.MethodGet:
			_, _ = w.Write([]byte(`{"dbPath":"C:/db.sqlite","ollamaModel":"old","jwtSecret":"<redacted>","roots":[{"label":"x"}]}`))
		case r.URL.Path == "/config" && r.Method == http.MethodPost:
			_ = json.NewDecoder(r.Body).Decode(&posted)
			_, _ = w.Write([]byte(`{}`))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	a, _, _ := appForServer(srv.URL)
	code := cmdConfigSet(a, []string{"--json", `{"ollamaModel":"llava"}`})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if posted["ollamaModel"] != "llava" || posted["dbPath"] != "C:/db.sqlite" {
		t.Errorf("posted = %v", posted)
	}
	if _, has := posted["jwtSecret"]; has {
		t.Error("redacted jwtSecret echoed back")
	}
	if _, has := posted["roots"]; has {
		t.Error("roots echoed back (would clobber credentials)")
	}
}

func TestUploadMultipart(t *testing.T) {
	dir := t.TempDir()
	f := filepath.Join(dir, "pic.jpg")
	if err := os.WriteFile(f, []byte("fake-image-bytes"), 0o600); err != nil {
		t.Fatal(err)
	}

	var gotFiles []string
	var gotDest string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mt, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if mt != "multipart/form-data" {
			t.Errorf("content type = %s", mt)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		for _, fh := range r.MultipartForm.File["files"] {
			gotFiles = append(gotFiles, fh.Filename)
			file, _ := fh.Open()
			b, _ := io.ReadAll(file)
			file.Close()
			if string(b) != "fake-image-bytes" {
				t.Errorf("file content = %q", b)
			}
		}
		gotDest = r.FormValue("destination")
		_, _ = w.Write([]byte(`{"success":true,"files":["/up/pic.jpg"]}`))
	}))
	defer srv.Close()

	a, out, _ := appForServer(srv.URL)
	code := cmdUpload(a, []string{f, "--dest", "D:/media/in"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if len(gotFiles) != 1 || gotFiles[0] != "pic.jpg" || gotDest != "D:/media/in" {
		t.Errorf("files = %v dest = %q", gotFiles, gotDest)
	}
	if !strings.Contains(out.String(), "success") {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestDepsDownloadWaitUntilInstalled(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.HasSuffix(r.URL.Path, "/download"):
			w.WriteHeader(http.StatusAccepted)
			_, _ = w.Write([]byte(`{"state":"downloading"}`))
		case r.URL.Path == "/api/deps/status":
			calls++
			state := "downloading"
			if calls >= 2 {
				state = "installed"
			}
			_, _ = w.Write([]byte(`[{"id":"m1","category":"model","name":"M","state":"` + state + `"}]`))
		default:
			t.Errorf("unexpected %s", r.URL.Path)
		}
	}))
	defer srv.Close()

	a, out, _ := appForServer(srv.URL)
	// Shrink the 2s poll by using a tiny timeout budget? The loop sleeps 2s
	// between polls; keep the test fast by making the first poll terminal.
	calls = 1 // second status call (first inside loop) reports installed
	code := cmdDepsDownload(a, []string{"m1", "--wait", "--timeout", "30s"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	if !strings.Contains(out.String(), `"installed"`) {
		t.Errorf("stdout = %s", out.String())
	}
}

func TestFSScan(t *testing.T) {
	srv, reqs := newRecordingServer(t, http.StatusOK, `{"count":3}`)
	a, _, _ := appForServer(srv.URL)
	code := cmdFSScan(a, []string{"D:/media", "--recursive"})
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	got := (*reqs)[0]
	if got.Path != "/api/fs/scan" || !strings.Contains(got.Body, `"recursive":true`) {
		t.Errorf("request = %+v", got)
	}
	_ = time.Second
}
