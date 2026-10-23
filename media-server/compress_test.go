package main

import (
	"compress/gzip"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"testing/fstest"
)

func serveVia(t *testing.T, h http.Handler, remote, accept string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, "/x", nil)
	req.RemoteAddr = remote
	if accept != "" {
		req.Header.Set("Accept-Encoding", accept)
	}
	rec := httptest.NewRecorder()
	withCompression(h).ServeHTTP(rec, req)
	return rec
}

func TestCompressionGzipsJSONForRemoteClients(t *testing.T) {
	body := strings.Repeat(`{"path":"/library/some/long/path.jpg"},`, 2000)
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Length", "999")
		io.WriteString(w, body)
	})
	rec := serveVia(t, h, "192.168.1.20:5555", "gzip, deflate, br")
	if rec.Header().Get("Content-Encoding") != "gzip" {
		t.Fatalf("not gzipped: %v", rec.Header())
	}
	if rec.Header().Get("Content-Length") != "" {
		t.Fatal("stale Content-Length kept on a gzipped body")
	}
	if rec.Body.Len() >= len(body)/4 {
		t.Fatalf("compressed %d -> %d", len(body), rec.Body.Len())
	}
	zr, err := gzip.NewReader(rec.Body)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(zr)
	if string(got) != body {
		t.Fatal("round-trip mismatch")
	}
}

func TestCompressionSkips(t *testing.T) {
	cases := []struct {
		name, remote, accept, ct string
		status                   int
	}{
		{"loopback", "127.0.0.1:1", "gzip", "application/json", 200},
		{"loopback v6", "[::1]:1", "gzip", "application/json", 200},
		{"no accept", "10.0.0.2:1", "", "application/json", 200},
		{"sse", "10.0.0.2:1", "gzip", "text/event-stream", 200},
		{"image", "10.0.0.2:1", "gzip", "image/jpeg", 200},
		{"video range", "10.0.0.2:1", "gzip", "video/mp4", 206},
	}
	for _, c := range cases {
		h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", c.ct)
			w.WriteHeader(c.status)
			io.WriteString(w, "payload")
		})
		rec := serveVia(t, h, c.remote, c.accept)
		if rec.Header().Get("Content-Encoding") != "" || rec.Body.String() != "payload" {
			t.Errorf("%s: should pass through untouched (enc=%q body=%q)", c.name,
				rec.Header().Get("Content-Encoding"), rec.Body.String())
		}
	}
}

// SSE through the middleware must still flush each event immediately.
func TestCompressionKeepsSSEFlushable(t *testing.T) {
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if _, ok := w.(http.Flusher); !ok {
			t.Fatal("wrapped writer lost http.Flusher")
		}
		io.WriteString(w, "data: 1\n\n")
		w.(http.Flusher).Flush()
	})
	rec := serveVia(t, h, "10.0.0.2:1", "gzip")
	if !rec.Flushed || rec.Body.String() != "data: 1\n\n" {
		t.Fatalf("flushed=%v body=%q", rec.Flushed, rec.Body.String())
	}
}

// Templates that never set a Content-Type are sniffed, then compressed.
func TestCompressionSniffsUntypedHTML(t *testing.T) {
	page := "<!doctype html><html><body>" + strings.Repeat("<p>row</p>", 500) + "</body></html>"
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, page) })
	rec := serveVia(t, h, "10.0.0.2:1", "gzip")
	if rec.Header().Get("Content-Encoding") != "gzip" {
		t.Fatalf("untyped HTML not compressed: %v", rec.Header())
	}
}

func TestSPAFilesRevalidateWith304(t *testing.T) {
	fsys := fstest.MapFS{"renderer.js": {Data: []byte("console.log('app')")}, "index.html": {Data: []byte("<html></html>")}}
	h := lokiSPAHandler(fsys)
	req := httptest.NewRequest(http.MethodGet, "/app/renderer.js", nil)
	rec := httptest.NewRecorder()
	h(rec, req)
	etag := rec.Header().Get("ETag")
	if rec.Code != 200 || etag == "" {
		t.Fatalf("first load: code=%d etag=%q", rec.Code, etag)
	}
	req = httptest.NewRequest(http.MethodGet, "/app/renderer.js", nil)
	req.Header.Set("If-None-Match", etag)
	rec = httptest.NewRecorder()
	h(rec, req)
	if rec.Code != http.StatusNotModified || rec.Body.Len() != 0 {
		t.Fatalf("revalidation: code=%d body=%d bytes", rec.Code, rec.Body.Len())
	}
}
