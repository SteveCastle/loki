package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
)

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

func TestDownloadFile_HappyPath(t *testing.T) {
	body := []byte("hello world")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	got, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex(body), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got != int64(len(body)) {
		t.Errorf("bytes=%d want %d", got, len(body))
	}
	b, _ := os.ReadFile(dst)
	if string(b) != string(body) {
		t.Errorf("content=%q want %q", b, body)
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf("partial still exists")
	}
}

func TestDownloadFile_ResumesFromPartial(t *testing.T) {
	body := []byte("0123456789ABCDEF")
	var rangeRequests int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			atomic.AddInt32(&rangeRequests, 1)
			// "bytes=8-" → serve last half
			w.Header().Set("Content-Range", "bytes 8-15/16")
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(body[8:])
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	if err := os.WriteFile(dst+".partial", body[:8], 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex(body), nil); err != nil {
		t.Fatal(err)
	}
	if atomic.LoadInt32(&rangeRequests) != 1 {
		t.Errorf("expected 1 Range request, got %d", rangeRequests)
	}
	b, _ := os.ReadFile(dst)
	if string(b) != string(body) {
		t.Errorf("got %q want %q", b, body)
	}
}

func TestDownloadFile_ChecksumMismatchDeletesPartial(t *testing.T) {
	body := []byte("payload")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	_, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex([]byte("wrong-payload")), nil)
	if err == nil {
		t.Fatal("expected checksum error")
	}
	if !strings.Contains(err.Error(), "checksum") {
		t.Errorf("err = %v, want checksum error", err)
	}
	for _, p := range []string{dst, dst + ".partial"} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Errorf("%s exists after mismatch", p)
		}
	}
}

func TestDownloadFile_CancelDeletesPartial(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000000")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(make([]byte, 64*1024))
		// Hang for the rest until client cancels.
		<-r.Context().Done()
	}))
	defer srv.Close()

	dir := t.TempDir()
	dst := filepath.Join(dir, "file.bin")
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		_, err := downloadFile(ctx, srv.URL, dst, sha256Hex([]byte("noop")), nil)
		errCh <- err
	}()
	cancel()
	if err := <-errCh; err == nil {
		t.Fatal("expected cancellation error")
	}
	if _, err := os.Stat(dst + ".partial"); !os.IsNotExist(err) {
		t.Errorf(".partial not cleaned up after cancel")
	}
}

func TestInstallModel_FailsForUnknownID(t *testing.T) {
	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })
	if err := InstallModel(context.Background(), "no-such-model", nil); err == nil {
		t.Fatal("expected error for unknown id")
	}
}

// Round-trip: install a fake one-file model and read the .meta.json back.
func TestInstallModel_WritesMetaAndFiles(t *testing.T) {
	body := []byte("fake-onnx-bytes")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dir := t.TempDir()
	SetDataDirForTest(dir)
	t.Cleanup(func() { SetDataDirForTest("") })

	// Substitute Manifest for this test.
	oldManifest := Manifest
	Manifest = []Model{
		{
			ID: "fake", Name: "Fake", Version: "1.0", SizeBytes: int64(len(body)),
			Files: []File{{URL: srv.URL, RelPath: "model.bin", SHA256: sha256Hex(body)}},
		},
	}
	defer func() { Manifest = oldManifest }()

	if err := InstallModel(context.Background(), "fake", nil); err != nil {
		t.Fatal(err)
	}
	meta, err := os.ReadFile(filepath.Join(ModelDir("fake"), ".meta.json"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(meta), `"version":"1.0"`) {
		t.Errorf("meta missing version: %s", meta)
	}
	got, _ := io.ReadAll(strings.NewReader(string(body)))
	b, _ := os.ReadFile(filepath.Join(ModelDir("fake"), "model.bin"))
	if string(b) != string(got) {
		t.Errorf("model.bin content mismatch")
	}
}
