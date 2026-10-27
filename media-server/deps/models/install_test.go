package models

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fakeModel swaps in a one-model manifest served by srv and a private data dir.
func fakeModel(t *testing.T, files ...File) {
	t.Helper()
	SetDataDirForTest(t.TempDir())
	old := Manifest
	Manifest = []Model{{ID: "fake", Name: "Fake", Version: "1.0", Files: files}}
	t.Cleanup(func() { Manifest = old; SetDataDirForTest("") })
}

// A file left complete by an earlier, interrupted install is not fetched
// again; only the missing one is.
func TestInstallModel_SkipsFilesAlreadyPresent(t *testing.T) {
	a, b := []byte("file-a-bytes"), []byte("file-b-bytes")
	var hits sync.Map
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n, _ := hits.LoadOrStore(r.URL.Path, new(int32))
		atomic.AddInt32(n.(*int32), 1)
		if r.URL.Path == "/a" {
			_, _ = w.Write(a)
		} else {
			_, _ = w.Write(b)
		}
	}))
	defer srv.Close()
	fakeModel(t,
		File{URL: srv.URL + "/a", RelPath: "a.bin", SHA256: sha256Hex(a), SizeBytes: int64(len(a))},
		File{URL: srv.URL + "/b", RelPath: "b.bin", SHA256: sha256Hex(b), SizeBytes: int64(len(b))},
	)
	// a.bin is on disk (no marker: an install from before marks existed).
	if err := os.MkdirAll(ModelDir("fake"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ModelDir("fake"), "a.bin"), a, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := InstallModel(context.Background(), "fake", nil); err != nil {
		t.Fatal(err)
	}
	if _, fetched := hits.Load("/a"); fetched {
		t.Error("a.bin was downloaded again although it was installed and verified")
	}
	if n, ok := hits.Load("/b"); !ok || atomic.LoadInt32(n.(*int32)) != 1 {
		t.Error("b.bin should be downloaded exactly once")
	}
	if !IsInstalled("fake") {
		t.Error("model not installed")
	}
	if !strings.Contains(string(mustRead(t, filepath.Join(ModelDir("fake"), filesMarker))), "a.bin") {
		t.Error("files marker does not record a.bin")
	}

	// A corrupt copy of a file is not trusted.
	if err := os.Remove(filepath.Join(ModelDir("fake"), ".meta.json")); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(ModelDir("fake"), filesMarker)); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ModelDir("fake"), "a.bin"), bytes.Repeat([]byte("x"), len(a)), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := InstallModel(context.Background(), "fake", nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(mustRead(t, filepath.Join(ModelDir("fake"), "a.bin")), a) {
		t.Error("corrupt a.bin was kept")
	}
}

// Two concurrent installs of one model download it once: the second waits on
// the lock, re-checks, and finds it installed.
func TestInstallModel_ConcurrentCallersDownloadOnce(t *testing.T) {
	body := []byte("model-weights")
	var gets int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&gets, 1)
		<-release // hold the first download until both callers are in
		_, _ = w.Write(body)
	}))
	defer srv.Close()
	fakeModel(t, File{URL: srv.URL, RelPath: "m.bin", SHA256: sha256Hex(body)})

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range errs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = Tracker.InstallSync(context.Background(), "fake", nil)
		}(i)
	}
	// Wait until the first download is in flight, then let it finish.
	deadline := time.Now().Add(5 * time.Second)
	for atomic.LoadInt32(&gets) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if n := atomic.LoadInt32(&gets); n != 1 {
		t.Errorf("downloads = %d, want 1", n)
	}
	if snap, ok := Tracker.Snapshot("fake"); !ok || snap.State != StateInstalled {
		t.Errorf("tracker row = %+v, want installed", snap)
	}
}

// A connection dropped mid-file keeps the partial; the next attempt resumes
// it with a Range request instead of starting from zero.
func TestDownloadFile_ResumesAfterNetworkError(t *testing.T) {
	body := bytes.Repeat([]byte("0123456789"), 20000) // 200 KB
	var ranges []string
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		ranges = append(ranges, r.Header.Get("Range"))
		first := len(ranges) == 1
		mu.Unlock()
		if first {
			// Promise the whole file, send part of it, drop the connection.
			w.Header().Set("Content-Length", strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(body[:100000])
			w.(http.Flusher).Flush()
			conn, _, _ := w.(http.Hijacker).Hijack()
			_ = conn.Close()
			return
		}
		var start int
		if rg := r.Header.Get("Range"); strings.HasPrefix(rg, "bytes=") {
			start, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
			w.Header().Set("Content-Range", "bytes "+strconv.Itoa(start)+"-"+strconv.Itoa(len(body)-1)+"/"+strconv.Itoa(len(body)))
			w.WriteHeader(http.StatusPartialContent)
		}
		_, _ = w.Write(body[start:])
	}))
	defer srv.Close()

	dst := filepath.Join(t.TempDir(), "file.bin")
	if _, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex(body), nil); err == nil {
		t.Fatal("expected the dropped connection to fail the first attempt")
	}
	info, err := os.Stat(dst + ".partial")
	if err != nil || info.Size() == 0 {
		t.Fatalf("partial not kept after a network error: %v", err)
	}
	if _, err := downloadFileWithRetry(context.Background(), srv.URL, dst, sha256Hex(body), nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(mustRead(t, dst), body) {
		t.Error("resumed file differs")
	}
	mu.Lock()
	defer mu.Unlock()
	if len(ranges) != 2 || ranges[1] != "bytes="+strconv.FormatInt(info.Size(), 10)+"-" {
		t.Errorf("requests' Range headers = %q, want a resume from %d", ranges, info.Size())
	}
}

// A stale partial whose bytes don't match is discarded and the file is
// fetched once more from scratch.
func TestDownloadFileWithRetry_StalePartialStartsOver(t *testing.T) {
	body := []byte("0123456789ABCDEF")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Range") != "" {
			w.Header().Set("Content-Range", "bytes 8-15/16")
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write(body[8:])
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	dst := filepath.Join(t.TempDir(), "file.bin")
	if err := os.WriteFile(dst+".partial", []byte("garbage!"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := downloadFileWithRetry(context.Background(), srv.URL, dst, sha256Hex(body), nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(mustRead(t, dst), body) {
		t.Error("file differs")
	}
}

// A partial that already holds the whole file gets 416: it is verified and
// committed without appending the error body.
func TestDownloadFile_CompletePartialRangeNotSatisfiable(t *testing.T) {
	body := []byte("complete-file")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
		_, _ = w.Write([]byte("range not satisfiable"))
	}))
	defer srv.Close()
	dst := filepath.Join(t.TempDir(), "file.bin")
	if err := os.WriteFile(dst+".partial", body, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := downloadFile(context.Background(), srv.URL, dst, sha256Hex(body), nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(mustRead(t, dst), body) {
		t.Error("file differs")
	}
}

// A model with no files for this OS fails clearly instead of "installing"
// an empty folder.
func TestInstallModel_UnavailableOnThisOS(t *testing.T) {
	fakeModel(t, File{URL: "http://unused", RelPath: "x", SHA256: "00", OS: "plan9"})
	err := InstallModel(context.Background(), "fake", nil)
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err = %v, want ErrUnavailable", err)
	}
	if Available("fake") {
		t.Error("Available = true")
	}
}

// A caller waiting for another install of the same model gives up when its
// context ends.
func TestInstallModel_WaiterHonoursCancel(t *testing.T) {
	fakeModel(t, File{URL: "http://unused", RelPath: "x", SHA256: "00"})
	release, err := acquireInstall(context.Background(), "fake")
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if err := InstallModel(ctx, "fake", nil); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want deadline exceeded", err)
	}
}

func mustRead(t *testing.T, p string) []byte {
	t.Helper()
	b, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	return b
}
