package tasks

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/jobqueue"
)

// recordInstalls stands in for the downloader: it records each id and fails
// (offline), with a private data dir so nothing counts as installed.
func recordInstalls(t *testing.T) *[]string {
	t.Helper()
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	var mu sync.Mutex
	calls := &[]string{}
	prev := installAIDep
	installAIDep = func(_ context.Context, _ func(string, ...any), id string) error {
		mu.Lock()
		*calls = append(*calls, id)
		mu.Unlock()
		return errors.New("offline")
	}
	t.Cleanup(func() { installAIDep = prev })
	return calls
}

func TestEnsureModelsSkipsInstalled(t *testing.T) {
	calls := recordInstalls(t)
	prev := modelInstalled
	modelInstalled = func(string) bool { return true }
	t.Cleanup(func() { modelInstalled = prev })
	if err := EnsureModels(context.Background(), nil, "wd-eva02-large-tagger-v3", "yunet"); err != nil {
		t.Fatal(err)
	}
	if len(*calls) != 0 {
		t.Errorf("installed models were installed again: %v", *calls)
	}
}

func TestEnsureModelsDownloadsMissingWithFriendlyLog(t *testing.T) {
	calls := recordInstalls(t)
	var log []string
	err := EnsureModels(context.Background(), func(s string) { log = append(log, s) }, "wd-eva02-large-tagger-v3")
	if err == nil || !strings.Contains(err.Error(), "could not install") {
		t.Fatalf("err = %v", err)
	}
	if strings.Join(*calls, ",") != "wd-eva02-large-tagger-v3" {
		t.Errorf("installs = %v", *calls)
	}
	if len(log) == 0 || !strings.Contains(log[0], "Downloading the Auto-tagging model") || !strings.Contains(log[0], "1.26 GB") || !strings.Contains(log[0], "first use only") {
		t.Errorf("log = %q", log)
	}
}

func TestEnsureModelsUnknownAndUnavailable(t *testing.T) {
	recordInstalls(t)
	if err := EnsureModels(context.Background(), nil, "no-such-model"); err == nil {
		t.Error("unknown id: expected an error")
	}
	old := models.Manifest
	models.Manifest = append([]models.Model{{ID: "elsewhere", Name: "Elsewhere", Version: "1",
		Files: []models.File{{URL: "http://unused", RelPath: "x", OS: "plan9"}}}}, old...)
	t.Cleanup(func() { models.Manifest = old })
	if err := EnsureModels(context.Background(), nil, "elsewhere"); err == nil || !strings.Contains(err.Error(), "not available") {
		t.Errorf("platform-less model: err = %v", err)
	}
}

// Two jobs ensuring one model at once download it once, through the real
// downloader and the Dependencies tracker (the second waits on the install
// lock, re-checks, and finds it installed).
func TestEnsureModelsConcurrentDownloadOnce(t *testing.T) {
	body := []byte("weights")
	sum := sha256.Sum256(body)
	var gets int32
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		atomic.AddInt32(&gets, 1)
		<-release
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	old := models.Manifest
	models.Manifest = []models.Model{{ID: "ensure-fake", Name: "Fake", Feature: "Testing", Version: "1",
		Files: []models.File{{URL: srv.URL, RelPath: "m.bin", SHA256: hex.EncodeToString(sum[:]), SizeBytes: int64(len(body))}}}}
	t.Cleanup(func() { models.Manifest = old })
	prev := installAIDep
	installAIDep = installDep // the real one, against the test server
	t.Cleanup(func() { installAIDep = prev })

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for i := range errs {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			errs[i] = EnsureModels(context.Background(), nil, "ensure-fake")
		}(i)
	}
	for deadline := time.Now().Add(5 * time.Second); atomic.LoadInt32(&gets) == 0 && time.Now().Before(deadline); {
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
	if !modelInstalled("ensure-fake") {
		t.Error("not installed")
	}
}

// Interactive paths never block on a download: they start it in the
// background and say so.
func TestStartModelInstallsReportsDownloading(t *testing.T) {
	recordInstalls(t)
	var started []string
	prev := startModelInstall
	startModelInstall = func(id string) { started = append(started, id) }
	t.Cleanup(func() { startModelInstall = prev })

	m := TextSearchModel()
	_, _, err := TextQueryVector(context.Background(), "a cat")
	if !IsModelsDownloading(err) || !strings.Contains(err.Error(), "visual search model") {
		t.Fatalf("err = %v", err)
	}
	if strings.Join(started, ",") != m.ID {
		t.Errorf("started = %v, want %s", started, m.ID)
	}
}

// The embed and autotag ops download their model on first use instead of
// failing with "install it from Dependencies".
func TestPrepareOpsEnsureTheirModel(t *testing.T) {
	for _, tc := range []struct {
		op   string
		prep func(*ItemRun) (*ItemProcessor, error)
		want string
	}{
		{"embed", prepareEmbedOp, ActiveEmbedModel().ID},
		{"autotag", prepareAutotagOp, ActiveTaggerModel().ID},
	} {
		t.Run(tc.op, func(t *testing.T) {
			calls := recordInstalls(t)
			db := setupItemOpsDB(t)
			q, j := newItemOpsJob(t, db, tc.op, nil, "")
			_, err := tc.prep(&ItemRun{Job: j, Queue: q, Workers: 1})
			if err == nil || !strings.Contains(err.Error(), "could not install") {
				t.Fatalf("err = %v", err)
			}
			if strings.Join(*calls, ",") != tc.want {
				t.Errorf("installs = %v, want %s", *calls, tc.want)
			}
		})
	}
}

// In a combined job, an op that can't be prepared is dropped with a warning:
// the others still run, and the job ends failed (not complete).
func TestRunItemOpsDropsOpThatFailsPrepare(t *testing.T) {
	db := setupItemOpsDB(t)
	paths := writeTempMedia(t, db, 3)
	ok := registerTestOp(t, "test-op-ok")
	registerTestOp(t, "test-op-broken", func(op *ItemOp) {
		op.Prepare = func(*ItemRun) (*ItemProcessor, error) { return nil, errors.New("model download failed") }
	})
	q, j := newItemOpsJob(t, db, "process", nil, strings.Join(paths, "\n"))
	err := runItemOps(j, q, []string{"test-op-broken", "test-op-ok"}, false)
	if err == nil || !strings.Contains(err.Error(), "test-op-broken") {
		t.Fatalf("err = %v", err)
	}
	if n := countStored(ok); n != len(paths) {
		t.Errorf("healthy op processed %d item(s), want %d", n, len(paths))
	}
	if st := q.GetJob(j.ID).State; st != jobqueue.StateError {
		t.Errorf("job state = %v, want error (an op never ran)", st)
	}
	if !strings.Contains(strings.Join(j.Stdout, "\n"), "test-op-broken: skipped") {
		t.Errorf("no skip warning in the job log: %q", j.Stdout)
	}
}
