package tasks

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/deps/models"
)

// offlineInstalls makes on-demand dependency installs fail instantly, with a
// private data dir, so a test never downloads anything.
func offlineInstalls(t *testing.T) {
	t.Helper()
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	prev := installAIDep
	installAIDep = func(context.Context, func(string, ...any), string) error { return errors.New("offline") }
	t.Cleanup(func() { installAIDep = prev })
}

// fakeInstall stands in for the downloader: it records the ids, drops the
// tool binary (a copy of the test executable) or the model files into the
// dependency dir, and stamps the model as installed.
func fakeInstall(t *testing.T, calls *[]string) {
	t.Helper()
	prev := installAIDep
	installAIDep = func(_ context.Context, _ func(string, ...any), id string) error {
		*calls = append(*calls, id)
		m, ok := models.Lookup(id)
		if !ok {
			return errors.New("unknown dependency " + id)
		}
		dir := models.ModelDir(id)
		for _, f := range m.EffectiveFiles() {
			dst := filepath.Join(dir, f.RelPath)
			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				return err
			}
			if m.EffectiveCategory() == "tool" {
				self, _ := os.Executable()
				if err := copyFileForTest(self, dst); err != nil {
					return err
				}
			} else if err := os.WriteFile(dst, []byte("x"), 0o644); err != nil {
				return err
			}
		}
		if err := os.WriteFile(filepath.Join(dir, ".meta.json"), []byte(`{"version":"`+m.Version+`"}`), 0o644); err != nil {
			return err
		}
		models.RebuildState()
		return nil
	}
	t.Cleanup(func() { installAIDep = prev })
}

func TestResolveAIToolInstallsOnDemand(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		t.Skip("engines are Windows/Linux only")
	}
	t.Setenv("PATH", t.TempDir())
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	var calls []string
	fakeInstall(t, &calls)

	db := setupItemOpsDB(t)
	q, j := newItemOpsJob(t, db, "retouch", nil, "")
	run, err := resolveAITool(context.Background(), q, j.ID, "retouch", retouchTool)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(calls, ","); got != "loki-retouch,qwen-image-2.1" {
		t.Errorf("installs = %s", got)
	}
	if !strings.HasPrefix(run.Bin, models.ModelDir("loki-retouch")) {
		t.Errorf("bin = %s, want inside the dependency dir", run.Bin)
	}
	want := "LOKI_MODELS=" + models.ModelDir("qwen-image-2.1")
	found := false
	for _, e := range run.Env {
		found = found || e == want
	}
	if !found {
		t.Errorf("env %v missing %s", run.Env, want)
	}

	// A second resolve is a pure lookup: nothing is installed again.
	calls = nil
	if _, err := resolveAITool(context.Background(), q, j.ID, "retouch", retouchTool); err != nil || len(calls) != 0 {
		t.Errorf("second resolve: err=%v installs=%v", err, calls)
	}
}

func TestResolveAIToolPathBinaryWins(t *testing.T) {
	installFakeAI(t)
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	prev := installAIDep
	installAIDep = func(_ context.Context, _ func(string, ...any), id string) error {
		t.Errorf("must not install %s when the binary is on PATH", id)
		return nil
	}
	t.Cleanup(func() { installAIDep = prev })

	db := setupItemOpsDB(t)
	q, j := newItemOpsJob(t, db, "reshoot", nil, "")
	run, err := resolveAITool(context.Background(), q, j.ID, "reshoot", reshootTool)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(run.Bin, "loki-reshoot") || strings.HasPrefix(run.Bin, models.ModelDir("loki-reshoot")) {
		t.Errorf("bin = %s", run.Bin)
	}
	for _, e := range run.Env {
		if strings.HasPrefix(e, "LOKI_MODELS=") {
			t.Errorf("LOKI_MODELS set although the model dependency is not installed: %s", e)
		}
	}
}

func TestAIDependencyManifest(t *testing.T) {
	for _, spec := range []aiToolSpec{retouchTool, reshootTool} {
		tool, ok := models.Lookup(spec.Bin)
		if !ok || tool.EffectiveCategory() != "tool" {
			t.Fatalf("%s: missing tool entry", spec.Bin)
		}
		for _, f := range tool.Files {
			if f.SHA256URL == "" || f.Archive != "zip" || !f.Exec || !strings.Contains(f.ArchiveMember, spec.Bin) {
				t.Errorf("%s %s: %+v", spec.Bin, f.OS, f)
			}
		}
		if _, ok := models.Lookup(spec.Model); !ok {
			t.Errorf("%s: missing model entry %s", spec.Bin, spec.Model)
		}
	}
}
