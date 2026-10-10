package tasks

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"time"

	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

// aiToolSpec ties a loki-* engine to its dependency ids in deps/models.
type aiToolSpec struct {
	Bin   string // executable name, also the "tool" dependency id
	Model string // model dependency id
}

var (
	retouchTool = aiToolSpec{Bin: "loki-retouch", Model: "qwen-image-2.1"}
	reshootTool = aiToolSpec{Bin: "loki-reshoot", Model: "minimax-h3-ref2va"}
)

// installAIDep installs a dependency on demand; a variable so tests never touch the network.
var installAIDep = installDep

// aiToolRun is what a task needs to launch an engine.
type aiToolRun struct {
	Bin string
	Env []string // extra environment (LOKI_MODELS, PATH with the bundled ffmpeg)
}

// resolveAITool finds (and if necessary installs) an engine so the task just
// works:
//
//  1. A binary on PATH wins: it is the user's own build or install, and it
//     finds its models the way the CLI documents (next to itself, ./models,
//     $LOKI_MODELS), downloading them itself if needed. Nothing is installed.
//  2. Otherwise the dependency-managed copy is used, installed on first use
//     (a ~50 MB download) together with its model files (17 GB for retouch, 42 GB
//     for reshoot; resumable, logged to the job, cancelled with the job). The
//     Dependencies page and setup wizard can pre-fetch both so a first job does
//     not wait.
//
// Whenever the model dependency is installed, LOKI_MODELS points at it so the
// engine and the Dependencies page share one copy of the weights.
func resolveAITool(ctx context.Context, q *jobqueue.Queue, jobID, tag string, spec aiToolSpec) (aiToolRun, error) {
	run := aiToolRun{}
	logf := func(format string, a ...any) { q.PushJobStdout(jobID, tag+": "+fmt.Sprintf(format, a...)) }

	if p, err := exec.LookPath(spec.Bin); err == nil {
		run.Bin = p
	} else {
		if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
			return run, fmt.Errorf("%s needs an NVIDIA GPU with CUDA and is only available on Windows and Linux", spec.Bin)
		}
		exe := spec.Bin + platform.BinaryExtension()
		p, err := deps.ModelPath(spec.Bin, exe)
		if err != nil {
			logf("installing %s (one-time)", spec.Bin)
			if ierr := installAIDep(ctx, logf, spec.Bin); ierr != nil {
				return run, fmt.Errorf("could not install %s: %w (it can also be put on PATH)", spec.Bin, ierr)
			}
			if p, err = deps.ModelPath(spec.Bin, exe); err != nil {
				return run, fmt.Errorf("%s was installed but %s is missing", spec.Bin, exe)
			}
		}
		run.Bin = p

		if !dirHasModel(spec.Model) {
			m, _ := models.Lookup(spec.Model)
			logf("downloading the %s model files (%.0f GB, one-time; resumable, also available on the Dependencies page)", m.Name, float64(m.EffectiveSizeBytes())/1e9)
			if ierr := installAIDep(ctx, logf, spec.Model); ierr != nil {
				return run, fmt.Errorf("could not install %s: %w", m.Name, ierr)
			}
		}
	}

	run.Env = aiToolEnv(spec)
	return run, nil
}

// aiToolEnv is the extra environment every engine run gets.
func aiToolEnv(spec aiToolSpec) []string {
	var env []string
	// Share the dependency-managed weights with the engine when present.
	if dir := models.ModelDir(spec.Model); dirHasModel(spec.Model) {
		env = append(env, "LOKI_MODELS="+dir)
	}
	// The server bundles ffmpeg/ffprobe; hand them to engines that need them.
	if ff := deps.BundledOrEmpty("ffmpeg"); ff != "" {
		env = append(env, "PATH="+filepath.Dir(ff)+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	return env
}

// locateAITool finds an engine the way resolveAITool does but never installs
// anything: for quick, side-effect-free calls such as the prompt preview. A
// variable so tests can stand in a fake engine.
var locateAITool = func(spec aiToolSpec) (aiToolRun, bool) {
	if p, err := exec.LookPath(spec.Bin); err == nil {
		return aiToolRun{Bin: p, Env: aiToolEnv(spec)}, true
	}
	if p, err := deps.ModelPath(spec.Bin, spec.Bin+platform.BinaryExtension()); err == nil {
		return aiToolRun{Bin: p, Env: aiToolEnv(spec)}, true
	}
	return aiToolRun{}, false
}

// dirHasModel reports whether a model dependency is completely installed.
func dirHasModel(id string) bool {
	models.RebuildState()
	return models.Cached()[id] == models.StatusInstalled
}

// installDep installs one dependency synchronously, logging progress roughly
// every 5% (and never more often than every 2 s) to the job.
func installDep(ctx context.Context, logf func(string, ...any), id string) error {
	var lastPct int64 = -5
	var lastFile string
	var lastAt time.Time
	err := models.InstallModel(ctx, id, func(file string, done, total int64) {
		pct := int64(0)
		if total > 0 {
			pct = done * 100 / total
		}
		if file != lastFile || (pct >= lastPct+5 && time.Since(lastAt) > 2*time.Second) {
			logf("  %s %d%% (%.1f/%.1f GB)", file, pct, float64(done)/1e9, float64(total)/1e9)
			lastFile, lastPct, lastAt = file, pct, time.Now()
		}
	})
	if err != nil {
		return err
	}
	models.RebuildState()
	return nil
}
