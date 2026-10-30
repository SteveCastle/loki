package tasks

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/platform"
)

// aiToolSpec ties a loki-* engine to its dependency ids in deps/models.
type aiToolSpec struct {
	Bin   string // executable name, also the "tool" dependency id
	Model string // model dependency id
	// Extra are further model dependencies a run needs on top of Model (a
	// second model that shares Model's encoders / VAEs). All of them are
	// installed on demand and searched through LOKI_MODELS, in this order.
	Extra []string
}

// modelIDs is every model dependency of the spec: Model first, then Extra.
func (s aiToolSpec) modelIDs() []string {
	return append([]string{s.Model}, s.Extra...)
}

var (
	retouchTool = aiToolSpec{Bin: "loki-retouch", Model: "qwen-image-2.1"}
	reshootTool = aiToolSpec{Bin: "loki-reshoot", Model: "minimax-h3-ref2va"}
)

// reshootSpec is the engine spec for a reshoot model: the fast model
// (FastH3 Omni) adds its own diffusion model to MiniMax H3's text encoder
// and VAEs, so it needs both dependencies.
func reshootSpec(model, dit string) aiToolSpec {
	// a fast-recipe extra model (`dit`, Config.ReshootModels) needs no FastH3 download
	if model == reshootModelFast && dit == "" {
		s := reshootTool
		s.Extra = []string{"fasth3-omni"}
		return s
	}
	return reshootTool
}

// installAIDep installs one dependency on demand (EnsureModels' downloader);
// a variable so tests never touch the network.
var installAIDep = installDep

// aiToolRun is what a task needs to launch an engine: a local executable, or
// a lease on a remote GPU worker (engines.go) that runs the same command line.
type aiToolRun struct {
	Bin string
	Env []string // extra environment (LOKI_MODELS, PATH with the bundled ffmpeg)

	lease *engineLease // nil: a local run outside the dispatcher (strategy local)
	light bool         // a model-free call (--show-prompt): not queued on the worker
}

// Remote reports whether the engine runs on a worker of a pool.
func (t aiToolRun) Remote() bool { return t.lease != nil && !t.lease.local() }

// Where names the place the engine runs, for job logs.
func (t aiToolRun) Where() string {
	if !t.Remote() {
		return "this machine"
	}
	return t.lease.t.key
}

// Release frees the run's worker slot; call it when the task is done with
// the engine (a deferred call right after resolveAITool).
func (t aiToolRun) Release() { t.lease.Release() }

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
// Whenever the model dependencies are installed, LOKI_MODELS lists their
// folders so the engine and the Dependencies page share one copy of the
// weights.
//
// With worker pools configured (Config.Engines) the run may go to a remote
// worker instead: the lease is taken here and held until Release.
func resolveAITool(ctx context.Context, q *jobqueue.Queue, jobID, tag string, spec aiToolSpec) (aiToolRun, error) {
	run := aiToolRun{}
	logf := func(format string, a ...any) { q.PushJobStdout(jobID, tag+": "+fmt.Sprintf(format, a...)) }

	if !engineRunsLocalOnly(spec.Bin) {
		lease, err := acquireEngine(ctx, spec.Bin, func(s string) { logf("%s", s) }, nil)
		if err != nil {
			return run, err
		}
		if !lease.local() {
			logf("running on GPU worker %s", lease.t.key)
			return aiToolRun{Bin: spec.Bin, lease: lease}, nil
		}
		run.lease = lease
		r, err := resolveLocalAITool(ctx, logf, spec)
		if err != nil {
			lease.Release()
			return run, err
		}
		r.lease = lease
		return r, nil
	}
	return resolveLocalAITool(ctx, logf, spec)
}

func resolveLocalAITool(ctx context.Context, logf func(string, ...any), spec aiToolSpec) (aiToolRun, error) {
	run := aiToolRun{}

	if p, err := exec.LookPath(spec.Bin); err == nil {
		run.Bin = p
	} else {
		if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
			return run, fmt.Errorf("%s needs an NVIDIA GPU with CUDA and is only available on Windows and Linux", spec.Bin)
		}
		exe := spec.Bin + platform.BinaryExtension()
		p, err := deps.ModelPath(spec.Bin, exe)
		if err != nil {
			if ierr := EnsureModels(ctx, func(s string) { logf("%s", s) }, spec.Bin); ierr != nil {
				return run, fmt.Errorf("%w (%s can also be put on PATH)", ierr, spec.Bin)
			}
			if p, err = deps.ModelPath(spec.Bin, exe); err != nil {
				return run, fmt.Errorf("%s was installed but %s is missing", spec.Bin, exe)
			}
		}
		run.Bin = p

		if err := EnsureModels(ctx, func(s string) { logf("%s", s) }, spec.modelIDs()...); err != nil {
			return run, err
		}
	}

	run.Env = aiToolEnv(spec)
	return run, nil
}

// aiToolEnv is the extra environment every engine run gets.
func aiToolEnv(spec aiToolSpec) []string {
	var env []string
	// Share the dependency-managed weights with the engine when present: an
	// OS path list of every installed model folder, searched in order.
	var dirs []string
	for _, id := range spec.modelIDs() {
		if dirHasModel(id) {
			dirs = append(dirs, models.ModelDir(id))
		}
	}
	if len(dirs) > 0 {
		env = append(env, "LOKI_MODELS="+strings.Join(dirs, string(os.PathListSeparator)))
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
// With pools, an engine routed away from this machine (or missing here) is
// answered by a reachable worker node.
var locateAITool = func(spec aiToolSpec) (aiToolRun, bool) {
	if engineDisp.route(spec.Bin) == appconfig.EngineStrategyPool {
		if t, ok := remotePreviewTool(spec); ok {
			return t, true
		}
	}
	if p, err := exec.LookPath(spec.Bin); err == nil {
		return aiToolRun{Bin: p, Env: aiToolEnv(spec)}, true
	}
	if p, err := deps.ModelPath(spec.Bin, spec.Bin+platform.BinaryExtension()); err == nil {
		return aiToolRun{Bin: p, Env: aiToolEnv(spec)}, true
	}
	return remotePreviewTool(spec)
}

// dirHasModel reports whether a model dependency is completely installed.
func dirHasModel(id string) bool { return modelInstalled(id) }

// installDep installs one dependency synchronously, logging progress roughly
// every 5% (and never more often than every 2 s) to the job. It runs as a
// tracked install, so the Dependencies page shows it (see EnsureModels).
func installDep(ctx context.Context, logf func(string, ...any), id string) error {
	var lastPct int64 = -5
	var lastFile string
	var lastAt time.Time
	err := models.Tracker.InstallSync(ctx, id, func(file string, done, total int64) {
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

// helpCache caches engines' --help text, keyed by path plus size and mtime
// so a rebuilt or updated engine is probed again.
var (
	helpCacheMu sync.Mutex
	helpCache   = map[string]string{}
)

// probeAIToolHelp returns an engine's --help text; a variable so tests never
// launch a real engine.
var probeAIToolHelp = func(ctx context.Context, tool aiToolRun) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 20*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, tool.Bin, "--help")
	cmd.Env = append(os.Environ(), tool.Env...)
	platform.HideSubprocessWindow(cmd)
	out, err := cmd.Output()
	return string(out), err
}

// aiToolSupportsMeta reports whether the engine takes --meta (stamping a
// caller's recipe into its generation record). Engines built before it
// reject unknown flags, so the flag is only passed when --help lists it.
func aiToolSupportsMeta(ctx context.Context, tool aiToolRun) bool {
	return aiToolHelpHas(ctx, tool, "--meta")
}

// aiToolHelpHas reports whether the engine's --help lists flag (engines
// reject flags they do not know, so newer ones are only passed when listed).
// A failed probe reports false and is not cached, so it is retried next job.
func aiToolHelpHas(ctx context.Context, tool aiToolRun, flag string) bool {
	if tool.Remote() {
		help, err := remoteHelp(ctx, tool)
		return err == nil && helpListsFlag(help, flag)
	}
	key := tool.Bin
	if st, err := os.Stat(tool.Bin); err == nil {
		key = fmt.Sprintf("%s|%d|%d", tool.Bin, st.Size(), st.ModTime().UnixNano())
	}
	helpCacheMu.Lock()
	help, known := helpCache[key]
	helpCacheMu.Unlock()
	if !known {
		var err error
		if help, err = probeAIToolHelp(ctx, tool); err != nil {
			return false
		}
		helpCacheMu.Lock()
		helpCache[key] = help
		helpCacheMu.Unlock()
	}
	return helpListsFlag(help, flag)
}

// helpListsFlag reports whether help mentions flag as a whole option name
// (so "--model" does not match "--models-dir").
func helpListsFlag(help, flag string) bool {
	for i := 0; ; {
		k := strings.Index(help[i:], flag)
		if k < 0 {
			return false
		}
		end := i + k + len(flag)
		if end == len(help) {
			return true
		}
		c := help[end]
		if !(c == '-' || c == '_' || c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9') {
			return true
		}
		i = end
	}
}

// prepareMeta readies a task's recipe for the engine: it is dropped (with a
// log line) when the engine is too old to take --meta, and a long one goes
// through a temp file (*metaFile) like a long prompt. The returned func
// removes that file.
func prepareMeta(ctx context.Context, q *jobqueue.Queue, jobID, tag string, tool aiToolRun, meta, metaFile *string) (func(), error) {
	noop := func() {}
	if *meta == "" {
		return noop, nil
	}
	if !aiToolSupportsMeta(ctx, tool) {
		q.PushJobStdout(jobID, tag+": this engine build cannot stamp the Transform recipe (update it to save tasks from results); running without it")
		*meta = ""
		return noop, nil
	}
	f, err := writePromptFile(*meta)
	if err != nil || f == "" {
		return noop, err
	}
	*metaFile = f
	return func() { os.Remove(f) }, nil
}
