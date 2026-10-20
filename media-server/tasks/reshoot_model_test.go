package tasks

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/jobqueue"
)

// loki-reshoot runs two models: h3 (MiniMax H3, 20 steps, the engine's
// default) and fast (FastH3 Omni, distilled for 8 steps, `--model fast`).

func TestReshootModelStepsDefault(t *testing.T) {
	parse := func(args ...string) reshootParams {
		return reshootParamsFromOptions(ParseOptions(&jobqueue.Job{Arguments: args}, reshootOptions))
	}
	cases := []struct {
		args  []string
		model string
		steps int
	}{
		{nil, "h3", 20},
		{[]string{"--model", "h3"}, "h3", 20},
		{[]string{"--model", "fast"}, "fast", 8},
		{[]string{"--model", "FAST"}, "fast", 8},
		{[]string{"--model", "fast", "--steps", "12"}, "fast", 12},
		{[]string{"--model", "fast", "--steps", "0"}, "fast", 8},
		{[]string{"--model", "h3", "--steps", "32"}, "h3", 32},
		{[]string{"--model", "turbo"}, "h3", 20},
	}
	for _, c := range cases {
		p := parse(c.args...)
		if p.Model != c.model || p.Steps != c.steps {
			t.Errorf("%q: model %q steps %d; want %q %d", c.args, p.Model, p.Steps, c.model, c.steps)
		}
	}
	// Direct maps (no ParseOptions defaults) behave the same.
	if p := reshootParamsFromOptions(map[string]any{"model": "fast"}); p.Steps != 8 {
		t.Errorf("fast without steps: %d", p.Steps)
	}
}

func TestBuildReshootArgsModel(t *testing.T) {
	base := reshootParams{Model: "h3", Duration: 5, Native: true, Megapixels: 0.4, Steps: 20, Seed: 3, RefSize: "match", Shake: "subtle", Prompt: "x"}
	h3 := buildReshootArgs(base, []string{"a.png"}, nil, nil, "o.mp4", "")
	for _, a := range h3 {
		if a == "--model" {
			t.Errorf("h3 args must not carry --model (old engines reject it): %q", h3)
		}
	}
	fast := base
	fast.Model, fast.Steps = "fast", 8
	got := buildReshootArgs(fast, []string{"a.png"}, nil, nil, "o.mp4", "")
	if strings.Join(got[:2], " ") != "--model fast" {
		t.Errorf("fast args = %q", got)
	}
	if v, _ := argValue(got, "--steps"); v != "8" {
		t.Errorf("fast --steps = %q", v)
	}
	if strings.Join(got[2:], "|") != strings.Join(buildReshootArgs(func() reshootParams { p := base; p.Steps = 8; return p }(), []string{"a.png"}, nil, nil, "o.mp4", ""), "|") {
		t.Errorf("fast args differ beyond --model: %q", got)
	}
}

func TestReshootSpec(t *testing.T) {
	if s := reshootSpec("h3"); len(s.Extra) != 0 || s.Model != "minimax-h3-ref2va" || s.Bin != "loki-reshoot" {
		t.Errorf("h3 spec = %+v", s)
	}
	s := reshootSpec("fast")
	if got := strings.Join(s.modelIDs(), ","); got != "minimax-h3-ref2va,fasth3-omni" || s.Bin != "loki-reshoot" {
		t.Errorf("fast spec models = %s", got)
	}
	// The shared spec is never mutated.
	if len(reshootTool.Extra) != 0 {
		t.Errorf("reshootTool mutated: %+v", reshootTool)
	}
}

func TestFastH3Manifest(t *testing.T) {
	fast, ok := models.Lookup("fasth3-omni")
	if !ok {
		t.Fatal("fasth3-omni missing from the manifest")
	}
	if len(fast.Files) != 1 || fast.Files[0].RelPath != "fasth3_omniref_8step_pruned_int8_convrot.safetensors" {
		t.Errorf("files = %+v", fast.Files)
	}
	if fast.SizeBytes != fast.EffectiveSizeBytes() {
		t.Errorf("size_bytes %d != file sizes %d", fast.SizeBytes, fast.EffectiveSizeBytes())
	}
	h3, _ := models.Lookup("minimax-h3-ref2va")
	for _, f := range h3.Files {
		if f.RelPath == fast.Files[0].RelPath {
			t.Errorf("fast DiT collides with an h3 file name")
		}
	}
	if len(fast.Consumers) != 1 || fast.Consumers[0] != "reshoot" {
		t.Errorf("consumers = %v", fast.Consumers)
	}
}

func TestHelpListsFlag(t *testing.T) {
	cases := []struct {
		help, flag string
		want       bool
	}{
		{"  --model <MODEL>  h3 | fast", "--model", true},
		{"  --model=MODEL", "--model", true},
		{"--model", "--model", true},
		{"  --models-dir <DIR>", "--model", false},
		{"  --modelx", "--model", false},
		{"  --models-dir <DIR>\n  --model <M>", "--model", true},
		{"  --meta-file <FILE>", "--meta", false},
		{"  --meta <JSON>\n  --meta-file <FILE>", "--meta", true},
	}
	for _, c := range cases {
		if got := helpListsFlag(c.help, c.flag); got != c.want {
			t.Errorf("helpListsFlag(%q, %q) = %v", c.help, c.flag, got)
		}
	}
}

func TestReshootFastTaskFake(t *testing.T) {
	argsLog := installFakeAI(t)
	withHelp(t, "  --model <MODEL>  h3 | fast\n  --meta <JSON>\n", nil)
	db := setupItemOpsDB(t)
	img := writeFileForTest(t, filepath.Join(t.TempDir(), "hero.png"), "x")
	q, j := newItemOpsJob(t, db, "reshoot", []string{"--prompt", "x", "--model", "fast", "--seed", "1"}, img)
	if err := reshootTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("reshoot: %v\n%s", err, jobLog(q, j.ID))
	}
	a := readFakeArgs(t, argsLog)[0]
	if v, _ := argValue(a, "--model"); v != "fast" {
		t.Errorf("--model = %q in %q", v, a)
	}
	if v, _ := argValue(a, "--steps"); v != "8" {
		t.Errorf("--steps = %q", v)
	}
	if got := q.GetJob(j.ID); got.ProgressTotal != 10 || got.ProgressDone != 10 {
		t.Errorf("progress = %d/%d; want 10/10", got.ProgressDone, got.ProgressTotal)
	}
	if !strings.Contains(jobLog(q, j.ID), "model FastH3 Omni, 8 steps") {
		t.Errorf("log:\n%s", jobLog(q, j.ID))
	}
}

func TestReshootFastTaskOldEngineFails(t *testing.T) {
	argsLog := installFakeAI(t)
	withHelp(t, "Usage: loki-reshoot [OPTIONS]\n  --steps <N>\n  --meta <JSON>\n", nil)
	db := setupItemOpsDB(t)
	img := writeFileForTest(t, filepath.Join(t.TempDir(), "hero.png"), "x")
	q, j := newItemOpsJob(t, db, "reshoot", []string{"--prompt", "x", "--model", "fast"}, img)
	if err := reshootTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("expected an error for an engine without --model")
	}
	if !strings.Contains(jobLog(q, j.ID), "too old for the fast model") {
		t.Errorf("log:\n%s", jobLog(q, j.ID))
	}
	if q.GetJob(j.ID).State != jobqueue.StateError {
		t.Errorf("state = %v", q.GetJob(j.ID).State)
	}
	if _, err := os.Stat(argsLog); err == nil {
		t.Errorf("the engine was run: %q", readFakeArgs(t, argsLog))
	}

	// The same old engine still runs H3 jobs (no --help probe needed).
	q, j = newItemOpsJob(t, db, "reshoot", []string{"--prompt", "x"}, img)
	if err := reshootTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("h3 on an old engine: %v\n%s", err, jobLog(q, j.ID))
	}
	if a := readFakeArgs(t, argsLog)[0]; hasArg(a, "--model") {
		t.Errorf("h3 args carry --model: %q", a)
	}
}

func TestResolveAIToolFastInstallsBothModels(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		t.Skip("engines are Windows/Linux only")
	}
	t.Setenv("PATH", t.TempDir())
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	var calls []string
	fakeInstall(t, &calls)

	db := setupItemOpsDB(t)
	q, j := newItemOpsJob(t, db, "reshoot", nil, "")

	// H3 first: the fast model is not installed.
	run, err := resolveAITool(context.Background(), q, j.ID, "reshoot", reshootSpec("h3"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(calls, ","); got != "loki-reshoot,minimax-h3-ref2va" {
		t.Errorf("h3 installs = %s", got)
	}
	if !containsArg(run.Env, "LOKI_MODELS="+models.ModelDir("minimax-h3-ref2va")) {
		t.Errorf("h3 env = %v", run.Env)
	}

	// Fast: only its own DiT is added, and LOKI_MODELS lists both folders.
	calls = nil
	run, err = resolveAITool(context.Background(), q, j.ID, "reshoot", reshootSpec("fast"))
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(calls, ","); got != "fasth3-omni" {
		t.Errorf("fast installs = %s", got)
	}
	want := "LOKI_MODELS=" + models.ModelDir("minimax-h3-ref2va") + string(os.PathListSeparator) + models.ModelDir("fasth3-omni")
	if !containsArg(run.Env, want) {
		t.Errorf("fast env %v missing %s", run.Env, want)
	}
	if !strings.Contains(jobLog(q, j.ID), "FastH3 Omni") {
		t.Errorf("install not logged:\n%s", jobLog(q, j.ID))
	}

	// locateAITool (prompt preview) hands the engine the same list.
	if tool, ok := locateAITool(reshootSpec("fast")); !ok || !containsArg(tool.Env, want) {
		t.Errorf("locate: ok=%v env=%v", ok, tool.Env)
	}
}

func TestResolveAIToolFastFromScratch(t *testing.T) {
	if runtime.GOOS != "windows" && runtime.GOOS != "linux" {
		t.Skip("engines are Windows/Linux only")
	}
	t.Setenv("PATH", t.TempDir())
	models.SetDataDirForTest(t.TempDir())
	t.Cleanup(func() { models.SetDataDirForTest(""); models.RebuildState() })
	var calls []string
	fakeInstall(t, &calls)
	db := setupItemOpsDB(t)
	q, j := newItemOpsJob(t, db, "reshoot", nil, "")
	if _, err := resolveAITool(context.Background(), q, j.ID, "reshoot", reshootSpec("fast")); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(calls, ","); got != "loki-reshoot,minimax-h3-ref2va,fasth3-omni" {
		t.Errorf("installs = %s", got)
	}
}
