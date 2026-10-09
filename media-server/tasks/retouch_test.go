package tasks

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
)

// ---------------------------------------------------------------------------
// Fake loki-retouch / loki-reshoot.
//
// The test binary doubles as the fake CLI: it is copied into a temp dir as
// loki-retouch(.exe) / loki-reshoot(.exe), that dir is put first on PATH, and
// TestMain diverts to fakeAIMain when LOKI_FAKE_AI=1 is in the environment.
// The fake mimics the real contract: `step i/n` lines on stderr, an info line,
// the -o file written, its path printed on stdout, exit 0.
//
//	FAKE_AI_ARGS   file the fake appends its argv to (one JSON array per line)
//	FAKE_AI_SLEEP  milliseconds to sleep per step (cancellation tests)
//	FAKE_AI_FAIL   "1" = exit 1 without output
// ---------------------------------------------------------------------------

func TestMain(m *testing.M) {
	if os.Getenv("LOKI_FAKE_AI") == "1" {
		os.Exit(fakeAIMain(os.Args))
	}
	os.Exit(m.Run())
}

func fakeAIMain(argv []string) int {
	reshoot := strings.Contains(strings.ToLower(filepath.Base(argv[0])), "reshoot")
	args := argv[1:]
	if f := os.Getenv("FAKE_AI_ARGS"); f != "" {
		b, _ := json.Marshal(args)
		af, err := os.OpenFile(f, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
		if err == nil {
			af.Write(append(b, '\n'))
			af.Close()
		}
	}
	if os.Getenv("FAKE_AI_FAIL") == "1" {
		fmt.Fprintln(os.Stderr, "error: fake failure")
		return 1
	}
	out, steps := "", 20
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "-o":
			if i+1 < len(args) {
				out = args[i+1]
			}
		case "--steps":
			if i+1 < len(args) {
				steps, _ = strconv.Atoi(args[i+1])
			}
		}
	}
	if out == "" {
		fmt.Fprintln(os.Stderr, "error: no -o")
		return 2
	}
	sleep, _ := strconv.Atoi(os.Getenv("FAKE_AI_SLEEP"))
	fmt.Fprintln(os.Stderr, "fake info: loading models")
	start := 0
	if reshoot {
		start = 1
	}
	for i := start; i <= steps; i++ {
		fmt.Fprintf(os.Stderr, "step %d/%d\n", i, steps)
		if sleep > 0 {
			time.Sleep(time.Duration(sleep) * time.Millisecond)
		}
	}
	if err := os.WriteFile(out, []byte("fake output"), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		return 1
	}
	fmt.Println(out)
	return 0
}

var (
	fakeBinOnce sync.Once
	fakeBinDir  string
	fakeBinErr  error
)

// installFakeAI puts the fake loki-retouch / loki-reshoot first on PATH and
// returns the file their argv gets logged to.
func installFakeAI(t *testing.T) string {
	t.Helper()
	fakeBinOnce.Do(func() {
		self, err := os.Executable()
		if err != nil {
			fakeBinErr = err
			return
		}
		dir, err := os.MkdirTemp("", "loki-fake-ai-")
		if err != nil {
			fakeBinErr = err
			return
		}
		ext := ""
		if runtime.GOOS == "windows" {
			ext = ".exe"
		}
		for _, name := range []string{"loki-retouch", "loki-reshoot"} {
			dst := filepath.Join(dir, name+ext)
			if err := os.Link(self, dst); err != nil {
				if err := copyFileForTest(self, dst); err != nil {
					fakeBinErr = err
					return
				}
			}
		}
		fakeBinDir = dir
	})
	if fakeBinErr != nil {
		t.Fatalf("install fake AI binaries: %v", fakeBinErr)
	}
	t.Setenv("PATH", fakeBinDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("LOKI_FAKE_AI", "1")
	argsLog := filepath.Join(t.TempDir(), "args.jsonl")
	t.Setenv("FAKE_AI_ARGS", argsLog)
	t.Setenv("FAKE_AI_SLEEP", "")
	t.Setenv("FAKE_AI_FAIL", "")
	return argsLog
}

func copyFileForTest(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func readFakeArgs(t *testing.T, path string) [][]string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fake args: %v", err)
	}
	var out [][]string
	for _, line := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if line == "" {
			continue
		}
		var a []string
		if err := json.Unmarshal([]byte(line), &a); err != nil {
			t.Fatal(err)
		}
		out = append(out, a)
	}
	return out
}

func argValue(args []string, flag string) (string, bool) {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1], true
		}
		if strings.HasPrefix(a, flag+"=") {
			return a[len(flag)+1:], true
		}
	}
	return "", false
}

func argValues(args []string, flag string) []string {
	var out []string
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			out = append(out, args[i+1])
		}
	}
	return out
}

func hasArg(args []string, flag string) bool {
	for _, a := range args {
		if a == flag {
			return true
		}
	}
	return false
}

func jobLog(q *jobqueue.Queue, id string) string {
	return strings.Join(q.GetJob(id).Stdout, "\n")
}

func mediaRowExists(t *testing.T, db *sql.DB, path string) bool {
	t.Helper()
	var n int
	if err := db.QueryRow(`SELECT COUNT(*) FROM media WHERE path = ?`, path).Scan(&n); err != nil {
		t.Fatal(err)
	}
	return n > 0
}

func writeFileForTest(t *testing.T, path, content string) string {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

// ---------------------------------------------------------------------------
// Pure helpers
// ---------------------------------------------------------------------------

func TestRetouchSuffix(t *testing.T) {
	cases := []struct {
		preset string
		video  bool
		at     float64
		want   string
	}{
		{"4kify", false, 0, "_4k"},
		{"4kify-phone", false, 0, "_phone"},
		{"upscale", false, 0, "_up"},
		{"restore", false, 0, "_restored"},
		{"", false, 0, "_edit"},
		{"", true, 1.5, "_t1.50_edit"},
		{"4kify", true, 12, "_t12.00_4k"},
	}
	for _, c := range cases {
		if got := retouchSuffix(c.preset, c.video, c.at); got != c.want {
			t.Errorf("retouchSuffix(%q,%v,%v) = %q; want %q", c.preset, c.video, c.at, got, c.want)
		}
	}
}

func TestRetouchParseStepLine(t *testing.T) {
	cases := []struct {
		line string
		i, n int
		ok   bool
	}{
		{"step 0/25", 0, 25, true},
		{"  step 12/25 (1.2 s/it)", 12, 25, true},
		{"step 25/25", 25, 25, true},
		{"loading model", 0, 0, false},
		{"restep 1/2", 0, 0, false},
		{"step 1/0", 0, 0, false},
		{"step x/5", 0, 0, false},
	}
	for _, c := range cases {
		i, n, ok := parseStepLine(c.line)
		if ok != c.ok || i != c.i || n != c.n {
			t.Errorf("parseStepLine(%q) = %d,%d,%v; want %d,%d,%v", c.line, i, n, ok, c.i, c.n, c.ok)
		}
	}
}

func TestRetouchParamsFromOptions(t *testing.T) {
	// Defaults straight from the option table.
	p := retouchParamsFromOptions(ParseOptions(&jobqueue.Job{}, retouchOptions))
	if p.Steps != 25 || p.Seed != -1 || p.Preset != "" || p.Combine || p.Snap || p.Seq || p.Time != 0 {
		t.Errorf("defaults = %+v", p)
	}

	// Clamping and parsing via ParseOptions' "--name value" syntax.
	j := &jobqueue.Job{Arguments: []string{
		"--preset", "Restore", "--steps", "200", "--seed", "42",
		"--refs", "C:\\a.png\r\n\n  \"C:\\b c.jpg\"  \n",
		"--longedge", "1536.7", "--scale", "-2", "--time", "-3",
		"--combine", "--snap", "true", "--seq", "false",
	}}
	p = retouchParamsFromOptions(ParseOptions(j, retouchOptions))
	if p.Preset != "restore" {
		t.Errorf("preset = %q", p.Preset)
	}
	if p.Steps != 80 {
		t.Errorf("steps clamp hi = %d; want 80", p.Steps)
	}
	if p.Seed != 42 {
		t.Errorf("seed = %d", p.Seed)
	}
	if len(p.Refs) != 2 || p.Refs[0] != `C:\a.png` || p.Refs[1] != `C:\b c.jpg` {
		t.Errorf("refs = %q", p.Refs)
	}
	if p.LongEdge != 1536 || p.Scale != 0 || p.Time != 0 {
		t.Errorf("longedge/scale/time = %d/%v/%v", p.LongEdge, p.Scale, p.Time)
	}
	if !p.Combine || !p.Snap || p.Seq {
		t.Errorf("bools combine=%v snap=%v seq=%v; want true,true,false", p.Combine, p.Snap, p.Seq)
	}

	p = retouchParamsFromOptions(map[string]any{"steps": 2.0, "preset": "none"})
	if p.Steps != 4 {
		t.Errorf("steps clamp lo = %d; want 4", p.Steps)
	}
	if p.Preset != "" {
		t.Errorf("preset none = %q; want empty", p.Preset)
	}
	if p.Seed != -1 {
		t.Errorf("missing seed = %d; want -1", p.Seed)
	}
}

func TestRetouchResolveSeed(t *testing.T) {
	if s, r := resolveSeed(7); s != 7 || r {
		t.Errorf("resolveSeed(7) = %d,%v", s, r)
	}
	if s, r := resolveSeed(0); s != 0 || r {
		t.Errorf("resolveSeed(0) = %d,%v", s, r)
	}
	for i := 0; i < 50; i++ {
		s, r := resolveSeed(-1)
		if !r || s < 0 || s >= 1<<31 {
			t.Fatalf("resolveSeed(-1) = %d,%v; want random in [0,2^31)", s, r)
		}
	}
}

func TestBuildRetouchArgs(t *testing.T) {
	p := retouchParams{
		Preset: "4kify", Prompt: "-make it night\nwith rain", Append: "keep the face",
		Scale: 1.5, Size: "same", LongEdge: 2048, Snap: true, Seq: true, Steps: 30, Seed: 9,
	}
	args := buildRetouchArgs(p, `C:\in\a.png`, []string{`C:\r1.png`, `C:\r2.png`}, `C:\in\a_4k.png`, "")
	want := []string{
		"--preset", "4kify",
		"--prompt=-make it night\nwith rain",
		"--append=keep the face",
		"--size", "same",
		"--scale", "1.5",
		"--long-edge", "2048",
		"--snap", "--seq",
		"--steps", "30",
		"--seed", "9",
		"--ref", `C:\r1.png`, "--ref", `C:\r2.png`,
		"-o", `C:\in\a_4k.png`, `C:\in\a.png`,
	}
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Errorf("args =\n%q\nwant\n%q", args, want)
	}

	// Minimal: no optional flags; prompt file replaces the inline prompt.
	p = retouchParams{Prompt: "x", Steps: 25, Seed: 0}
	args = buildRetouchArgs(p, "in.png", nil, "out.png", "prompt.txt")
	want = []string{"--prompt-file", "prompt.txt", "--steps", "25", "--seed", "0", "-o", "out.png", "in.png"}
	if strings.Join(args, "|") != strings.Join(want, "|") {
		t.Errorf("minimal args = %q; want %q", args, want)
	}
}

func TestRetouchLongPromptFile(t *testing.T) {
	f, err := writePromptFile("short")
	if err != nil || f != "" {
		t.Fatalf("short prompt -> file %q err %v; want none", f, err)
	}
	long := strings.Repeat("a detailed prompt line\n", 300) // > 4000 bytes
	f, err = writePromptFile(long)
	if err != nil || f == "" {
		t.Fatalf("long prompt -> file %q err %v", f, err)
	}
	defer os.Remove(f)
	b, _ := os.ReadFile(f)
	if string(b) != long {
		t.Errorf("prompt file content mismatch (%d vs %d bytes)", len(b), len(long))
	}
}

func TestFourKifyParams(t *testing.T) {
	j := &jobqueue.Job{Arguments: []string{"--phone", "--steps", "12", "--time", "3.5", "--prompt64", "IG1ha2UgaXQgPGltYWdlMT4g"}}
	p, err := fourKifyParams(ParseOptions(j, fourKifyOptions))
	if err != nil {
		t.Fatal(err)
	}
	if p.Preset != "4kify-phone" || p.Steps != 12 || p.Time != 3.5 || p.Prompt != "make it <image1>" || p.Seed != -1 {
		t.Errorf("4kify params = %+v", p)
	}
	p, _ = fourKifyParams(ParseOptions(&jobqueue.Job{}, fourKifyOptions))
	if p.Preset != "4kify" || p.Steps != 25 || p.Prompt != "" {
		t.Errorf("4kify defaults = %+v", p)
	}
	if _, err := fourKifyParams(map[string]any{"prompt64": "!!!"}); err == nil {
		t.Error("invalid base64 must error")
	}
}

func TestRetouchParseOptionsBoolLiteral(t *testing.T) {
	opts := []TaskOption{{Name: "native", Type: "bool", Default: true}, {Name: "x", Type: "string"}}
	r := ParseOptions(&jobqueue.Job{Arguments: []string{"--native", "false", "--x", "y"}}, opts)
	if r["native"] != false || r["x"] != "y" {
		t.Errorf("--native false -> %v, x=%v", r["native"], r["x"])
	}
	r = ParseOptions(&jobqueue.Job{Arguments: []string{"--native", "--x", "y"}}, opts)
	if r["native"] != true {
		t.Errorf("--native -> %v", r["native"])
	}
}

// ---------------------------------------------------------------------------
// End-to-end with the fake binary
// ---------------------------------------------------------------------------

func TestRetouchTaskTerminalFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "photo.png"), "png")
	// Pre-existing output: the new file must not clobber it.
	writeFileForTest(t, filepath.Join(dir, "photo_restored.png"), "old")

	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "restore", "--steps", "10", "--seed", "123"}, in)
	if err := retouchTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("retouch: %v\n%s", err, jobLog(q, j.ID))
	}
	want := filepath.Join(dir, "photo_restored_1.png")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("output %s missing: %v\n%s", want, err, jobLog(q, j.ID))
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "photo_restored.png")); string(b) != "old" {
		t.Error("existing output was overwritten")
	}
	got := q.GetJob(j.ID)
	if got.State != jobqueue.StateCompleted {
		t.Errorf("state = %v", got.State)
	}
	if got.ProgressTotal != 11 || got.ProgressDone != 11 {
		t.Errorf("progress = %d/%d; want 11/11", got.ProgressDone, got.ProgressTotal)
	}
	if len(got.OutputFiles) != 1 || got.OutputFiles[0] != want {
		t.Errorf("output files = %v", got.OutputFiles)
	}
	if !mediaRowExists(t, db, want) {
		t.Error("terminal output not inserted into media table")
	}
	log := jobLog(q, j.ID)
	for _, s := range []string{"retouch: seed 123", "retouch: fake info: loading models", "retouch: step 0/10", "retouch: step 5/10", "retouch: step 10/10", "retouch: output " + want} {
		if !strings.Contains(log, s) {
			t.Errorf("log missing %q:\n%s", s, log)
		}
	}
	if strings.Contains(log, "step 3/10") {
		t.Errorf("step lines not throttled:\n%s", log)
	}
	calls := readFakeArgs(t, argsLog)
	if len(calls) != 1 {
		t.Fatalf("calls = %d", len(calls))
	}
	a := calls[0]
	if v, _ := argValue(a, "--preset"); v != "restore" {
		t.Errorf("--preset = %q", v)
	}
	if v, _ := argValue(a, "-o"); v != want {
		t.Errorf("-o = %q", v)
	}
	if a[len(a)-1] != in {
		t.Errorf("input arg = %q", a[len(a)-1])
	}
}

func TestRetouchTaskRandomSeedAndPerItemFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	a := writeFileForTest(t, filepath.Join(dir, "a.jpg"), "x")
	b := writeFileForTest(t, filepath.Join(dir, "b.webp"), "x")
	txt := writeFileForTest(t, filepath.Join(dir, "notes.txt"), "x")
	long := strings.Repeat("make it brighter please. ", 200)

	q, j := newItemOpsJob(t, db, "retouch", []string{"--prompt", long, "--steps", "4"}, a+"\n"+txt+"\n"+b)
	if err := retouchTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("retouch: %v\n%s", err, jobLog(q, j.ID))
	}
	for _, f := range []string{"a_edit.png", "b_edit.png"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("%s missing", f)
		}
	}
	log := jobLog(q, j.ID)
	if !strings.Contains(log, "skipping unsupported file notes.txt") {
		t.Errorf("txt not skipped:\n%s", log)
	}
	if !strings.Contains(log, "(random)") {
		t.Errorf("random seed not logged:\n%s", log)
	}
	calls := readFakeArgs(t, argsLog)
	if len(calls) != 2 {
		t.Fatalf("calls = %d; want 2", len(calls))
	}
	s0, _ := argValue(calls[0], "--seed")
	s1, _ := argValue(calls[1], "--seed")
	n0, _ := strconv.Atoi(s0)
	n1, _ := strconv.Atoi(s1)
	if n1 != n0+1 {
		t.Errorf("seeds %s,%s; want consecutive", s0, s1)
	}
	if !strings.Contains(log, fmt.Sprintf("retouch: seed %d", n0)) {
		t.Errorf("seed %d not logged", n0)
	}
	if _, ok := argValue(calls[0], "--prompt-file"); !ok {
		t.Errorf("long prompt not passed via --prompt-file: %q", calls[0])
	}
	got := q.GetJob(j.ID)
	if got.ProgressTotal != 3*5 || got.ProgressDone != 3*5 {
		t.Errorf("progress = %d/%d; want 15/15", got.ProgressDone, got.ProgressTotal)
	}
}

func TestRetouchTaskCombineAndRefsFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	a := writeFileForTest(t, filepath.Join(dir, "me.png"), "x")
	b := writeFileForTest(t, filepath.Join(dir, "jacket.png"), "x")
	c := writeFileForTest(t, filepath.Join(dir, "clip.mp4"), "x")
	ref := writeFileForTest(t, filepath.Join(dir, "style.jpg"), "x")
	refVid := writeFileForTest(t, filepath.Join(dir, "style.mov"), "x")

	q, j := newItemOpsJob(t, db, "retouch", []string{
		"--prompt", "put the jacket of <image2> on <image1>", "--combine", "true",
		"--refs", ref + "\n" + refVid, "--steps", "4", "--seed", "5",
	}, a+"\n"+b+"\n"+c)
	if err := retouchTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("retouch: %v\n%s", err, jobLog(q, j.ID))
	}
	calls := readFakeArgs(t, argsLog)
	if len(calls) != 1 {
		t.Fatalf("combine made %d calls; want 1", len(calls))
	}
	refs := argValues(calls[0], "--ref")
	if strings.Join(refs, "|") != b+"|"+ref {
		t.Errorf("--ref = %q; want [%s %s]", refs, b, ref)
	}
	if calls[0][len(calls[0])-1] != a {
		t.Errorf("primary input = %q", calls[0][len(calls[0])-1])
	}
	if _, err := os.Stat(filepath.Join(dir, "me_edit.png")); err != nil {
		t.Error("me_edit.png missing")
	}
	log := jobLog(q, j.ID)
	if !strings.Contains(log, "not a still image") || !strings.Contains(log, "skipping non-image reference clip.mp4") {
		t.Errorf("skips not logged:\n%s", log)
	}
}

func TestRetouchTaskChainedFake(t *testing.T) {
	installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "pic.png"), "x")

	q := jobqueue.NewQueueWithDB(db)
	ids, err := q.AddWorkflow(jobqueue.Workflow{Tasks: []jobqueue.WorkflowTask{
		{ID: "step-retouch", Command: "retouch", Arguments: []string{"--prompt", "add snow", "--steps", "4"}, Input: in},
		{ID: "step-wait", Command: "wait", Dependencies: []string{"step-retouch"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	j, err := q.ClaimJob()
	if err != nil || j == nil || j.ID != ids[0] {
		t.Fatalf("claim: %v %v", j, err)
	}
	if err := retouchTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("retouch: %v\n%s", err, jobLog(q, j.ID))
	}
	want := filepath.Join(dir, ".loki-temp", j.ID, "pic_edit.png")
	if _, err := os.Stat(want); err != nil {
		t.Fatalf("chained output %s missing\n%s", want, jobLog(q, j.ID))
	}
	if _, err := os.Stat(filepath.Join(dir, "pic_edit.png")); err == nil {
		t.Error("chained step wrote beside the original")
	}
	if mediaRowExists(t, db, want) {
		t.Error("intermediate output must not be added to the library")
	}
	got := q.GetJob(j.ID)
	if len(got.OutputFiles) != 1 || got.OutputFiles[0] != want {
		t.Errorf("output files = %v", got.OutputFiles)
	}
}

func TestRetouchTaskMissingBinary(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	offlineInstalls(t)
	db := setupItemOpsDB(t)
	in := writeFileForTest(t, filepath.Join(t.TempDir(), "a.png"), "x")
	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "upscale"}, in)
	if err := retouchTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("expected error")
	}
	if q.GetJob(j.ID).State != jobqueue.StateError {
		t.Errorf("state = %v; want error", q.GetJob(j.ID).State)
	}
	if !strings.Contains(jobLog(q, j.ID), "retouch: could not install loki-retouch") {
		t.Errorf("log:\n%s", jobLog(q, j.ID))
	}

	// The legacy alias resolves the same binary.
	q, j = newItemOpsJob(t, db, "4kify", nil, in)
	if err := fourKifyTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("4kify: expected error")
	}
	if !strings.Contains(jobLog(q, j.ID), "could not install loki-retouch") {
		t.Errorf("4kify log:\n%s", jobLog(q, j.ID))
	}
}

func TestRetouchTaskNeedsPromptOrPreset(t *testing.T) {
	installFakeAI(t)
	db := setupItemOpsDB(t)
	in := writeFileForTest(t, filepath.Join(t.TempDir(), "a.png"), "x")
	q, j := newItemOpsJob(t, db, "retouch", nil, in)
	if err := retouchTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(jobLog(q, j.ID), "retouch: needs a prompt or a preset") {
		t.Errorf("log:\n%s", jobLog(q, j.ID))
	}
	if q.GetJob(j.ID).State != jobqueue.StateError {
		t.Errorf("state = %v", q.GetJob(j.ID).State)
	}
}

func TestRetouchTaskBinaryFailureFake(t *testing.T) {
	installFakeAI(t)
	t.Setenv("FAKE_AI_FAIL", "1")
	db := setupItemOpsDB(t)
	in := writeFileForTest(t, filepath.Join(t.TempDir(), "a.png"), "x")
	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "restore"}, in)
	if err := retouchTask(j, q, &sync.Mutex{}); err == nil {
		t.Fatal("expected error")
	}
	log := jobLog(q, j.ID)
	if !strings.Contains(log, "retouch: error: fake failure") || !strings.Contains(log, "failed for a.png") {
		t.Errorf("log:\n%s", log)
	}
}

func TestFourKifyTaskFake(t *testing.T) {
	argsLog := installFakeAI(t)
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "wall.jpg"), "x")
	q, j := newItemOpsJob(t, db, "4kify", []string{"--phone", "--steps", "6"}, in)
	if err := fourKifyTask(j, q, &sync.Mutex{}); err != nil {
		t.Fatalf("4kify: %v\n%s", err, jobLog(q, j.ID))
	}
	if _, err := os.Stat(filepath.Join(dir, "wall_phone.png")); err != nil {
		t.Errorf("wall_phone.png missing\n%s", jobLog(q, j.ID))
	}
	a := readFakeArgs(t, argsLog)[0]
	if v, _ := argValue(a, "--preset"); v != "4kify-phone" {
		t.Errorf("--preset = %q", v)
	}
	if v, _ := argValue(a, "--steps"); v != "6" {
		t.Errorf("--steps = %q", v)
	}
}

func TestRetouchTaskCancelFake(t *testing.T) {
	installFakeAI(t)
	t.Setenv("FAKE_AI_SLEEP", "1000") // 80 steps x 1 s: far longer than the test
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := writeFileForTest(t, filepath.Join(dir, "a.png"), "x")
	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "restore", "--steps", "80"}, in)

	done := make(chan error, 1)
	go func() { done <- retouchTask(j, q, &sync.Mutex{}) }()

	// Wait until the fake reports its first step, then cancel.
	deadline := time.Now().Add(20 * time.Second)
	for !strings.Contains(jobLog(q, j.ID), "step 0/80") {
		if time.Now().After(deadline) {
			t.Fatalf("fake never started:\n%s", jobLog(q, j.ID))
		}
		time.Sleep(20 * time.Millisecond)
	}
	start := time.Now()
	if err := q.CancelJob(j.ID); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err == nil {
			t.Error("canceled task returned nil")
		}
		if time.Since(start) > 15*time.Second {
			t.Errorf("cancel took %v", time.Since(start))
		}
	case <-time.After(30 * time.Second):
		t.Fatal("task did not stop after cancel")
	}
	if !strings.Contains(jobLog(q, j.ID), "retouch: task canceled") {
		t.Errorf("log:\n%s", jobLog(q, j.ID))
	}
	if _, err := os.Stat(filepath.Join(dir, "a_restored.png")); err == nil {
		t.Error("canceled run produced output")
	}
}

// ---------------------------------------------------------------------------
// Real smoke run (LOKI_AI_E2E=1): the actual loki-retouch on the GPU.
// ---------------------------------------------------------------------------

func TestRetouchRealE2E(t *testing.T) {
	if os.Getenv("LOKI_AI_E2E") != "1" {
		t.Skip("set LOKI_AI_E2E=1 to run the real loki-retouch (needs a GPU with ~20 GB free)")
	}
	db := setupItemOpsDB(t)
	dir := t.TempDir()
	in := filepath.Join(dir, "gradient.png")
	img := image.NewRGBA(image.Rect(0, 0, 256, 320))
	for y := 0; y < 320; y++ {
		for x := 0; x < 256; x++ {
			img.Set(x, y, color.RGBA{uint8(x), uint8(y * 255 / 320), uint8(255 - x), 255})
		}
	}
	f, err := os.Create(in)
	if err != nil {
		t.Fatal(err)
	}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(err)
	}
	f.Close()

	q, j := newItemOpsJob(t, db, "retouch", []string{"--preset", "restore", "--steps", "8", "--seed", "1"}, in)
	start := time.Now()
	err = retouchTask(j, q, &sync.Mutex{})
	t.Logf("job log:\n%s", jobLog(q, j.ID))
	if err != nil {
		t.Fatalf("retouch: %v", err)
	}
	out := filepath.Join(dir, "gradient_restored.png")
	st, err := os.Stat(out)
	if err != nil {
		t.Fatalf("output missing: %v", err)
	}
	of, _ := os.Open(out)
	cfg, _, derr := image.DecodeConfig(of)
	of.Close()
	if derr != nil {
		t.Fatalf("output not a PNG: %v", derr)
	}
	t.Logf("output %s: %dx%d, %d bytes, %v", out, cfg.Width, cfg.Height, st.Size(), time.Since(start))
	got := q.GetJob(j.ID)
	if got.ProgressDone != got.ProgressTotal {
		t.Errorf("progress = %d/%d", got.ProgressDone, got.ProgressTotal)
	}
}
