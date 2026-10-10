package tasks

import (
	"bytes"
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/mediaext"
	"github.com/stevecastle/shrike/platform"
)

// --- retouch (loki-retouch) ---
//
// Wraps the standalone `loki-retouch` CLI (Qwen Image 2.1 image editing:
// edit by prompt, upscale, restore, composite, wallpaper presets). Shaped like
// the ffmpeg media-altering tasks: one new `<name><suffix>.png` per input
// beside the original (or in the workflow temp dir when chained), registered
// as an output file so downstream steps and the library pick it up. Videos
// are handled by sampling one frame at --time seconds. The legacy `4kify`
// task (fourkify.go) is an alias onto the same runner.
//
// The binary is resolved from PATH; its models live in a models/ folder next
// to it. It uses ~20 GB of VRAM, so all diffusion tasks share one host bucket
// (HostBucketGPUDiffusion) and the machine-wide local-compute slot.

// aiLongPromptBytes is the prompt size above which the prompt goes through a
// temp file (--prompt-file) instead of the command line.
const aiLongPromptBytes = 4000

var retouchPresetChoices = []string{"", "4kify", "4kify-phone", "upscale", "restore"}

var retouchOptions = []TaskOption{
	{Name: "preset", Label: "Preset", Type: "enum", Choices: retouchPresetChoices, Default: "", Description: "4kify (3840x2160 wallpaper), 4kify-phone (1296x2800), upscale (faithful super-resolution), restore (denoise/deblock at the same size); empty = plain edit by prompt"},
	{Name: "prompt", Label: "Prompt", Type: "string", Default: "", Description: "Edit instruction (replaces the preset's prompt). <image1> is the input, <image2>... the references"},
	{Name: "append", Label: "Append to Prompt", Type: "string", Default: "", Description: "Text appended to the final prompt (extends a preset)"},
	{Name: "scale", Label: "Scale", Type: "number", Default: 0.0, Description: "Output size as a multiple of the input (2 = 2x each side); 0 = default"},
	{Name: "size", Label: "Size", Type: "string", Default: "", Description: "Output size WxH, or 'same' for the input's size; empty = default"},
	{Name: "longedge", Label: "Long Edge", Type: "number", Default: 0.0, Description: "Output long edge in pixels, ratio kept; 0 = default"},
	{Name: "snap", Label: "Snap to 16", Type: "bool", Default: false, Description: "Deliver the model's native multiple-of-16 size instead of resampling to the exact size"},
	{Name: "steps", Label: "Steps", Type: "number", Default: 25.0, Description: "Sampling steps (4-80; more is slower)"},
	{Name: "seed", Label: "Seed", Type: "number", Default: -1.0, Description: "Noise seed; -1 = random (logged)"},
	{Name: "refs", Label: "Reference Images", Type: "string", Default: "", Description: "Extra reference images (<image2>, <image3>...), one absolute path per line"},
	{Name: "combine", Label: "Combine Inputs", Type: "bool", Default: false, Description: "With several input images: the first is <image1>, the rest become references, one output"},
	{Name: "time", Label: "Video Time", Type: "number", Default: 0.0, Description: "Seconds into a video input to sample the frame from (images ignore it)"},
	{Name: "seq", Label: "Sequence Mode", Type: "bool", Default: false, Description: "Frames of a clip: stricter faithful prompt, framing pinned, one seed for all inputs"},
}

// retouchParams is the normalized (defaulted, clamped) option set of one
// retouch run. Seed < 0 means "pick one" (see resolveSeed).
type retouchParams struct {
	Preset   string
	Prompt   string
	Append   string
	Scale    float64
	Size     string
	LongEdge int
	Snap     bool
	Steps    int
	Seed     int64
	Refs     []string
	Combine  bool
	Time     float64
	Seq      bool
}

func optString(opts map[string]any, k string) string {
	s, _ := opts[k].(string)
	return s
}

func optFloat(opts map[string]any, k string) float64 {
	switch v := opts[k].(type) {
	case float64:
		return v
	case int:
		return float64(v)
	case int64:
		return float64(v)
	}
	return 0
}

func optBool(opts map[string]any, k string) bool {
	b, _ := opts[k].(bool)
	return b
}

func clampInt(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

func clampFloat(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}

// splitPathList splits a newline-separated path list (the `refs` option),
// trimming blanks and surrounding quotes.
func splitPathList(raw string) []string {
	return parseInputPaths(strings.ReplaceAll(raw, "\r", "\n"))
}

// retouchParamsFromOptions normalizes ParseOptions output for retouch.
func retouchParamsFromOptions(opts map[string]any) retouchParams {
	p := retouchParams{
		Preset:  strings.ToLower(strings.TrimSpace(optString(opts, "preset"))),
		Prompt:  strings.TrimSpace(optString(opts, "prompt")),
		Append:  strings.TrimSpace(optString(opts, "append")),
		Scale:   optFloat(opts, "scale"),
		Size:    strings.TrimSpace(optString(opts, "size")),
		Snap:    optBool(opts, "snap"),
		Combine: optBool(opts, "combine"),
		Seq:     optBool(opts, "seq"),
		Refs:    splitPathList(optString(opts, "refs")),
	}
	if p.Preset == "none" {
		p.Preset = ""
	}
	if p.Scale < 0 {
		p.Scale = 0
	}
	if le := optFloat(opts, "longedge"); le > 0 {
		p.LongEdge = int(le)
	}
	steps := int(optFloat(opts, "steps"))
	if steps == 0 {
		steps = 25
	}
	p.Steps = clampInt(steps, 4, 80)
	p.Seed = int64(optFloat(opts, "seed"))
	if _, ok := opts["seed"]; !ok {
		p.Seed = -1
	}
	p.Time = optFloat(opts, "time")
	if p.Time < 0 {
		p.Time = 0
	}
	return p
}

// resolveSeed returns seed unchanged when >= 0, otherwise a random seed in
// [0, 2^31). The bool reports whether it was randomly chosen.
func resolveSeed(seed int64) (int64, bool) {
	if seed >= 0 {
		return seed, false
	}
	return rand.Int64N(1 << 31), true
}

// retouchSuffix is the file-name suffix (before .png) for a retouch output.
func retouchSuffix(preset string, isVideo bool, t float64) string {
	var s string
	switch preset {
	case "4kify":
		s = "_4k"
	case "4kify-phone":
		s = "_phone"
	case "upscale":
		s = "_up"
	case "restore":
		s = "_restored"
	default:
		s = "_edit"
	}
	if isVideo {
		return fmt.Sprintf("_t%.2f%s", t, s)
	}
	return s
}

func formatFloatArg(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

// buildRetouchArgs builds the loki-retouch argument list for one job item.
// Free-text values use the `--flag=value` form so a value that starts with
// '-' can never be mistaken for a flag. When promptFile is non-empty the
// prompt is read from it (--prompt-file) instead of passed inline.
func buildRetouchArgs(p retouchParams, input string, refs []string, output, promptFile string) []string {
	var args []string
	if p.Preset != "" {
		args = append(args, "--preset", p.Preset)
	}
	if promptFile != "" {
		args = append(args, "--prompt-file", promptFile)
	} else if p.Prompt != "" {
		args = append(args, "--prompt="+p.Prompt)
	}
	if p.Append != "" {
		args = append(args, "--append="+p.Append)
	}
	if p.Size != "" {
		args = append(args, "--size", p.Size)
	}
	if p.Scale > 0 {
		args = append(args, "--scale", formatFloatArg(p.Scale))
	}
	if p.LongEdge > 0 {
		args = append(args, "--long-edge", strconv.Itoa(p.LongEdge))
	}
	if p.Snap {
		args = append(args, "--snap")
	}
	if p.Seq {
		args = append(args, "--seq")
	}
	args = append(args, "--steps", strconv.Itoa(p.Steps))
	if p.Seed >= 0 {
		args = append(args, "--seed", strconv.FormatInt(p.Seed, 10))
	}
	for _, r := range refs {
		args = append(args, "--ref", r)
	}
	args = append(args, "-o", output, input)
	return args
}

var stepLineRe = regexp.MustCompile(`^\s*step (\d+)/(\d+)`)

// parseStepLine recognizes the CLIs' non-TTY progress lines ("step i/n").
func parseStepLine(line string) (i, n int, ok bool) {
	m := stepLineRe.FindStringSubmatch(line)
	if m == nil {
		return 0, 0, false
	}
	i, err1 := strconv.Atoi(m[1])
	n, err2 := strconv.Atoi(m[2])
	if err1 != nil || err2 != nil || n <= 0 {
		return 0, 0, false
	}
	return i, n, true
}

// writePromptFile writes a long prompt to a temp file when it exceeds
// aiLongPromptBytes. Returns "" (no file) for short prompts. The caller
// removes the file.
func writePromptFile(prompt string) (string, error) {
	if len(prompt) <= aiLongPromptBytes {
		return "", nil
	}
	f, err := os.CreateTemp("", "loki-prompt-*.txt")
	if err != nil {
		return "", err
	}
	if _, err := f.WriteString(prompt); err != nil {
		f.Close()
		os.Remove(f.Name())
		return "", err
	}
	if err := f.Close(); err != nil {
		os.Remove(f.Name())
		return "", err
	}
	return f.Name(), nil
}

// lineWriter is an io.Writer that hands complete lines to fn. Used as a
// command's Stdout/Stderr so os/exec owns the copying goroutines (and
// Cmd.WaitDelay can bound them if a grandchild keeps a pipe open).
type lineWriter struct {
	mu  sync.Mutex
	buf []byte
	fn  func(string)
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.buf = append(w.buf, p...)
	for {
		i := bytes.IndexAny(w.buf, "\r\n")
		if i < 0 {
			break
		}
		line := string(w.buf[:i])
		w.buf = w.buf[i+1:]
		if line != "" {
			w.fn(line)
		}
	}
	return len(p), nil
}

func (w *lineWriter) Flush() {
	w.mu.Lock()
	defer w.mu.Unlock()
	if len(w.buf) > 0 {
		w.fn(string(w.buf))
		w.buf = nil
	}
}

// runAICLI runs one loki-* CLI invocation, streaming stderr into the job log
// (prefixed with tag, step lines throttled) and reporting step progress via
// onStep. Returns the stdout lines (the written path).
func runAICLI(ctx context.Context, q *jobqueue.Queue, jobID, tag string, tool aiToolRun, args []string, onStep func(i, n int)) ([]string, error) {
	cmd := exec.CommandContext(ctx, tool.Bin, args...)
	if len(tool.Env) > 0 {
		cmd.Env = append(os.Environ(), tool.Env...)
	}
	platform.HideSubprocessWindow(cmd)
	cmd.WaitDelay = 10 * time.Second

	var outLines []string
	var outMu sync.Mutex
	stdout := &lineWriter{fn: func(s string) {
		outMu.Lock()
		outLines = append(outLines, strings.TrimSpace(s))
		outMu.Unlock()
	}}
	firstStep := true
	stderr := &lineWriter{fn: func(s string) {
		if i, n, ok := parseStepLine(s); ok {
			if onStep != nil {
				onStep(i, n)
			}
			if firstStep || i%5 == 0 || i >= n {
				_ = q.PushJobStdout(jobID, tag+": "+strings.TrimSpace(s))
			}
			firstStep = false
			return
		}
		_ = q.PushJobStdout(jobID, tag+": "+s)
	}}
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	err := cmd.Run()
	stdout.Flush()
	stderr.Flush()
	return outLines, err
}

// aiOutputDir returns the directory a diffusion task writes into for the
// source abs: beside it when terminal (or outside a workflow), else the
// workflow temp dir (stripping any existing .loki-temp so chains stay flat).
func aiOutputDir(j *jobqueue.Job, terminal bool, abs string) (string, error) {
	if j.WorkflowID != "" && !terminal {
		dir := filepath.Join(stripLokiTemp(abs), ".loki-temp", j.ID)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", err
		}
		return dir, nil
	}
	return filepath.Dir(abs), nil
}

func absPath(p string) string {
	if a, err := filepath.Abs(p); err == nil {
		return filepath.FromSlash(a)
	}
	return p
}

func retouchTask(j *jobqueue.Job, q *jobqueue.Queue, mu *sync.Mutex) error {
	p := retouchParamsFromOptions(ParseOptions(j, retouchOptions))
	return runRetouchJob(j, q, "retouch", p)
}

// retouchItem is one loki-retouch invocation: a primary input and its
// per-item extra references (the `combine` inputs; `refs` are added later).
type retouchItem struct {
	src  string
	refs []string
}

// planRetouchRuns turns a job's files into loki-retouch invocations. Without
// combine every file is its own run; with combine (several files) the first
// is <image1> and the other still images ride along as references of ONE
// run. refs are the `refs` option's extra references (existing still images
// only), appended to every run after the item's own. Shared by the task and
// the prompt preview so both always agree on what the model sees. logf gets
// one line per skipped file.
func planRetouchRuns(files []string, p retouchParams, logf func(string)) (items []retouchItem, refs []string) {
	for _, r := range p.Refs {
		r = absPath(r)
		if !isImageExt(filepath.Ext(r)) {
			logf("skipping reference (not a still image): " + r)
			continue
		}
		if _, err := os.Stat(r); err != nil {
			logf("skipping missing reference: " + r)
			continue
		}
		refs = append(refs, r)
	}
	if p.Combine && len(files) > 1 {
		first := absPath(files[0])
		var extra []string
		for _, f := range files[1:] {
			f = absPath(f)
			if !isImageExt(filepath.Ext(f)) {
				logf("combine: skipping non-image reference " + filepath.Base(f))
				continue
			}
			extra = append(extra, f)
		}
		return []retouchItem{{src: first, refs: extra}}, refs
	}
	for _, f := range files {
		items = append(items, retouchItem{src: f})
	}
	return items, refs
}

// runRetouchJob is the shared per-file loop behind `retouch` and the legacy
// `4kify` alias. tag prefixes every log line.
func runRetouchJob(j *jobqueue.Job, q *jobqueue.Queue, tag string, p retouchParams) error {
	ctx := j.Ctx
	if ctx == nil {
		ctx = context.Background()
	}
	fail := func(msg string, err error) error {
		q.PushJobStdout(j.ID, tag+": "+msg)
		q.ErrorJob(j.ID)
		if err == nil {
			err = fmt.Errorf("%s: %s", tag, msg)
		}
		return err
	}

	if p.Prompt == "" && p.Preset == "" {
		return fail("needs a prompt or a preset", nil)
	}
	validPreset := false
	for _, c := range retouchPresetChoices {
		if p.Preset == c {
			validPreset = true
		}
	}
	if !validPreset {
		return fail("unknown preset "+strconv.Quote(p.Preset)+" (choose 4kify, 4kify-phone, upscale or restore)", nil)
	}

	bin, err := resolveAITool(ctx, q, j.ID, tag, retouchTool)
	if err != nil {
		return fail(err.Error(), err)
	}

	res, rerr := resolveJobItemsRaw(j, q)
	if rerr != nil {
		return fail("failed to resolve input: "+rerr.Error(), rerr)
	}
	if res.FromQuery {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: query: %s", tag, res.Query))
	}
	files := res.Paths

	items, refs := planRetouchRuns(files, p, func(msg string) { q.PushJobStdout(j.ID, tag+": "+msg) })

	if len(items) == 0 {
		q.PushJobStdout(j.ID, tag+": no files to process")
		q.CompleteJob(j.ID)
		return nil
	}

	seed, random := resolveSeed(p.Seed)
	if random {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: seed %d (random)", tag, seed))
	} else {
		q.PushJobStdout(j.ID, fmt.Sprintf("%s: seed %d", tag, seed))
	}

	promptFile, perr := writePromptFile(p.Prompt)
	if perr != nil {
		return fail("failed to write prompt file: "+perr.Error(), perr)
	}
	if promptFile != "" {
		defer os.Remove(promptFile)
	}

	units := p.Steps + 1
	totalUnits := len(items) * units
	_ = q.SetJobProgress(j.ID, 0, totalUnits)

	// Every job is a workflow, so "in a workflow" can't mean "a save step
	// follows". When nothing downstream consumes this job's output (a plain
	// context-menu run, or the last step of a chain), write the finished file
	// beside the original and, like the save step, announce it so the library
	// reloads and the cursor jumps to the first new file. The announcement
	// fires once at the end (also on a partial run) so a batch doesn't hop the
	// cursor per image. With a later step, output stays in .loki-temp for it.
	terminal := !q.HasDependents(j.ID)
	var created []string
	defer func() {
		if terminal {
			broadcastMediaCreated(created)
		}
	}()

	ran := 0 // items actually sent to the CLI (skips don't advance the seed)
	for idx, item := range items {
		select {
		case <-ctx.Done():
			return fail("task canceled", ctx.Err())
		default:
		}
		if q.PauseRequested(j.ID) {
			q.PushJobStdout(j.ID, fmt.Sprintf("%s: paused at %d/%d - resume to continue", tag, idx, len(items)))
			return jobqueue.ErrPaused
		}

		abs := absPath(item.src)
		base := filepath.Base(abs)
		ext := filepath.Ext(abs)
		name := strings.TrimSuffix(base, ext)

		isVideo := mediaext.IsVideo(abs)
		if !isImageExt(ext) && !isVideo {
			q.PushJobStdout(j.ID, tag+": skipping unsupported file "+base)
			_ = q.SetJobProgress(j.ID, (idx+1)*units, totalUnits)
			continue
		}

		outputDir, derr := aiOutputDir(j, terminal, abs)
		if derr != nil {
			return fail("failed to create temp dir: "+derr.Error(), derr)
		}
		output := filepath.Join(outputDir, name+retouchSuffix(p.Preset, isVideo, p.Time)+".png")
		if terminal {
			if _, err := os.Stat(output); err == nil {
				output = resolveConflict(output)
			}
		}

		input := abs
		frameDir := ""
		if isVideo {
			frame, ferr := sampleVideoFrame(ctx, abs, p.Time)
			if ferr != nil {
				return fail(fmt.Sprintf("failed to sample frame at %.2fs from %s: %v", p.Time, base, ferr), ferr)
			}
			frameDir = filepath.Dir(frame)
			input = frame
			q.PushJobStdout(j.ID, fmt.Sprintf("%s: sampled frame at %.2fs from %s", tag, p.Time, base))
		}

		ip := p
		ip.Seed = seed
		if !p.Seq {
			// Same convention as the CLI's own batch mode: seed+1 per input.
			ip.Seed = seed + int64(ran)
		}
		ran++
		itemRefs := append(append([]string{}, item.refs...), refs...)
		args := buildRetouchArgs(ip, input, itemRefs, output, promptFile)

		msg := fmt.Sprintf("%s: running on %s -> %s", tag, base, filepath.Base(output))
		if len(itemRefs) > 0 {
			msg += fmt.Sprintf(" (+%d reference(s))", len(itemRefs))
		}
		q.PushJobStdout(j.ID, msg)

		base0 := idx * units
		out, waitErr := runAICLI(ctx, q, j.ID, tag, bin, args, func(i, n int) {
			done := base0 + i*(units-1)/n
			if done > totalUnits-1 {
				done = totalUnits - 1
			}
			_ = q.SetJobProgress(j.ID, done, totalUnits)
		})
		if frameDir != "" {
			_ = os.RemoveAll(frameDir)
		}
		for _, l := range out {
			if l != "" {
				q.PushJobStdout(j.ID, tag+": output "+l)
			}
		}

		if ctx.Err() != nil {
			return fail("task canceled", ctx.Err())
		}
		if waitErr != nil {
			return fail("failed for "+base+": "+waitErr.Error(), waitErr)
		}
		info, statErr := os.Stat(output)
		if statErr != nil {
			return fail("no output produced for "+base, nil)
		}
		// A finished file the chain ends on belongs in the library right away:
		// without a media row it is invisible to path/tag queries until
		// someone re-ingests the folder. Intermediate (.loki-temp) output is
		// never added — the step that places it owns that.
		if terminal {
			if err := insertMediaRecord(q.Db, output, info.Size()); err != nil {
				q.PushJobStdout(j.ID, tag+": failed to add to library: "+err.Error())
			}
		}

		q.PushJobStdout(j.ID, tag+": completed for "+base)
		q.RegisterOutputFile(j.ID, output, abs)
		created = append(created, output)
		_ = q.SetJobProgress(j.ID, (idx+1)*units, totalUnits)
	}

	q.CompleteJob(j.ID)
	return nil
}
