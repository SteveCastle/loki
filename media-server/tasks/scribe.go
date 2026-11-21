package tasks

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/platform"
)

// --- Text generation: loki-scribe ---
//
// Every image -> text and text -> text job (descriptions, LoRA captions)
// runs loki-scribe, the standalone engine for Qwen3.5/3.8 GGUF models. A
// caller asks for one answer (generateText); concurrent requests with the
// same model and prompt are coalesced into one engine run of up to
// Config.TextBatch images, which the engine decodes TextParallel at a time
// without reloading anything. A run goes where the engine dispatcher routes
// loki-scribe: to a resident `loki-scribe --worker` process on this machine
// (kept loaded between runs, released when idle or when an image/video
// engine needs the GPU), or to a worker of a GPU pool like every other
// loki-* engine.

// scribeEngine is the executable (and "tool" dependency id).
const scribeEngine = "loki-scribe"

// HostBucketScribe is the job-queue bucket of text-generation jobs.
const HostBucketScribe = "scribe"

// scribeDefaultModel is the managed default model's dependency id.
const scribeDefaultModel = "qwen3.8-27b"

// scribeSpec is loki-scribe and its weights as dependencies: the managed
// default model is only installed when no other model is configured (the
// engine finds or downloads a configured one itself).
func scribeSpec(model string) aiToolSpec {
	if strings.TrimSpace(model) == "" {
		return aiToolSpec{Bin: scribeEngine, Model: scribeDefaultModel}
	}
	return aiToolSpec{Bin: scribeEngine}
}

// InferenceHost is the concurrency bucket of every text-generation job.
func InferenceHost() string { return HostBucketScribe }

// InferenceHostIsLocal reports whether text generation may run on this
// machine's GPU (then its jobs also hold the shared local-compute slot).
func InferenceHostIsLocal() bool {
	return appconfig.ParseRoute(engineDisp.route(scribeEngine)).Local
}

// textRequest is one generation.
type textRequest struct {
	Prompt string
	System string
	// Image is a local image file (optional).
	Image string
	// Model overrides Config.TextModel (a GGUF path, HF publisher/repo or name).
	Model     string
	MaxTokens int
	// Temperature overrides the engine's sampling temperature (0 = its
	// default). Never 0 for real: greedy decoding makes Qwen3 models fall into
	// endless repetition.
	Temperature float64
	// NoRepeat turns on the engine's DRY loop guard (scribeDRYArgs).
	NoRepeat bool
	// MinImageTokens (optional) upscales an image smaller than this many
	// vision tokens before reading it (capped at Config.TextMaxImageTokens).
	MinImageTokens int
	// OnFinish (optional) gets the engine's finish reason: "stop", or
	// "length" when the answer ran into MaxTokens without ending.
	OnFinish func(finish string)
}

// textMaxOutputTokens caps generation for descriptions and captions: a few
// paragraphs at most, so a model that misses its stop token ends in seconds.
const textMaxOutputTokens = 1024

// textBatchWindow is how long the first request of a batch waits for others.
const textBatchWindow = 400 * time.Millisecond

type textResult struct {
	text string
	err  error
}

type textItem struct {
	req  textRequest
	ctx  context.Context
	logf func(string)
	done chan textResult
}

type textGroup struct {
	key   string
	items []*textItem
	timer *time.Timer
}

var textBatches = struct {
	sync.Mutex
	groups map[string]*textGroup
}{groups: map[string]*textGroup{}}

// generateText runs one request through the batcher and waits for its answer.
// logf (may be nil) gets the engine's progress lines for this request.
func generateText(ctx context.Context, logf func(string), req textRequest) (string, error) {
	if logf == nil {
		logf = func(string) {}
	}
	cfg := appconfig.Get()
	if strings.TrimSpace(req.Model) == "" {
		req.Model = cfg.TextModel
	}
	if req.MaxTokens <= 0 {
		req.MaxTokens = textMaxOutputTokens
	}
	it := &textItem{req: req, ctx: ctx, logf: logf, done: make(chan textResult, 1)}
	key := strings.Join([]string{req.Model, req.System, req.Prompt, strconv.Itoa(req.MaxTokens), strconv.FormatBool(req.Image != ""), strconv.FormatFloat(req.Temperature, 'f', -1, 64), strconv.FormatBool(req.NoRepeat)}, "\x00")
	batch := max(1, cfg.TextBatch)
	if req.Image == "" {
		batch = 1 // a text-only request is one engine run
	}
	textBatches.Lock()
	g := textBatches.groups[key]
	if g == nil {
		g = &textGroup{key: key}
		textBatches.groups[key] = g
		g.timer = time.AfterFunc(textBatchWindow, func() { flushTextGroup(g) })
	}
	g.items = append(g.items, it)
	full := len(g.items) >= batch
	textBatches.Unlock()
	if full {
		flushTextGroup(g)
	}
	select {
	case r := <-it.done:
		return r.text, r.err
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// flushTextGroup detaches a group (once) and runs it.
func flushTextGroup(g *textGroup) {
	textBatches.Lock()
	if textBatches.groups[g.key] != g {
		textBatches.Unlock()
		return
	}
	delete(textBatches.groups, g.key)
	g.timer.Stop()
	items := g.items
	textBatches.Unlock()
	go runTextBatch(items)
}

// scribeResult is one --json result line of loki-scribe.
type scribeResult struct {
	Index  int    `json:"index"`
	Input  string `json:"input"`
	Text   string `json:"text"`
	Finish string `json:"finish"`
	Error  string `json:"error"`
}

var scribeLogPrefix = regexp.MustCompile(`^\[(.+?)\] `)

func runTextBatch(items []*textItem) {
	// requests whose callers gave up are dropped before the run
	live := items[:0:0]
	for _, it := range items {
		if it.ctx.Err() != nil {
			it.done <- textResult{err: it.ctx.Err()}
			continue
		}
		live = append(live, it)
	}
	if len(live) == 0 {
		return
	}
	// the run stops when every caller has gone
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		for _, it := range live {
			select {
			case <-it.ctx.Done():
			case <-ctx.Done():
				return
			}
		}
		cancel()
	}()

	cfg := appconfig.Get()
	r := live[0].req
	maxImage := max(64, cfg.TextMaxImageTokens)
	args := []string{"--json", "--max-tokens", strconv.Itoa(r.MaxTokens), "--parallel", strconv.Itoa(max(1, cfg.TextParallel)), "--max-image-tokens", strconv.Itoa(maxImage)}
	if r.MinImageTokens > 0 {
		args = append(args, "--min-image-tokens", strconv.Itoa(min(r.MinImageTokens, maxImage)))
	}
	if r.Model != "" {
		args = append(args, "--model", r.Model)
	}
	if strings.TrimSpace(cfg.TextMMProj) != "" {
		args = append(args, "--mmproj", cfg.TextMMProj)
	}
	if r.System != "" {
		args = append(args, "-s", r.System)
	}
	if r.Temperature > 0 {
		args = append(args, "--temp", strconv.FormatFloat(r.Temperature, 'f', -1, 64))
	}
	if r.NoRepeat {
		args = append(args, scribeDRYArgs...)
	}
	args = append(args, "-p", r.Prompt)
	// log lines name their input as the engine saw it (a worker's own path),
	// so they are matched by file name
	byInput := map[string]*textItem{}
	for _, it := range live {
		if it.req.Image != "" {
			args = append(args, it.req.Image)
			byInput[scribeLogKey(it.req.Image)] = it
		}
	}
	var distinct []func(string)
	seen := map[string]bool{}
	for _, it := range live {
		k := fmt.Sprintf("%p", it.logf)
		if !seen[k] {
			seen[k] = true
			distinct = append(distinct, it.logf)
		}
	}
	onLog := func(line string) {
		line = strings.TrimRight(line, "\r\n")
		if strings.TrimSpace(line) == "" {
			return
		}
		// "[<input>] ..." lines belong to one request; the rest to every job of the run
		if m := scribeLogPrefix.FindStringSubmatch(line); m != nil {
			if it := byInput[scribeLogKey(m[1])]; it != nil {
				it.logf("[text] " + filepath.Base(it.req.Image) + ": " + strings.TrimPrefix(line, m[0]))
				return
			}
		}
		for _, f := range distinct {
			f("[text] " + line)
		}
	}
	out, err := runScribe(ctx, args, onLog)
	results := map[int]scribeResult{}
	for _, l := range out {
		var res scribeResult
		if json.Unmarshal([]byte(strings.TrimSpace(l)), &res) == nil && (res.Text != "" || res.Finish != "") {
			results[res.Index] = res
		}
	}
	for i, it := range live {
		res, ok := results[i]
		switch {
		case ok && res.Error != "":
			it.done <- textResult{err: errors.New(res.Error)}
		case ok:
			if res.Finish == "length" {
				// no end token: the text is cut off, or the model fell into a loop
				it.logf(fmt.Sprintf("[text] %s: stopped at the %d-token limit without finishing (raise the max length if the text is cut short)", filepath.Base(it.req.Image), it.req.MaxTokens))
			}
			if it.req.OnFinish != nil {
				it.req.OnFinish(res.Finish)
			}
			it.done <- textResult{text: res.Text}
		case err != nil:
			it.done <- textResult{err: err}
		default:
			it.done <- textResult{err: errors.New("loki-scribe returned no result for this request")}
		}
	}
}

// scribeLogKey is the file name of a path as it appears in an engine log
// line (either separator, possibly JSON-escaped).
func scribeLogKey(p string) string {
	p = strings.ReplaceAll(p, "\\\\", "/")
	p = strings.ReplaceAll(p, "\\", "/")
	if i := strings.LastIndex(p, "/"); i >= 0 {
		p = p[i+1:]
	}
	return strings.ToLower(p)
}

// runScribe runs one engine command line where the dispatcher routes
// loki-scribe and returns its stdout lines.
func runScribe(ctx context.Context, args []string, onLog func(string)) ([]string, error) {
	if engineRunsLocalOnly(scribeEngine) {
		return scribeLocal.run(ctx, args, onLog)
	}
	lease, err := acquireEngine(ctx, scribeEngine, onLog, nil)
	if err != nil {
		return nil, err
	}
	defer lease.Release()
	if lease.local() {
		return scribeLocal.run(ctx, args, onLog)
	}
	onLog("running on GPU worker " + lease.t.key)
	tool := aiToolRun{Bin: scribeEngine, lease: lease}
	return runRemote(ctx, tool, remoteScribeArgs(scribeSupportedArgs(ctx, tool, args, onLog)), false, onLog, onLog)
}

// scribeDRYArgs is the DRY loop guard for transcription (validated on a
// manga page that greedy decoding looped on: it finished cleanly at 400
// tokens). A repeat of the answer's own text longer than 16 tokens is
// penalized 0.8 * 1.75^(extra tokens), so a line can recur a few times but
// not run away; short recurring pieces (names, PANEL / SFX labels) are free.
var scribeDRYArgs = []string{"--dry-multiplier", "0.8", "--dry-base", "1.75", "--dry-allowed-length", "16"}

// scribeSupportedArgs drops the DRY flags when the engine that will run them
// predates them (an engine rejects flags it does not know; a pool worker
// can lag this machine's build).
func scribeSupportedArgs(ctx context.Context, tool aiToolRun, args []string, onLog func(string)) []string {
	if !slices.Contains(args, "--dry-multiplier") || aiToolHelpHas(ctx, tool, "--dry-multiplier") {
		return args
	}
	onLog("this loki-scribe has no loop guard (--dry-multiplier); running without it - update the engine")
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		if strings.HasPrefix(args[i], "--dry-") {
			i++ // and its value
			continue
		}
		out = append(out, args[i])
	}
	return out
}

// remoteScribeArgs keeps a worker from being sent this machine's model
// files: a --model / --mmproj that is a local file becomes the name the
// worker resolves on its own (LM Studio / Hugging Face layout
// publisher/repo/file.gguf when the path has it, else the file name).
func remoteScribeArgs(args []string) []string {
	out := append([]string(nil), args...)
	for i := 0; i+1 < len(out); i++ {
		if out[i] != "--model" && out[i] != "--mmproj" {
			continue
		}
		p := out[i+1]
		if st, err := os.Stat(p); err != nil || st.IsDir() {
			continue
		}
		parts := strings.Split(filepath.ToSlash(filepath.Clean(p)), "/")
		if len(parts) >= 4 && strings.EqualFold(parts[len(parts)-4], "models") {
			out[i+1] = strings.Join(parts[len(parts)-3:], "/")
		} else {
			out[i+1] = filepath.Base(p)
		}
	}
	return out
}

// scribeWarmModel is the configured text model as a worker resolves it.
func scribeWarmModel() string {
	m := strings.TrimSpace(appconfig.Get().TextModel)
	if m == "" {
		return ""
	}
	return remoteScribeArgs([]string{"--model", m})[1]
}

// scribeConcurrency is how many describe items should be in flight at once
// so the batcher can fill its runs: a batch per GPU slot that can take one.
func scribeConcurrency() int {
	batch := max(1, appconfig.Get().TextBatch)
	if engineRunsLocalOnly(scribeEngine) {
		return batch
	}
	cands, _ := engineDisp.candidates(scribeEngine)
	slots := 0
	for _, t := range cands {
		slots += t.slots()
	}
	return batch * max(1, slots)
}

// --- the resident local engine

// scribeIdle is how long the local engine stays loaded without work.
const scribeIdle = 90 * time.Second

type scribeProc struct {
	mu      sync.Mutex // one run at a time; held while starting or stopping
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	events  chan map[string]any
	exited  chan struct{}
	bin     string // the running engine binary (its --help gates newer flags)
	logMu   sync.Mutex
	onLog   func(string)
	logDone chan string
	used    atomic.Int64 // unix nanos of the last run's end
	reaper  sync.Once
	seq     atomic.Int64
}

var scribeLocal = &scribeProc{}

// gpuHeavyWait blocks while an image/video engine holds the GPU.
func gpuHeavyWait(ctx context.Context, onLog func(string)) error {
	said := false
	for gpuHeavy.Load() > 0 {
		if !said {
			onLog("waiting for the GPU (an image or video job is running)")
			said = true
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return nil
}

func (p *scribeProc) running() bool {
	if p.cmd == nil {
		return false
	}
	select {
	case <-p.exited:
		return false
	default:
		return true
	}
}

// start launches `loki-scribe --worker` (the caller holds p.mu).
func (p *scribeProc) start(ctx context.Context, onLog func(string)) error {
	tool, err := resolveLocalAITool(ctx, func(f string, a ...any) { onLog(fmt.Sprintf(f, a...)) }, scribeSpec(appconfig.Get().TextModel))
	if err != nil {
		return err
	}
	cmd := exec.Command(tool.Bin, "--worker")
	p.bin = tool.Bin
	cmd.Env = append(os.Environ(), tool.Env...)
	platform.HideSubprocessWindow(cmd)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("starting loki-scribe: %w", err)
	}
	p.cmd, p.stdin = cmd, stdin
	p.events = make(chan map[string]any, 64)
	p.exited = make(chan struct{})
	p.logDone = make(chan string, 16)
	events, exited, logDone := p.events, p.exited, p.logDone
	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 1<<20), 64<<20)
		for sc.Scan() {
			var ev map[string]any
			if json.Unmarshal(sc.Bytes(), &ev) == nil {
				events <- ev
			}
		}
		close(events)
	}()
	go func() {
		sc := bufio.NewScanner(stderr)
		sc.Buffer(make([]byte, 1<<20), 16<<20)
		sc.Split(scanLinesOrCRTasks)
		for sc.Scan() {
			line := sc.Text()
			if id, ok := strings.CutPrefix(line, workerSentinel); ok {
				select {
				case logDone <- strings.TrimSpace(id):
				default:
				}
				continue
			}
			p.logMu.Lock()
			fn := p.onLog
			p.logMu.Unlock()
			if fn != nil {
				fn(line)
			}
		}
	}()
	go func() {
		_ = cmd.Wait()
		close(exited)
	}()
	select {
	case ev, ok := <-events:
		if !ok || ev["event"] != "ready" {
			p.kill()
			return errors.New("loki-scribe did not start in worker mode")
		}
	case <-time.After(2 * time.Minute):
		p.kill()
		return errors.New("loki-scribe did not become ready")
	case <-ctx.Done():
		p.kill()
		return ctx.Err()
	}
	p.reaper.Do(func() { go p.reap() })
	log.Printf("loki-scribe: resident engine started")
	return nil
}

// kill ends the process (the caller holds p.mu).
func (p *scribeProc) kill() {
	if p.cmd == nil {
		return
	}
	_ = p.stdin.Close()
	select {
	case <-p.exited:
	case <-time.After(5 * time.Second):
		_ = p.cmd.Process.Kill()
		<-p.exited
	}
	p.cmd = nil
}

// reap releases the engine (and its VRAM) after scribeIdle without work.
func (p *scribeProc) reap() {
	for {
		time.Sleep(15 * time.Second)
		if !p.mu.TryLock() {
			continue // a run is in progress
		}
		if p.running() && time.Since(time.Unix(0, p.used.Load())) > scribeIdle {
			log.Printf("loki-scribe: idle; releasing the resident engine and its GPU memory")
			p.kill()
		}
		p.mu.Unlock()
	}
}

// close stops the resident engine, waiting for a run in progress to finish.
func (p *scribeProc) close() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.running() {
		log.Printf("loki-scribe: releasing the resident engine (the GPU is needed elsewhere)")
	}
	p.kill()
}

// run sends one command line and waits for its result lines.
func (p *scribeProc) run(ctx context.Context, args []string, onLog func(string)) ([]string, error) {
	if err := gpuHeavyWait(ctx, onLog); err != nil {
		return nil, err
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	defer func() { p.used.Store(time.Now().UnixNano()) }()
	if !p.running() {
		p.kill()
		onLog("loading the text model (loki-scribe)")
		if err := p.start(ctx, onLog); err != nil {
			return nil, err
		}
	}
	args = scribeSupportedArgs(ctx, aiToolRun{Bin: p.bin}, args, onLog)
	p.logMu.Lock()
	p.onLog = onLog
	p.logMu.Unlock()
	defer func() {
		p.logMu.Lock()
		p.onLog = nil
		p.logMu.Unlock()
	}()
	id := fmt.Sprintf("t%d", p.seq.Add(1))
	b, _ := json.Marshal(map[string]any{"id": id, "args": args})
	if _, err := p.stdin.Write(append(b, '\n')); err != nil {
		p.kill()
		return nil, fmt.Errorf("loki-scribe stopped: %w", err)
	}
	var out []string
	cancelled := false
	var killAt <-chan time.Time
	for {
		select {
		case ev, ok := <-p.events:
			if !ok {
				<-p.exited
				p.kill()
				return out, errors.New("loki-scribe exited during the run")
			}
			if ev["id"] != id {
				continue
			}
			switch ev["event"] {
			case "stdout":
				if l, ok := ev["line"].(string); ok {
					out = append(out, strings.Split(l, "\n")...)
				}
			case "done":
				select {
				case <-p.logDone:
				case <-time.After(2 * time.Second):
				}
				for len(p.logDone) > 0 {
					<-p.logDone
				}
				if okv, _ := ev["ok"].(bool); !okv {
					msg, _ := ev["error"].(string)
					if c, _ := ev["cancelled"].(bool); c {
						return out, context.Canceled
					}
					return out, fmt.Errorf("loki-scribe: %s", msg)
				}
				return out, nil
			}
		case <-ctx.Done():
			if !cancelled {
				cancelled = true
				c, _ := json.Marshal(map[string]string{"cancel": id})
				_, _ = p.stdin.Write(append(c, '\n'))
				killAt = time.After(30 * time.Second)
			}
		case <-killAt:
			p.kill()
			return out, ctx.Err()
		}
	}
}

// releaseScribeIdle stops the resident engine when it has nothing to do, so
// other local GPU work (transcription, ONNX models) gets its memory.
func releaseScribeIdle() {
	p := scribeLocal
	if !p.mu.TryLock() {
		return
	}
	defer p.mu.Unlock()
	if p.running() {
		log.Printf("loki-scribe: releasing the idle resident engine for other GPU work")
		p.kill()
	}
}

// workerSentinel closes each request's stderr in worker mode (CONVENTIONS.md).
const workerSentinel = "\x1eloki-worker done"

// scanLinesOrCRTasks splits on \n or \r (progress lines redraw with \r).
func scanLinesOrCRTasks(data []byte, atEOF bool) (int, []byte, error) {
	for i, b := range data {
		if b == '\n' || b == '\r' {
			return i + 1, data[:i], nil
		}
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}
