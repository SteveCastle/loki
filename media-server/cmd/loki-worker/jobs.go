package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

// job is one engine run on this worker.
type job struct {
	mu     sync.Mutex
	st     workerapi.JobStatus
	spec   workerapi.JobSpec
	dir    string
	logs   []string
	cancel context.CancelFunc
	ctx    context.Context
	done   chan struct{}
	// onLog, when set, sees every log line as it arrives (the RunPod stream).
	onLog func(string)
}

func (j *job) status(logFrom int) workerapi.JobStatus {
	j.mu.Lock()
	defer j.mu.Unlock()
	st := j.st
	if logFrom < 0 {
		logFrom = 0
	}
	if logFrom > len(j.logs) {
		logFrom = len(j.logs)
	}
	// at most 2000 lines per poll; the rest comes with the next one
	end := min(len(j.logs), logFrom+2000)
	st.Logs = append([]string(nil), j.logs[logFrom:end]...)
	st.LogStart, st.LogNext = logFrom, end
	st.Stdout = append([]string(nil), j.st.Stdout...)
	st.Outputs = append([]workerapi.OutputFile(nil), j.st.Outputs...)
	return st
}

var stepRe = regexp.MustCompile(`^\s*(?:step|track) (\d+)/(\d+)`)

func (j *job) log(line string) {
	line = placeholder(line, j.dir, workerapi.JobDir)
	j.mu.Lock()
	j.logs = append(j.logs, line)
	if m := stepRe.FindStringSubmatch(line); m != nil {
		j.st.Step, _ = strconv.Atoi(m[1])
		j.st.Steps, _ = strconv.Atoi(m[2])
	}
	fn := j.onLog
	j.mu.Unlock()
	if fn != nil {
		fn(line)
	}
}

// slot is one GPU: its engines (started on demand) and the job running on it.
type slot struct {
	gpu     int
	mu      sync.Mutex // held while a job runs
	engines map[string]*engine
	busy    string
	totalMB int
}

// worker is the node: its engines, slots, queue and jobs.
type worker struct {
	cfg     config
	tools   map[string]string // engine -> executable
	slots   []*slot
	started time.Time

	mu      sync.Mutex
	jobs    map[string]*job
	queue   []*job
	wake    chan struct{}
	ready   bool
	helps   map[string]string
	gpuInfo []workerapi.GPU
	gpuAt   time.Time

	sessions *sessions
}

func newWorker(cfg config, tools map[string]string, gpus []workerapi.GPU) *worker {
	w := &worker{cfg: cfg, tools: tools, started: time.Now(), jobs: map[string]*job{}, wake: make(chan struct{}, 1), helps: map[string]string{}, gpuInfo: gpus}
	for _, g := range gpus {
		s := &slot{gpu: g.Index, engines: map[string]*engine{}, totalMB: g.TotalMB}
		for t, bin := range tools {
			s.engines[t] = newEngine(t, bin, g.Index, cfg.engineEnv)
		}
		w.slots = append(w.slots, s)
	}
	w.sessions = newSessions(w)
	return w
}

// coexist reports whether several engines may keep models on one GPU at once.
func (w *worker) coexist(s *slot) bool {
	switch w.cfg.coexist {
	case "always":
		return true
	case "never":
		return false
	}
	return s.totalMB >= 60000
}

// submit registers a job (inputs already staged) and queues it.
func (w *worker) submit(j *job) {
	w.mu.Lock()
	w.jobs[j.st.ID] = j
	if j.spec.Light {
		w.mu.Unlock()
		go w.runLight(j)
		return
	}
	w.queue = append(w.queue, j)
	w.updateQueuePos()
	w.mu.Unlock()
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

func (w *worker) updateQueuePos() {
	for i, q := range w.queue {
		q.mu.Lock()
		q.st.QueuePos = i
		q.mu.Unlock()
	}
}

// dispatch hands queued jobs to free slots, preferring a slot whose engine
// already holds the job's models.
func (w *worker) dispatch(ctx context.Context) {
	for {
		w.mu.Lock()
		for len(w.queue) > 0 {
			j := w.queue[0]
			s := w.freeSlot(j.spec.Tool)
			if s == nil {
				break
			}
			w.queue = w.queue[1:]
			s.busy = j.st.ID
			w.updateQueuePos()
			go w.runOn(s, j)
		}
		w.mu.Unlock()
		select {
		case <-ctx.Done():
			return
		case <-w.wake:
		case <-time.After(2 * time.Second):
		}
	}
}

// freeSlot picks an idle slot for tool (the caller holds w.mu).
func (w *worker) freeSlot(tool string) *slot {
	var best *slot
	for _, s := range w.slots {
		if s.busy != "" {
			continue
		}
		if e := s.engines[tool]; e != nil && len(e.Resident()) > 0 {
			return s
		}
		if best == nil {
			best = s
		}
	}
	return best
}

func (w *worker) release(s *slot) {
	w.mu.Lock()
	s.busy = ""
	w.mu.Unlock()
	select {
	case w.wake <- struct{}{}:
	default:
	}
}

func (w *worker) finish(j *job, state workerapi.JobState, msg string) {
	j.mu.Lock()
	if !j.st.State.Finished() {
		j.st.State = state
		if msg != "" {
			j.st.Error = msg
		}
		j.st.Ended = time.Now()
	}
	j.mu.Unlock()
	select {
	case <-j.done:
	default:
		close(j.done)
	}
}

func (w *worker) runOn(s *slot, j *job) {
	defer w.release(s)
	s.mu.Lock()
	defer s.mu.Unlock()
	if j.ctx.Err() != nil {
		w.finish(j, workerapi.JobCancelled, "cancelled before it started")
		return
	}
	e := s.engines[j.spec.Tool]
	if e == nil {
		w.finish(j, workerapi.JobFailed, j.spec.Tool+" is not installed on this worker")
		return
	}
	if !w.coexist(s) {
		// one engine's models at a time on a small GPU
		for t, other := range s.engines {
			if t != j.spec.Tool && other.State() != "stopped" {
				w.sessions.dropEngine(other)
				other.Stop()
			}
		}
	}
	j.mu.Lock()
	j.st.State, j.st.Started, j.st.GPU, j.st.QueuePos = workerapi.JobRunning, time.Now(), s.gpu, 0
	j.mu.Unlock()
	args := expandArgs(j.spec.Args, j.dir)
	req := map[string]any{"id": j.st.ID, "args": args}
	r, err := e.Call(j.ctx, req, j.log)
	if err != nil {
		w.finish(j, workerapi.JobFailed, err.Error())
		return
	}
	w.complete(j, r)
}

// runLight runs a model-free call (--show-prompt, --help) as its own process.
func (w *worker) runLight(j *job) {
	bin := w.tools[j.spec.Tool]
	if bin == "" {
		w.finish(j, workerapi.JobFailed, j.spec.Tool+" is not installed on this worker")
		return
	}
	ctx, cancel := context.WithTimeout(j.ctx, 2*time.Minute)
	defer cancel()
	j.mu.Lock()
	j.st.State, j.st.Started = workerapi.JobRunning, time.Now()
	j.mu.Unlock()
	cmd := exec.CommandContext(ctx, bin, expandArgs(j.spec.Args, j.dir)...)
	cmd.Env = append(append(os.Environ(), w.cfg.engineEnv...), envOf(j.spec.Env)...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	for _, l := range strings.Split(strings.ReplaceAll(stderr.String(), "\r", "\n"), "\n") {
		if l != "" {
			j.log(l)
		}
	}
	r := &reply{OK: err == nil, Secs: time.Since(j.st.Started).Seconds()}
	for _, l := range strings.Split(strings.TrimRight(stdout.String(), "\r\n"), "\n") {
		r.Stdout = append(r.Stdout, strings.TrimRight(l, "\r"))
	}
	if err != nil {
		r.Error = strings.TrimSpace(lastLine(stderr.String()))
		if r.Error == "" {
			r.Error = err.Error()
		}
	}
	w.complete(j, r)
}

func lastLine(s string) string {
	s = strings.TrimRight(s, "\r\n")
	if k := strings.LastIndexAny(s, "\r\n"); k >= 0 {
		return s[k+1:]
	}
	return s
}

// complete records an engine's reply: the result lines and the output files.
func (w *worker) complete(j *job, r *reply) {
	for i, l := range r.Stdout {
		r.Stdout[i] = placeholder(l, j.dir, workerapi.JobDir)
	}
	j.mu.Lock()
	j.st.Stdout, j.st.Secs, j.st.OOM = r.Stdout, r.Secs, r.OOM
	j.mu.Unlock()
	switch {
	case r.Cancelled || j.ctx.Err() != nil:
		w.finish(j, workerapi.JobCancelled, "cancelled")
		return
	case !r.OK:
		w.finish(j, workerapi.JobFailed, placeholder(r.Error, j.dir, workerapi.JobDir))
		return
	}
	outs, err := collectOutputs(j.dir, j.spec.Inputs)
	if err == nil {
		err = w.deliver(j, outs)
	}
	if err != nil {
		w.finish(j, workerapi.JobFailed, err.Error())
		return
	}
	j.mu.Lock()
	j.st.Outputs = outs
	j.mu.Unlock()
	w.finish(j, workerapi.JobDone, "")
}

// deliver pushes outputs to the job's sink when it has one (serverless).
func (w *worker) deliver(j *job, outs []workerapi.OutputFile) error {
	sink := j.spec.Outputs
	if sink == nil {
		return nil
	}
	if len(sink.PutURLs) > 0 {
		if len(outs) > len(sink.PutURLs) {
			return fmt.Errorf("%d output files but only %d upload slots", len(outs), len(sink.PutURLs))
		}
		for i := range outs {
			if err := putFile(j.ctx, sink.PutURLs[i], filepath.Join(j.dir, filepath.FromSlash(outs[i].Path))); err != nil {
				return fmt.Errorf("uploading %s: %w", outs[i].Path, err)
			}
			outs[i].Slot = i
		}
		return nil
	}
	if sink.Inline {
		var total int64
		for _, o := range outs {
			total += o.Size
		}
		limit := sink.MaxInline
		if limit <= 0 {
			limit = workerapi.InlineLimit
		}
		if total > limit {
			return fmt.Errorf("the outputs (%d bytes) are too large to return inline; give the pool a transfer bucket", total)
		}
		for i := range outs {
			b, err := os.ReadFile(filepath.Join(j.dir, filepath.FromSlash(outs[i].Path)))
			if err != nil {
				return err
			}
			outs[i].Data = b
		}
	}
	return nil
}

func putFile(ctx context.Context, url, path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, url, f)
	if err != nil {
		return err
	}
	req.ContentLength = st.Size()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("upload answered %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return nil
}

// collectOutputs lists what a run left in its job directory: everything
// under out/, and new files under in/ (an engine's default output beside
// its input).
func collectOutputs(dir string, inputs []workerapi.Input) ([]workerapi.OutputFile, error) {
	sent := map[string]bool{}
	for _, in := range inputs {
		sent[in.Path] = true
	}
	var outs []workerapi.OutputFile
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		if sent[rel] || !(strings.HasPrefix(rel, "out/") || strings.HasPrefix(rel, "in/")) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		outs = append(outs, workerapi.OutputFile{Path: rel, Size: info.Size(), Slot: -1})
		return nil
	})
	sort.Slice(outs, func(a, b int) bool { return outs[a].Path < outs[b].Path })
	return outs, err
}

// stage prepares a job directory: inputs fetched by URL or given inline
// (multipart uploads are already written by the handler) and output folders.
func stage(ctx context.Context, dir string, spec *workerapi.JobSpec) error {
	for _, d := range spec.Dirs {
		rel, ok := workerapi.CleanRel(d)
		if !ok {
			return fmt.Errorf("bad directory %q", d)
		}
		if err := os.MkdirAll(filepath.Join(dir, filepath.FromSlash(rel)), 0755); err != nil {
			return err
		}
	}
	for _, in := range spec.Inputs {
		rel, ok := workerapi.CleanRel(in.Path)
		if !ok {
			return fmt.Errorf("bad input path %q", in.Path)
		}
		dst := filepath.Join(dir, filepath.FromSlash(rel))
		switch {
		case in.Data != nil:
			if _, err := workerapi.WriteFileAtomic(dst, bytes.NewReader(in.Data)); err != nil {
				return err
			}
		case in.URL != "":
			if err := fetchURL(ctx, in.URL, dst); err != nil {
				return fmt.Errorf("fetching %s: %w", in.Path, err)
			}
		default:
			if _, err := os.Stat(dst); err != nil {
				return fmt.Errorf("input %s was not uploaded", in.Path)
			}
		}
	}
	return nil
}

func fetchURL(ctx context.Context, url, dst string) error {
	var last error
	for attempt := 0; attempt < 3; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return err
		}
		resp, err := http.DefaultClient.Do(req)
		if err == nil {
			if resp.StatusCode/100 == 2 {
				_, err = workerapi.WriteFileAtomic(dst, resp.Body)
				resp.Body.Close()
				if err == nil {
					return nil
				}
			} else {
				b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
				resp.Body.Close()
				err = fmt.Errorf("%d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
			}
		}
		last = err
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt+1) * 2 * time.Second):
		}
	}
	return last
}

// expandArgs replaces the $JOB placeholder with the job directory.
func expandArgs(args []string, dir string) []string {
	d := filepath.ToSlash(dir)
	out := make([]string, len(args))
	for i, a := range args {
		out[i] = strings.ReplaceAll(a, workerapi.JobDir, d)
	}
	return out
}

var placeholderTok = regexp.MustCompile(`\$(?:JOB|SESSION)[^\s"',;]*`)

// placeholder turns this machine's paths under dir back into token-relative
// ones ($JOB/out/0/x.png) with forward slashes, in a plain or a JSON line.
func placeholder(line, dir, token string) string {
	back := filepath.FromSlash(dir)
	forms := []string{
		strings.ReplaceAll(back, `\`, `\\`), // JSON-escaped (Windows)
		filepath.ToSlash(dir),
		back,
	}
	hit := false
	for _, f := range forms {
		if f != "" && strings.Contains(line, f) {
			line = strings.ReplaceAll(line, f, token)
			hit = true
		}
	}
	if !hit || !strings.Contains(line, `\`) {
		return line
	}
	return placeholderTok.ReplaceAllStringFunc(line, func(m string) string {
		m = strings.ReplaceAll(m, `\\`, "/")
		return strings.ReplaceAll(m, `\`, "/")
	})
}

func envOf(m map[string]string) []string {
	var out []string
	for k, v := range m {
		if strings.HasPrefix(k, "LOKI_") {
			out = append(out, k+"="+v)
		}
	}
	return out
}

// janitor drops finished jobs (and their files) after the keep period.
func (w *worker) janitor(ctx context.Context) {
	t := time.NewTicker(time.Minute)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
		w.mu.Lock()
		var drop []*job
		for id, j := range w.jobs {
			j.mu.Lock()
			old := j.st.State.Finished() && time.Since(j.st.Ended) > w.cfg.keep
			j.mu.Unlock()
			if old {
				drop = append(drop, j)
				delete(w.jobs, id)
			}
		}
		w.mu.Unlock()
		for _, j := range drop {
			_ = os.RemoveAll(j.dir)
		}
		w.sessions.expire()
	}
}

func (w *worker) remove(id string) bool {
	w.mu.Lock()
	j := w.jobs[id]
	if j != nil {
		delete(w.jobs, id)
		for i, q := range w.queue {
			if q == j {
				w.queue = append(w.queue[:i], w.queue[i+1:]...)
				w.updateQueuePos()
				break
			}
		}
	}
	w.mu.Unlock()
	if j == nil {
		return false
	}
	j.cancel()
	go func() {
		select {
		case <-j.done:
		case <-time.After(time.Minute):
		}
		_ = os.RemoveAll(j.dir)
	}()
	return true
}

func (w *worker) cancelJob(id string) bool {
	w.mu.Lock()
	j := w.jobs[id]
	queued := false
	if j != nil {
		for i, q := range w.queue {
			if q == j {
				w.queue = append(w.queue[:i], w.queue[i+1:]...)
				w.updateQueuePos()
				queued = true
				break
			}
		}
	}
	w.mu.Unlock()
	if j == nil {
		return false
	}
	j.cancel()
	if queued {
		w.finish(j, workerapi.JobCancelled, "cancelled")
	}
	return true
}

func (w *worker) newJob(spec workerapi.JobSpec) (*job, error) {
	spec.Tool = workerapi.EngineName(spec.Tool)
	if _, ok := w.tools[spec.Tool]; !ok {
		return nil, fmt.Errorf("%s is not available on this worker (has: %s)", spec.Tool, strings.Join(w.toolNames(), ", "))
	}
	id := fmt.Sprintf("j%d-%s", time.Now().UnixMilli(), randID(3))
	dir := filepath.Join(w.cfg.dataDir, "jobs", id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	j := &job{spec: spec, dir: dir, ctx: ctx, cancel: cancel, done: make(chan struct{})}
	j.st = workerapi.JobStatus{ID: id, Tool: spec.Tool, State: workerapi.JobQueued, Worker: w.cfg.name, GPU: -1, Created: time.Now()}
	return j, nil
}

func (w *worker) toolNames() []string {
	var n []string
	for t := range w.tools {
		n = append(n, t)
	}
	sort.Strings(n)
	return n
}

// info describes the worker (help: include each engine's --help).
func (w *worker) info(help bool) workerapi.Info {
	w.mu.Lock()
	running := 0
	for _, s := range w.slots {
		if s.busy != "" {
			running++
		}
	}
	queued := len(w.queue)
	ready := w.ready
	w.mu.Unlock()
	inf := workerapi.Info{Worker: w.cfg.name, Version: version, Protocol: workerapi.Protocol, Mode: w.cfg.mode, Ready: ready, Slots: len(w.slots), Running: running, Queued: queued, Uptime: time.Since(w.started).Seconds(), Sessions: w.sessions.count()}
	inf.GPUs = w.gpus()
	for _, name := range w.toolNames() {
		ti := workerapi.ToolInfo{Name: name, Path: w.tools[name]}
		for _, s := range w.slots {
			e := s.engines[name]
			ti.States = append(ti.States, e.State())
			ti.Resident = append(ti.Resident, e.Resident()...)
			if e.version != "" {
				ti.Version, ti.Model = e.version, e.model
			}
		}
		if help {
			ti.Help = w.help(name)
		}
		inf.Tools = append(inf.Tools, ti)
	}
	return inf
}

func (w *worker) gpus() []workerapi.GPU {
	w.mu.Lock()
	defer w.mu.Unlock()
	if time.Since(w.gpuAt) > 5*time.Second {
		if g, err := queryGPUs(); err == nil && len(g) > 0 {
			free := map[int]int{}
			for _, x := range g {
				free[x.Index] = x.FreeMB
			}
			for i := range w.gpuInfo {
				if f, ok := free[w.gpuInfo[i].Index]; ok {
					w.gpuInfo[i].FreeMB = f
				}
			}
		}
		w.gpuAt = time.Now()
	}
	out := append([]workerapi.GPU(nil), w.gpuInfo...)
	for i := range out {
		for _, s := range w.slots {
			if s.gpu == out[i].Index {
				out[i].Busy = s.busy
			}
		}
	}
	return out
}

// help is an engine's --help text, cached.
func (w *worker) help(tool string) string {
	w.mu.Lock()
	h, ok := w.helps[tool]
	w.mu.Unlock()
	if ok {
		return h
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, w.tools[tool], "--help").Output()
	if err != nil {
		log.Printf("%s --help: %v", tool, err)
		return ""
	}
	w.mu.Lock()
	w.helps[tool] = string(out)
	w.mu.Unlock()
	return string(out)
}

// warm fetches and loads models in the background (one slot each).
func (w *worker) warm(req workerapi.WarmRequest) error {
	var tools []string
	for _, t := range req.Tools {
		t = workerapi.EngineName(t)
		if _, ok := w.tools[t]; !ok {
			return fmt.Errorf("%s is not available on this worker", t)
		}
		tools = append(tools, t)
	}
	go func() {
		for _, t := range tools {
			if err := w.warmNow(context.Background(), t, req.Model); err != nil {
				log.Printf("warming %s: %v", t, err)
			}
		}
	}()
	return nil
}

// warmNow loads a tool's models on the first slot, waiting for it.
func (w *worker) warmNow(ctx context.Context, tool, model string) error {
	if len(w.slots) == 0 {
		return fmt.Errorf("no GPU")
	}
	s := w.slots[0]
	s.mu.Lock()
	defer s.mu.Unlock()
	log.Printf("warming %s on gpu %d (downloads its models when missing)", tool, s.gpu)
	req := map[string]any{"cmd": "warm"}
	if model != "" {
		req["model"] = model
	}
	r, err := s.engines[tool].Call(ctx, req, func(l string) { log.Printf("%s: %s", tool, l) })
	if err != nil {
		return err
	}
	if !r.OK {
		return fmt.Errorf("%s", r.Error)
	}
	return nil
}
