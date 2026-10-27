package main

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
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

// sentinel closes each request's stderr log (see the engines' worker.rs).
const sentinel = "\x1eloki-worker done"

// engine is one loki-* process in --worker mode on one GPU: it keeps its
// models resident between requests and answers one request at a time.
type engine struct {
	tool string
	bin  string
	gpu  int
	env  []string

	mu       sync.Mutex // one request at a time
	state    atomic.Value
	proc     *exec.Cmd
	stdin    io.WriteCloser
	events   chan map[string]any
	exited   chan struct{}
	exitErr  error
	version  string
	model    string
	resident []workerapi.Resident
	// gen counts process starts: sessions opened in an earlier process are gone.
	gen int

	logMu   sync.Mutex
	onLog   func(string)
	logDone chan string // sentinel ids seen on stderr

	lastUsed time.Time
}

func newEngine(tool, bin string, gpu int, env []string) *engine {
	e := &engine{tool: tool, bin: bin, gpu: gpu, env: env}
	e.state.Store("stopped")
	return e
}

func (e *engine) State() string { return e.state.Load().(string) }

// start launches the process and waits for its ready event (the caller holds e.mu).
func (e *engine) start() error {
	cmd := exec.Command(e.bin, "--worker")
	cmd.Env = append(append(os.Environ(), e.env...), "CUDA_DEVICE_ORDER=PCI_BUS_ID", fmt.Sprintf("CUDA_VISIBLE_DEVICES=%d", e.gpu))
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
	e.state.Store("starting")
	if err := cmd.Start(); err != nil {
		e.state.Store("stopped")
		return fmt.Errorf("starting %s: %w", e.tool, err)
	}
	e.proc, e.stdin = cmd, stdin
	e.events = make(chan map[string]any, 64)
	e.exited = make(chan struct{})
	e.logDone = make(chan string, 16)
	e.gen++
	events, exited, logDone := e.events, e.exited, e.logDone
	go func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 1<<20), 64<<20)
		for sc.Scan() {
			var ev map[string]any
			if json.Unmarshal(sc.Bytes(), &ev) != nil {
				log.Printf("%s[gpu %d]: unexpected stdout: %s", e.tool, e.gpu, sc.Text())
				continue
			}
			events <- ev
		}
		close(events)
	}()
	go func() {
		sc := bufio.NewScanner(stderr)
		sc.Buffer(make([]byte, 1<<20), 16<<20)
		sc.Split(scanLinesOrCR)
		for sc.Scan() {
			line := sc.Text()
			if id, ok := strings.CutPrefix(line, sentinel); ok {
				select {
				case logDone <- strings.TrimSpace(id):
				default:
				}
				continue
			}
			e.logMu.Lock()
			fn := e.onLog
			e.logMu.Unlock()
			if fn != nil {
				fn(line)
			} else if strings.TrimSpace(line) != "" {
				log.Printf("%s[gpu %d]: %s", e.tool, e.gpu, line)
			}
		}
	}()
	go func() {
		err := cmd.Wait()
		e.exitErr = err
		close(exited)
	}()
	select {
	case ev, ok := <-events:
		if !ok || ev["event"] != "ready" {
			e.kill()
			return fmt.Errorf("%s did not start in worker mode (an engine build without --worker?)", e.tool)
		}
		e.version, _ = ev["version"].(string)
		e.model, _ = ev["model"].(string)
	case <-time.After(2 * time.Minute):
		e.kill()
		return fmt.Errorf("%s did not become ready", e.tool)
	}
	e.state.Store("idle")
	log.Printf("%s[gpu %d]: ready (%s %s)", e.tool, e.gpu, e.version, e.model)
	return nil
}

// kill ends the process (the caller holds e.mu).
func (e *engine) kill() {
	if e.proc == nil {
		return
	}
	if e.stdin != nil {
		_ = e.stdin.Close()
	}
	select {
	case <-e.exited:
	case <-time.After(3 * time.Second):
		_ = e.proc.Process.Kill()
		<-e.exited
	}
	e.proc = nil
	e.resident = nil
	e.state.Store("stopped")
}

// Stop ends the process once it is idle, freeing all of its VRAM.
func (e *engine) Stop() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.proc != nil {
		log.Printf("%s[gpu %d]: stopping (releasing its GPU memory)", e.tool, e.gpu)
	}
	e.kill()
}

func (e *engine) running() bool {
	if e.proc == nil {
		return false
	}
	select {
	case <-e.exited:
		return false
	default:
		return true
	}
}

// reply is a finished request.
type reply struct {
	OK        bool
	Error     string
	OOM       bool
	Cancelled bool
	Secs      float64
	Stdout    []string
	Reply     map[string]any
}

var errEngineExited = errors.New("engine exited")

// Call sends one request and waits for its done event, handing every stderr
// line of the request to onLog. Cancelling ctx sends a cancel; an engine
// that does not stop within 30 s is killed.
func (e *engine) Call(ctx context.Context, req map[string]any, onLog func(string)) (*reply, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if !e.running() {
		if e.proc != nil {
			e.kill()
		}
		if err := e.start(); err != nil {
			return nil, err
		}
	}
	id, _ := req["id"].(string)
	if id == "" {
		id = fmt.Sprintf("r%d", time.Now().UnixNano())
		req["id"] = id
	}
	e.logMu.Lock()
	e.onLog = onLog
	e.logMu.Unlock()
	defer func() {
		e.logMu.Lock()
		e.onLog = nil
		e.logMu.Unlock()
	}()
	e.state.Store("busy")
	defer func() {
		if e.running() {
			e.state.Store("idle")
		}
		e.lastUsed = time.Now()
	}()
	b, _ := json.Marshal(req)
	if _, err := e.stdin.Write(append(b, '\n')); err != nil {
		e.kill()
		return nil, fmt.Errorf("%w: %v", errEngineExited, err)
	}
	r := &reply{}
	cancelSent := false
	var killAt <-chan time.Time
	for {
		select {
		case ev, ok := <-e.events:
			if !ok {
				<-e.exited
				e.kill()
				return nil, fmt.Errorf("%w (%v)", errEngineExited, e.exitErr)
			}
			if ev["id"] != id {
				continue
			}
			switch ev["event"] {
			case "stdout":
				if l, ok := ev["line"].(string); ok {
					// a multi-line result (a prompt) arrives as one event
					r.Stdout = append(r.Stdout, strings.Split(l, "\n")...)
				}
			case "done":
				r.OK, _ = ev["ok"].(bool)
				r.Error, _ = ev["error"].(string)
				r.OOM, _ = ev["oom"].(bool)
				r.Cancelled, _ = ev["cancelled"].(bool)
				r.Secs, _ = ev["secs"].(float64)
				r.Reply, _ = ev["reply"].(map[string]any)
				e.resident = parseResident(ev["resident"])
				// the last log lines may still be in flight on stderr
				select {
				case <-e.logDone:
				case <-time.After(2 * time.Second):
				}
				for len(e.logDone) > 0 {
					<-e.logDone
				}
				return r, nil
			}
		case <-ctx.Done():
			if !cancelSent {
				cancelSent = true
				c, _ := json.Marshal(map[string]string{"cancel": id})
				_, _ = e.stdin.Write(append(c, '\n'))
				killAt = time.After(30 * time.Second)
			}
		case <-killAt:
			log.Printf("%s[gpu %d]: did not stop after cancel; killing it", e.tool, e.gpu)
			e.kill()
			return &reply{Cancelled: true, Error: "cancelled"}, nil
		}
	}
}

func parseResident(v any) []workerapi.Resident {
	arr, _ := v.([]any)
	var out []workerapi.Resident
	for _, x := range arr {
		m, _ := x.(map[string]any)
		key, _ := m["key"].(string)
		mb, _ := m["mb"].(float64)
		out = append(out, workerapi.Resident{Key: key, MB: int(mb)})
	}
	return out
}

// Resident is the models the engine held after its last request.
func (e *engine) Resident() []workerapi.Resident {
	if e.State() == "stopped" {
		return nil
	}
	return e.resident
}

// scanLinesOrCR splits on \n or \r (progress bars redraw with \r).
func scanLinesOrCR(data []byte, atEOF bool) (int, []byte, error) {
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
