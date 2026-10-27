// Command loki-worker runs the loki-* GPU engines (loki-retouch,
// loki-reshoot, loki-dream, loki-segment) on a GPU host and serves them to
// the Lowkey Media Server: one resident engine process per GPU keeps its
// models in VRAM between jobs, jobs queue per node, and files travel with
// the jobs. It runs as an HTTP server (a persistent node: a RunPod pod, an
// EC2 instance, a container anywhere, a load-balanced serverless endpoint)
// or as a RunPod queue-based serverless worker (picked automatically when
// RunPod's worker environment is present). See tools/worker/README.md.
package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/stevecastle/shrike/workerapi"
)

// version is stamped at build time (-ldflags "-X main.version=...").
var version = "dev"

type config struct {
	addr      string
	token     string
	name      string
	mode      string // "http" | "runpod"
	dataDir   string
	tools     []string
	binDir    string
	gpus      string
	preload   []string
	coexist   string
	keep      time.Duration
	engineEnv []string
}

func envOr(k, def string) string {
	if v := strings.TrimSpace(os.Getenv(k)); v != "" {
		return v
	}
	return def
}

func splitList(s string) []string {
	var out []string
	for _, p := range strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == ';' }) {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

func randID(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)
	log.SetPrefix("loki-worker: ")

	host, _ := os.Hostname()
	defAddr := ":" + envOr("PORT", "8080")
	var cfg config
	var tools, preload string
	var keepHours float64
	var showVersion, noToken bool
	flag.StringVar(&cfg.addr, "addr", envOr("LOKI_WORKER_ADDR", defAddr), "HTTP listen address (env LOKI_WORKER_ADDR; RunPod's PORT is honored)")
	flag.StringVar(&cfg.token, "token", os.Getenv("LOKI_WORKER_TOKEN"), "bearer token callers must send (env LOKI_WORKER_TOKEN)")
	flag.BoolVar(&noToken, "no-token", envOr("LOKI_WORKER_NO_TOKEN", "") == "1", "serve without a token (only on a private network; env LOKI_WORKER_NO_TOKEN=1)")
	flag.StringVar(&cfg.name, "name", envOr("LOKI_WORKER_NAME", envOr("RUNPOD_POD_ID", host)), "this worker's name in status and logs (env LOKI_WORKER_NAME)")
	flag.StringVar(&cfg.mode, "mode", envOr("LOKI_WORKER_MODE", "auto"), "http, runpod (queue-based serverless), or auto (runpod when RunPod's worker environment is set)")
	flag.StringVar(&cfg.dataDir, "data", envOr("LOKI_WORKER_DATA", filepath.Join(os.TempDir(), "loki-worker")), "folder for job and session files (env LOKI_WORKER_DATA)")
	flag.StringVar(&tools, "tools", envOr("LOKI_WORKER_TOOLS", ""), "engines to serve, comma separated (default: every loki-* engine found; env LOKI_WORKER_TOOLS)")
	flag.StringVar(&cfg.binDir, "bin", envOr("LOKI_WORKER_BIN", ""), "folder holding the engines (default: next to loki-worker, then PATH; env LOKI_WORKER_BIN)")
	flag.StringVar(&cfg.gpus, "gpus", envOr("LOKI_WORKER_GPUS", "auto"), "GPU indices to use, comma separated, or auto (all; env LOKI_WORKER_GPUS)")
	flag.StringVar(&preload, "preload", envOr("LOKI_WORKER_PRELOAD", ""), "engines whose models are fetched and loaded at start, before /ping says ready (env LOKI_WORKER_PRELOAD)")
	flag.StringVar(&cfg.coexist, "coexist", envOr("LOKI_WORKER_COEXIST", "auto"), "keep several engines' models on one GPU: auto (GPUs with 60 GB or more), always, never (env LOKI_WORKER_COEXIST)")
	flag.Float64Var(&keepHours, "keep", 1, "hours a finished job's files stay for download")
	flag.BoolVar(&showVersion, "version", false, "print the version and exit")
	flag.Parse()
	if showVersion {
		fmt.Println("loki-worker", version)
		return
	}
	cfg.tools = splitList(tools)
	cfg.preload = splitList(preload)
	cfg.keep = time.Duration(keepHours * float64(time.Hour))
	if cfg.mode == "auto" {
		cfg.mode = "http"
		if _, ok := runpodFromEnv(); ok {
			cfg.mode = "runpod"
		}
	}
	// the engines read their models from (and download missing ones to) LOKI_MODELS;
	// on RunPod a network volume keeps them across workers
	models := os.Getenv("LOKI_MODELS")
	if models == "" || models == "/models" { // unset, or the image's default
		if st, err := os.Stat("/runpod-volume"); err == nil && st.IsDir() {
			models = "/runpod-volume/models"
		}
	}
	if models != "" {
		first := filepath.SplitList(models)[0]
		_ = os.MkdirAll(first, 0755)
		cfg.engineEnv = append(cfg.engineEnv, "LOKI_MODELS="+models)
		if os.Getenv("LOKI_MODELS_DOWNLOAD") == "" {
			cfg.engineEnv = append(cfg.engineEnv, "LOKI_MODELS_DOWNLOAD="+first)
		}
		log.Printf("models: %s", models)
	}
	if err := os.MkdirAll(cfg.dataDir, 0755); err != nil {
		log.Fatalf("data folder: %v", err)
	}

	found := findTools(cfg)
	if len(found) == 0 {
		log.Fatalf("no loki-* engine found (looked next to loki-worker%s and on PATH)", map[bool]string{true: ", in " + cfg.binDir, false: ""}[cfg.binDir != ""])
	}
	gpus, err := pickGPUs(cfg.gpus)
	if err != nil {
		log.Fatalf("GPUs: %v", err)
	}
	for _, g := range gpus {
		log.Printf("gpu %d: %s (%d MB)", g.Index, g.Name, g.TotalMB)
	}
	for t, p := range found {
		log.Printf("engine %s: %s", t, p)
	}

	w := newWorker(cfg, found, gpus)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go w.dispatch(ctx)
	go w.janitor(ctx)
	go func() {
		for _, t := range cfg.preload {
			t = workerapi.EngineName(t)
			if _, ok := found[t]; !ok {
				log.Printf("preload: %s is not installed here", t)
				continue
			}
			if err := w.warmNow(ctx, t, ""); err != nil {
				log.Printf("preload %s: %v", t, err)
			}
		}
		w.mu.Lock()
		w.ready = true
		w.mu.Unlock()
		log.Printf("ready")
	}()

	if cfg.mode == "runpod" {
		e, ok := runpodFromEnv()
		if !ok {
			log.Fatalf("--mode runpod needs RunPod's worker environment (RUNPOD_WEBHOOK_GET_JOB, RUNPOD_WEBHOOK_POST_OUTPUT)")
		}
		w.serveRunPod(ctx, e)
		shutdown(w)
		return
	}

	if cfg.token == "" && !noToken {
		if !isLoopback(cfg.addr) {
			log.Fatalf("set LOKI_WORKER_TOKEN (or --token): this worker listens on %s and runs whatever it is sent. Use --no-token only on a private network.", cfg.addr)
		}
		log.Printf("no token: only reachable from this machine (%s)", cfg.addr)
	}
	srv := &http.Server{Addr: cfg.addr, Handler: w.routes(), ReadHeaderTimeout: 30 * time.Second}
	go func() {
		<-ctx.Done()
		sctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = srv.Shutdown(sctx)
	}()
	log.Printf("%s %s serving %d engine(s) on %d GPU(s) at %s", cfg.name, version, len(found), len(gpus), cfg.addr)
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("listen: %v", err)
	}
	shutdown(w)
}

func shutdown(w *worker) {
	for _, s := range w.slots {
		for _, e := range s.engines {
			e.Stop()
		}
	}
}

func isLoopback(addr string) bool {
	h, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if h == "localhost" {
		return true
	}
	ip := net.ParseIP(h)
	return ip != nil && ip.IsLoopback()
}

// findTools locates the engines: --bin, next to loki-worker, then PATH.
func findTools(cfg config) map[string]string {
	want := workerapi.Engines
	if len(cfg.tools) > 0 {
		want = nil
		for _, t := range cfg.tools {
			want = append(want, workerapi.EngineName(t))
		}
	}
	var dirs []string
	if cfg.binDir != "" {
		dirs = append(dirs, cfg.binDir)
	}
	if exe, err := os.Executable(); err == nil {
		dirs = append(dirs, filepath.Dir(exe))
	}
	ext := ""
	if os.PathSeparator == '\\' {
		ext = ".exe"
	}
	found := map[string]string{}
	for _, t := range want {
		for _, d := range dirs {
			p := filepath.Join(d, t+ext)
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				found[t] = p
				break
			}
		}
		if found[t] == "" {
			if p, err := exec.LookPath(t); err == nil {
				found[t] = p
			}
		}
	}
	return found
}

// queryGPUs asks nvidia-smi for the GPUs.
func queryGPUs() ([]workerapi.GPU, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, err
	}
	var gpus []workerapi.GPU
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		f := strings.Split(line, ",")
		if len(f) < 4 {
			continue
		}
		idx, _ := strconv.Atoi(strings.TrimSpace(f[0]))
		total, _ := strconv.Atoi(strings.TrimSpace(f[2]))
		free, _ := strconv.Atoi(strings.TrimSpace(f[3]))
		gpus = append(gpus, workerapi.GPU{Index: idx, Name: strings.TrimSpace(f[1]), TotalMB: total, FreeMB: free})
	}
	return gpus, nil
}

func pickGPUs(spec string) ([]workerapi.GPU, error) {
	all, err := queryGPUs()
	if err != nil {
		log.Printf("nvidia-smi: %v (assuming one GPU)", err)
		all = []workerapi.GPU{{Index: 0, Name: "GPU 0"}}
	}
	if spec == "" || spec == "auto" || spec == "all" {
		if len(all) == 0 {
			return nil, fmt.Errorf("no NVIDIA GPU found")
		}
		return all, nil
	}
	var out []workerapi.GPU
	for _, s := range splitList(spec) {
		i, err := strconv.Atoi(s)
		if err != nil {
			return nil, fmt.Errorf("bad GPU index %q", s)
		}
		g := workerapi.GPU{Index: i, Name: fmt.Sprintf("GPU %d", i)}
		for _, a := range all {
			if a.Index == i {
				g = a
			}
		}
		out = append(out, g)
	}
	return out, nil
}
