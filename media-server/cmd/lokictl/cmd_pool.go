package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/workerapi"
	"github.com/stevecastle/shrike/workerapi/provision"
)

// GPU worker pools: where the loki-* engines run (GET/PUT /api/engines), and
// creating RunPod workers for them (RunPod's REST API, called from here with
// your RunPod key; the server only stores what it needs to use them).

func init() {
	register(command{group: "pool", name: "status", summary: "Engine placement and every GPU worker's health (GET /api/engines)", run: cmdPoolStatus})
	register(command{group: "pool", name: "check", summary: "Probe every GPU worker now (POST /api/engines/check)", run: func(a *App, args []string) int {
		var out any
		if err := a.Client.DoJSON("POST", "/api/engines/check", nil, &out); err != nil {
			return a.Fail(err)
		}
		return a.PrintJSON(out)
	}})
	register(command{group: "pool", name: "strategy", args: "local|pool|hybrid [--fallback=true|false]",
		summary: "Where engines run: this GPU, the pools, or both", run: cmdPoolStrategy})
	register(command{group: "pool", name: "local-only", args: "on|off",
		summary: "Run every engine on this machine, keeping the pools configured", run: cmdPoolLocalOnly})
	register(command{group: "pool", name: "route", args: "<engine> local|pool|pool:NAME|hybrid|default",
		summary: "Override the strategy for one engine (e.g. keep segment local)", run: cmdPoolRoute})
	register(command{group: "pool", name: "add-node", args: "<pool> <url> [--token T] [--name N]",
		summary: "Add a loki-worker node (tested first)", run: cmdPoolAddNode})
	register(command{group: "pool", name: "add-runpod", args: "<pool> --endpoint ID --api-key KEY [--max-jobs N] [--transfer-root LABEL | --bucket B ...]",
		summary: "Add a RunPod serverless endpoint running the worker image", run: cmdPoolAddRunPod})
	register(command{group: "pool", name: "remove", args: "<pool> [<node-url>]",
		summary: "Remove a pool, or one node of it (cloud resources stay: see destroy)", run: cmdPoolRemove})
	register(command{group: "pool", name: "warm", args: "<pool|all> <engine>... [--model fast]",
		summary: "Download and load engines' models on a pool's workers ahead of use", run: cmdPoolWarm})
	register(command{group: "pool", name: "create", args: "runpod-pods|runpod-serverless <pool> [flags]",
		summary: "Create RunPod GPU workers and add them as a pool (needs RUNPOD_API_KEY)", run: cmdPoolCreate})
	register(command{group: "pool", name: "scale", args: "<pool> --count N",
		summary: "Add or remove RunPod pods of a pool made by pool create", run: cmdPoolScale})
	register(command{group: "pool", name: "destroy", args: "<pool>",
		summary: "Delete a pool's RunPod pods / endpoint and remove the pool", run: cmdPoolDestroy})
}

type enginesDoc struct {
	Config appconfig.EnginesConfig `json:"config"`
	Status map[string]any          `json:"status"`
}

func getEngines(a *App) (*enginesDoc, error) {
	var d enginesDoc
	if err := a.Client.DoJSON("GET", "/api/engines", nil, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

func putEngines(a *App, e appconfig.EnginesConfig) int {
	var out any
	if err := a.Client.DoJSON("PUT", "/api/engines", e, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

func findPool(e *appconfig.EnginesConfig, name string) *appconfig.WorkerPool {
	for i := range e.Pools {
		if strings.EqualFold(e.Pools[i].Name, name) {
			return &e.Pools[i]
		}
	}
	return nil
}

func flags(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	return fs
}

// parseInterleaved lets flags follow positionals ("add-node gpu URL --token T").
func parseInterleaved(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for len(args) > 0 {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		args = fs.Args()
		if len(args) > 0 {
			pos = append(pos, args[0])
			args = args[1:]
		}
	}
	return pos, nil
}

func cmdPoolStatus(a *App, args []string) int {
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	if !a.Table {
		return a.PrintJSON(d)
	}
	// table: one row per target
	targets, _ := d.Status["targets"].([]any)
	var rows []map[string]any
	for _, t := range targets {
		m, _ := t.(map[string]any)
		row := map[string]any{"target": m["key"], "kind": m["kind"], "healthy": m["healthy"], "busy": fmt.Sprintf("%v/%v", m["running"], m["slots"]), "error": m["error"]}
		if info, ok := m["info"].(map[string]any); ok {
			var warm []string
			tools, _ := info["tools"].([]any)
			for _, ti := range tools {
				tm, _ := ti.(map[string]any)
				if r, _ := tm["resident"].([]any); len(r) > 0 {
					warm = append(warm, fmt.Sprint(tm["name"]))
				}
			}
			row["warm"] = strings.Join(warm, ",")
			row["version"] = info["version"]
		}
		rows = append(rows, row)
	}
	return a.PrintJSON(rows)
}

func cmdPoolStrategy(a *App, args []string) int {
	fs := flags("pool strategy")
	fallback := fs.String("fallback", "", "true: run here when no worker is reachable (strategy pool)")
	pos, err := parseInterleaved(fs, args)
	if err != nil || len(pos) != 1 {
		return a.Usage(fs, "usage: lokictl pool strategy local|pool|hybrid [--fallback=true|false]")
	}
	s := strings.ToLower(pos[0])
	if s != "local" && s != "pool" && s != "hybrid" {
		return a.Usage(fs, "strategy must be local, pool or hybrid")
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	d.Config.Strategy = s
	switch *fallback {
	case "true":
		d.Config.Fallback = true
	case "false":
		d.Config.Fallback = false
	}
	return putEngines(a, d.Config)
}

func cmdPoolLocalOnly(a *App, args []string) int {
	if len(args) != 1 || (args[0] != "on" && args[0] != "off") {
		return a.Usage(nil, "usage: lokictl pool local-only on|off")
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	d.Config.LocalOnly = args[0] == "on"
	return putEngines(a, d.Config)
}

func cmdPoolRoute(a *App, args []string) int {
	if len(args) != 2 {
		return a.Usage(nil, "usage: lokictl pool route <engine> local|pool|pool:NAME|hybrid|default")
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	e := workerapi.EngineName(args[0])
	if d.Config.Routes == nil {
		d.Config.Routes = map[string]string{}
	}
	if args[1] == "default" {
		delete(d.Config.Routes, e)
	} else {
		d.Config.Routes[e] = args[1]
	}
	return putEngines(a, d.Config)
}

func cmdPoolAddNode(a *App, args []string) int {
	fs := flags("pool add-node")
	token := fs.String("token", "", "the worker's LOKI_WORKER_TOKEN")
	name := fs.String("name", "", "a display name (default: the URL)")
	skip := fs.Bool("no-test", false, "add without testing the connection")
	pos, err := parseInterleaved(fs, args)
	if err != nil || len(pos) != 2 {
		return a.Usage(fs, "usage: lokictl pool add-node <pool> <url> [--token T] [--name N]")
	}
	if !*skip {
		var probe map[string]any
		if err := a.Client.DoJSON("POST", "/api/engines/probe", map[string]string{"kind": "nodes", "url": pos[1], "token": *token}, &probe); err != nil {
			return a.Fail(fmt.Errorf("the worker did not answer (use --no-test to add it anyway): %w", err))
		}
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	p := findPool(&d.Config, pos[0])
	if p == nil {
		d.Config.Pools = append(d.Config.Pools, appconfig.WorkerPool{Name: pos[0], Kind: appconfig.PoolKindNodes})
		p = &d.Config.Pools[len(d.Config.Pools)-1]
	}
	if p.Kind == appconfig.PoolKindRunPod {
		return a.Usage(fs, "pool "+p.Name+" is a RunPod endpoint; give nodes their own pool")
	}
	p.Nodes = append(p.Nodes, appconfig.WorkerNode{Name: *name, URL: pos[1], Token: *token})
	if d.Config.Strategy == "" || d.Config.Strategy == appconfig.EngineStrategyLocal {
		d.Config.Strategy = appconfig.EngineStrategyPool
	}
	return putEngines(a, d.Config)
}

func transferFlags(fs *flag.FlagSet) func() *appconfig.PoolTransfer {
	root := fs.String("transfer-root", "", "an S3 storage root (by label) to pass files through")
	bucket := fs.String("bucket", "", "...or a bucket of its own")
	endpoint := fs.String("s3-endpoint", "", "the bucket's S3 endpoint (empty: AWS)")
	region := fs.String("region", "", "the bucket's region")
	prefix := fs.String("prefix", "", "key prefix in the bucket")
	ak := fs.String("access-key", os.Getenv("AWS_ACCESS_KEY_ID"), "bucket access key (default $AWS_ACCESS_KEY_ID)")
	sk := fs.String("secret-key", os.Getenv("AWS_SECRET_ACCESS_KEY"), "bucket secret key (default $AWS_SECRET_ACCESS_KEY)")
	return func() *appconfig.PoolTransfer {
		if *root == "" && *bucket == "" {
			return nil
		}
		return &appconfig.PoolTransfer{Root: *root, Bucket: *bucket, Endpoint: *endpoint, Region: *region, Prefix: *prefix, AccessKey: *ak, SecretKey: *sk}
	}
}

func cmdPoolAddRunPod(a *App, args []string) int {
	fs := flags("pool add-runpod")
	endpoint := fs.String("endpoint", "", "the endpoint id (required)")
	key := fs.String("api-key", os.Getenv("RUNPOD_API_KEY"), "RunPod API key (default $RUNPOD_API_KEY)")
	maxJobs := fs.Int("max-jobs", 4, "jobs in flight (match the endpoint's max workers)")
	transfer := transferFlags(fs)
	pos, err := parseInterleaved(fs, args)
	if err != nil || len(pos) != 1 || *endpoint == "" || *key == "" {
		return a.Usage(fs, "usage: lokictl pool add-runpod <pool> --endpoint ID --api-key KEY [--max-jobs N] [--transfer-root LABEL]")
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	if p := findPool(&d.Config, pos[0]); p != nil {
		return a.Usage(fs, "pool "+pos[0]+" exists; remove it first or pick another name")
	}
	d.Config.Pools = append(d.Config.Pools, appconfig.WorkerPool{Name: pos[0], Kind: appconfig.PoolKindRunPod, EndpointID: *endpoint, APIKey: *key, MaxJobs: *maxJobs, Transfer: transfer()})
	if d.Config.Strategy == "" || d.Config.Strategy == appconfig.EngineStrategyLocal {
		d.Config.Strategy = appconfig.EngineStrategyPool
	}
	return putEngines(a, d.Config)
}

func cmdPoolRemove(a *App, args []string) int {
	if len(args) < 1 || len(args) > 2 {
		return a.Usage(nil, "usage: lokictl pool remove <pool> [<node-url>]")
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	p := findPool(&d.Config, args[0])
	if p == nil {
		return a.Fail(fmt.Errorf("no pool %q", args[0]))
	}
	if len(args) == 2 {
		var keep []appconfig.WorkerNode
		for _, n := range p.Nodes {
			if strings.TrimRight(n.URL, "/") != strings.TrimRight(args[1], "/") {
				keep = append(keep, n)
			}
		}
		p.Nodes = keep
	} else {
		var keep []appconfig.WorkerPool
		for _, x := range d.Config.Pools {
			if !strings.EqualFold(x.Name, args[0]) {
				keep = append(keep, x)
			}
		}
		d.Config.Pools = keep
	}
	return putEngines(a, d.Config)
}

func cmdPoolWarm(a *App, args []string) int {
	fs := flags("pool warm")
	model := fs.String("model", "", "variant: fast (loki-reshoot's FastH3 Omni)")
	pos, err := parseInterleaved(fs, args)
	if err != nil || len(pos) < 2 {
		return a.Usage(fs, "usage: lokictl pool warm <pool|all> <engine>... [--model fast]")
	}
	pool := pos[0]
	if pool == "all" {
		pool = ""
	}
	var engines []string
	for _, e := range pos[1:] {
		engines = append(engines, workerapi.EngineName(e))
	}
	var out any
	if err := a.Client.DoJSON("POST", "/api/engines/warm", map[string]any{"pool": pool, "engines": engines, "model": *model}, &out); err != nil {
		return a.Fail(err)
	}
	return a.PrintJSON(out)
}

type listFlag []string

func (l *listFlag) String() string { return strings.Join(*l, ",") }
func (l *listFlag) Set(v string) error {
	for _, s := range strings.Split(v, ",") {
		if s = strings.TrimSpace(s); s != "" {
			*l = append(*l, s)
		}
	}
	return nil
}

// runpodBase overrides RunPod's REST address (tests).
var runpodBase = ""

func runpodClient(key string) (*provision.RunPod, error) {
	if key == "" {
		return nil, fmt.Errorf("a RunPod API key is needed: --runpod-key or $RUNPOD_API_KEY (runpod.io → Settings → API Keys)")
	}
	return &provision.RunPod{APIKey: key, Base: runpodBase}, nil
}

func cmdPoolCreate(a *App, args []string) int {
	if len(args) < 2 {
		return a.Usage(nil, "usage: lokictl pool create runpod-pods|runpod-serverless <pool> [flags] (see --help of each)")
	}
	kind, rest := args[0], args[1:]
	fs := flags("pool create " + kind)
	key := fs.String("runpod-key", os.Getenv("RUNPOD_API_KEY"), "RunPod API key (default $RUNPOD_API_KEY)")
	image := fs.String("image", provision.DefaultImage, "worker image")
	var gpus, preload, dcs, only listFlag
	fs.Var(&only, "engines", "serve only these engines (e.g. retouch): the workers never load another engine's models, so they stay warm for these; route the engines here with pool route")
	fs.Var(&gpus, "gpu", "RunPod GPU type, repeatable or comma separated (default: "+strings.Join(provision.DefaultGPUs, ", ")+")")
	fs.Var(&dcs, "datacenter", "allowed data centers (e.g. EU-RO-1), repeatable")
	netVol := fs.String("network-volume", "", "a RunPod network volume id: models downloaded once, shared by every worker")
	registryAuth := fs.String("registry-auth", "", "RunPod registry credential id for a private image")
	// pods
	count := fs.Int("count", 1, "pods: how many")
	cloud := fs.String("cloud", "SECURE", "pods: SECURE or COMMUNITY")
	volume := fs.Int("volume-gb", 150, "pods: pod volume for the models (without --network-volume)")
	fs.Var(&preload, "preload", "pods: engines whose models load at start, e.g. retouch,segment")
	// serverless
	maxWorkers := fs.Int("max-workers", 3, "serverless: most workers at once")
	minWorkers := fs.Int("min-workers", 0, "serverless: workers kept running (0 = scale to zero)")
	idle := fs.Int("idle", 120, "serverless: seconds a worker (and its loaded models) stays after its last job")
	execSecs := fs.Int("max-job-secs", 3600, "serverless: execution timeout per job")
	endpointKey := fs.String("endpoint-key", os.Getenv("RUNPOD_ENDPOINT_KEY"), "serverless: the key the server keeps to run jobs (a restricted key with write access to the endpoint; default $RUNPOD_ENDPOINT_KEY, else --runpod-key)")
	transfer := transferFlags(fs)
	pos, err := parseInterleaved(fs, rest)
	if err != nil || len(pos) != 1 {
		return a.Usage(fs, "usage: lokictl pool create "+kind+" <pool> [flags]")
	}
	name := pos[0]
	var engines []string
	for _, e := range only {
		engines = append(engines, workerapi.EngineName(e))
	}
	var env map[string]string
	if len(engines) > 0 {
		env = map[string]string{"LOKI_WORKER_TOOLS": strings.Join(engines, ",")}
	}
	rp, err := runpodClient(*key)
	if err != nil {
		return a.Usage(fs, err.Error())
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	if findPool(&d.Config, name) != nil {
		return a.Usage(fs, "pool "+name+" exists already")
	}
	ctx := context.Background()
	switch kind {
	case "runpod-pods":
		var warm []string
		for _, e := range preload {
			warm = append(warm, workerapi.EngineName(e))
		}
		pool := appconfig.WorkerPool{Name: name, Kind: appconfig.PoolKindNodes, Engines: engines, Provider: &appconfig.PoolProvider{Name: "runpod"}}
		var made []*provision.Pod
		for i := 0; i < *count; i++ {
			token := workerapi.NewToken()
			p, err := rp.CreatePod(ctx, provision.PodSpec{Name: fmt.Sprintf("loki-%s-%d", name, i+1), Image: *image, GPUs: gpus, Cloud: *cloud, VolumeGB: *volume, NetworkVolumeID: *netVol, DataCenters: dcs, Token: token, Preload: warm, Env: env, RegistryAuthID: *registryAuth})
			if err != nil {
				for _, m := range made {
					_ = rp.DeletePod(ctx, m.ID)
				}
				return a.Fail(fmt.Errorf("creating pod %d: %w", i+1, err))
			}
			made = append(made, p)
			pool.Nodes = append(pool.Nodes, appconfig.WorkerNode{Name: p.ID, URL: p.URL, Token: token})
			pool.Provider.PodIDs = append(pool.Provider.PodIDs, p.ID)
		}
		d.Config.Pools = append(d.Config.Pools, pool)
	case "runpod-serverless":
		e, err := rp.CreateServerless(ctx, provision.ServerlessSpec{Name: "loki-" + name, Image: *image, GPUs: gpus, WorkersMin: *minWorkers, WorkersMax: *maxWorkers, IdleSeconds: *idle, ExecutionSecs: *execSecs, NetworkVolumeID: *netVol, Flashboot: true, Env: env, RegistryAuthID: *registryAuth})
		if err != nil {
			return a.Fail(err)
		}
		runKey := *endpointKey
		if runKey == "" {
			runKey = *key
		}
		d.Config.Pools = append(d.Config.Pools, appconfig.WorkerPool{Name: name, Kind: appconfig.PoolKindRunPod, Engines: engines, EndpointID: e.ID, APIKey: runKey, MaxJobs: *maxWorkers, Transfer: transfer(),
			Provider: &appconfig.PoolProvider{Name: "runpod", EndpointID: e.ID, TemplateID: e.TemplateID}})
		if transfer() == nil {
			fmt.Fprintln(a.ErrOut, `{"note":"no transfer bucket: files up to 7 MB travel inline (stills work; video references need --transfer-root or --bucket)"}`)
		}
	default:
		return a.Usage(fs, "kind must be runpod-pods or runpod-serverless")
	}
	if d.Config.Strategy == "" || d.Config.Strategy == appconfig.EngineStrategyLocal {
		d.Config.Strategy = appconfig.EngineStrategyPool
	}
	fmt.Fprintln(a.ErrOut, `{"note":"workers take a few minutes to start and download their models on first use; watch them with: lokictl -o table pool status"}`)
	return putEngines(a, d.Config)
}

func cmdPoolScale(a *App, args []string) int {
	fs := flags("pool scale")
	key := fs.String("runpod-key", os.Getenv("RUNPOD_API_KEY"), "RunPod API key (default $RUNPOD_API_KEY)")
	count := fs.Int("count", -1, "pods the pool should have")
	image := fs.String("image", provision.DefaultImage, "worker image for new pods")
	var gpus listFlag
	fs.Var(&gpus, "gpu", "GPU types for new pods")
	volume := fs.Int("volume-gb", 150, "pod volume for new pods")
	pos, err := parseInterleaved(fs, args)
	if err != nil || len(pos) != 1 || *count < 0 {
		return a.Usage(fs, "usage: lokictl pool scale <pool> --count N")
	}
	rp, err := runpodClient(*key)
	if err != nil {
		return a.Usage(fs, err.Error())
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	p := findPool(&d.Config, pos[0])
	if p == nil || p.Provider == nil || p.Provider.Name != "runpod" || p.Kind != appconfig.PoolKindNodes {
		return a.Fail(fmt.Errorf("pool %q was not made by `lokictl pool create runpod-pods`", pos[0]))
	}
	ctx := context.Background()
	for len(p.Nodes) < *count {
		token := workerapi.NewToken()
		pod, err := rp.CreatePod(ctx, provision.PodSpec{Name: fmt.Sprintf("loki-%s-%d", p.Name, len(p.Nodes)+1), Image: *image, GPUs: gpus, VolumeGB: *volume, Token: token})
		if err != nil {
			return a.Fail(err)
		}
		p.Nodes = append(p.Nodes, appconfig.WorkerNode{Name: pod.ID, URL: pod.URL, Token: token})
		p.Provider.PodIDs = append(p.Provider.PodIDs, pod.ID)
	}
	for len(p.Nodes) > *count {
		last := p.Nodes[len(p.Nodes)-1]
		if err := rp.DeletePod(ctx, last.Name); err != nil {
			return a.Fail(err)
		}
		p.Nodes = p.Nodes[:len(p.Nodes)-1]
		p.Provider.PodIDs = removeString(p.Provider.PodIDs, last.Name)
	}
	return putEngines(a, d.Config)
}

func removeString(l []string, s string) []string {
	var out []string
	for _, x := range l {
		if x != s {
			out = append(out, x)
		}
	}
	return out
}

func cmdPoolDestroy(a *App, args []string) int {
	fs := flags("pool destroy")
	key := fs.String("runpod-key", os.Getenv("RUNPOD_API_KEY"), "RunPod API key (default $RUNPOD_API_KEY)")
	pos, err := parseInterleaved(fs, args)
	if err != nil || len(pos) != 1 {
		return a.Usage(fs, "usage: lokictl pool destroy <pool>")
	}
	d, err := getEngines(a)
	if err != nil {
		return a.Fail(err)
	}
	p := findPool(&d.Config, pos[0])
	if p == nil {
		return a.Fail(fmt.Errorf("no pool %q", pos[0]))
	}
	if p.Provider != nil && p.Provider.Name == "runpod" {
		rp, err := runpodClient(*key)
		if err != nil {
			return a.Usage(fs, err.Error())
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancel()
		var errs []string
		for _, id := range p.Provider.PodIDs {
			if err := rp.DeletePod(ctx, id); err != nil {
				errs = append(errs, err.Error())
			}
		}
		if p.Provider.EndpointID != "" {
			if err := rp.DeleteEndpoint(ctx, p.Provider.EndpointID); err != nil {
				errs = append(errs, err.Error())
			}
		}
		if p.Provider.TemplateID != "" {
			if err := rp.DeleteTemplate(ctx, p.Provider.TemplateID); err != nil {
				errs = append(errs, err.Error())
			}
		}
		if len(errs) > 0 {
			return a.Fail(fmt.Errorf("some RunPod resources were not deleted (the pool is kept): %s", strings.Join(errs, "; ")))
		}
	}
	var keep []appconfig.WorkerPool
	for _, x := range d.Config.Pools {
		if !strings.EqualFold(x.Name, pos[0]) {
			keep = append(keep, x)
		}
	}
	d.Config.Pools = keep
	return putEngines(a, d.Config)
}
