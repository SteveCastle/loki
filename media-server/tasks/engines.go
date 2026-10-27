package tasks

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/workerapi"
)

// --- Engine placement: where a loki-* run happens ---
//
// The GPU engines (loki-retouch, loki-reshoot, loki-dream, loki-segment) run
// on "targets": this machine's GPU (the local target, which execs the CLI as
// it always has) and the workers of the configured pools — loki-worker nodes
// reached over HTTP and RunPod serverless endpoints. Config.Engines picks
// which targets an engine may use (strategy local / pool / hybrid, with
// per-engine routes). A task leases a target for its whole run
// (resolveAITool) and releases it when done; a target takes as many leases
// at once as it has GPU slots, so the gpu-diffusion queue bucket is sized to
// the sum of them. Leases prefer a worker whose engine already holds the
// models (warm), then the least loaded one.

// engineTarget is one place engines run.
type engineTarget struct {
	key  string // "local", "<pool>/<node>", "<pool>"
	pool string
	kind string // "local" | "node" | "runpod"

	node     *workerapi.Node
	rp       *workerapi.RunPod
	transfer *appconfig.PoolTransfer
	engines  []string // engines the pool is limited to (nil: any)
	maxJobs  int      // runpod

	mu       sync.Mutex
	healthy  bool
	checked  time.Time
	lastErr  string
	info     *workerapi.Info
	rpHealth *workerapi.RunPodHealth
	inflight int
	helps    map[string]string // engine -> --help, for flag probes
	tr       workerapi.Transfer
}

func (t *engineTarget) slots() int {
	switch t.kind {
	case "local":
		return 1
	case "runpod":
		return max(1, t.maxJobs)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.info != nil && t.info.Slots > 0 {
		return t.info.Slots
	}
	return 1
}

// serves reports whether the target can run the engine as far as we know.
func (t *engineTarget) serves(engine string) bool {
	if len(t.engines) > 0 {
		ok := false
		for _, e := range t.engines {
			if workerapi.EngineName(e) == engine {
				ok = true
			}
		}
		if !ok {
			return false
		}
	}
	if t.kind != "node" {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.info == nil || t.info.Tool(engine) != nil
}

func (t *engineTarget) warm(engine string) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.info != nil && t.info.Warm(engine)
}

func (t *engineTarget) isHealthy() bool {
	if t.kind == "local" {
		return true
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	// unknown (not checked yet) counts as healthy: the first run finds out
	return t.healthy || t.checked.IsZero()
}

func (t *engineTarget) markDown(err error) {
	t.mu.Lock()
	t.healthy, t.checked, t.lastErr = false, time.Now(), err.Error()
	t.mu.Unlock()
}

// engineDispatcher owns the targets and their leases.
type engineDispatcher struct {
	mu      sync.Mutex
	cfg     appconfig.EnginesConfig
	local   *engineTarget
	targets []*engineTarget // pool targets
	q       *jobqueue.Queue
	changed chan struct{} // closed and replaced whenever a lease frees or health changes
	health  sync.Once
}

var engineDisp = &engineDispatcher{
	cfg:     appconfig.EnginesConfig{Strategy: appconfig.EngineStrategyLocal},
	local:   &engineTarget{key: "local", kind: "local", healthy: true},
	changed: make(chan struct{}),
}

// ConfigureEngines applies Config.Engines: rebuilds the pool targets
// (keeping the state of unchanged ones) and sizes the GPU bucket. Called at
// startup and after every config save (ApplyHostLimits).
func ConfigureEngines(q *jobqueue.Queue, cfg appconfig.EnginesConfig) {
	appconfig.NormalizeEngines(&cfg)
	d := engineDisp
	d.mu.Lock()
	old := map[string]*engineTarget{}
	for _, t := range d.targets {
		old[t.key+"|"+targetIdentity(t)] = t
	}
	var targets []*engineTarget
	for _, p := range cfg.Pools {
		if p.Disabled {
			continue
		}
		switch p.Kind {
		case appconfig.PoolKindRunPod:
			if strings.TrimSpace(p.EndpointID) == "" {
				continue
			}
			t := &engineTarget{key: p.Name, pool: p.Name, kind: "runpod", rp: &workerapi.RunPod{Endpoint: p.EndpointID, APIKey: p.APIKey}, transfer: p.Transfer, engines: p.Engines, maxJobs: p.MaxJobs}
			targets = append(targets, reuseTarget(old, t))
		default:
			for _, n := range p.Nodes {
				if n.Disabled {
					continue
				}
				t := &engineTarget{key: p.Name + "/" + n.Name, pool: p.Name, kind: "node", node: &workerapi.Node{Name: n.Name, URL: n.URL, Token: n.Token}, engines: p.Engines}
				targets = append(targets, reuseTarget(old, t))
			}
		}
	}
	d.cfg, d.targets = cfg, targets
	if q != nil {
		d.q = q
	}
	d.mu.Unlock()
	d.resize()
	d.notify()
	if len(targets) > 0 {
		d.health.Do(func() { go d.healthLoop() })
		go d.checkAll(context.Background())
	}
}

func targetIdentity(t *engineTarget) string {
	if t.node != nil {
		return t.node.URL + "|" + t.node.Token
	}
	if t.rp != nil {
		return t.rp.Endpoint + "|" + t.rp.APIKey + "|" + fmt.Sprint(t.maxJobs, t.engines, t.transfer)
	}
	return ""
}

func reuseTarget(old map[string]*engineTarget, t *engineTarget) *engineTarget {
	if o, ok := old[t.key+"|"+targetIdentity(t)]; ok {
		o.engines, o.maxJobs, o.transfer = t.engines, t.maxJobs, t.transfer
		return o
	}
	return t
}

// resize sets the gpu-diffusion bucket to the slots of every target in use.
func (d *engineDispatcher) resize() {
	d.mu.Lock()
	q, cfg := d.q, d.cfg
	n := 0
	if cfg.Strategy != appconfig.EngineStrategyPool || cfg.Fallback || d.anyRouteLocal() {
		n += d.local.slots()
	}
	if cfg.Strategy != appconfig.EngineStrategyLocal || len(cfg.Routes) > 0 {
		for _, t := range d.targets {
			n += t.slots()
		}
	}
	d.mu.Unlock()
	if q != nil && n > 0 {
		q.SetHostLimit(HostBucketGPUDiffusion, n)
	}
}

func (d *engineDispatcher) anyRouteLocal() bool {
	for _, r := range d.cfg.Routes {
		if r == "local" || r == appconfig.EngineStrategyHybrid {
			return true
		}
	}
	return false
}

func (d *engineDispatcher) notify() {
	d.mu.Lock()
	close(d.changed)
	d.changed = make(chan struct{})
	d.mu.Unlock()
}

// route is the placement rule for one engine: "local", "pool", "pool:<name>"
// or "hybrid".
func (d *engineDispatcher) route(engine string) string {
	d.mu.Lock()
	defer d.mu.Unlock()
	if r, ok := d.cfg.Routes[engine]; ok && r != "" {
		return r
	}
	return d.cfg.Strategy
}

// engineRunsLocalOnly reports whether the engine never leaves this machine
// (the default): callers then skip the dispatcher entirely.
func engineRunsLocalOnly(engine string) bool {
	return engineDisp.route(workerapi.EngineName(engine)) == appconfig.EngineStrategyLocal
}

// EngineLocalOnly is engineRunsLocalOnly for other packages.
func EngineLocalOnly(engine string) bool { return engineRunsLocalOnly(engine) }

// candidates are the targets an engine may use now (healthy ones), and
// whether the local GPU is one of them.
func (d *engineDispatcher) candidates(engine string) (out []*engineTarget, unreachable []string) {
	r := d.route(engine)
	d.mu.Lock()
	targets, fallback := d.targets, d.cfg.Fallback
	d.mu.Unlock()
	pool := ""
	switch {
	case r == "local":
		return []*engineTarget{d.local}, nil
	case strings.HasPrefix(r, "pool:"):
		pool = strings.TrimPrefix(r, "pool:")
	}
	for _, t := range targets {
		if pool != "" && !strings.EqualFold(t.pool, pool) {
			continue
		}
		if !t.serves(engine) {
			continue
		}
		if !t.isHealthy() {
			t.mu.Lock()
			unreachable = append(unreachable, t.key+": "+t.lastErr)
			t.mu.Unlock()
			continue
		}
		out = append(out, t)
	}
	if r == appconfig.EngineStrategyHybrid || (len(out) == 0 && fallback) {
		out = append([]*engineTarget{d.local}, out...)
	}
	return out, unreachable
}

// engineLease holds one slot of a target for a task's run.
type engineLease struct {
	t      *engineTarget
	engine string
	once   sync.Once
	// tried are targets a failed submit already went to (not retried).
	tried map[*engineTarget]bool
}

func (l *engineLease) local() bool { return l == nil || l.t.kind == "local" }

// Release frees the slot (safe to call more than once, and on nil).
func (l *engineLease) Release() {
	if l == nil {
		return
	}
	l.once.Do(func() {
		l.t.mu.Lock()
		l.t.inflight--
		l.t.mu.Unlock()
		engineDisp.notify()
	})
}

// pick chooses the best free target (the caller holds no locks).
func pick(cands []*engineTarget, engine string, skip map[*engineTarget]bool) *engineTarget {
	type scored struct {
		t     *engineTarget
		score float64
	}
	var free []scored
	for _, t := range cands {
		if skip[t] {
			continue
		}
		slots := t.slots()
		t.mu.Lock()
		in := t.inflight
		t.mu.Unlock()
		if in >= slots {
			continue
		}
		s := float64(in) / float64(slots)
		switch t.kind {
		case "local":
			s -= 2 // no transfer: first when free
		case "node":
			if t.warm(engine) {
				s -= 1
			}
		case "runpod":
			s += 1 // may cold-start
		}
		free = append(free, scored{t, s})
	}
	if len(free) == 0 {
		return nil
	}
	sort.SliceStable(free, func(a, b int) bool { return free[a].score < free[b].score })
	return free[0].t
}

// acquireEngine leases a target for an engine run, waiting while every
// candidate is busy. logf gets a line when it has to wait.
func acquireEngine(ctx context.Context, engine string, logf func(string), skip map[*engineTarget]bool) (*engineLease, error) {
	engine = workerapi.EngineName(engine)
	d := engineDisp
	waited := false
	for {
		cands, unreachable := d.candidates(engine)
		if len(cands) == 0 {
			msg := "no GPU worker can run " + engine
			if len(unreachable) > 0 {
				msg += " (unreachable: " + strings.Join(unreachable, "; ") + ")"
			} else {
				msg += ": add a worker to a pool on the Workers page, or set the engines strategy back to local"
			}
			return nil, errors.New(msg)
		}
		d.mu.Lock()
		changed := d.changed
		d.mu.Unlock()
		if t := pick(cands, engine, skip); t != nil {
			t.mu.Lock()
			ok := t.inflight < max(1, slotsLocked(t))
			if ok {
				t.inflight++
			}
			t.mu.Unlock()
			if ok {
				return &engineLease{t: t, engine: engine, tried: map[*engineTarget]bool{}}, nil
			}
			continue
		}
		if !waited && logf != nil {
			waited = true
			logf(fmt.Sprintf("all %d GPU worker(s) for %s are busy; waiting for one", len(cands), engine))
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-changed:
		case <-time.After(3 * time.Second):
		}
	}
}

// slotsLocked is slots() for a caller already holding t.mu.
func slotsLocked(t *engineTarget) int {
	switch t.kind {
	case "local":
		return 1
	case "runpod":
		return max(1, t.maxJobs)
	}
	if t.info != nil && t.info.Slots > 0 {
		return t.info.Slots
	}
	return 1
}

// holdLocalGPU takes the local GPU for work that runs here whatever the
// strategy (splat training): it waits for a local engine lease so the two
// never share the GPU. With the default local strategy the queue's
// gpu-diffusion bucket already serializes them and this is a no-op.
func holdLocalGPU(ctx context.Context) (func(), error) {
	d := engineDisp
	d.mu.Lock()
	onlyLocal := d.cfg.Strategy == appconfig.EngineStrategyLocal && len(d.cfg.Routes) == 0
	d.mu.Unlock()
	if onlyLocal {
		return func() {}, nil
	}
	for {
		d.mu.Lock()
		changed := d.changed
		d.mu.Unlock()
		t := d.local
		t.mu.Lock()
		if t.inflight < 1 {
			t.inflight++
			t.mu.Unlock()
			l := &engineLease{t: t}
			return l.Release, nil
		}
		t.mu.Unlock()
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-changed:
		case <-time.After(3 * time.Second):
		}
	}
}

// --- health

func (d *engineDispatcher) healthLoop() {
	for {
		time.Sleep(15 * time.Second)
		d.checkAll(context.Background())
	}
}

func (d *engineDispatcher) checkAll(ctx context.Context) {
	d.mu.Lock()
	targets := append([]*engineTarget(nil), d.targets...)
	d.mu.Unlock()
	var wg sync.WaitGroup
	for _, t := range targets {
		wg.Add(1)
		go func(t *engineTarget) {
			defer wg.Done()
			t.check(ctx)
		}(t)
	}
	wg.Wait()
	d.resize()
	d.notify()
}

func (t *engineTarget) check(ctx context.Context) {
	switch t.kind {
	case "node":
		info, err := t.node.Info(ctx, false)
		t.mu.Lock()
		defer t.mu.Unlock()
		was := t.healthy
		t.checked = time.Now()
		if err != nil {
			t.healthy, t.lastErr = false, err.Error()
			if was {
				log.Printf("engines: worker %s is unreachable: %v", t.key, err)
			}
			return
		}
		if t.info != nil && t.info.Version != info.Version {
			t.helps = nil // a new build may take new flags
		}
		t.info, t.healthy, t.lastErr = info, info.Ready, ""
		if !info.Ready {
			t.lastErr = "starting (preloading models)"
		}
		if !was && t.healthy {
			log.Printf("engines: worker %s is up (%d GPU slot(s), engines %s)", t.key, info.Slots, toolList(info))
		}
	case "runpod":
		h, err := t.rp.Health(ctx)
		t.mu.Lock()
		defer t.mu.Unlock()
		t.checked = time.Now()
		if err != nil {
			t.healthy, t.lastErr = false, err.Error()
			return
		}
		t.rpHealth, t.healthy, t.lastErr = h, true, ""
	}
}

func toolList(info *workerapi.Info) string {
	var n []string
	for _, t := range info.Tools {
		n = append(n, t.Name)
	}
	return strings.Join(n, ", ")
}

// help returns an engine's --help as the target's build prints it.
func (t *engineTarget) help(ctx context.Context, engine string) (string, error) {
	t.mu.Lock()
	h, ok := t.helps[engine]
	t.mu.Unlock()
	if ok {
		return h, nil
	}
	switch t.kind {
	case "node":
		info, err := t.node.Info(ctx, true)
		if err != nil {
			return "", err
		}
		t.mu.Lock()
		t.helps = map[string]string{}
		for _, ti := range info.Tools {
			t.helps[ti.Name] = ti.Help
		}
		h = t.helps[engine]
		t.mu.Unlock()
		return h, nil
	case "runpod":
		// a serverless worker's help would cost a (possibly cold) job: the
		// image is built from this repository, so it has every current flag
		return "--meta --model --show-prompt", nil
	}
	return "", nil
}
