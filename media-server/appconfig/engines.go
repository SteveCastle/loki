package appconfig

import (
	"encoding/json"
	"log"
	"os"
	"strings"
)

// Where the loki-* GPU engines (loki-retouch, loki-reshoot, loki-dream,
// loki-segment) run.
//
//   - "local" (the default): on this machine's GPU, as they always have.
//   - "pool": on remote GPU workers (loki-worker nodes and RunPod
//     serverless endpoints); this machine needs no GPU.
//   - "hybrid": this machine's GPU is one more worker beside the pools.
//
// Routes overrides the strategy per engine ("loki-segment": "local" keeps
// interactive selection on this machine while generation goes to a pool).
const (
	EngineStrategyLocal  = "local"
	EngineStrategyPool   = "pool"
	EngineStrategyHybrid = "hybrid"
)

// Worker pool kinds.
const (
	// PoolKindNodes: loki-worker HTTP servers this server talks to directly
	// (RunPod pods, cloud VMs, a container on another PC, a load-balanced
	// serverless endpoint).
	PoolKindNodes = "nodes"
	// PoolKindRunPod: a RunPod queue-based serverless endpoint running the
	// loki-worker image; scales from zero, files travel through a bucket.
	PoolKindRunPod = "runpod"
)

// EnginesConfig is the engines section of the config.
type EnginesConfig struct {
	Strategy string            `json:"strategy,omitempty"`
	Pools    []WorkerPool      `json:"pools"`
	Routes   map[string]string `json:"routes"`
	// Fallback lets a run go to this machine's GPU when no pool worker is
	// reachable (strategy "pool" only; "hybrid" always may).
	Fallback bool `json:"fallback,omitempty"`
}

// WorkerPool is a group of interchangeable GPU workers.
type WorkerPool struct {
	Name string `json:"name"`
	Kind string `json:"kind"` // PoolKindNodes | PoolKindRunPod
	// Disabled pools are kept in the config but never used.
	Disabled bool `json:"disabled,omitempty"`
	// Engines this pool serves (empty: every engine its workers report).
	Engines []string `json:"engines,omitempty"`

	// nodes
	Nodes []WorkerNode `json:"nodes,omitempty"`

	// runpod
	EndpointID string `json:"endpointId,omitempty"`
	APIKey     string `json:"apiKey,omitempty"`
	// MaxJobs caps jobs in flight on the endpoint (default 4; match the
	// endpoint's max workers).
	MaxJobs int `json:"maxJobs,omitempty"`

	// Transfer is how files reach workers that this server cannot address
	// (serverless). Nodes take uploads directly and ignore it.
	Transfer *PoolTransfer `json:"transfer,omitempty"`

	// Provider records what `lokictl pool create` made, so `lokictl pool
	// destroy` can take it down again.
	Provider *PoolProvider `json:"provider,omitempty"`

	// FromEnv marks a pool supplied by LOWKEY_ENGINES / LOWKEY_WORKER_NODES:
	// used at runtime, never written to the config file.
	FromEnv bool `json:"fromEnv,omitempty"`
}

// WorkerNode is one loki-worker HTTP server.
type WorkerNode struct {
	Name  string `json:"name,omitempty"`
	URL   string `json:"url"`
	Token string `json:"token,omitempty"`
	// Disabled nodes stay listed but get no work (draining).
	Disabled bool `json:"disabled,omitempty"`
}

// PoolTransfer is the bucket serverless jobs pass files through: either a
// configured S3 storage root by label, or a bucket of its own.
type PoolTransfer struct {
	Root      string `json:"root,omitempty"`
	Endpoint  string `json:"endpoint,omitempty"`
	Region    string `json:"region,omitempty"`
	Bucket    string `json:"bucket,omitempty"`
	Prefix    string `json:"prefix,omitempty"`
	AccessKey string `json:"accessKey,omitempty"`
	SecretKey string `json:"secretKey,omitempty"`
}

// PoolProvider is the cloud resources a pool was provisioned with.
type PoolProvider struct {
	Name       string   `json:"name"` // "runpod"
	PodIDs     []string `json:"podIds,omitempty"`
	TemplateID string   `json:"templateId,omitempty"`
	EndpointID string   `json:"endpointId,omitempty"`
}

// persistedEngines is the section as the config file stores it: without the
// pools that came from the environment.
func persistedEngines(e EnginesConfig) EnginesConfig {
	var keep []WorkerPool
	for _, p := range e.Pools {
		if !p.FromEnv {
			keep = append(keep, p)
		}
	}
	e.Pools = keep
	return e
}

// NormalizeEngines fills defaults and drops junk (in place).
func NormalizeEngines(e *EnginesConfig) {
	switch strings.ToLower(strings.TrimSpace(e.Strategy)) {
	case EngineStrategyPool:
		e.Strategy = EngineStrategyPool
	case EngineStrategyHybrid:
		e.Strategy = EngineStrategyHybrid
	default:
		e.Strategy = EngineStrategyLocal
	}
	var pools []WorkerPool
	seen := map[string]bool{}
	for _, p := range e.Pools {
		p.Name = strings.TrimSpace(p.Name)
		if p.Name == "" || seen[strings.ToLower(p.Name)] {
			continue
		}
		seen[strings.ToLower(p.Name)] = true
		p.Kind = strings.ToLower(strings.TrimSpace(p.Kind))
		if p.Kind != PoolKindRunPod {
			p.Kind = PoolKindNodes
		}
		var nodes []WorkerNode
		for _, n := range p.Nodes {
			n.URL = strings.TrimRight(strings.TrimSpace(n.URL), "/")
			if n.URL == "" {
				continue
			}
			if n.Name == "" {
				n.Name = n.URL
			}
			nodes = append(nodes, n)
		}
		p.Nodes = nodes
		if p.MaxJobs <= 0 {
			p.MaxJobs = 4
		}
		pools = append(pools, p)
	}
	e.Pools = pools
	routes := map[string]string{}
	for k, v := range e.Routes {
		k = strings.ToLower(strings.TrimSpace(k))
		if k == "" {
			continue
		}
		if !strings.HasPrefix(k, "loki-") {
			k = "loki-" + k
		}
		routes[k] = strings.TrimSpace(v)
	}
	if len(routes) == 0 {
		routes = nil
	}
	e.Routes = routes
}

// applyEnginesEnv: LOWKEY_ENGINES (the whole section as JSON) replaces the
// file's; LOWKEY_ENGINES_STRATEGY sets just the strategy; LOWKEY_WORKER_NODES
// ("https://a:8080|token,https://b:8080|token") adds a "nodes" pool named
// "env" — the Docker-friendly ways to point a server at workers.
func applyEnginesEnv(c *Config) {
	if v := strings.TrimSpace(os.Getenv("LOWKEY_ENGINES")); v != "" {
		var e EnginesConfig
		if err := json.Unmarshal([]byte(v), &e); err != nil {
			log.Printf("Warning: LOWKEY_ENGINES is not valid JSON (%v); ignored", err)
		} else {
			for i := range e.Pools {
				e.Pools[i].FromEnv = true
			}
			c.Engines = e
		}
	}
	if v := strings.TrimSpace(os.Getenv("LOWKEY_WORKER_NODES")); v != "" {
		p := WorkerPool{Name: "env", Kind: PoolKindNodes, FromEnv: true}
		for _, item := range strings.Split(v, ",") {
			url, token, _ := strings.Cut(strings.TrimSpace(item), "|")
			if url != "" {
				p.Nodes = append(p.Nodes, WorkerNode{URL: url, Token: token})
			}
		}
		if len(p.Nodes) > 0 {
			c.Engines.Pools = append(c.Engines.Pools, p)
			if c.Engines.Strategy == "" || c.Engines.Strategy == EngineStrategyLocal {
				c.Engines.Strategy = EngineStrategyPool
			}
		}
	}
	if v := strings.TrimSpace(os.Getenv("LOWKEY_ENGINES_STRATEGY")); v != "" {
		c.Engines.Strategy = v
	}
	NormalizeEngines(&c.Engines)
}
