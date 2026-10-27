package appconfig

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestNormalizeEngines(t *testing.T) {
	e := EnginesConfig{Strategy: "HYBRID", Routes: map[string]string{"segment": "local", " ": "x"}, Pools: []WorkerPool{
		{Name: "gpu", Nodes: []WorkerNode{{URL: "https://a:8080/"}, {URL: " "}}},
		{Name: "GPU"}, // duplicate name
		{Name: ""},
		{Name: "sls", Kind: "RunPod", EndpointID: "e"},
	}}
	NormalizeEngines(&e)
	if e.Strategy != EngineStrategyHybrid {
		t.Errorf("strategy %q", e.Strategy)
	}
	if len(e.Pools) != 2 || e.Pools[0].Kind != PoolKindNodes || e.Pools[1].Kind != PoolKindRunPod || e.Pools[1].MaxJobs != 4 {
		t.Fatalf("pools %+v", e.Pools)
	}
	if n := e.Pools[0].Nodes; len(n) != 1 || n[0].URL != "https://a:8080" || n[0].Name != n[0].URL {
		t.Errorf("nodes %+v", n)
	}
	if e.Routes["loki-segment"] != "local" || len(e.Routes) != 1 {
		t.Errorf("routes %v", e.Routes)
	}
	var zero EnginesConfig
	NormalizeEngines(&zero)
	if zero.Strategy != EngineStrategyLocal {
		t.Errorf("default strategy %q", zero.Strategy)
	}
}

// Pools from LOWKEY_WORKER_NODES are used but never written to config.json.
func TestEnvPoolsNotPersisted(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	old := getConfigPath
	getConfigPath = func() (string, error) { return path, nil }
	t.Cleanup(func() { getConfigPath = old })
	t.Setenv("LOWKEY_WORKER_NODES", "https://a:8080|tok,https://b:8080|tok2")

	c := defaultConfig()
	c.Engines.Pools = []WorkerPool{{Name: "mine", Nodes: []WorkerNode{{URL: "https://c:8080", Token: "t3"}}}}
	applyEnginesEnv(&c)
	if c.Engines.Strategy != EngineStrategyPool || len(c.Engines.Pools) != 2 || len(c.Engines.Pools[1].Nodes) != 2 || c.Engines.Pools[1].Nodes[1].Token != "tok2" {
		t.Fatalf("env pool not applied: %+v", c.Engines)
	}
	if _, err := Save(c); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	var onDisk Config
	if err := json.Unmarshal(b, &onDisk); err != nil {
		t.Fatal(err)
	}
	if len(onDisk.Engines.Pools) != 1 || onDisk.Engines.Pools[0].Name != "mine" {
		t.Fatalf("on disk: %+v", onDisk.Engines.Pools)
	}
	if got := Get().Engines.Pools; len(got) != 2 {
		t.Fatalf("in memory the env pool must stay: %+v", got)
	}
}

// Turning a switch off must reach the file: Save merges key by key, so a
// false that marshals to nothing would leave the stored true behind.
func TestEngineSwitchesOffPersist(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.json")
	old := getConfigPath
	getConfigPath = func() (string, error) { return path, nil }
	t.Cleanup(func() { getConfigPath = old })

	c := defaultConfig()
	c.Engines.LocalOnly, c.Engines.Fallback = true, true
	if _, err := Save(c); err != nil {
		t.Fatal(err)
	}
	c.Engines.LocalOnly, c.Engines.Fallback = false, false
	if _, err := Save(c); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(path)
	var onDisk Config
	if err := json.Unmarshal(b, &onDisk); err != nil {
		t.Fatal(err)
	}
	if onDisk.Engines.LocalOnly || onDisk.Engines.Fallback {
		t.Fatalf("switches still on in the file: %+v", onDisk.Engines)
	}
}
