package main

import (
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestEngineSecretsRoundTrip(t *testing.T) {
	stored := appconfig.EnginesConfig{Strategy: "pool", Pools: []appconfig.WorkerPool{
		{Name: "gpu", Kind: "nodes", Nodes: []appconfig.WorkerNode{{URL: "https://a:8080", Token: "tok-a"}, {URL: "https://b:8080", Token: "tok-b"}},
			Provider: &appconfig.PoolProvider{Name: "runpod", PodIDs: []string{"p1"}}},
		{Name: "sls", Kind: "runpod", EndpointID: "ep", APIKey: "rp-key", Transfer: &appconfig.PoolTransfer{Bucket: "b", AccessKey: "ak", SecretKey: "sk"}},
	}}
	red := redactEngines(stored)
	if red.Pools[0].Nodes[0].Token != redactedPlaceholder || red.Pools[1].APIKey != redactedPlaceholder || red.Pools[1].Transfer.SecretKey != redactedPlaceholder {
		t.Fatalf("not redacted: %+v", red)
	}
	if stored.Pools[0].Nodes[0].Token != "tok-a" || stored.Pools[1].Transfer.SecretKey != "sk" {
		t.Fatal("redaction changed the stored config")
	}
	// a redacted read posted back, with a new node and the provider record dropped
	red.Pools[0].Nodes = append(red.Pools[0].Nodes, appconfig.WorkerNode{URL: "https://c:8080", Token: "tok-c"})
	red.Pools[0].Provider = nil
	got := mergeEngineSecrets(red, stored)
	if got.Pools[0].Nodes[0].Token != "tok-a" || got.Pools[0].Nodes[1].Token != "tok-b" || got.Pools[0].Nodes[2].Token != "tok-c" {
		t.Fatalf("node tokens: %+v", got.Pools[0].Nodes)
	}
	if got.Pools[1].APIKey != "rp-key" || got.Pools[1].Transfer.AccessKey != "ak" || got.Pools[1].Transfer.SecretKey != "sk" {
		t.Fatalf("runpod secrets: %+v %+v", got.Pools[1], got.Pools[1].Transfer)
	}
	if got.Pools[0].Provider == nil || got.Pools[0].Provider.PodIDs[0] != "p1" {
		t.Fatal("the provider record was lost")
	}
	// a placeholder with nothing stored behind it is never saved
	fresh := mergeEngineSecrets(appconfig.EnginesConfig{Pools: []appconfig.WorkerPool{{Name: "new", Nodes: []appconfig.WorkerNode{{URL: "https://x", Token: redactedPlaceholder}}}}}, stored)
	if strings.Contains(fresh.Pools[0].Nodes[0].Token, "redacted") {
		t.Fatal("a placeholder token was kept")
	}
	// GET /api/config redacts the section too
	cfg := redactConfig(appconfig.Config{Engines: stored})
	if cfg.Engines.Pools[1].APIKey != redactedPlaceholder {
		t.Fatal("/api/config leaks the RunPod key")
	}
}
