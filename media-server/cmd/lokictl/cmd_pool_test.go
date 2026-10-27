package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/workerapi/provision"
)

// fakeEnginesServer serves GET/PUT /api/engines and the probe.
func fakeEnginesServer(t *testing.T, cfg appconfig.EnginesConfig) (*httptest.Server, func() appconfig.EnginesConfig) {
	var mu sync.Mutex
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.URL.Path == "/api/engines" && r.Method == http.MethodGet:
			_ = json.NewEncoder(w).Encode(map[string]any{"config": cfg, "status": map[string]any{}})
		case r.URL.Path == "/api/engines" && r.Method == http.MethodPut:
			var next appconfig.EnginesConfig
			_ = json.NewDecoder(r.Body).Decode(&next)
			cfg = next
			_ = json.NewEncoder(w).Encode(map[string]any{"config": cfg})
		case r.URL.Path == "/api/engines/probe":
			var p map[string]string
			_ = json.NewDecoder(r.Body).Decode(&p)
			if p["url"] == "http://down" {
				w.WriteHeader(http.StatusBadGateway)
				_, _ = w.Write([]byte(`{"error":"connection refused"}`))
				return
			}
			_, _ = w.Write([]byte(`{"ok":true}`))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, func() appconfig.EnginesConfig { mu.Lock(); defer mu.Unlock(); return cfg }
}

func TestPoolAddNodeAndRoute(t *testing.T) {
	srv, got := fakeEnginesServer(t, appconfig.EnginesConfig{Strategy: "local"})
	a, _, _ := appForServer(srv.URL)
	if code := cmdPoolAddNode(a, []string{"gpu", "https://abc-8080.proxy.runpod.net", "--token", "tok", "--name", "pod1"}); code != 0 {
		t.Fatalf("add-node exit %d", code)
	}
	c := got()
	if c.Strategy != "pool" || len(c.Pools) != 1 || c.Pools[0].Nodes[0].Token != "tok" || c.Pools[0].Nodes[0].Name != "pod1" {
		t.Fatalf("config after add-node: %+v", c)
	}
	if code := cmdPoolRoute(a, []string{"segment", "local"}); code != 0 || got().Routes["loki-segment"] != "local" {
		t.Fatalf("route: %d %+v", code, got().Routes)
	}
	if code := cmdPoolAddNode(a, []string{"gpu", "http://down"}); code == 0 {
		t.Fatal("an unreachable node was added without --no-test")
	}
	if code := cmdPoolRemove(a, []string{"gpu", "https://abc-8080.proxy.runpod.net"}); code != 0 || len(got().Pools[0].Nodes) != 0 {
		t.Fatalf("remove node: %+v", got())
	}
}

// pool create runpod-pods makes one pod per --count with its own token and
// adds them as nodes at RunPod's proxy URL; destroy deletes them.
func TestPoolCreateAndDestroyRunPodPods(t *testing.T) {
	var mu sync.Mutex
	var created []map[string]any
	var deleted []string
	rp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if r.Header.Get("Authorization") != "Bearer rpk" {
			w.WriteHeader(401)
			return
		}
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/pods":
			var b map[string]any
			_ = json.NewDecoder(r.Body).Decode(&b)
			created = append(created, b)
			_ = json.NewEncoder(w).Encode(map[string]string{"id": "pod" + string(rune('A'+len(created)-1))})
		case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/pods/"):
			deleted = append(deleted, strings.TrimPrefix(r.URL.Path, "/pods/"))
		default:
			t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
		}
	}))
	defer rp.Close()
	old := runpodBase
	runpodBase = rp.URL
	defer func() { runpodBase = old }()

	srv, got := fakeEnginesServer(t, appconfig.EnginesConfig{})
	a, _, _ := appForServer(srv.URL)
	if code := cmdPoolCreate(a, []string{"runpod-pods", "gpu", "--count", "2", "--runpod-key", "rpk", "--preload", "retouch"}); code != 0 {
		t.Fatalf("create exit %d", code)
	}
	c := got()
	if len(c.Pools) != 1 || len(c.Pools[0].Nodes) != 2 || c.Pools[0].Provider == nil || len(c.Pools[0].Provider.PodIDs) != 2 {
		t.Fatalf("pool after create: %+v", c)
	}
	n := c.Pools[0].Nodes[0]
	if n.URL != provision.PodURL("podA") || n.Token == "" || n.Token == c.Pools[0].Nodes[1].Token {
		t.Fatalf("nodes: %+v", c.Pools[0].Nodes)
	}
	env := created[0]["env"].(map[string]any)
	if env["LOKI_WORKER_TOKEN"] != n.Token || env["LOKI_WORKER_PRELOAD"] != "loki-retouch" {
		t.Fatalf("pod env %v", env)
	}
	if code := cmdPoolDestroy(a, []string{"gpu", "--runpod-key", "rpk"}); code != 0 {
		t.Fatalf("destroy exit %d", code)
	}
	if len(deleted) != 2 || len(got().Pools) != 0 {
		t.Fatalf("deleted %v, pools %+v", deleted, got().Pools)
	}
}
