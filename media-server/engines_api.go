package main

// GPU worker pools: where the loki-* engines run (Config.Engines), their
// live status, and the GPU Engines tab of the Config page. Untagged so every platform main
// registers it (registerEngineRoutes).

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/tasks"
)

// redactEngines replaces every worker token, API key and bucket secret with
// the redaction placeholder (a deep copy: the caller's config is untouched).
func redactEngines(e appconfig.EnginesConfig) appconfig.EnginesConfig {
	red := func(s string) string {
		if s != "" {
			return redactedPlaceholder
		}
		return ""
	}
	pools := make([]appconfig.WorkerPool, len(e.Pools))
	for i, p := range e.Pools {
		p.APIKey = red(p.APIKey)
		p.Nodes = append([]appconfig.WorkerNode(nil), p.Nodes...)
		for k := range p.Nodes {
			p.Nodes[k].Token = red(p.Nodes[k].Token)
		}
		if p.Transfer != nil {
			t := *p.Transfer
			t.AccessKey, t.SecretKey = red(t.AccessKey), red(t.SecretKey)
			p.Transfer = &t
		}
		pools[i] = p
	}
	e.Pools = pools
	return e
}

// mergeEngineSecrets restores redacted secrets in a posted engines section
// from the stored one (pools matched by name, nodes by URL); a placeholder
// with nothing to restore is dropped rather than saved.
func mergeEngineSecrets(in, stored appconfig.EnginesConfig) appconfig.EnginesConfig {
	keep := func(v, old string) string {
		if strings.TrimSpace(v) == redactedPlaceholder {
			return old
		}
		return strings.TrimSpace(v)
	}
	for i := range in.Pools {
		var old *appconfig.WorkerPool
		for k := range stored.Pools {
			if strings.EqualFold(stored.Pools[k].Name, in.Pools[i].Name) {
				old = &stored.Pools[k]
			}
		}
		p := &in.Pools[i]
		var oldKey string
		var oldTr appconfig.PoolTransfer
		if old != nil {
			oldKey = old.APIKey
			if old.Transfer != nil {
				oldTr = *old.Transfer
			}
		}
		p.APIKey = keep(p.APIKey, oldKey)
		if p.Transfer != nil {
			p.Transfer.AccessKey = keep(p.Transfer.AccessKey, oldTr.AccessKey)
			p.Transfer.SecretKey = keep(p.Transfer.SecretKey, oldTr.SecretKey)
		}
		for k := range p.Nodes {
			oldTok := ""
			if old != nil {
				for _, n := range old.Nodes {
					if strings.TrimRight(n.URL, "/") == strings.TrimRight(p.Nodes[k].URL, "/") {
						oldTok = n.Token
					}
				}
			}
			p.Nodes[k].Token = keep(p.Nodes[k].Token, oldTok)
		}
		if p.Provider == nil && old != nil {
			p.Provider = old.Provider
		}
	}
	return in
}

// saveEngines persists a new engines section and applies it at once.
func saveEngines(deps *Dependencies, e appconfig.EnginesConfig) error {
	cfg := currentConfig
	// routes the new section drops are written as "" (= follow the
	// strategy): the config file merges objects key by key
	for k := range cfg.Engines.Routes {
		if _, ok := e.Routes[k]; !ok {
			if e.Routes == nil {
				e.Routes = map[string]string{}
			}
			e.Routes[k] = ""
		}
	}
	appconfig.NormalizeEngines(&e)
	cfg.Engines = e
	if _, err := appconfig.Save(cfg); err != nil {
		return err
	}
	currentConfig = cfg
	tasks.ApplyHostLimits(deps.Queue, cfg)
	return nil
}

func registerEngineRoutes(mux *http.ServeMux, deps *Dependencies) {
	admin := func(fn http.HandlerFunc) http.HandlerFunc {
		return renderer.ApplyMiddlewares(fn, renderer.RoleAdmin)
	}
	// the Workers page became the Config page's GPU Engines tab
	mux.HandleFunc("GET /workers", admin(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/config#engines", http.StatusFound)
	}))
	// GET: the engines section (secrets redacted) and the live status;
	// ?local=1 adds this machine's engine installs (versions cost a
	// subprocess the first time).
	mux.HandleFunc("GET /api/engines", admin(func(w http.ResponseWriter, r *http.Request) {
		out := map[string]any{"config": redactEngines(currentConfig.Engines), "status": tasks.EnginesStatus()}
		if r.URL.Query().Get("local") == "1" {
			out["local"] = tasks.LocalEngines()
		}
		writeJSON(w, out)
	}))
	// PUT: replace the engines section (redacted secrets keep their values).
	mux.HandleFunc("PUT /api/engines", admin(func(w http.ResponseWriter, r *http.Request) {
		var in appconfig.EnginesConfig
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
			httpError(w, "bad json: "+err.Error(), http.StatusBadRequest)
			return
		}
		in = mergeEngineSecrets(in, currentConfig.Engines)
		if err := saveEngines(deps, in); err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 20*time.Second)
		defer cancel()
		writeJSON(w, map[string]any{"config": redactEngines(currentConfig.Engines), "status": tasks.CheckEngines(ctx)})
	}))
	// POST check: probe every worker now.
	mux.HandleFunc("POST /api/engines/check", admin(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		writeJSON(w, tasks.CheckEngines(ctx))
	}))
	// POST warm {pool, engines, model}: preload models on the workers.
	mux.HandleFunc("POST /api/engines/warm", admin(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Pool    string   `json:"pool"`
			Engines []string `json:"engines"`
			Model   string   `json:"model"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || len(req.Engines) == 0 {
			httpError(w, `give {"engines": ["loki-retouch", ...]}`, http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), time.Minute)
		defer cancel()
		writeJSON(w, map[string]any{"notes": tasks.WarmEngines(ctx, req.Pool, req.Engines, req.Model)})
	}))
	// POST probe {kind, url, token, endpointId, apiKey}: check a worker before adding it.
	mux.HandleFunc("POST /api/engines/probe", admin(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Kind       string `json:"kind"`
			URL        string `json:"url"`
			Token      string `json:"token"`
			EndpointID string `json:"endpointId"`
			APIKey     string `json:"apiKey"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			httpError(w, "bad json", http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()
		info, err := tasks.ProbeWorker(ctx, req.Kind, strings.TrimSpace(req.URL), strings.TrimSpace(req.Token), strings.TrimSpace(req.EndpointID), strings.TrimSpace(req.APIKey))
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		writeJSON(w, map[string]any{"ok": true, "info": info})
	}))
}
