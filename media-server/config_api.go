package main

// GET /api/config — JSON view of the active configuration with secrets
// redacted, so API clients (lokictl) can read-modify-write via POST /config.
// Untagged so every platform main registers it.

import (
	"encoding/json"
	"net/http"

	"github.com/stevecastle/shrike/appconfig"
)

// redactConfig returns a copy of cfg with every secret replaced by
// "<redacted>" (empty secrets stay empty). The Roots slice is cloned so the
// caller's config is never mutated.
func redactConfig(cfg appconfig.Config) appconfig.Config {
	red := func(s string) string {
		if s != "" {
			return "<redacted>"
		}
		return ""
	}
	cfg.JWTSecret = red(cfg.JWTSecret)
	cfg.DiscordToken = red(cfg.DiscordToken)
	cfg.RunPodAPIKey = red(cfg.RunPodAPIKey)
	cfg.LMStudioAPIKey = red(cfg.LMStudioAPIKey)
	cfg.LlamaCppAPIKey = red(cfg.LlamaCppAPIKey)
	roots := make([]appconfig.StorageRoot, len(cfg.Roots))
	copy(roots, cfg.Roots)
	for i := range roots {
		roots[i].AccessKey = red(roots[i].AccessKey)
		roots[i].SecretKey = red(roots[i].SecretKey)
	}
	cfg.Roots = roots
	return cfg
}

func configGetAPIHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, `{"error":"Use GET"}`, http.StatusMethodNotAllowed)
			return
		}
		cfg, _, err := appconfig.Load()
		if err != nil {
			http.Error(w, `{"error":"failed to load config"}`, http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(redactConfig(cfg))
	}
}
