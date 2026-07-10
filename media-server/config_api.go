package main

// GET /api/config — JSON view of the active configuration with secrets
// redacted, so API clients (lokictl) can read-modify-write via POST /config.
// Untagged so every platform main registers it.

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
)

// redactedPlaceholder is what GET /api/config and the config page show in
// place of stored secrets. POST /config treats it as "keep the stored value"
// so redacted reads can be posted back without destroying credentials.
const redactedPlaceholder = "<redacted>"

// redactRoots returns a copy of roots with S3 credentials replaced by the
// redaction placeholder (empty credentials stay empty).
func redactRoots(roots []appconfig.StorageRoot) []appconfig.StorageRoot {
	out := make([]appconfig.StorageRoot, len(roots))
	copy(out, roots)
	for i := range out {
		if out[i].AccessKey != "" {
			out[i].AccessKey = redactedPlaceholder
		}
		if out[i].SecretKey != "" {
			out[i].SecretKey = redactedPlaceholder
		}
	}
	return out
}

// redactConfig returns a copy of cfg with every secret replaced by
// the redaction placeholder (empty secrets stay empty). The Roots slice is
// cloned so the caller's config is never mutated.
func redactConfig(cfg appconfig.Config) appconfig.Config {
	red := func(s string) string {
		if s != "" {
			return redactedPlaceholder
		}
		return ""
	}
	cfg.JWTSecret = red(cfg.JWTSecret)
	cfg.DiscordToken = red(cfg.DiscordToken)
	cfg.RunPodAPIKey = red(cfg.RunPodAPIKey)
	cfg.LMStudioAPIKey = red(cfg.LMStudioAPIKey)
	cfg.LlamaCppAPIKey = red(cfg.LlamaCppAPIKey)
	cfg.Roots = redactRoots(cfg.Roots)
	return cfg
}

// mergeIncomingRoots resolves redaction placeholders in a posted roots list by
// recovering the real credentials from the currently stored roots. Incoming
// roots are matched to stored ones by S3 identity (bucket+endpoint+prefix)
// first, then by label. A placeholder with no stored counterpart is dropped so
// the literal "<redacted>" is never persisted as a credential.
func mergeIncomingRoots(incoming, existing []appconfig.StorageRoot) []appconfig.StorageRoot {
	out := make([]appconfig.StorageRoot, len(incoming))
	copy(out, incoming)
	for i := range out {
		if out[i].AccessKey != redactedPlaceholder && out[i].SecretKey != redactedPlaceholder {
			continue
		}
		var match *appconfig.StorageRoot
		for j := range existing {
			if existing[j].Type == "s3" && out[i].Type == "s3" &&
				existing[j].Bucket == out[i].Bucket &&
				existing[j].Endpoint == out[i].Endpoint &&
				existing[j].Prefix == out[i].Prefix {
				match = &existing[j]
				break
			}
		}
		if match == nil {
			for j := range existing {
				if existing[j].Label == out[i].Label {
					match = &existing[j]
					break
				}
			}
		}
		if out[i].AccessKey == redactedPlaceholder {
			out[i].AccessKey = ""
			if match != nil {
				out[i].AccessKey = match.AccessKey
			}
		}
		if out[i].SecretKey == redactedPlaceholder {
			out[i].SecretKey = ""
			if match != nil {
				out[i].SecretKey = match.SecretKey
			}
		}
	}
	return out
}

// keepStoredIfRedacted returns the stored value when the incoming one is the
// redaction placeholder, otherwise the (trimmed) incoming value. Used by the
// config POST handler for top-level secrets so redacted reads round-trip.
func keepStoredIfRedacted(incoming, stored string) string {
	v := strings.TrimSpace(incoming)
	if v == redactedPlaceholder {
		return stored
	}
	return v
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
