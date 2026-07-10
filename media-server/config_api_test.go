package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestRedactConfig(t *testing.T) {
	var cfg appconfig.Config
	cfg.DBPath = "C:/data/media.db"
	cfg.OllamaModel = "llava"
	cfg.JWTSecret = "super-secret"
	cfg.DiscordToken = "tok"
	cfg.RunPodAPIKey = "rp"
	cfg.LMStudioAPIKey = "lm"
	cfg.LlamaCppAPIKey = "lc"
	cfg.Roots = []appconfig.StorageRoot{
		{Type: "s3", Label: "bucket", AccessKey: "AK", SecretKey: "SK"},
		{Type: "local", Label: "disk", Path: "D:/media"},
	}

	got := redactConfig(cfg)

	for name, v := range map[string]string{
		"JWTSecret":      got.JWTSecret,
		"DiscordToken":   got.DiscordToken,
		"RunPodAPIKey":   got.RunPodAPIKey,
		"LMStudioAPIKey": got.LMStudioAPIKey,
		"LlamaCppAPIKey": got.LlamaCppAPIKey,
		"Roots[0].AccessKey": got.Roots[0].AccessKey,
		"Roots[0].SecretKey": got.Roots[0].SecretKey,
	} {
		if v != "<redacted>" {
			t.Errorf("%s = %q, want <redacted>", name, v)
		}
	}
	// Empty secrets stay empty; non-secrets untouched; original not mutated.
	if got.Roots[1].SecretKey != "" {
		t.Errorf("empty SecretKey redacted to %q", got.Roots[1].SecretKey)
	}
	if got.DBPath != "C:/data/media.db" || got.OllamaModel != "llava" || got.Roots[1].Path != "D:/media" {
		t.Errorf("non-secret fields changed: %+v", got)
	}
	if cfg.JWTSecret != "super-secret" || cfg.Roots[0].SecretKey != "SK" {
		t.Errorf("redactConfig mutated its input")
	}
}

func TestConfigGetAPIHandler_MethodGuard(t *testing.T) {
	deps := &Dependencies{}
	req := httptest.NewRequest(http.MethodPost, "/api/config", nil)
	rr := httptest.NewRecorder()
	configGetAPIHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusMethodNotAllowed {
		t.Errorf("POST status = %d, want 405", rr.Code)
	}
}

func TestMergeIncomingRoots_RecoversRedactedCreds(t *testing.T) {
	existing := []appconfig.StorageRoot{
		{Type: "s3", Label: "Bucket A", Bucket: "a", Endpoint: "https://s3.example.com", AccessKey: "AK-A", SecretKey: "SK-A"},
		{Type: "s3", Label: "Bucket B", Bucket: "b", Endpoint: "https://s3.example.com", AccessKey: "AK-B", SecretKey: "SK-B"},
		{Type: "local", Label: "Disk", Path: "D:/media"},
	}
	incoming := []appconfig.StorageRoot{
		// Same bucket, renamed label, redacted creds → recovered by S3 identity.
		{Type: "s3", Label: "Renamed A", Bucket: "a", Endpoint: "https://s3.example.com", AccessKey: "<redacted>", SecretKey: "<redacted>"},
		// User typed a fresh secret; access key still redacted.
		{Type: "s3", Label: "Bucket B", Bucket: "b", Endpoint: "https://s3.example.com", AccessKey: "<redacted>", SecretKey: "NEW-SK"},
		// Brand-new root posted with placeholders (no stored counterpart):
		// placeholders must be dropped, never persisted literally.
		{Type: "s3", Label: "New", Bucket: "new", AccessKey: "<redacted>", SecretKey: "<redacted>"},
		{Type: "local", Label: "Disk", Path: "D:/media"},
	}

	got := mergeIncomingRoots(incoming, existing)

	if got[0].AccessKey != "AK-A" || got[0].SecretKey != "SK-A" {
		t.Errorf("root 0 creds = %q/%q, want recovered AK-A/SK-A", got[0].AccessKey, got[0].SecretKey)
	}
	if got[0].Label != "Renamed A" {
		t.Errorf("root 0 label = %q, want the incoming rename kept", got[0].Label)
	}
	if got[1].AccessKey != "AK-B" || got[1].SecretKey != "NEW-SK" {
		t.Errorf("root 1 creds = %q/%q, want AK-B/NEW-SK", got[1].AccessKey, got[1].SecretKey)
	}
	if got[2].AccessKey != "" || got[2].SecretKey != "" {
		t.Errorf("root 2 creds = %q/%q, want placeholders dropped", got[2].AccessKey, got[2].SecretKey)
	}
	if got[3] != incoming[3] {
		t.Errorf("local root changed: %+v", got[3])
	}
	// Input slices untouched.
	if incoming[0].AccessKey != "<redacted>" || existing[0].AccessKey != "AK-A" {
		t.Error("mergeIncomingRoots mutated its inputs")
	}
}

func TestKeepStoredIfRedacted(t *testing.T) {
	if got := keepStoredIfRedacted("<redacted>", "stored"); got != "stored" {
		t.Errorf("redacted → %q, want stored", got)
	}
	if got := keepStoredIfRedacted("  new-value  ", "stored"); got != "new-value" {
		t.Errorf("new value → %q, want trimmed new-value", got)
	}
	if got := keepStoredIfRedacted("", "stored"); got != "" {
		t.Errorf("empty → %q, want empty (caller's non-empty guard keeps stored)", got)
	}
}
