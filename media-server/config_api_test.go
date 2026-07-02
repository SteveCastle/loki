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
