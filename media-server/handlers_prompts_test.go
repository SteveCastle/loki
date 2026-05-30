package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestDescribePromptHandlerReturnsCurrentDefault(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = "the-current-default"
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/describe", nil)
	rec := httptest.NewRecorder()

	describePromptHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var body struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response body not valid json: %v", err)
	}
	if body.Prompt != "the-current-default" {
		t.Errorf("prompt = %q, want %q", body.Prompt, "the-current-default")
	}
}

func TestDescribePromptHandlerRejectsNonGet(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/api/prompts/describe", nil)
	rec := httptest.NewRecorder()

	describePromptHandler(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", rec.Code)
	}
}

func TestDescribePromptHandlerEmptyDefault(t *testing.T) {
	old := appconfig.Get()
	cfg := old
	cfg.DescribePrompt = ""
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })

	req := httptest.NewRequest(http.MethodGet, "/api/prompts/describe", nil)
	rec := httptest.NewRecorder()

	describePromptHandler(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var body struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("response body not valid json: %v", err)
	}
	if body.Prompt != "" {
		t.Errorf("prompt = %q, want empty string", body.Prompt)
	}
}
