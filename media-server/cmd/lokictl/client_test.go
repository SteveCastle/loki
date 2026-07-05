package main

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestDoJSONSendsAuthAndDecodes(t *testing.T) {
	var gotAuth, gotAccept, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotAccept = r.Header.Get("Accept")
		b, _ := io.ReadAll(r.Body)
		gotBody = string(b)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "tok123", 5*time.Second)
	var out struct {
		OK bool `json:"ok"`
	}
	if err := c.DoJSON("POST", "/x", map[string]int{"a": 1}, &out); err != nil {
		t.Fatalf("DoJSON: %v", err)
	}
	if gotAuth != "Bearer tok123" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotAccept != "application/json" {
		t.Errorf("Accept = %q", gotAccept)
	}
	if gotBody != `{"a":1}` {
		t.Errorf("body = %q", gotBody)
	}
	if !out.OK {
		t.Errorf("decode failed: %+v", out)
	}
}

func TestDoJSONErrorMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":"Authentication required"}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "", 5*time.Second)
	err := c.DoJSON("GET", "/x", nil, nil)
	var apiErr *APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("expected APIError, got %T: %v", err, err)
	}
	if apiErr.Status != 401 || !strings.Contains(apiErr.Hint, "lokictl login") {
		t.Errorf("apiErr = %+v", apiErr)
	}
}

func TestDoJSONConnectionRefusedHint(t *testing.T) {
	c := NewClient("http://127.0.0.1:1", "", 2*time.Second)
	err := c.DoJSON("GET", "/health", nil, nil)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "lokictl health") {
		t.Errorf("error lacks hint: %v", err)
	}
}

func TestLoginStoresConfig(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("LOKICTL_CONFIG_DIR", dir)
	t.Setenv("LOKICTL_SERVER", "")
	t.Setenv("LOKICTL_TOKEN", "")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/auth/login" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var creds map[string]string
		_ = json.NewDecoder(r.Body).Decode(&creds)
		if creds["username"] != "admin" || creds["password"] != "pw" {
			t.Errorf("creds = %v", creds)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok","token":"jwt-abc","setup_required":false}`))
	}))
	defer srv.Close()

	var out, errOut strings.Builder
	code := run([]string{"--server", srv.URL, "login", "--password", "pw"}, &out, &errOut)
	if code != 0 {
		t.Fatalf("exit = %d; stderr = %s", code, errOut.String())
	}
	cfg := loadCLIConfig()
	if cfg.Token != "jwt-abc" || cfg.Server != srv.URL {
		t.Errorf("saved config = %+v", cfg)
	}
	if !strings.Contains(out.String(), `"status": "ok"`) {
		t.Errorf("stdout = %q", out.String())
	}
}

func TestUnknownCommandExits2(t *testing.T) {
	var out, errOut strings.Builder
	code := run([]string{"frobnicate"}, &out, &errOut)
	if code != 2 {
		t.Errorf("exit = %d", code)
	}
	if !strings.Contains(errOut.String(), "unknown command") {
		t.Errorf("stderr = %q", errOut.String())
	}
}

func TestHelpListsCommands(t *testing.T) {
	var out, errOut strings.Builder
	code := run([]string{"help"}, &out, &errOut)
	if code != 0 {
		t.Fatalf("exit = %d", code)
	}
	for _, want := range []string{"lokictl login", "lokictl health", "lokictl api"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("help missing %q", want)
		}
	}
}
