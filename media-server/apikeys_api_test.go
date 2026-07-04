package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/auth"
	"github.com/stevecastle/shrike/renderer"
)

func newAPIKeyTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	for _, stmt := range []string{
		`CREATE TABLE users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE NOT NULL,
			password_hash TEXT NOT NULL,
			created_at INTEGER
		)`,
		`CREATE TABLE api_keys (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id INTEGER NOT NULL,
			name TEXT NOT NULL,
			key_hash TEXT UNIQUE NOT NULL,
			prefix TEXT NOT NULL,
			created_at INTEGER,
			last_used_at INTEGER
		)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("create table: %v", err)
		}
	}
	svc := auth.NewAuthService(db, "test-secret")
	if err := svc.Register("steve", "pw"); err != nil {
		t.Fatalf("register: %v", err)
	}
	return &Dependencies{DB: db, Auth: svc}
}

func doAPIKeys(t *testing.T, deps *Dependencies, method, target, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, target, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	apiKeysHandler(deps).ServeHTTP(rr, req)
	return rr
}

func TestAPIKeysHandler_CreateListRevoke(t *testing.T) {
	deps := newAPIKeyTestDeps(t)

	// Create
	rr := doAPIKeys(t, deps, http.MethodPost, "/auth/keys", `{"name":"ci","username":"steve"}`)
	if rr.Code != http.StatusCreated {
		t.Fatalf("create status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var created struct {
		Status   string `json:"status"`
		Key      string `json:"key"`
		ID       int64  `json:"id"`
		Username string `json:"username"`
		Prefix   string `json:"prefix"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &created); err != nil {
		t.Fatalf("create response not JSON: %v", err)
	}
	if !strings.HasPrefix(created.Key, auth.APIKeyPrefix) || created.Username != "steve" {
		t.Fatalf("unexpected create response: %+v", created)
	}

	// The minted key authenticates through the shared credential dispatch.
	claims, err := verifyCredential(deps, created.Key)
	if err != nil || claims.Username != "steve" {
		t.Fatalf("verifyCredential: claims = %+v, err = %v", claims, err)
	}

	// List shows metadata but never the key itself.
	rr = doAPIKeys(t, deps, http.MethodGet, "/auth/keys", "")
	if rr.Code != http.StatusOK {
		t.Fatalf("list status = %d", rr.Code)
	}
	if strings.Contains(rr.Body.String(), created.Key) {
		t.Error("list response leaks the plaintext key")
	}
	var list struct {
		Keys []auth.APIKey `json:"keys"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &list); err != nil || len(list.Keys) != 1 {
		t.Fatalf("list = %+v, err = %v", list, err)
	}

	// Revoke
	rr = doAPIKeys(t, deps, http.MethodDelete, "/auth/keys?id=1", "")
	if rr.Code != http.StatusOK {
		t.Fatalf("revoke status = %d; body = %s", rr.Code, rr.Body.String())
	}
	if _, err := verifyCredential(deps, created.Key); err == nil {
		t.Error("revoked key still verifies")
	}
}

func TestAPIKeysHandler_DefaultsToCallerUsername(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	token, err := deps.Auth.Login("steve", "pw")
	if err != nil {
		t.Fatalf("login: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, "/auth/keys", strings.NewReader(`{"name":"mine"}`))
	req.Header.Set("Authorization", "Bearer "+token)
	rr := httptest.NewRecorder()
	apiKeysHandler(deps).ServeHTTP(rr, req)
	if rr.Code != http.StatusCreated {
		t.Fatalf("status = %d; body = %s", rr.Code, rr.Body.String())
	}
	var created struct {
		Username string `json:"username"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &created); err != nil || created.Username != "steve" {
		t.Fatalf("username = %q, err = %v", created.Username, err)
	}
}

func TestAPIKeysHandler_Validation(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	cases := []struct {
		method, target, body string
		want                 int
	}{
		{http.MethodPost, "/auth/keys", `{not json`, http.StatusBadRequest},
		{http.MethodPost, "/auth/keys", `{"name":"","username":"steve"}`, http.StatusBadRequest},
		{http.MethodPost, "/auth/keys", `{"name":"k"}`, http.StatusBadRequest}, // no username, unauthenticated request
		{http.MethodPost, "/auth/keys", `{"name":"k","username":"nobody"}`, http.StatusBadRequest},
		{http.MethodDelete, "/auth/keys", "", http.StatusBadRequest},
		{http.MethodPut, "/auth/keys", "", http.StatusMethodNotAllowed},
	}
	for _, c := range cases {
		rr := doAPIKeys(t, deps, c.method, c.target, c.body)
		if rr.Code != c.want {
			t.Errorf("%s %s body=%q: status = %d, want %d", c.method, c.target, c.body, rr.Code, c.want)
		}
	}
}

// TestAuthMiddleware_APIKey drives the real middleware: an lk_ key must pass
// a RoleAdmin route via both Authorization: Bearer and X-API-Key, and bad or
// revoked keys must 401 for JSON clients.
func TestAuthMiddleware_APIKey(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	rrKey := doAPIKeys(t, deps, http.MethodPost, "/auth/keys", `{"name":"mw","username":"steve"}`)
	var created struct {
		Key string `json:"key"`
		ID  int64  `json:"id"`
	}
	if err := json.Unmarshal(rrKey.Body.Bytes(), &created); err != nil || created.Key == "" {
		t.Fatalf("create key: %v; body = %s", err, rrKey.Body.String())
	}

	protected := authMiddleware(deps, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), renderer.RoleAdmin)

	send := func(setAuth func(*http.Request)) int {
		req := httptest.NewRequest(http.MethodGet, "/media/api", nil)
		req.Header.Set("Accept", "application/json")
		setAuth(req)
		rr := httptest.NewRecorder()
		protected.ServeHTTP(rr, req)
		return rr.Code
	}

	if got := send(func(r *http.Request) { r.Header.Set("Authorization", "Bearer "+created.Key) }); got != http.StatusOK {
		t.Errorf("Bearer API key: status = %d, want 200", got)
	}
	if got := send(func(r *http.Request) { r.Header.Set("X-API-Key", created.Key) }); got != http.StatusOK {
		t.Errorf("X-API-Key: status = %d, want 200", got)
	}
	if got := send(func(r *http.Request) {}); got != http.StatusUnauthorized {
		t.Errorf("no credentials: status = %d, want 401", got)
	}
	if got := send(func(r *http.Request) { r.Header.Set("X-API-Key", "lk_wrong") }); got != http.StatusUnauthorized {
		t.Errorf("bad key: status = %d, want 401", got)
	}

	if err := deps.Auth.DeleteAPIKey(created.ID); err != nil {
		t.Fatalf("revoke: %v", err)
	}
	if got := send(func(r *http.Request) { r.Header.Set("X-API-Key", created.Key) }); got != http.StatusUnauthorized {
		t.Errorf("revoked key: status = %d, want 401", got)
	}
}

func TestRequestAuthToken(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	if got := requestAuthToken(req); got != "" {
		t.Errorf("no headers: got %q", got)
	}
	req.Header.Set("Authorization", "Bearer abc")
	if got := requestAuthToken(req); got != "abc" {
		t.Errorf("bearer: got %q", got)
	}
	req.Header.Set("X-API-Key", "lk_xyz")
	if got := requestAuthToken(req); got != "lk_xyz" {
		t.Errorf("X-API-Key should win: got %q", got)
	}
}
