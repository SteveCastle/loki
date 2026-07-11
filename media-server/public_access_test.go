package main

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/renderer"
)

// setPublicAccess flips the in-memory config flag for the duration of a test.
func setPublicAccess(t *testing.T, on bool) {
	t.Helper()
	old := appconfig.Get()
	cfg := old
	cfg.AllowPublicAccess = on
	appconfig.Set(cfg)
	t.Cleanup(func() { appconfig.Set(old) })
}

func ok200() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

func loginToken(t *testing.T, deps *Dependencies) string {
	t.Helper()
	tok, err := deps.Auth.Login("steve", "pw")
	if err != nil {
		t.Fatalf("login: %v", err)
	}
	return tok
}

func TestRolePublicRead_FlagOff(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	setPublicAccess(t, false)
	h := authMiddleware(deps, ok200(), renderer.RolePublicRead)

	// Browser-shaped request: redirected to login.
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/swipe", nil))
	if rr.Code != http.StatusFound {
		t.Fatalf("anonymous browser request: got %d, want 302", rr.Code)
	}

	// JSON-shaped request: 401.
	req := httptest.NewRequest(http.MethodGet, "/api/media/query", nil)
	req.Header.Set("Accept", "application/json")
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous JSON request: got %d, want 401", rr.Code)
	}
}

func TestRolePublicRead_FlagOn(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	setPublicAccess(t, true)
	h := authMiddleware(deps, ok200(), renderer.RolePublicRead)

	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, "/swipe", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("anonymous request with flag on: got %d, want 200", rr.Code)
	}
}

func TestRolePublicRead_FlagOn_AdminStillWorks(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	setPublicAccess(t, true)
	h := authMiddleware(deps, ok200(), renderer.RolePublicRead)

	req := httptest.NewRequest(http.MethodGet, "/api/media/query", nil)
	req.Header.Set("Authorization", "Bearer "+loginToken(t, deps))
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("admin request with flag on: got %d, want 200", rr.Code)
	}
}

func TestRolePublicRead_FlagOn_AdminRoutesStayLocked(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	setPublicAccess(t, true)
	h := authMiddleware(deps, ok200(), renderer.RoleAdmin)

	req := httptest.NewRequest(http.MethodPost, "/create", nil)
	req.Header.Set("Accept", "application/json")
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous write with flag on: got %d, want 401", rr.Code)
	}
}

func TestRequireAuthWhenPublic_CookieAndJSONShape(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	setPublicAccess(t, true)
	h := requireAuthWhenPublic(deps, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	// Anonymous while public: 401 JSON regardless of Accept header (never
	// a login redirect — these branches are only reached from fetch()).
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPut, "/api/settings", nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous: got %d, want 401", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Fatalf("anonymous: Content-Type = %q, want application/json", ct)
	}

	// Cookie credential: allowed.
	req := httptest.NewRequest(http.MethodPut, "/api/settings", nil)
	req.AddCookie(&http.Cookie{Name: "auth_token", Value: loginToken(t, deps)})
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("cookie auth: got %d, want 200", rr.Code)
	}
}

func TestRequireAuthWhenPublic(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	h := requireAuthWhenPublic(deps, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	// Flag off: open (Electron/wizard call these unauthenticated).
	setPublicAccess(t, false)
	rr := httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/deps/models/x/download", nil))
	if rr.Code != http.StatusOK {
		t.Fatalf("flag off anonymous: got %d, want 200", rr.Code)
	}

	// Flag on: anonymous blocked.
	setPublicAccess(t, true)
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, httptest.NewRequest(http.MethodPost, "/api/deps/models/x/download", nil))
	if rr.Code != http.StatusUnauthorized {
		t.Fatalf("flag on anonymous: got %d, want 401", rr.Code)
	}

	// Flag on: admin allowed.
	req := httptest.NewRequest(http.MethodPost, "/api/deps/models/x/download", nil)
	req.Header.Set("Authorization", "Bearer "+loginToken(t, deps))
	rr = httptest.NewRecorder()
	h.ServeHTTP(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("flag on admin: got %d, want 200", rr.Code)
	}
}
