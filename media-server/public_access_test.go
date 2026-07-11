package main

import (
	"net"
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

func TestIsPublicIP(t *testing.T) {
	blocked := []string{
		"127.0.0.1", "::1", // loopback
		"169.254.169.254",                       // link-local (cloud metadata)
		"10.0.0.5", "192.168.1.1", "172.16.0.1", // private
		"100.64.1.1", // CGNAT
		"0.0.0.0",    // unspecified
	}
	for _, s := range blocked {
		if isPublicIP(netParse(s)) {
			t.Errorf("isPublicIP(%s) = true, want false (must be blocked)", s)
		}
	}
	public := []string{"8.8.8.8", "1.1.1.1", "93.184.216.34"}
	for _, s := range public {
		if !isPublicIP(netParse(s)) {
			t.Errorf("isPublicIP(%s) = false, want true", s)
		}
	}
}

func TestMediaReadAllowed(t *testing.T) {
	deps := newAPIKeyTestDeps(t)
	// media table for the row-existence branch
	deps.DB.Exec(`CREATE TABLE IF NOT EXISTS media (path TEXT PRIMARY KEY)`)
	deps.DB.Exec(`INSERT INTO media (path) VALUES ('/curated/local/item.jpg')`)

	anon := func(path string) bool {
		req := newReqNoAuth()
		return mediaReadAllowed(deps, req, path)
	}

	// No storage backend configured in this deps → BackendFor is nil.
	if anon("http://169.254.169.254/") {
		t.Error("anon must not be allowed an http URL (SSRF)")
	}
	if anon("https://example.com/x.jpg") {
		t.Error("anon must not be allowed an https URL (SSRF)")
	}
	if anon("/etc/passwd") {
		t.Error("anon must not read an arbitrary local path")
	}
	if !anon("/curated/local/item.jpg") {
		t.Error("anon should be allowed a curated library row")
	}

	// Admin bypasses everything.
	adminReq := newReqNoAuth()
	adminReq.Header.Set("Authorization", "Bearer "+loginToken(t, deps))
	if !mediaReadAllowed(deps, adminReq, "http://169.254.169.254/") {
		t.Error("admin should be unrestricted")
	}
}

func netParse(s string) net.IP { return net.ParseIP(s) }

func newReqNoAuth() *http.Request {
	return httptest.NewRequest(http.MethodGet, "/", nil)
}
