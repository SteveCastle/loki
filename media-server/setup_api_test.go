package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/auth"
	"github.com/stevecastle/shrike/media"
)

// newSetupTestDeps builds a Dependencies around an in-memory database with
// the default admin created, and points appconfig at a throwaway config file
// with a clean in-memory config.
func newSetupTestDeps(t *testing.T) *Dependencies {
	t.Helper()
	t.Setenv("LOWKEY_CONFIG_PATH", filepath.Join(t.TempDir(), "config.json"))
	// Neutralize any ambient provisioning env so tests control it explicitly.
	t.Setenv("LOWKEY_ADMIN_USER", "")
	t.Setenv("LOWKEY_ADMIN_PASSWORD", "")
	appconfig.Set(appconfig.Config{JWTSecret: "test-secret"})

	db, err := sql.Open("sqlite", ":memory:")
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	if err := media.InitializeSchema(db); err != nil {
		t.Fatalf("schema: %v", err)
	}
	a := auth.NewAuthService(db, "test-secret")
	if err := a.CreateDefaultUser(); err != nil {
		t.Fatalf("default user: %v", err)
	}
	return &Dependencies{DB: db, Auth: a}
}

func newSetupTestMux(t *testing.T, d *Dependencies) *http.ServeMux {
	t.Helper()
	mux := http.NewServeMux()
	registerSetupRoutes(mux, d)
	return mux
}

func setupPost(t *testing.T, mux *http.ServeMux, path string, body any, cookie string) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	if cookie != "" {
		req.AddCookie(&http.Cookie{Name: "auth_token", Value: cookie})
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

func setupGet(t *testing.T, mux *http.ServeMux, path string, cookie string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodGet, path, nil)
	if cookie != "" {
		req.AddCookie(&http.Cookie{Name: "auth_token", Value: cookie})
	}
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	return w
}

func decodeBody(t *testing.T, w *httptest.ResponseRecorder, v any) {
	t.Helper()
	if err := json.Unmarshal(w.Body.Bytes(), v); err != nil {
		t.Fatalf("decode %q: %v", w.Body.String(), err)
	}
}

// ---------------------------------------------------------------------------

func TestInferSetupComplete(t *testing.T) {
	d := newSetupTestDeps(t)

	inferSetupComplete(d)
	if appconfig.Get().SetupComplete {
		t.Fatal("default-admin-only install must not be inferred complete")
	}

	if err := d.Auth.Register("steve", "hunter22"); err != nil {
		t.Fatalf("register: %v", err)
	}
	inferSetupComplete(d)
	if !appconfig.Get().SetupComplete {
		t.Fatal("install with a real account must be inferred complete")
	}
}

func TestProvisionAdminFromEnv(t *testing.T) {
	d := newSetupTestDeps(t)
	t.Setenv("LOWKEY_ADMIN_USER", "nasadmin")
	t.Setenv("LOWKEY_ADMIN_PASSWORD", "hunter22")

	newSetupTestMux(t, d) // registerSetupRoutes runs provisioning + inference

	if setupRequired, _ := d.Auth.IsSetupRequired(); setupRequired {
		t.Fatal("env-provisioned account not created")
	}
	if !appconfig.Get().SetupComplete {
		t.Fatal("setup not inferred complete after env provisioning")
	}
	if _, err := d.Auth.Login("nasadmin", "hunter22"); err != nil {
		t.Fatalf("provisioned account login failed: %v", err)
	}
}

func TestProvisionAdminFromEnvNeverOverwrites(t *testing.T) {
	d := newSetupTestDeps(t)
	if err := d.Auth.Register("steve", "original-pass"); err != nil {
		t.Fatal(err)
	}
	t.Setenv("LOWKEY_ADMIN_USER", "steve")
	t.Setenv("LOWKEY_ADMIN_PASSWORD", "attacker-pass")

	provisionAdminFromEnv(d)

	if _, err := d.Auth.Login("steve", "original-pass"); err != nil {
		t.Fatalf("existing account password was disturbed: %v", err)
	}
	if _, err := d.Auth.Login("steve", "attacker-pass"); err == nil {
		t.Fatal("env provisioning overwrote an existing account's password")
	}
}

func TestSetupGate(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	if w := setupGet(t, mux, "/setup/api/state", ""); w.Code != http.StatusOK {
		t.Fatalf("state while setup incomplete = %d, want 200", w.Code)
	}

	cfg := appconfig.Get()
	cfg.SetupComplete = true
	appconfig.Set(cfg)

	if w := setupGet(t, mux, "/setup/api/state", ""); w.Code != http.StatusForbidden {
		t.Fatalf("state after completion unauthenticated = %d, want 403", w.Code)
	}

	if err := d.Auth.Register("steve", "hunter22"); err != nil {
		t.Fatalf("register: %v", err)
	}
	token, err := d.Auth.Login("steve", "hunter22")
	if err != nil {
		t.Fatalf("login: %v", err)
	}
	if w := setupGet(t, mux, "/setup/api/state", token); w.Code != http.StatusOK {
		t.Fatalf("state after completion authenticated = %d, want 200", w.Code)
	}
}

func TestLoginRedirectTarget(t *testing.T) {
	newSetupTestDeps(t)
	if got := loginRedirectTarget("/login"); got != "/setup" {
		t.Fatalf("incomplete setup redirect = %q, want /setup", got)
	}
	cfg := appconfig.Get()
	cfg.SetupComplete = true
	appconfig.Set(cfg)
	if got := loginRedirectTarget("/login"); got != "/login" {
		t.Fatalf("complete setup redirect = %q, want /login", got)
	}
}

func TestSetupState(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	w := setupGet(t, mux, "/setup/api/state", "")
	if w.Code != http.StatusOK {
		t.Fatalf("state = %d: %s", w.Code, w.Body.String())
	}
	var state struct {
		SetupComplete bool `json:"setupComplete"`
		HasRealUsers  bool `json:"hasRealUsers"`
		ModelGroups   []struct {
			ID        string   `json:"id"`
			SizeBytes int64    `json:"sizeBytes"`
			Models    []string `json:"models"`
		} `json:"modelGroups"`
		DefaultDBPath string `json:"defaultDBPath"`
	}
	decodeBody(t, w, &state)
	if state.SetupComplete || state.HasRealUsers {
		t.Fatalf("fresh install state = %+v, want incomplete with no real users", state)
	}
	if len(state.ModelGroups) != 4 {
		t.Fatalf("model groups = %d, want 4", len(state.ModelGroups))
	}
	for _, g := range state.ModelGroups {
		if g.SizeBytes <= 0 {
			t.Errorf("group %s has size %d, want > 0 (manifest lookup broken?)", g.ID, g.SizeBytes)
		}
	}
	if state.DefaultDBPath == "" {
		t.Fatal("defaultDBPath missing")
	}
}

func TestSetupBrowse(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	if w := setupPost(t, mux, "/setup/api/browse", map[string]string{"path": "relative/dir"}, ""); w.Code != http.StatusBadRequest {
		t.Fatalf("relative path = %d, want 400", w.Code)
	}

	dir := t.TempDir()
	for _, sub := range []string{"beta", "Alpha"} {
		if err := os.Mkdir(filepath.Join(dir, sub), 0755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "not-a-dir.txt"), []byte("x"), 0644); err != nil {
		t.Fatal(err)
	}

	w := setupPost(t, mux, "/setup/api/browse", map[string]string{"path": dir}, "")
	if w.Code != http.StatusOK {
		t.Fatalf("browse = %d: %s", w.Code, w.Body.String())
	}
	var resp setupBrowseResponse
	decodeBody(t, w, &resp)
	if !resp.Exists || !resp.Writable {
		t.Fatalf("temp dir reported exists=%v writable=%v", resp.Exists, resp.Writable)
	}
	if len(resp.Entries) != 2 || resp.Entries[0].Name != "Alpha" || resp.Entries[1].Name != "beta" {
		t.Fatalf("entries = %+v, want case-insensitive-sorted dirs only", resp.Entries)
	}

	// Nonexistent: reported, not an error, so the UI can offer to create it.
	w = setupPost(t, mux, "/setup/api/browse", map[string]string{"path": filepath.Join(dir, "missing")}, "")
	if w.Code != http.StatusOK {
		t.Fatalf("missing dir browse = %d", w.Code)
	}
	decodeBody(t, w, &resp)
	if resp.Exists {
		t.Fatal("missing dir reported as existing")
	}

	// Empty path lists starting locations.
	w = setupPost(t, mux, "/setup/api/browse", map[string]string{"path": ""}, "")
	decodeBody(t, w, &resp)
	if len(resp.Entries) == 0 {
		t.Fatal("empty path returned no locations")
	}
}

func TestSetupMkdir(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	if w := setupPost(t, mux, "/setup/api/mkdir", map[string]string{"path": "relative"}, ""); w.Code != http.StatusBadRequest {
		t.Fatalf("relative mkdir = %d, want 400", w.Code)
	}

	target := filepath.Join(t.TempDir(), "media", "library")
	w := setupPost(t, mux, "/setup/api/mkdir", map[string]string{"path": target}, "")
	if w.Code != http.StatusOK {
		t.Fatalf("mkdir = %d: %s", w.Code, w.Body.String())
	}
	if info, err := os.Stat(target); err != nil || !info.IsDir() {
		t.Fatalf("nested dir not created: %v", err)
	}
}

func TestSetupDatabase(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	var switched []string
	orig := switchDatabaseFn
	switchDatabaseFn = func(p string) error { switched = append(switched, p); return nil }
	t.Cleanup(func() { switchDatabaseFn = orig })

	if w := setupPost(t, mux, "/setup/api/database", map[string]string{"path": "relative.db"}, ""); w.Code != http.StatusBadRequest {
		t.Fatalf("relative db path = %d, want 400", w.Code)
	}

	dir := t.TempDir()
	dbPath := filepath.Join(dir, "media.db")
	w := setupPost(t, mux, "/setup/api/database", map[string]string{"path": dbPath}, "")
	if w.Code != http.StatusOK {
		t.Fatalf("database = %d: %s", w.Code, w.Body.String())
	}
	var resp struct {
		Path         string `json:"path"`
		Existed      bool   `json:"existed"`
		HasRealUsers bool   `json:"hasRealUsers"`
	}
	decodeBody(t, w, &resp)
	if resp.Existed {
		t.Fatal("fresh path reported as existing")
	}
	if resp.HasRealUsers {
		t.Fatal("default-admin DB reported real users")
	}
	if len(switched) != 1 || switched[0] != dbPath {
		t.Fatalf("switch calls = %v, want [%s]", switched, dbPath)
	}
	if got := appconfig.Get().DBPath; got != dbPath {
		t.Fatalf("persisted dbPath = %q, want %q", got, dbPath)
	}

	// A directory means "use the default filename inside it".
	w = setupPost(t, mux, "/setup/api/database", map[string]string{"path": dir}, "")
	decodeBody(t, w, &resp)
	if want := filepath.Join(dir, "media.db"); resp.Path != want {
		t.Fatalf("dir request resolved to %q, want %q", resp.Path, want)
	}

	// Pre-existing file is detected so the wizard can say "found a library".
	existing := filepath.Join(dir, "old.db")
	if err := os.WriteFile(existing, []byte("not empty"), 0644); err != nil {
		t.Fatal(err)
	}
	w = setupPost(t, mux, "/setup/api/database", map[string]string{"path": existing}, "")
	decodeBody(t, w, &resp)
	if !resp.Existed {
		t.Fatal("existing file not detected")
	}

	// Switch failure surfaces as a 500 with the reason.
	switchDatabaseFn = func(p string) error { return fmt.Errorf("disk on fire") }
	w = setupPost(t, mux, "/setup/api/database", map[string]string{"path": dbPath}, "")
	if w.Code != http.StatusInternalServerError || !strings.Contains(w.Body.String(), "disk on fire") {
		t.Fatalf("switch failure = %d %q", w.Code, w.Body.String())
	}
}

func TestSetupStorage(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	if w := setupPost(t, mux, "/setup/api/storage", map[string]any{"roots": []any{}}, ""); w.Code != http.StatusBadRequest {
		t.Fatalf("empty roots = %d, want 400", w.Code)
	}

	if w := setupPost(t, mux, "/setup/api/storage", map[string]any{
		"roots": []map[string]any{{"type": "local", "path": "relative/media"}},
	}, ""); w.Code != http.StatusBadRequest {
		t.Fatalf("relative local root = %d, want 400", w.Code)
	}

	if w := setupPost(t, mux, "/setup/api/storage", map[string]any{
		"roots": []map[string]any{{"type": "s3", "label": "cloud"}},
	}, ""); w.Code != http.StatusBadRequest {
		t.Fatalf("s3 without bucket = %d, want 400", w.Code)
	}

	dir := filepath.Join(t.TempDir(), "library")
	w := setupPost(t, mux, "/setup/api/storage", map[string]any{
		"roots": []map[string]any{{"type": "local", "path": dir, "label": "Library"}},
	}, "")
	if w.Code != http.StatusOK {
		t.Fatalf("valid local root = %d: %s", w.Code, w.Body.String())
	}
	if info, err := os.Stat(dir); err != nil || !info.IsDir() {
		t.Fatalf("storage dir not created: %v", err)
	}
	roots := appconfig.Get().Roots
	if len(roots) != 1 || roots[0].Path != dir || !roots[0].Default {
		t.Fatalf("persisted roots = %+v, want one default root at %s", roots, dir)
	}
	if d.Storage == nil {
		t.Fatal("registry not swapped onto deps")
	}
}

func TestSetupTestS3Failure(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	w := setupPost(t, mux, "/setup/api/storage/test-s3", map[string]string{
		"endpoint": "http://127.0.0.1:1", "region": "us-east-1",
		"bucket": "nope", "accessKey": "x", "secretKey": "y",
	}, "")
	if w.Code != http.StatusBadRequest {
		t.Fatalf("unreachable s3 = %d, want 400", w.Code)
	}
	if !strings.Contains(w.Body.String(), "connection failed") {
		t.Fatalf("error body = %q, want connection failure message", w.Body.String())
	}
}

func TestSetupCompleteLocksGate(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := newSetupTestMux(t, d)

	if w := setupPost(t, mux, "/setup/api/complete", map[string]string{}, ""); w.Code != http.StatusOK {
		t.Fatalf("complete = %d", w.Code)
	}
	if !appconfig.Get().SetupComplete {
		t.Fatal("flag not set")
	}
	if w := setupGet(t, mux, "/setup/api/state", ""); w.Code != http.StatusForbidden {
		t.Fatalf("state after complete = %d, want 403", w.Code)
	}
}

func TestUserCreateLockdown(t *testing.T) {
	d := newSetupTestDeps(t)
	mux := http.NewServeMux()
	mux.HandleFunc("/auth/users", userManagementHandler(d))

	// First account: open (this is how the wizard and Electron bootstrap).
	w := setupPost(t, mux, "/auth/users", map[string]string{"username": "steve", "password": "hunter22"}, "")
	if w.Code != http.StatusCreated {
		t.Fatalf("first account = %d: %s", w.Code, w.Body.String())
	}

	// Second account unauthenticated: locked.
	w = setupPost(t, mux, "/auth/users", map[string]string{"username": "mallory", "password": "x"}, "")
	if w.Code != http.StatusForbidden {
		t.Fatalf("unauthenticated second account = %d, want 403", w.Code)
	}

	// Second account with a valid token: allowed.
	token, err := d.Auth.Login("steve", "hunter22")
	if err != nil {
		t.Fatalf("login: %v", err)
	}
	w = setupPost(t, mux, "/auth/users", map[string]string{"username": "friend", "password": "hunter33"}, token)
	if w.Code != http.StatusCreated {
		t.Fatalf("authenticated second account = %d: %s", w.Code, w.Body.String())
	}
}
