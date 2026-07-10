package main

// setup_api.go — first-run setup wizard: the /setup page and its
// /setup/api/* endpoints. No build tag: it references symbols the
// platform-gated mains define (Dependencies, switchDatabase), which is safe
// because exactly one of those files compiles per GOOS.
//
// Gating contract: while appconfig.Get().SetupComplete is false every
// endpoint here is open (there is no account yet to authenticate with) and
// the auth middleware funnels page requests to /setup. Once true, these
// endpoints require an authenticated admin, same as the rest of the UI.

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/deps/status"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/storage"
	"github.com/stevecastle/shrike/tasks"
)

// switchDatabaseFn is a seam so tests can exercise the database endpoint
// without swapping the process-global queue/runners.
var switchDatabaseFn = func(path string) error { return switchDatabase(path) }

// inferSetupComplete marks setup complete for installs that predate the
// wizard: if a real (non-default) account already exists in the active
// database, the owner has been through account setup and must never be
// funneled back into the first-run flow.
func inferSetupComplete(d *Dependencies) {
	cfg := appconfig.Get()
	if cfg.SetupComplete {
		return
	}
	setupRequired, err := d.Auth.IsSetupRequired()
	if err != nil || setupRequired {
		return
	}
	cfg.SetupComplete = true
	if _, err := appconfig.Save(cfg); err != nil {
		log.Printf("setup: failed to persist inferred setupComplete: %v", err)
	} else {
		log.Printf("setup: existing account found, marking first-run setup complete")
	}
}

// provisionAdminFromEnv creates the initial account from LOWKEY_ADMIN_USER /
// LOWKEY_ADMIN_PASSWORD when the install is still on the default admin, so
// headless deployments (Docker, NAS) can provision without clicking through
// the wizard. inferSetupComplete then marks setup complete. No-op once a
// real account exists — it never overwrites an existing user's password.
func provisionAdminFromEnv(d *Dependencies) {
	user := strings.TrimSpace(os.Getenv("LOWKEY_ADMIN_USER"))
	pass := os.Getenv("LOWKEY_ADMIN_PASSWORD")
	if user == "" || pass == "" {
		return
	}
	setupRequired, err := d.Auth.IsSetupRequired()
	if err != nil || !setupRequired {
		return
	}
	if err := d.Auth.Register(user, pass); err != nil {
		log.Printf("setup: LOWKEY_ADMIN_USER provisioning failed: %v", err)
		return
	}
	log.Printf("setup: account %q provisioned from LOWKEY_ADMIN_USER", user)
}

// setupAuthed reports whether the request carries a valid credential
// (Bearer JWT, lk_ API key, or the auth_token cookie).
func setupAuthed(d *Dependencies, r *http.Request) bool {
	if tok := requestAuthToken(r); tok != "" {
		if _, err := verifyCredential(d, tok); err == nil {
			return true
		}
	}
	if c, err := r.Cookie("auth_token"); err == nil {
		if _, err := d.Auth.VerifyToken(c.Value); err == nil {
			return true
		}
	}
	return false
}

// setupGateAPI protects a setup JSON endpoint: open during first-run,
// admin-only afterwards.
func setupGateAPI(d *Dependencies, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !appconfig.Get().SetupComplete || setupAuthed(d, r) {
			next(w, r)
			return
		}
		httpError(w, "setup is complete; authentication required", http.StatusForbidden)
	}
}

// setupGatePage protects the /setup page itself: open during first-run,
// redirect to /login afterwards when unauthenticated.
func setupGatePage(d *Dependencies, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !appconfig.Get().SetupComplete || setupAuthed(d, r) {
			next(w, r)
			return
		}
		http.Redirect(w, r, "/login", http.StatusFound)
	}
}

// loginRedirectTarget routes unauthenticated page requests to the wizard
// until first-run setup finishes, and to the given login URL afterwards.
// Called from each platform main's authMiddleware.
func loginRedirectTarget(fallback string) string {
	if !appconfig.Get().SetupComplete {
		return "/setup"
	}
	return fallback
}

// registerSetupRoutes wires the wizard onto the mux. Called from each
// platform main after the auth service exists; also runs the one-time
// upgrade inference so pre-wizard installs never see the wizard.
func registerSetupRoutes(mux *http.ServeMux, d *Dependencies) {
	provisionAdminFromEnv(d)
	inferSetupComplete(d)

	page := func(fn http.HandlerFunc) http.HandlerFunc {
		return renderer.ApplyMiddlewares(setupGatePage(d, fn), renderer.RolePublic)
	}
	api := func(fn http.HandlerFunc) http.HandlerFunc {
		return renderer.ApplyMiddlewares(setupGateAPI(d, fn), renderer.RolePublic)
	}

	mux.HandleFunc("GET /setup", page(setupPageHandler()))
	mux.HandleFunc("GET /setup/api/state", api(setupStateHandler(d)))
	mux.HandleFunc("POST /setup/api/browse", api(setupBrowseHandler()))
	mux.HandleFunc("POST /setup/api/mkdir", api(setupMkdirHandler()))
	mux.HandleFunc("POST /setup/api/database", api(setupDatabaseHandler(d)))
	mux.HandleFunc("POST /setup/api/storage/test-s3", api(setupTestS3Handler()))
	mux.HandleFunc("POST /setup/api/storage", api(setupStorageHandler(d)))
	mux.HandleFunc("POST /setup/api/complete", api(setupCompleteHandler()))
}

func setupPageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if err := renderer.Templates().ExecuteTemplate(w, "setup", nil); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

type setupModelGroup struct {
	ID          string   `json:"id"`
	Title       string   `json:"title"`
	Desc        string   `json:"desc"`
	Recommended bool     `json:"recommended"`
	SizeBytes   int64    `json:"sizeBytes"`
	Models      []string `json:"models"`
}

// setupModelGroups maps wizard feature cards to downloadable model IDs.
// Sizes are summed from the models manifest at request time so the wizard
// always shows current numbers.
func setupModelGroups() []setupModelGroup {
	groups := []setupModelGroup{
		{ID: "visual-search", Title: "Visual & text search",
			Desc:        "Find media by describing it, or by visual similarity to another image.",
			Recommended: true, Models: []string{"siglip2-base-patch16-224"}},
		{ID: "autotag", Title: "Auto-tagging",
			Desc:        "Automatically suggest tags for images so your library organizes itself.",
			Recommended: true, Models: []string{"wd-eva02-large-tagger-v3"}},
		{ID: "faces", Title: "Face & character recognition",
			Desc:   "Group photos and artwork by the people and characters in them.",
			Models: []string{"yunet", "sface", "anime-head", "ccip"}},
		{ID: "transcription", Title: "Audio transcription",
			Desc:   "Generate searchable transcripts for videos and audio files.",
			Models: []string{"faster-whisper"}},
	}
	sizes := map[string]int64{}
	for _, m := range models.Manifest {
		sizes[m.ID] = m.EffectiveSizeBytes()
	}
	for i := range groups {
		var total int64
		for _, id := range groups[i].Models {
			total += sizes[id]
		}
		groups[i].SizeBytes = total
	}
	return groups
}

func setupStateHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg := appconfig.Get()
		setupRequired, _ := d.Auth.IsSetupRequired()

		// Redact S3 secrets: the wizard only needs to show which roots exist.
		roots := make([]appconfig.StorageRoot, len(cfg.Roots))
		copy(roots, cfg.Roots)
		for i := range roots {
			if roots[i].SecretKey != "" {
				roots[i].SecretKey = "•••"
			}
		}

		home, _ := os.UserHomeDir()
		writeJSON(w, map[string]any{
			"setupComplete": cfg.SetupComplete,
			"defaultDBPath": appconfig.DefaultDBPath(),
			"activeDBPath":  cfg.DBPath,
			"dataDir":       platform.GetDataDir(),
			"homeDir":       home,
			"os":            runtime.GOOS,
			"roots":         roots,
			"hasRealUsers":  !setupRequired,
			"authed":        setupAuthed(d, r),
			"deps":          status.Snapshot(),
			"modelGroups":   setupModelGroups(),
		})
	}
}

// ---------------------------------------------------------------------------
// Directory browsing (setup-scoped: unlike /api/fs/list this browses the
// whole machine, because its whole job is picking locations before any
// storage root exists)
// ---------------------------------------------------------------------------

type setupBrowseEntry struct {
	Name string `json:"name"`
	Path string `json:"path"`
}

type setupBrowseResponse struct {
	Path     string             `json:"path"`
	Parent   *string            `json:"parent"`
	Entries  []setupBrowseEntry `json:"entries"`
	Exists   bool               `json:"exists"`
	Writable bool               `json:"writable"`
}

// listBrowseLocations returns the top-level starting points shown when the
// picker opens with no path: home, the app data dir, and each drive (or the
// filesystem root on unix).
func listBrowseLocations() []setupBrowseEntry {
	var out []setupBrowseEntry
	if home, err := os.UserHomeDir(); err == nil {
		out = append(out, setupBrowseEntry{Name: "Home", Path: home})
	}
	out = append(out, setupBrowseEntry{Name: "App data", Path: platform.GetDataDir()})
	if runtime.GOOS == "windows" {
		for c := 'A'; c <= 'Z'; c++ {
			drive := string(c) + `:\`
			if _, err := os.Stat(drive); err == nil {
				out = append(out, setupBrowseEntry{Name: string(c) + ":", Path: drive})
			}
		}
	} else {
		out = append(out, setupBrowseEntry{Name: "/", Path: "/"})
	}
	return out
}

func dirWritable(dir string) bool {
	f, err := os.CreateTemp(dir, ".lowkey-setup-*")
	if err != nil {
		return false
	}
	name := f.Name()
	_ = f.Close()
	_ = os.Remove(name)
	return true
}

func browseDir(path string) (setupBrowseResponse, error) {
	resp := setupBrowseResponse{Path: path, Entries: []setupBrowseEntry{}}
	if path == "" {
		resp.Entries = listBrowseLocations()
		resp.Exists = true
		return resp, nil
	}
	if !filepath.IsAbs(path) {
		return resp, errors.New("path must be absolute")
	}
	path = filepath.Clean(path)
	resp.Path = path

	if parent := filepath.Dir(path); parent != path {
		resp.Parent = &parent
	} else {
		// Drive/filesystem root: "up" returns to the locations list.
		empty := ""
		resp.Parent = &empty
	}

	info, err := os.Stat(path)
	if err != nil {
		return resp, nil // exists:false — caller may offer to create it
	}
	if !info.IsDir() {
		return resp, errors.New("path is a file, not a directory")
	}
	resp.Exists = true
	resp.Writable = dirWritable(path)

	entries, err := os.ReadDir(path)
	if err != nil {
		// Readable-but-listing-failed (permissions): report existence only.
		return resp, nil
	}
	for _, e := range entries {
		if !e.IsDir() || strings.HasPrefix(e.Name(), ".") {
			continue
		}
		resp.Entries = append(resp.Entries, setupBrowseEntry{
			Name: e.Name(),
			Path: filepath.Join(path, e.Name()),
		})
	}
	sort.Slice(resp.Entries, func(i, j int) bool {
		return strings.ToLower(resp.Entries[i].Name) < strings.ToLower(resp.Entries[j].Name)
	})
	return resp, nil
}

func setupBrowseHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path string `json:"path"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		resp, err := browseDir(strings.TrimSpace(req.Path))
		if err != nil {
			httpError(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, resp)
	}
}

func setupMkdirHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path string `json:"path"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		p := strings.TrimSpace(req.Path)
		if p == "" || !filepath.IsAbs(p) {
			httpError(w, "path must be absolute", http.StatusBadRequest)
			return
		}
		if err := os.MkdirAll(p, 0755); err != nil {
			httpError(w, fmt.Sprintf("could not create folder: %v", err), http.StatusBadRequest)
			return
		}
		resp, err := browseDir(p)
		if err != nil {
			httpError(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeJSON(w, resp)
	}
}

// ---------------------------------------------------------------------------
// Database step
// ---------------------------------------------------------------------------

func setupDatabaseHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path string `json:"path"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		p := strings.TrimSpace(req.Path)
		if p == "" || !filepath.IsAbs(p) {
			httpError(w, "database path must be absolute", http.StatusBadRequest)
			return
		}
		p = filepath.Clean(p)
		// A directory (existing, or a path spelled with a trailing separator)
		// means "put the database in here with the default name".
		if info, err := os.Stat(p); (err == nil && info.IsDir()) ||
			strings.HasSuffix(req.Path, "/") || strings.HasSuffix(req.Path, string(os.PathSeparator)) {
			p = filepath.Join(p, "media.db")
		}

		_, statErr := os.Stat(p)
		existed := statErr == nil

		if err := switchDatabaseFn(p); err != nil {
			httpError(w, fmt.Sprintf("could not open database: %v", err), http.StatusInternalServerError)
			return
		}

		cfg := appconfig.Get()
		cfg.DBPath = p
		if _, err := appconfig.Save(cfg); err != nil {
			httpError(w, fmt.Sprintf("database opened but config save failed: %v", err), http.StatusInternalServerError)
			return
		}

		// switchDatabase replaces d.Auth (same pointer as the global deps),
		// so this reflects the freshly opened database.
		setupRequired, _ := d.Auth.IsSetupRequired()
		writeJSON(w, map[string]any{
			"path":         p,
			"existed":      existed,
			"hasRealUsers": !setupRequired,
		})
	}
}

// ---------------------------------------------------------------------------
// Storage step
// ---------------------------------------------------------------------------

func setupTestS3Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Endpoint  string `json:"endpoint"`
			Region    string `json:"region"`
			Bucket    string `json:"bucket"`
			Prefix    string `json:"prefix"`
			AccessKey string `json:"accessKey"`
			SecretKey string `json:"secretKey"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		if strings.TrimSpace(req.Bucket) == "" {
			httpError(w, "bucket is required", http.StatusBadRequest)
			return
		}
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Second)
		defer cancel()
		b, err := storage.NewS3Backend(ctx, storage.S3Config{
			Label:     "setup-test",
			Endpoint:  req.Endpoint,
			Region:    req.Region,
			Bucket:    req.Bucket,
			Prefix:    req.Prefix,
			AccessKey: req.AccessKey,
			SecretKey: req.SecretKey,
		})
		if err != nil {
			httpError(w, fmt.Sprintf("connection failed: %v", err), http.StatusBadRequest)
			return
		}
		if err := b.Ping(ctx); err != nil {
			httpError(w, fmt.Sprintf("connection failed: %v", err), http.StatusBadRequest)
			return
		}
		writeJSON(w, map[string]any{"ok": true})
	}
}

func setupStorageHandler(d *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Roots []appconfig.StorageRoot `json:"roots"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}
		if len(req.Roots) == 0 {
			httpError(w, "add at least one storage location", http.StatusBadRequest)
			return
		}
		// The state endpoint redacts S3 secrets; a resubmitted root carrying
		// the redaction placeholder keeps its stored secret.
		existing := appconfig.Get().Roots
		for i := range req.Roots {
			if req.Roots[i].Type == "s3" && req.Roots[i].SecretKey == "•••" {
				for _, old := range existing {
					if old.Type == "s3" && old.Bucket == req.Roots[i].Bucket && old.Endpoint == req.Roots[i].Endpoint {
						req.Roots[i].SecretKey = old.SecretKey
						break
					}
				}
			}
		}
		defaults := 0
		for i := range req.Roots {
			root := &req.Roots[i]
			switch root.Type {
			case "local", "":
				root.Type = "local"
				if strings.TrimSpace(root.Path) == "" || !filepath.IsAbs(root.Path) {
					httpError(w, fmt.Sprintf("storage folder %q needs an absolute path", root.Label), http.StatusBadRequest)
					return
				}
				if err := os.MkdirAll(root.Path, 0755); err != nil {
					httpError(w, fmt.Sprintf("could not create %q: %v", root.Path, err), http.StatusBadRequest)
					return
				}
			case "s3":
				if strings.TrimSpace(root.Bucket) == "" {
					httpError(w, fmt.Sprintf("S3 root %q needs a bucket", root.Label), http.StatusBadRequest)
					return
				}
			default:
				httpError(w, fmt.Sprintf("unknown storage type %q", root.Type), http.StatusBadRequest)
				return
			}
			if root.Label == "" {
				if root.Type == "s3" {
					root.Label = root.Bucket
				} else {
					root.Label = filepath.Base(root.Path)
				}
			}
			if root.Default {
				defaults++
			}
		}
		if defaults == 0 {
			req.Roots[0].Default = true
		}

		reg, errs := storage.BuildRegistry(req.Roots)
		if len(errs) > 0 {
			msgs := make([]string, len(errs))
			for i, e := range errs {
				msgs[i] = e.Error()
			}
			httpError(w, strings.Join(msgs, "; "), http.StatusBadRequest)
			return
		}

		cfg := appconfig.Get()
		cfg.Roots = req.Roots
		if _, err := appconfig.Save(cfg); err != nil {
			httpError(w, fmt.Sprintf("config save failed: %v", err), http.StatusInternalServerError)
			return
		}
		d.Storage = reg
		tasks.SetStorageRegistry(reg)
		writeJSON(w, map[string]any{"status": "ok", "roots": len(req.Roots)})
	}
}

// ---------------------------------------------------------------------------
// Completion
// ---------------------------------------------------------------------------

func setupCompleteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cfg := appconfig.Get()
		cfg.SetupComplete = true
		if _, err := appconfig.Save(cfg); err != nil {
			httpError(w, fmt.Sprintf("config save failed: %v", err), http.StatusInternalServerError)
			return
		}
		writeJSON(w, map[string]any{"status": "ok"})
	}
}
