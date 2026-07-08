//go:build darwin
// +build darwin

package main

import (
	"context"
	"database/sql"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	_ "modernc.org/sqlite"

	"github.com/stevecastle/shrike/appconfig"
	"github.com/stevecastle/shrike/auth"
	"github.com/stevecastle/shrike/deps/bundled"
	"github.com/stevecastle/shrike/deps/models"
	"github.com/stevecastle/shrike/jobqueue"
	"github.com/stevecastle/shrike/media"
	"github.com/stevecastle/shrike/platform"
	"github.com/stevecastle/shrike/renderer"
	"github.com/stevecastle/shrike/runners"
	"github.com/stevecastle/shrike/storage"
	"github.com/stevecastle/shrike/stream"
	"github.com/stevecastle/shrike/tasks"
	"github.com/stevecastle/shrike/transcribe"
)

// -----------------------------------------------------------------------------
// Embed static assets under client/static; ** must recurse all sub-paths.
// -----------------------------------------------------------------------------

//go:embed client/static/**
var embeddedStatic embed.FS

//go:embed loki-static/**
var embeddedSPA embed.FS

// staticFS is the embedded filesystem rooted at client/static/.
var staticFS fs.FS

// spaFS is the embedded filesystem rooted at loki-static/.
var spaFS fs.FS

// -----------------------------------------------------------------------------
// http server so we can shut it down cleanly from onExit.
// -----------------------------------------------------------------------------
var srv *http.Server

// Global dependencies variable so we can access it from onExit
var deps *Dependencies

// Keep a copy of the currently loaded config in memory
var currentConfig appconfig.Config

// Global runners instance so we can shut it down when switching databases
var currentRunners *runners.Runners

// -----------------------------------------------------------------------------
// Dependencies struct to hold shared dependencies
// -----------------------------------------------------------------------------
type Dependencies struct {
	Queue   *jobqueue.Queue
	DB      *sql.DB
	Auth    *auth.AuthService
	Storage *storage.Registry
}

// -----------------------------------------------------------------------------
// Utility â€“ run from the folder that contains the executable so the templates
// and static files are found even when launched from elsewhere (during dev
// this still helps, but isn't strictly required for embedded files).
// -----------------------------------------------------------------------------
func init() {
	if exe, err := os.Executable(); err == nil {
		_ = os.Chdir(filepath.Dir(exe))
	}

	// Carve out the client/static subtree of the embedded FS so that
	// "/static/foo.js" maps directly to "foo.js".
	var err error
	staticFS, err = fs.Sub(embeddedStatic, "client/static")
	if err != nil {
		panic("lowkeymediaserver: fs.Sub failed: " + err.Error())
	}

	// Carve out the loki-static subtree for the SPA.
	spaFS, err = fs.Sub(embeddedSPA, "loki-static")
	if err != nil {
		panic("lowkeymediaserver: fs.Sub(loki-static) failed: " + err.Error())
	}
}

// -----------------------------------------------------------------------------
// Database initialization
// -----------------------------------------------------------------------------

// switchDatabase switches the application's active database and queue to the provided path
func switchDatabase(newDBPath string) error {
	if newDBPath == "" {
		return fmt.Errorf("newDBPath cannot be empty")
	}

	log.Printf("Switching database to: %s", newDBPath)

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(newDBPath), 0755); err != nil {
		return fmt.Errorf("failed to create database directory: %v", err)
	}

	// Open and ping the new DB first to validate
	newDB, err := sql.Open("sqlite", sqliteDSN(newDBPath))
	if err != nil {
		return fmt.Errorf("failed to open new database: %v", err)
	}
	if err := newDB.Ping(); err != nil {
		newDB.Close()
		return fmt.Errorf("failed to ping new database: %v", err)
	}

	// Initialize schema if tables don't exist
	if err := media.InitializeSchema(newDB); err != nil {
		log.Printf("warning: failed to initialize database schema on new database: %v", err)
	}

	// Ensure indexes on the new database
	if err := ensureIndexes(newDB); err != nil {
		log.Printf("warning: failed to ensure indexes on new database: %v", err)
	}

	// Prepare a new queue backed by the new DB
	newQueue := jobqueue.NewQueueWithDB(newDB)
	tasks.ApplyHostLimits(newQueue, currentConfig)

	// Shut down old runners first if they exist
	if currentRunners != nil {
		log.Println("Shutting down old runners...")
		currentRunners.Shutdown()
		log.Println("Old runners shut down successfully")
	}

	// Swap dependencies (this updates the global deps pointer that all handlers reference)
	oldDB := deps.DB
	deps.DB = newDB
	deps.Queue = newQueue

	// Random-sampler cache is per-DB. Reset (not Invalidate) so old-DB
	// paths can't leak into IN-list lookups against the new DB during the
	// rebuild window. Warm asynchronously so the first /swipe/api request
	// after the swap stays fast.
	media.ResetRandomSampleCache()
	media.WarmRandomSampleCache(newDB)

	// Start new runners for the new queue
	log.Println("Starting new runners for new queue...")
	currentRunners = runners.New(newQueue)
	log.Printf("New runners started. Current jobs in new queue: %d", len(newQueue.GetJobs()))

	// Recreate auth service with the new database connection
	log.Println("Recreating auth service for new database...")
	deps.Auth = auth.NewAuthService(newDB, currentConfig.JWTSecret)
	if err := deps.Auth.CreateDefaultUser(); err != nil {
		log.Printf("Failed to create default user on new database: %v", err)
	}

	// Close the old DB last
	if oldDB != nil {
		log.Println("Closing old database connection...")
		_ = oldDB.Close()
	}

	log.Printf("Database switch complete. Now using: %s", newDBPath)
	return nil
}

func initDB() (*sql.DB, error) {
	// Load config (creates default config if doesn't exist)
	cfg, _, err := appconfig.Load()
	if err != nil {
		return nil, err
	}
	currentConfig = cfg
	dbPath := cfg.DBPath
	log.Printf("Using database path from config: %s", dbPath)

	// A missing file means we are about to create a brand-new EMPTY library -
	// legitimate on first run, but catastrophic-looking when dbPath silently
	// changed (the Electron viewer shares config.json and writes the same
	// "dbPath" key). Shout about it so a wrong path is obvious in the log
	// instead of surfacing as "my library is empty and my login is gone".
	if _, statErr := os.Stat(dbPath); os.IsNotExist(statErr) {
		log.Printf("WARNING: no database exists at %s - creating a NEW EMPTY library (fresh users/auth included). If you expected an existing library, stop the server and fix dbPath in config.json.", dbPath)
	}

	// Ensure the directory exists
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create database directory: %v", err)
	}

	db, err := sql.Open("sqlite", sqliteDSN(dbPath))
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %v", err)
	}

	// Test the connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %v", err)
	}

	// Initialize schema if tables don't exist
	if err := media.InitializeSchema(db); err != nil {
		log.Printf("warning: failed to initialize database schema: %v", err)
	}

	// Best-effort: ensure helpful indexes exist
	if err := ensureIndexes(db); err != nil {
		log.Printf("warning: failed to ensure indexes: %v", err)
	}

	// Warm the random-sampler cache asynchronously so the first
	// /swipe/api request doesn't pay the SELECT DISTINCT scan.
	media.WarmRandomSampleCache(db)

	log.Printf("Connected to SQLite database at: %s", dbPath)
	return db, nil
}

// ensureIndexes creates recommended indexes if the related tables exist.
func ensureIndexes(db *sql.DB) error {
	// Helper to detect if a table exists
	tableExists := func(name string) bool {
		var cnt int
		_ = db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?`, name).Scan(&cnt)
		return cnt > 0
	}

	// Indexes for media_tag_by_category
	if tableExists("media_tag_by_category") {
		stmts := []string{
			"CREATE INDEX IF NOT EXISTS idx_mtbc_media_path ON media_tag_by_category(media_path)",
			"CREATE INDEX IF NOT EXISTS idx_mtbc_tag_label ON media_tag_by_category(tag_label)",
			"CREATE INDEX IF NOT EXISTS idx_mtbc_category_label ON media_tag_by_category(category_label)",
			"CREATE INDEX IF NOT EXISTS idx_mtbc_tag_category ON media_tag_by_category(tag_label, category_label)",
		}
		for _, s := range stmts {
			if _, err := db.Exec(s); err != nil {
				return fmt.Errorf("creating index failed: %w", err)
			}
		}
	}

	// Indexes for media
	if tableExists("media") {
		stmts := []string{
			"CREATE INDEX IF NOT EXISTS idx_media_path ON media(path)",
			"CREATE INDEX IF NOT EXISTS idx_media_has_description ON media(description) WHERE description IS NOT NULL AND description <> ''",
			"CREATE INDEX IF NOT EXISTS idx_media_has_hash ON media(hash) WHERE hash IS NOT NULL AND hash <> ''",
			"CREATE INDEX IF NOT EXISTS idx_media_has_size ON media(size) WHERE size IS NOT NULL",
		}
		for _, s := range stmts {
			if _, err := db.Exec(s); err != nil {
				return fmt.Errorf("creating index failed: %w", err)
			}
		}
	}

	return nil
}

// -----------------------------------------------------------------------------
// Web-handler helpers
// -----------------------------------------------------------------------------

type ListTemplateData struct{ Jobs []jobqueue.Job }
type DetailTemplateData struct{ Job *jobqueue.Job }

type Command struct {
	Command   string
	Arguments []string
}

func homeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// POST = legacy JSON workflow launch
		if r.Method == http.MethodPost {
			var c Command
			if err := json.NewDecoder(r.Body).Decode(&c); err != nil {
				http.Error(w, "bad json", http.StatusBadRequest)
				return
			}
			if len(c.Arguments) == 0 {
				http.Error(w, "arguments required: last element is the task input", http.StatusBadRequest)
				return
			}
			workflow := jobqueue.Workflow{
				Tasks: []jobqueue.WorkflowTask{
					{
						Command:   c.Command,
						Arguments: c.Arguments[:len(c.Arguments)-1],
						Input:     c.Arguments[len(c.Arguments)-1],
					},
				},
			}
			ids, err := deps.Queue.AddWorkflow(workflow)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			id := ""
			if len(ids) > 0 {
				id = ids[0]
			}

			// Send successful response for legacy POST
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusCreated)
			_ = json.NewEncoder(w).Encode(map[string]string{"id": id})
			return
		}

		// GET â€“ render quick jobs launcher
		if err := renderer.Templates().ExecuteTemplate(w, "home", nil); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func jobsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		data := ListTemplateData{Jobs: deps.Queue.GetJobs()}
		if err := renderer.Templates().ExecuteTemplate(w, "jobs", data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func jobsListHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		jobs := deps.Queue.GetJobs()
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(jobs); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func detailHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		job := deps.Queue.GetJob(id)
		if job == nil {
			http.NotFound(w, r)
			return
		}
		data := DetailTemplateData{Job: job}
		if err := renderer.Templates().ExecuteTemplate(w, "detail", data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

type CreateJobHandlerRequest struct {
	Input  string            `json:"input"`
	Fields map[string]string `json:"fields,omitempty"`
}

func createJobHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		var req CreateJobHandlerRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		args := ParseCommand(req.Input)
		if len(args) == 0 {
			http.Error(w, "Invalid input", http.StatusBadRequest)
			return
		}

		cmd, input := args[0], ""
		if len(args) > 1 {
			input = args[len(args)-1]
			args = args[1 : len(args)-1]
		} else {
			args = nil
		}

		args = appendFieldArgs(args, req.Fields)

		id, err := deps.Queue.AddWorkflow(jobqueue.Workflow{
			Tasks: []jobqueue.WorkflowTask{
				{
					Command:   cmd,
					Arguments: args,
					Input:     input,
				},
			},
		})
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		jobID := ""
		if len(id) > 0 {
			jobID = id[0]
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"id": jobID})
	}
}

func cancelHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		deps.Queue.CancelJob(r.PathValue("id"))

		// Send successful response
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("Job cancelled successfully"))
	}
}

func copyHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		newID, err := deps.Queue.CopyJob(r.PathValue("id"))
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Send successful response with new job ID
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]string{"id": newID, "message": "Job copied successfully"})
	}
}

func removeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}
		if err := deps.Queue.RemoveJob(r.PathValue("id")); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Send successful response
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("Job removed successfully"))
	}
}

func clearNonRunningJobsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		clearedCount, err := deps.Queue.ClearNonRunningJobs()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"cleared_count": clearedCount,
			"message":       fmt.Sprintf("Cleared %d non-running jobs", clearedCount),
		})
	}
}

func readJSONBody(r *http.Request, v any) error {
	defer r.Body.Close()
	return json.NewDecoder(r.Body).Decode(v)
}

// healthHandler provides system health information including stream connections
func healthHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set fully permissive CORS headers
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.Header().Set("Access-Control-Max-Age", "86400")

		// Handle preflight OPTIONS request
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		// Get stream connection statistics
		streamStats := stream.GetConnectionStats()

		// Get job queue statistics
		jobs := deps.Queue.GetJobs()
		jobStats := map[string]int{
			"total":       len(jobs),
			"pending":     0,
			"in_progress": 0,
			"completed":   0,
			"cancelled":   0,
			"error":       0,
		}

		for _, job := range jobs {
			switch job.State {
			case 0: // StatePending
				jobStats["pending"]++
			case 1: // StateInProgress
				jobStats["in_progress"]++
			case 2: // StateCompleted
				jobStats["completed"]++
			case 3: // StateCancelled
				jobStats["cancelled"]++
			case 4: // StateError
				jobStats["error"]++
			}
		}

		health := map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now().Unix(),
			"stream":    streamStats,
			"jobs":      jobStats,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(health); err != nil {
			log.Printf("Error encoding health response: %v", err)
		}
	}
}

// mediaHandler serves the main media browsing page
func mediaHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		const initialLimit = 25

		// Get search query from URL parameter
		searchQuery := r.URL.Query().Get("q")

		items, totalCount, hasMore, err := media.GetItems(deps.DB, 0, initialLimit, searchQuery)
		if err != nil {
			log.Printf("Error fetching media items: %v", err)
			http.Error(w, "Error fetching media items", http.StatusInternalServerError)
			return
		}

		data := media.TemplateData{
			MediaItems:         items,
			Offset:             len(items),
			HasMore:            hasMore,
			TotalCount:         totalCount,
			SearchQuery:        searchQuery,
			DefaultOllamaModel: currentConfig.OllamaModel,
		}

		if err := renderer.Templates().ExecuteTemplate(w, "media", data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

// mediaAPIHandler serves the JSON API for infinite scroll
func mediaAPIHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		// Parse query parameters
		offsetStr := r.URL.Query().Get("offset")
		limitStr := r.URL.Query().Get("limit")
		searchQuery := r.URL.Query().Get("q")
		singleStr := r.URL.Query().Get("single")

		// For the path parameter, use robust decoding to handle unicode characters
		var pathQuery string
		if rawPath := getRawQueryParam(r.URL.RawQuery, "path"); rawPath != "" {
			decoded, err := url.PathUnescape(rawPath)
			if err != nil {
				log.Printf("Error decoding path parameter: %v", err)
				http.Error(w, "Invalid path encoding", http.StatusBadRequest)
				return
			}
			pathQuery = decoded
		}

		// Check if this is a single item request by path
		if pathQuery != "" && singleStr == "true" {
			// Handle single item lookup by path
			item, err := media.GetItemByPath(deps.DB, pathQuery)
			if err != nil {
				log.Printf("Error fetching media item by path '%s': %v", pathQuery, err)
				http.Error(w, "Error fetching media item", http.StatusInternalServerError)
				return
			}

			var items []media.MediaItem
			if item != nil {
				items = append(items, *item)
			}

			response := media.APIResponse{
				Items:      items,
				HasMore:    false,
				TotalCount: len(items),
			}

			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(response); err != nil {
				log.Printf("Error encoding JSON response: %v", err)
			}
			return
		}

		// Handle regular pagination requests
		offset := 0
		limit := 25

		if offsetStr != "" {
			if parsed, err := strconv.Atoi(offsetStr); err == nil {
				offset = parsed
			}
		}

		if limitStr != "" {
			if parsed, err := strconv.Atoi(limitStr); err == nil && parsed > 0 && parsed <= 100 {
				limit = parsed
			}
		}

		items, totalCount, hasMore, err := media.GetItems(deps.DB, offset, limit, searchQuery)
		if err != nil {
			log.Printf("Error fetching media items: %v", err)
			http.Error(w, "Error fetching media items", http.StatusInternalServerError)
			return
		}

		response := media.APIResponse{
			Items:      items,
			HasMore:    hasMore,
			TotalCount: totalCount,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Error encoding JSON response: %v", err)
		}
	}
}

// mediaSuggestHandler serves suggestion data for typeahead search
func mediaSuggestHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		kind := strings.ToLower(strings.TrimSpace(r.URL.Query().Get("kind")))
		prefix := r.URL.Query().Get("prefix")
		limitStr := r.URL.Query().Get("limit")
		limit := 25
		if limitStr != "" {
			if parsed, err := strconv.Atoi(limitStr); err == nil && parsed > 0 && parsed <= 200 {
				limit = parsed
			}
		}

		type respSimple struct {
			Suggestions []string `json:"suggestions"`
		}

		type respTags struct {
			Tags        []media.MediaTag `json:"tags"`
			Suggestions []string         `json:"suggestions"`
		}

		switch kind {
		case "filters":
			suggestions := media.SuggestFilters()
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(respSimple{Suggestions: suggestions})
			return
		case "tag":
			// Return structured tags with categories for grouping, capped by
			// `limit` so a short substring against a 100k+ tag library can't
			// return tens of thousands of rows and freeze the typeahead.
			tags, err := media.SuggestTagsWithCategories(deps.DB, prefix, limit)
			if err != nil {
				log.Printf("suggest error kind=%s prefix=%q: %v", kind, prefix, err)
				http.Error(w, "suggest error", http.StatusInternalServerError)
				return
			}

			// Also populate simple suggestions for autocomplete (deduplicated)
			seen := make(map[string]bool)
			var suggestions []string
			for _, t := range tags {
				if !seen[t.Label] {
					suggestions = append(suggestions, t.Label)
					seen[t.Label] = true
				}
			}

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(respTags{Tags: tags, Suggestions: suggestions})
			return
		case "category":
			suggestions, err := media.SuggestCategoryLabels(deps.DB, prefix, limit)
			if err != nil {
				log.Printf("suggest error kind=%s prefix=%q: %v", kind, prefix, err)
				http.Error(w, "suggest error", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(respSimple{Suggestions: suggestions})
			return
		case "path":
			suggestions, err := media.SuggestPaths(deps.DB, prefix, limit)
			if err != nil {
				log.Printf("suggest error kind=%s prefix=%q: %v", kind, prefix, err)
				http.Error(w, "suggest error", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(respSimple{Suggestions: suggestions})
			return
		case "pathdir":
			suggestions, err := media.SuggestPathDirs(deps.DB, prefix, limit)
			if err != nil {
				log.Printf("suggest error kind=%s prefix=%q: %v", kind, prefix, err)
				http.Error(w, "suggest error", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(respSimple{Suggestions: suggestions})
			return
		default:
			http.Error(w, "unknown kind", http.StatusBadRequest)
			return
		}
	}
}

// mediaTagHandler handles adding/removing tags from media items
func mediaTagHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		type tagRequest struct {
			MediaPath     string `json:"media_path"`
			TagLabel      string `json:"tag_label"`
			CategoryLabel string `json:"category_label"`
			Action        string `json:"action"` // "add" or "remove"
		}

		var req tagRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		if req.MediaPath == "" || req.TagLabel == "" || req.CategoryLabel == "" {
			http.Error(w, "media_path, tag_label, and category_label are required", http.StatusBadRequest)
			return
		}

		if req.Action != "add" && req.Action != "remove" {
			http.Error(w, "action must be 'add' or 'remove'", http.StatusBadRequest)
			return
		}

		var err error
		if req.Action == "add" {
			err = media.AddTag(deps.DB, req.MediaPath, req.TagLabel, req.CategoryLabel)
		} else {
			err = media.RemoveTag(deps.DB, req.MediaPath, req.TagLabel, req.CategoryLabel)
		}

		if err != nil {
			log.Printf("tag error: %v", err)
			http.Error(w, "Failed to update tag", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}
}

// mediaHasTagHandler checks if a media item has a specific tag
func mediaHasTagHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		mediaPath := r.URL.Query().Get("media_path")
		tagLabel := r.URL.Query().Get("tag_label")
		categoryLabel := r.URL.Query().Get("category_label")

		if mediaPath == "" || tagLabel == "" || categoryLabel == "" {
			http.Error(w, "media_path, tag_label, and category_label are required", http.StatusBadRequest)
			return
		}

		hasTag, err := media.HasTag(deps.DB, mediaPath, tagLabel, categoryLabel)
		if err != nil {
			log.Printf("has tag error: %v", err)
			http.Error(w, "Failed to check tag", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"has_tag": hasTag})
	}
}

// -----------------------------------------------------------------------------
// Swipe view handlers â€“ TikTok-like mobile experience
// -----------------------------------------------------------------------------

// swipeTemplateData holds data for the swipe template
type swipeTemplateData struct {
	SearchQuery string `json:"search_query"`
}

// swipeHandler serves the swipe (TikTok-like) view page
func swipeHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		searchQuery := r.URL.Query().Get("q")

		data := swipeTemplateData{
			SearchQuery: searchQuery,
		}

		if err := renderer.Templates().ExecuteTemplate(w, "swipe", data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

// swipeAPIHandler serves randomized media items for the swipe view
func swipeAPIHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		// "More like this": rank by embedding similarity to an anchor item
		// instead of the seeded shuffle. Shared across platform builds.
		if maybeHandleSwipeSimilar(w, r, deps) {
			return
		}

		// Parse query parameters
		offsetStr := r.URL.Query().Get("offset")
		limitStr := r.URL.Query().Get("limit")
		searchQuery := r.URL.Query().Get("q")
		seedStr := r.URL.Query().Get("seed")

		offset := 0
		limit := 20
		seed := int64(0)

		if offsetStr != "" {
			if parsed, err := strconv.Atoi(offsetStr); err == nil {
				offset = parsed
			}
		}

		if limitStr != "" {
			if parsed, err := strconv.Atoi(limitStr); err == nil && parsed > 0 && parsed <= 50 {
				limit = parsed
			}
		}

		if seedStr != "" {
			if parsed, err := strconv.ParseInt(seedStr, 10, 64); err == nil {
				seed = parsed
			}
		}

		items, hasMore, err := media.GetRandomItems(deps.DB, offset, limit, searchQuery, seed)
		if err != nil {
			log.Printf("Error fetching random media items: %v", err)
			http.Error(w, "Error fetching media items", http.StatusInternalServerError)
			return
		}

		response := media.APIResponse{
			Items:   items,
			HasMore: hasMore,
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Error encoding JSON response: %v", err)
		}
	}
}

// swipeManifestHandler serves the PWA manifest for the swipe view
func swipeManifestHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		manifest := map[string]interface{}{
			"name":             "Lowkey Media Server Swipe",
			"short_name":       "Swipe",
			"description":      "Swipe through your media library",
			"start_url":        "/swipe",
			"display":          "standalone",
			"orientation":      "portrait",
			"background_color": "#000000",
			"theme_color":      "#000000",
			"icons": []map[string]interface{}{
				{
					"src":     "/static/icon-192.png",
					"sizes":   "192x192",
					"type":    "image/png",
					"purpose": "any maskable",
				},
				{
					"src":     "/static/icon-512.png",
					"sizes":   "512x512",
					"type":    "image/png",
					"purpose": "any maskable",
				},
			},
		}

		w.Header().Set("Content-Type", "application/manifest+json")
		_ = json.NewEncoder(w).Encode(manifest)
	}
}

// -----------------------------------------------------------------------------
// Tasks handler â€“ lists all registered tasks/commands
// -----------------------------------------------------------------------------

// TaskInfo represents a task for the API response
type TaskInfo struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

func tasksHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		taskMap := tasks.GetTasks()
		taskList := make([]TaskInfo, 0, len(taskMap))

		for _, t := range taskMap {
			taskList = append(taskList, TaskInfo{
				ID:   t.ID,
				Name: t.Name,
			})
		}

		// Sort by ID for consistent ordering
		sort.Slice(taskList, func(i, j int) bool {
			return taskList[i].ID < taskList[j].ID
		})

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"tasks": taskList})
	}
}

func editorHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		if err := renderer.Templates().ExecuteTemplate(w, "editor", nil); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

type WorkflowRequest struct {
	Tasks []jobqueue.WorkflowTask `json:"tasks"`
}

func workflowHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		var req WorkflowRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		workflow := jobqueue.Workflow{
			Tasks: req.Tasks,
		}

		ids, err := deps.Queue.AddWorkflow(workflow)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"ids": ids})
	}
}

// -----------------------------------------------------------------------------
// Ollama models handler â€“ lists available models via `ollama ls`
// -----------------------------------------------------------------------------

func ollamaModelsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		// Run `ollama ls --quiet` to get just model names, one per line.
		// Fallback to `ollama ls` parsing if --quiet is unavailable.
		var models []string

		run := func(args ...string) ([]byte, error) {
			cmd := exec.Command("ollama", args...)
			platform.HideSubprocessWindow(cmd)
			// Best-effort timeout via context is not critical here; rely on default.
			return cmd.Output()
		}

		// Try quiet first
		out, err := run("ls", "--quiet")
		if err != nil || len(out) == 0 {
			// Fallback to regular `ollama ls` and parse first column
			out, err = run("ls")
		}
		if err != nil {
			log.Printf("ollama ls error: %v", err)
			http.Error(w, "failed to list ollama models", http.StatusInternalServerError)
			return
		}

		for _, line := range strings.Split(string(out), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			// When not quiet, lines are like: "llama3:latest  4.7 GB  2 weeks ago"
			// Take the first whitespace-separated token.
			if strings.Contains(line, " ") {
				line = strings.Fields(line)[0]
			}
			// Some outputs include tags like name:tag â€“ keep as-is so user can choose full ref.
			models = append(models, line)
		}

		// Deduplicate while preserving order
		seen := map[string]struct{}{}
		unique := make([]string, 0, len(models))
		for _, m := range models {
			if _, ok := seen[m]; ok {
				continue
			}
			seen[m] = struct{}{}
			unique = append(unique, m)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"models": unique})
	}
}

// -----------------------------------------------------------------------------
// Stats page handler
// -----------------------------------------------------------------------------

type statsTemplateData struct {
	TotalMedia         int
	WithDescription    int
	WithHash           int
	WithSize           int
	WithTags           int
	WithoutDescription int
	WithoutHash        int
	WithoutSize        int
	WithoutTags        int
}

func statsHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		db := deps.DB

		var total, withDesc, withHash, withSize, withTags int

		// Single round-trip to fetch all counts
		err := db.QueryRow(`
            SELECT
                (SELECT COUNT(*) FROM media) AS total,
                (SELECT COUNT(*) FROM media WHERE description IS NOT NULL AND TRIM(description) <> '') AS with_desc,
                (SELECT COUNT(*) FROM media WHERE hash IS NOT NULL AND TRIM(hash) <> '') AS with_hash,
                (SELECT COUNT(*) FROM media WHERE size IS NOT NULL) AS with_size,
                (SELECT COUNT(*) FROM media m WHERE EXISTS (SELECT 1 FROM media_tag_by_category mtbc WHERE mtbc.media_path = m.path)) AS with_tags
        `).Scan(&total, &withDesc, &withHash, &withSize, &withTags)
		if err != nil {
			log.Printf("stats counts error: %v", err)
		}

		data := statsTemplateData{
			TotalMedia:         total,
			WithDescription:    withDesc,
			WithHash:           withHash,
			WithSize:           withSize,
			WithTags:           withTags,
			WithoutDescription: total - withDesc,
			WithoutHash:        total - withHash,
			WithoutSize:        total - withSize,
			WithoutTags:        total - withTags,
		}

		if err := renderer.Templates().ExecuteTemplate(w, "stats", data); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

func formatBytes(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %cB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// -----------------------------------------------------------------------------
// Config page handlers
// -----------------------------------------------------------------------------

type configTemplateData struct {
	Config                 appconfig.Config
	ConfigPath             string
	ActiveDBPath           string
	EmbeddingModels        []tasks.EmbedModel
	TaggerModels           []tasks.TaggerModel
	FaceModels             []tasks.FaceModel
	DirectMLInstalled      bool
	TranscriptionProviders []transcribe.Provider
	TranscriptionModels    []transcribe.ModelChoice
}

type updateConfigRequest struct {
	DBPath               string `json:"dbPath"`
	Port                 int    `json:"port"`
	DownloadPath         string `json:"downloadPath"`
	OllamaBaseURL        string `json:"ollamaBaseUrl"`
	OllamaModel          string `json:"ollamaModel"`
	DescribePrompt       string `json:"describePrompt"`
	InferenceProvider    string `json:"inferenceProvider"`
	RunPodEndpoint       string `json:"runpodEndpoint"`
	RunPodAPIKey         string `json:"runpodApiKey"`
	LMStudioBaseURL      string `json:"lmstudioBaseUrl"`
	LMStudioModel        string `json:"lmstudioModel"`
	LMStudioAPIKey       string `json:"lmstudioApiKey"`
	LlamaCppBaseURL      string `json:"llamacppBaseUrl"`
	LlamaCppModel        string `json:"llamacppModel"`
	LlamaCppAPIKey       string `json:"llamacppApiKey"`
	InferenceConcurrency struct {
		Ollama   int `json:"ollama"`
		RunPod   int `json:"runpod"`
		LMStudio int `json:"lmstudio"`
		LlamaCpp int `json:"llamacpp"`
	} `json:"inferenceConcurrency"`
	LocalComputeConcurrency int `json:"localComputeConcurrency"`
	OnnxModelPath             string                  `json:"onnxModelPath"`
	OnnxLabelsPath            string                  `json:"onnxLabelsPath"`
	OnnxConfigPath            string                  `json:"onnxConfigPath"`
	OnnxORTSharedLibPath      string                  `json:"onnxOrtSharedLibPath"`
	OnnxGeneralThreshold      float64                 `json:"onnxGeneralThreshold"`
	OnnxCharacterThreshold    float64                 `json:"onnxCharacterThreshold"`
	EmbeddingModel            string                  `json:"embeddingModel"`
	EmbeddingProvider         string                  `json:"embeddingProvider"`
	EmbeddingPerformance      string                  `json:"embeddingPerformance"`
	EmbeddingWorkers          int                     `json:"embeddingWorkers"`
	EmbeddingThreadsPerWorker int                     `json:"embeddingThreadsPerWorker"`
	AutotagModel              string                  `json:"autotagModel"`
	AutotagProvider           string                  `json:"autotagProvider"`
	AutotagPerformance        string                  `json:"autotagPerformance"`
	AutotagWorkers            int                     `json:"autotagWorkers"`
	AutotagThreadsPerWorker   int                     `json:"autotagThreadsPerWorker"`
	OnnxFileTimeoutSeconds    int                     `json:"onnxFileTimeoutSeconds"`
	FaceModel                 string                  `json:"faceModel"`
	FaceRouting               string                  `json:"faceRouting"`
	FaceProvider              string                  `json:"faceProvider"`
	FacePerformance           string                  `json:"facePerformance"`
	FaceWorkers               int                     `json:"faceWorkers"`
	FaceThreadsPerWorker      int                     `json:"faceThreadsPerWorker"`
	// nil = field absent from the POST (leave stored entries alone);
	// an explicit empty array clears the list.
	ByoFaceModels         []appconfig.ByoFaceModel `json:"byoFaceModels"`
	TranscriptionProvider string                   `json:"transcriptionProvider"`
	TranscriptionModel        string                  `json:"transcriptionModel"`
	TranscriptionLanguage     *string                 `json:"transcriptionLanguage"`
	TranscriptionVADFilter    *bool                   `json:"transcriptionVadFilter"`
	FasterWhisperPath         string                  `json:"fasterWhisperPath"`
	DiscordToken              string                  `json:"discordToken"`
	Roots                     []appconfig.StorageRoot `json:"roots"`
}

// -----------------------------------------------------------------------------
// File Upload handler
// -----------------------------------------------------------------------------

type uploadResponse struct {
	Success bool     `json:"success"`
	Files   []string `json:"files"`
	Message string   `json:"message,omitempty"`
	Error   string   `json:"error,omitempty"`
}

func uploadHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		// Limit upload size to 10GB
		r.Body = http.MaxBytesReader(w, r.Body, 10<<30)

		if err := r.ParseMultipartForm(32 << 20); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(uploadResponse{
				Success: false,
				Error:   "Failed to parse upload: " + err.Error(),
			})
			return
		}

		backend := deps.Storage.DefaultBackend()
		if backend == nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(uploadResponse{
				Success: false,
				Error:   "No storage backend configured. Add a storage root in Config.",
			})
			return
		}

		files := r.MultipartForm.File["files"]
		if len(files) == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(uploadResponse{
				Success: false,
				Error:   "No files provided",
			})
			return
		}

		ctx := r.Context()
		var uploadedPaths []string

		for _, fileHeader := range files {
			file, err := fileHeader.Open()
			if err != nil {
				log.Printf("Failed to open uploaded file %s: %v", fileHeader.Filename, err)
				continue
			}

			filename := filepath.Base(fileHeader.Filename)
			destPath := "uploads/" + filename

			for i := 1; ; i++ {
				exists, _ := backend.Exists(ctx, destPath)
				if !exists {
					break
				}
				ext := filepath.Ext(filename)
				base := strings.TrimSuffix(filename, ext)
				destPath = fmt.Sprintf("uploads/%s_%d%s", base, i, ext)
			}

			contentType := fileHeader.Header.Get("Content-Type")
			if contentType == "" {
				contentType = "application/octet-stream"
			}

			if err := backend.Upload(ctx, destPath, file, contentType); err != nil {
				file.Close()
				log.Printf("Failed to upload file %s: %v", destPath, err)
				continue
			}
			file.Close()

			uploadedPaths = append(uploadedPaths, destPath)
			log.Printf("Uploaded file: %s", destPath)
		}

		if len(uploadedPaths) == 0 {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(uploadResponse{
				Success: false,
				Error:   "Failed to save any files",
			})
			return
		}

		autoIngest := r.FormValue("autoIngest") != "false"
		if autoIngest && len(uploadedPaths) > 0 {
			ids, err := deps.Queue.AddWorkflow(jobqueue.Workflow{
				Tasks: []jobqueue.WorkflowTask{
					{
						Command:   "ingest",
						Arguments: nil,
						Input:     "uploads/",
					},
				},
			})
			if err != nil {
				log.Printf("Failed to create ingest job: %v", err)
			} else if len(ids) > 0 {
				log.Printf("Created ingest job %s for uploaded files", ids[0])
			}
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(uploadResponse{
			Success: true,
			Files:   uploadedPaths,
			Message: fmt.Sprintf("Uploaded %d file(s)", len(uploadedPaths)),
		})
	}
}

func configHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			cfg, cfgPath, err := appconfig.Load()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			currentConfig = cfg
			data := configTemplateData{
				Config:                 cfg,
				ConfigPath:             cfgPath,
				ActiveDBPath:           cfg.DBPath,
				EmbeddingModels:        tasks.EmbedModelList(),
				TaggerModels:           tasks.TaggerModelList(),
				FaceModels:             tasks.FaceModelList(),
				DirectMLInstalled:      tasks.DirectMLRuntimeInstalled(),
				TranscriptionProviders: transcribe.Providers(),
			}
			if p, err := transcribe.Active(); err == nil {
				data.TranscriptionModels = p.Models()
			}
			if err := renderer.Templates().ExecuteTemplate(w, "config", data); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
		case http.MethodPost:
			var req updateConfigRequest
			if err := readJSONBody(r, &req); err != nil {
				http.Error(w, "bad json", http.StatusBadRequest)
				return
			}
			req.DBPath = strings.TrimSpace(req.DBPath)
			if req.DBPath == "" {
				http.Error(w, "dbPath cannot be empty", http.StatusBadRequest)
				return
			}

			oldCfg := currentConfig
			oldDBPath := currentConfig.DBPath
			oldEmbeddingModel := currentConfig.EmbeddingModel
			oldFaceModel := currentConfig.FaceModel
			newCfg := currentConfig
			newCfg.DBPath = req.DBPath
			// Port: positive in-range values overwrite; 0 = field absent from a
			// partial POST, leave the stored value alone. Takes effect on restart.
			if req.Port > 0 && req.Port <= 65535 {
				newCfg.Port = req.Port
			}
			if strings.TrimSpace(req.DownloadPath) != "" {
				newCfg.DownloadPath = strings.TrimSpace(req.DownloadPath)
			}
			if strings.TrimSpace(req.OllamaBaseURL) != "" {
				newCfg.OllamaBaseURL = strings.TrimSpace(req.OllamaBaseURL)
			}
			if strings.TrimSpace(req.OllamaModel) != "" {
				newCfg.OllamaModel = strings.TrimSpace(req.OllamaModel)
			}
			if req.DescribePrompt != "" {
				newCfg.DescribePrompt = req.DescribePrompt
			}
			// Inference provider: assign unconditionally so changing tabs
			// (including selecting Off) takes immediate effect.
			if v := strings.ToLower(strings.TrimSpace(req.InferenceProvider)); v != "" {
				newCfg.InferenceProvider = v
			}
			// RunPod fields: only overwrite when the payload includes a
			// non-empty value. Matches the protective pattern used for
			// every other persisted credential field and prevents partial
			// callers (e.g. the Electron client posting just {dbPath} on
			// startup) from accidentally wiping saved RunPod creds.
			if v := strings.TrimSpace(req.RunPodEndpoint); v != "" {
				newCfg.RunPodEndpoint = v
			}
			if v := strings.TrimSpace(req.RunPodAPIKey); v != "" {
				newCfg.RunPodAPIKey = v
			}
			if v := strings.TrimSpace(req.LMStudioBaseURL); v != "" {
				newCfg.LMStudioBaseURL = v
			}
			if v := strings.TrimSpace(req.LMStudioModel); v != "" {
				newCfg.LMStudioModel = v
			}
			if v := strings.TrimSpace(req.LMStudioAPIKey); v != "" {
				newCfg.LMStudioAPIKey = v
			}
			if v := strings.TrimSpace(req.LlamaCppBaseURL); v != "" {
				newCfg.LlamaCppBaseURL = v
			}
			if v := strings.TrimSpace(req.LlamaCppModel); v != "" {
				newCfg.LlamaCppModel = v
			}
			if v := strings.TrimSpace(req.LlamaCppAPIKey); v != "" {
				newCfg.LlamaCppAPIKey = v
			}
			if req.InferenceConcurrency.Ollama > 0 {
				newCfg.InferenceConcurrency.Ollama = req.InferenceConcurrency.Ollama
			}
			if req.InferenceConcurrency.RunPod > 0 {
				newCfg.InferenceConcurrency.RunPod = req.InferenceConcurrency.RunPod
			}
			if req.InferenceConcurrency.LMStudio > 0 {
				newCfg.InferenceConcurrency.LMStudio = req.InferenceConcurrency.LMStudio
			}
			if req.InferenceConcurrency.LlamaCpp > 0 {
				newCfg.InferenceConcurrency.LlamaCpp = req.InferenceConcurrency.LlamaCpp
			}
			if req.LocalComputeConcurrency > 0 {
				newCfg.LocalComputeConcurrency = req.LocalComputeConcurrency
			}
			newCfg.OnnxTagger.ModelPath = strings.TrimSpace(req.OnnxModelPath)
			newCfg.OnnxTagger.LabelsPath = strings.TrimSpace(req.OnnxLabelsPath)
			newCfg.OnnxTagger.ConfigPath = strings.TrimSpace(req.OnnxConfigPath)
			newCfg.OnnxTagger.ORTSharedLibraryPath = strings.TrimSpace(req.OnnxORTSharedLibPath)
			if req.OnnxGeneralThreshold > 0 {
				newCfg.OnnxTagger.GeneralThreshold = req.OnnxGeneralThreshold
			}
			if req.OnnxCharacterThreshold > 0 {
				newCfg.OnnxTagger.CharacterThreshold = req.OnnxCharacterThreshold
			}
			if v := strings.TrimSpace(req.EmbeddingModel); v != "" {
				newCfg.EmbeddingModel = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.EmbeddingProvider)); v != "" {
				newCfg.EmbeddingProvider = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.EmbeddingPerformance)); v != "" {
				newCfg.EmbeddingPerformance = v
			}
			if req.EmbeddingWorkers > 0 {
				newCfg.EmbeddingWorkers = req.EmbeddingWorkers
			}
			if req.EmbeddingThreadsPerWorker > 0 {
				newCfg.EmbeddingThreadsPerWorker = req.EmbeddingThreadsPerWorker
			}
			if v := strings.TrimSpace(req.AutotagModel); v != "" {
				newCfg.AutotagModel = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.AutotagProvider)); v != "" {
				newCfg.AutotagProvider = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.AutotagPerformance)); v != "" {
				newCfg.AutotagPerformance = v
			}
			if req.AutotagWorkers > 0 {
				newCfg.AutotagWorkers = req.AutotagWorkers
			}
			if req.AutotagThreadsPerWorker > 0 {
				newCfg.AutotagThreadsPerWorker = req.AutotagThreadsPerWorker
			}
			if req.OnnxFileTimeoutSeconds != 0 {
				// non-zero so a partial POST doesn't clobber it; negative = disabled.
				newCfg.OnnxFileTimeoutSeconds = req.OnnxFileTimeoutSeconds
			}
			if v := strings.TrimSpace(req.FaceModel); v != "" {
				newCfg.FaceModel = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.FaceRouting)); v == "auto" || v == "single" {
				newCfg.FaceRouting = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.FaceProvider)); v != "" {
				newCfg.FaceProvider = v
			}
			if v := strings.ToLower(strings.TrimSpace(req.FacePerformance)); v != "" {
				newCfg.FacePerformance = v
			}
			if req.FaceWorkers > 0 {
				newCfg.FaceWorkers = req.FaceWorkers
			}
			if req.FaceThreadsPerWorker > 0 {
				newCfg.FaceThreadsPerWorker = req.FaceThreadsPerWorker
			}
			// nil = field omitted (partial POST) â†’ keep; an explicit empty
			// array is a deliberate "remove all BYO entries".
			if req.ByoFaceModels != nil {
				newCfg.ByoFaceModels = req.ByoFaceModels
			}
			// Transcription: provider/model use the protective non-empty
			// pattern; language and VAD are pointers so an explicit empty
			// language ("auto-detect") or unchecked VAD box persists, while
			// partial POSTs that omit the fields leave them alone.
			if v := strings.ToLower(strings.TrimSpace(req.TranscriptionProvider)); v != "" {
				newCfg.TranscriptionProvider = v
			}
			if v := strings.TrimSpace(req.TranscriptionModel); v != "" {
				newCfg.TranscriptionModel = v
			}
			if req.TranscriptionLanguage != nil {
				newCfg.TranscriptionLanguage = strings.TrimSpace(*req.TranscriptionLanguage)
			}
			if req.TranscriptionVADFilter != nil {
				newCfg.TranscriptionVADFilter = *req.TranscriptionVADFilter
			}
			if strings.TrimSpace(req.FasterWhisperPath) != "" {
				newCfg.FasterWhisperPath = strings.TrimSpace(req.FasterWhisperPath)
			}
			if strings.TrimSpace(req.DiscordToken) != "" {
				newCfg.DiscordToken = strings.TrimSpace(req.DiscordToken)
			}
			if req.Roots != nil {
				newCfg.Roots = req.Roots
			}
			cfgPath, err := appconfig.Save(newCfg)
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}

			dbChanged := req.DBPath != oldDBPath
			if dbChanged {
				if err := switchDatabase(req.DBPath); err != nil {
					http.Error(w, "failed to switch database: "+err.Error(), http.StatusInternalServerError)
					return
				}
				// Clear the auth cookie to force re-login after DB switch
				http.SetCookie(w, &http.Cookie{
					Name:     "auth_token",
					Value:    "",
					Path:     "/",
					HttpOnly: true,
					SameSite: http.SameSiteLaxMode,
					Expires:  time.Unix(0, 0),
					MaxAge:   -1,
				})
				log.Println("Auth session cleared due to database switch")
			}
			currentConfig = newCfg

			// If the active embedding model changed, rebuild the ANN index for
			// the new model in the background. Vectors are model-keyed in the DB
			// (no re-inference needed); this just reloads the stored vectors for
			// the now-active model. Done off the request goroutine because a
			// large library can take a while to load.
			if newCfg.EmbeddingModel != oldEmbeddingModel {
				go func(db *sql.DB) {
					model, n, err := tasks.RebuildActiveIndex(db, nil)
					if err != nil {
						log.Printf("embedding index rebuild after model switch failed (model %s): %v", model, err)
						return
					}
					log.Printf("embedding index rebuilt after model switch: %d vectors (model %s)", n, model)
				}(deps.DB)
			}

			// Same for the face index when the active recognizer changed. Face
			// vectors are model-keyed too, so this just reloads stored vectors.
			if newCfg.FaceModel != oldFaceModel {
				go func(db *sql.DB) {
					model, n, err := tasks.RebuildActiveFaceIndex(db, nil)
					if err != nil {
						log.Printf("face index rebuild after model switch failed (model %s): %v", model, err)
						return
					}
					log.Printf("face index rebuilt after model switch: %d faces (model %s)", n, model)
				}(deps.DB)
			}

			// Rebuild storage backends from new config
			newReg, regErrs := storage.BuildRegistry(newCfg.Roots)
			for _, regErr := range regErrs {
				log.Printf("Warning: storage backend init error: %v", regErr)
			}
			deps.Storage.ReplaceWithDefault(newReg.AllBackends(), newReg.DefaultIdx())

			// Re-apply per-bucket concurrency caps so UI changes take effect
			// immediately. Cheap idempotent operation â€” only future ClaimJob
			// calls consult the new value; in-flight jobs are untouched.
			tasks.ApplyHostLimits(deps.Queue, newCfg)

			// Determine if any config field actually changed
			changed := !reflect.DeepEqual(oldCfg, newCfg)

			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status":         "ok",
				"configPath":     cfgPath,
				"activeDBPath":   currentConfig.DBPath,
				"changed":        changed,
				"dbChanged":      dbChanged,
				"logoutRequired": dbChanged,
			})
		default:
			http.Error(w, "Use GET or POST", http.StatusMethodNotAllowed)
		}
	}
}

// getRawQueryParam extracts a parameter value from a raw query string without decoding.
// This allows us to use url.PathUnescape instead of url.QueryUnescape, which:
// 1. Properly handles complex unicode characters
// 2. Does not treat '+' as space (important for file paths that may contain '+')
// 3. Handles all valid percent-encoded sequences
func getRawQueryParam(rawQuery, key string) string {
	keyPrefix := key + "="
	for _, param := range strings.Split(rawQuery, "&") {
		if strings.HasPrefix(param, keyPrefix) {
			return strings.TrimPrefix(param, keyPrefix)
		}
	}
	return ""
}

// mediaFileHandler serves individual media files for preview
func mediaFileHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}

		// Get the file path from query parameter using robust decoding
		// We use getRawQueryParam + PathUnescape to properly handle:
		// 1. Complex unicode characters (properly decoded from percent-encoded UTF-8)
		// 2. Plus signs in paths (not treated as spaces, unlike QueryUnescape)
		// 3. All valid path characters
		rawPath := getRawQueryParam(r.URL.RawQuery, "path")
		if rawPath == "" {
			http.Error(w, "Missing path parameter", http.StatusBadRequest)
			return
		}

		filePath, err := url.PathUnescape(rawPath)
		if err != nil {
			log.Printf("Error decoding path parameter: %v (raw: %s)", err, rawPath)
			http.Error(w, "Invalid path encoding", http.StatusBadRequest)
			return
		}

		// Validate and sanitize the file path
		filePath = strings.TrimSpace(filePath)
		if filePath == "" {
			http.Error(w, "Empty file path", http.StatusBadRequest)
			return
		}

		// If local path, enforce absolute path to avoid traversal via relative inputs
		if !strings.HasPrefix(filePath, "http://") && !strings.HasPrefix(filePath, "https://") && !strings.HasPrefix(filePath, "s3://") {
			if !filepath.IsAbs(filePath) {
				http.Error(w, "Path must be absolute", http.StatusBadRequest)
				return
			}
		}

		// For remote URLs, proxy the request
		if strings.HasPrefix(filePath, "http://") || strings.HasPrefix(filePath, "https://") {
			proxyRemoteMedia(w, r, filePath)
			return
		}

		// For S3 paths, redirect to presigned URL
		if strings.HasPrefix(filePath, "s3://") {
			backend := deps.Storage.BackendFor(filePath)
			if backend == nil {
				http.Error(w, "No storage backend for path", http.StatusNotFound)
				return
			}
			presignedURL, err := backend.MediaURL(filePath)
			if err != nil {
				log.Printf("Failed to generate presigned URL for %s: %v", filePath, err)
				http.Error(w, "Failed to generate media URL", http.StatusInternalServerError)
				return
			}
			http.Redirect(w, r, presignedURL, http.StatusFound)
			return
		}

		// Handle local files
		// Clean the path for consistency
		filePath = filepath.Clean(filePath)

		// Check if file exists
		if !media.CheckFileExists(filePath) {
			log.Printf("File not found: %s", filePath)
			http.Error(w, "File not found", http.StatusNotFound)
			return
		}

		// Get file info for additional validation
		fileInfo, err := os.Stat(filePath)
		if err != nil {
			log.Printf("Error getting file info for '%s': %v", filePath, err)
			http.Error(w, "Cannot access file", http.StatusInternalServerError)
			return
		}

		// Check if it's actually a file (not a directory)
		if fileInfo.IsDir() {
			http.Error(w, "Path is a directory", http.StatusBadRequest)
			return
		}

		// Check file size (prevent serving extremely large files for preview)
		// For localhost serving of large videos, we allow up to 2GB
		const maxFileSize = 2 * 1024 * 1024 * 1024 // 2GB limit
		if fileInfo.Size() > maxFileSize {
			http.Error(w, "File too large for preview", http.StatusRequestEntityTooLarge)
			return
		}

		// Set appropriate content type based on file extension
		ext := strings.ToLower(filepath.Ext(filePath))
		contentType := getContentType(ext)
		w.Header().Set("Content-Type", contentType)

		// Set cache headers for better performance
		w.Header().Set("Cache-Control", "public, max-age=3600") // Cache for 1 hour
		etag := fmt.Sprintf(`"%s-%d-%d"`, filepath.Base(filePath), fileInfo.Size(), fileInfo.ModTime().Unix())
		w.Header().Set("ETag", etag)

		// Check If-None-Match header for caching
		if r.Header.Get("If-None-Match") == etag {
			w.WriteHeader(http.StatusNotModified)
			return
		}

		// Set content length
		w.Header().Set("Content-Length", strconv.FormatInt(fileInfo.Size(), 10))

		// Serve the file
		http.ServeFile(w, r, filePath)
	}
}

// proxyRemoteMedia proxies remote media files with timeout and size limits
func proxyRemoteMedia(w http.ResponseWriter, r *http.Request, remoteURL string) {
	// Create a client with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	// Create request to remote URL
	req, err := http.NewRequest("GET", remoteURL, nil)
	if err != nil {
		log.Printf("Error creating request for remote URL '%s': %v", remoteURL, err)
		http.Error(w, "Invalid remote URL", http.StatusBadRequest)
		return
	}

	// Set User-Agent to identify our requests
	req.Header.Set("User-Agent", "Lowkey-Media-Server/1.0")

	// Make the request
	resp, err := client.Do(req)
	if err != nil {
		log.Printf("Error fetching remote media '%s': %v", remoteURL, err)
		http.Error(w, "Failed to fetch remote media", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Check response status
	if resp.StatusCode != http.StatusOK {
		log.Printf("Remote media returned status %d for URL: %s", resp.StatusCode, remoteURL)
		http.Error(w, "Remote media not accessible", resp.StatusCode)
		return
	}

	// Check content length if provided
	if contentLengthStr := resp.Header.Get("Content-Length"); contentLengthStr != "" {
		if contentLength, err := strconv.ParseInt(contentLengthStr, 10, 64); err == nil {
			const maxFileSize = 100 * 1024 * 1024 // 100MB limit
			if contentLength > maxFileSize {
				http.Error(w, "Remote file too large for preview", http.StatusRequestEntityTooLarge)
				return
			}
		}
	}

	// Copy headers from remote response
	for key, values := range resp.Header {
		if key == "Content-Type" || key == "Content-Length" || key == "Cache-Control" || key == "ETag" {
			for _, value := range values {
				w.Header().Add(key, value)
			}
		}
	}

	// If no cache headers from remote, set our own
	if w.Header().Get("Cache-Control") == "" {
		w.Header().Set("Cache-Control", "public, max-age=1800") // 30 minutes for remote files
	}

	// Copy the response body with size limit
	const maxFileSize = 100 * 1024 * 1024 // 100MB limit
	limitedReader := &io.LimitedReader{R: resp.Body, N: maxFileSize}

	written, err := io.Copy(w, limitedReader)
	if err != nil {
		log.Printf("Error copying remote media response: %v", err)
		return
	}

	// Check if we hit the size limit
	if limitedReader.N <= 0 && written == maxFileSize {
		log.Printf("Remote file too large, truncated at %d bytes: %s", maxFileSize, remoteURL)
	}
}

// getContentType returns the appropriate MIME type for a file extension
func getContentType(ext string) string {
	switch ext {
	case ".jpg", ".jpeg":
		return "image/jpeg"
	case ".png":
		return "image/png"
	case ".gif":
		return "image/gif"
	case ".webp":
		return "image/webp"
	case ".bmp":
		return "image/bmp"
	case ".svg":
		return "image/svg+xml"
	case ".tiff":
		return "image/tiff"
	case ".ico":
		return "image/x-icon"
	case ".mp4":
		return "video/mp4"
	case ".webm":
		return "video/webm"
	case ".ogg":
		return "video/ogg"
	case ".avi":
		return "video/x-msvideo"
	case ".mov":
		return "video/quicktime"
	case ".wmv":
		return "video/x-ms-wmv"
	case ".flv":
		return "video/x-flv"
	case ".mkv":
		return "video/x-matroska"
	case ".m4v":
		return "video/x-m4v"
	default:
		return "application/octet-stream"
	}
}

// openPathHandler opens a local file or directory in the OS default application
func openPathHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		var req struct {
			Path string `json:"path"`
		}
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}

		path := strings.TrimSpace(req.Path)
		if path == "" {
			http.Error(w, "path cannot be empty", http.StatusBadRequest)
			return
		}

		// Validate it's an absolute path to prevent arbitrary command execution
		if !filepath.IsAbs(path) {
			http.Error(w, "path must be absolute", http.StatusBadRequest)
			return
		}

		// Use platform-specific method to open in default application
		if err := platform.OpenFile(path); err != nil {
			log.Printf("Error opening path '%s': %v", path, err)
			http.Error(w, "failed to open path", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
	}
}

func ParseCommand(input string) []string {
	var (
		result   []string
		current  strings.Builder
		inQuotes bool
	)
	for i := 0; i < len(input); i++ {
		c := input[i]
		switch c {
		case '"':
			inQuotes = !inQuotes
		case ' ':
			if inQuotes {
				current.WriteByte(c)
			} else if current.Len() > 0 {
				result = append(result, current.String())
				current.Reset()
			}
		default:
			current.WriteByte(c)
		}
	}
	if current.Len() > 0 {
		result = append(result, current.String())
	}
	return result
}

// -----------------------------------------------------------------------------
// Authentication Handlers
// -----------------------------------------------------------------------------

func authMiddleware(deps *Dependencies, next http.Handler, requiredRole renderer.AuthRole) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if requiredRole == renderer.RolePublic {
			next.ServeHTTP(w, r)
			return
		}

		// Check header credentials first (Authorization Bearer JWT or lk_
		// API key, or the X-API-Key header)
		if tokenString := requestAuthToken(r); tokenString != "" {
			if claims, err := verifyCredential(deps, tokenString); err == nil {
				// Check if user setup is required (logged in as default admin)
				if claims.Username == auth.DefaultAdminUsername {
					setupRequired, _ := deps.Auth.IsSetupRequired()
					if setupRequired {
						if r.Header.Get("Accept") == "application/json" {
							w.Header().Set("Content-Type", "application/json")
							w.WriteHeader(http.StatusForbidden)
							w.Write([]byte(`{"error":"setup_required","message":"Please create a new user account"}`))
						} else {
							http.Redirect(w, r, loginRedirectTarget("/login?setup=true"), http.StatusFound)
						}
						return
					}
				}
				next.ServeHTTP(w, r)
				return
			}
		}

		// Check cookie
		cookie, err := r.Cookie("auth_token")
		if err != nil {
			if r.Header.Get("Accept") == "application/json" {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
			} else {
				http.Redirect(w, r, loginRedirectTarget("/login?redirect="+url.QueryEscape(r.URL.RequestURI())), http.StatusFound)
			}
			return
		}

		// Verify token
		claims, err := deps.Auth.VerifyToken(cookie.Value)
		if err != nil {
			if r.Header.Get("Accept") == "application/json" {
				http.Error(w, "Unauthorized", http.StatusUnauthorized)
			} else {
				http.Redirect(w, r, loginRedirectTarget("/login?redirect="+url.QueryEscape(r.URL.RequestURI())), http.StatusFound)
			}
			return
		}

		// Check if user setup is required (logged in as default admin)
		if claims.Username == auth.DefaultAdminUsername {
			setupRequired, _ := deps.Auth.IsSetupRequired()
			if setupRequired {
				if r.Header.Get("Accept") == "application/json" {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusForbidden)
					w.Write([]byte(`{"error":"setup_required","message":"Please create a new user account"}`))
				} else {
					http.Redirect(w, r, loginRedirectTarget("/login?setup=true"), http.StatusFound)
				}
				return
			}
		}

		// For now, assume all authenticated users have Admin role
		// In the future, we would check the role from the token/claims
		next.ServeHTTP(w, r)
	})
}

func loginPageHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Use GET", http.StatusMethodNotAllowed)
			return
		}
		// Until first-run setup finishes there is no account to log into â€”
		// the wizard owns account creation.
		if !appconfig.Get().SetupComplete {
			http.Redirect(w, r, "/setup", http.StatusFound)
			return
		}
		if err := renderer.Templates().ExecuteTemplate(w, "login", nil); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	}
}

// getAllowedOrigin determines the allowed CORS origin based on the request
// Supports browser extensions (chrome-extension://, moz-extension://) and Electron renderer
func getAllowedOrigin(r *http.Request) string {
	origin := r.Header.Get("Origin")
	// Allow browser extensions
	if strings.HasPrefix(origin, "chrome-extension://") || strings.HasPrefix(origin, "moz-extension://") {
		return origin
	}
	// Default to Electron renderer origin
	return "http://localhost:1212"
}

func loginAPIHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers for all requests (including preflight)
		w.Header().Set("Access-Control-Allow-Origin", getAllowedOrigin(r))
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Use POST", http.StatusMethodNotAllowed)
			return
		}

		var creds struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}
		if err := json.NewDecoder(r.Body).Decode(&creds); err != nil {
			http.Error(w, "Invalid JSON", http.StatusBadRequest)
			return
		}

		token, err := deps.Auth.Login(creds.Username, creds.Password)
		if err != nil {
			http.Error(w, `{"error":"Invalid credentials"}`, http.StatusUnauthorized)
			return
		}

		http.SetCookie(w, &http.Cookie{
			Name:     "auth_token",
			Value:    token,
			Path:     "/",
			HttpOnly: true,
			SameSite: http.SameSiteLaxMode,
			Expires:  time.Now().Add(24 * time.Hour),
		})

		// Check if setup is required (logged in as default admin)
		setupRequired := false
		if creds.Username == auth.DefaultAdminUsername {
			setupRequired, _ = deps.Auth.IsSetupRequired()
		}

		w.Header().Set("Content-Type", "application/json")
		// Return token in response body as well for API clients
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"status":         "ok",
			"token":          token,
			"setup_required": setupRequired,
		})
	}
}

func logoutHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", getAllowedOrigin(r))
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		http.SetCookie(w, &http.Cookie{
			Name:     "auth_token",
			Value:    "",
			Path:     "/",
			HttpOnly: true,
			SameSite: http.SameSiteLaxMode,
			Expires:  time.Unix(0, 0),
			MaxAge:   -1,
		})
		if r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"status":"ok"}`))
		} else {
			http.Redirect(w, r, "/login", http.StatusFound)
		}
	}
}

func authStatusHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers
		w.Header().Set("Access-Control-Allow-Origin", getAllowedOrigin(r))
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Credentials", "true")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		// Check header credentials first (Authorization Bearer JWT or lk_
		// API key, or the X-API-Key header)
		if tokenString := requestAuthToken(r); tokenString != "" {
			if claims, err := verifyCredential(deps, tokenString); err == nil {
				w.Header().Set("Content-Type", "application/json")
				json.NewEncoder(w).Encode(map[string]interface{}{
					"loggedIn": true,
					"username": claims.Username,
				})
				return
			}
		}

		cookie, err := r.Cookie("auth_token")
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"loggedIn":false}`))
			return
		}

		claims, err := deps.Auth.VerifyToken(cookie.Value)
		if err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"loggedIn":false}`))
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"loggedIn": true,
			"username": claims.Username,
		})
	}
}

// streamHandler wraps the stream handler with CORS headers (public endpoint)
func streamHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set CORS headers for SSE
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Cache-Control")
		stream.StreamHandler(w, r)
	}
}

func userManagementHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			users, err := deps.Auth.ListUsers()
			if err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{"users": users})

		case http.MethodPost:
			// First-account creation is open (the setup wizard and Electron
			// onboarding run before any credential exists); once a real user
			// exists, only an authenticated caller may create more.
			if setupRequired, _ := deps.Auth.IsSetupRequired(); !setupRequired {
				if !setupAuthed(deps, r) {
					http.Error(w, `{"error":"unauthorized"}`, http.StatusForbidden)
					return
				}
			}
			var req struct {
				Username string `json:"username"`
				Password string `json:"password"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, "Invalid JSON", http.StatusBadRequest)
				return
			}
			if req.Username == "" || req.Password == "" {
				http.Error(w, "Username and password required", http.StatusBadRequest)
				return
			}
			if err := deps.Auth.Register(req.Username, req.Password); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.WriteHeader(http.StatusCreated)
			w.Write([]byte(`{"status":"created"}`))

		case http.MethodDelete:
			username := r.URL.Query().Get("username")
			if username == "" {
				http.Error(w, "Username required", http.StatusBadRequest)
				return
			}
			if err := deps.Auth.DeleteUser(username); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
			w.Write([]byte(`{"status":"deleted"}`))

		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

// -----------------------------------------------------------------------------
// main â€“ start server, then hand control to the menu-bar UI (cgo builds)
// or a headless signal wait (see tray_darwin.go / tray_darwin_nocgo.go).
// -----------------------------------------------------------------------------

func main() {
	// â€“â€“â€“ initialize database â€“â€“â€“
	db, err := initDB()
	if err != nil {
		log.Fatalf("Failed to initialize database: %v", err)
	}
	defer db.Close()

	// â€“â€“â€“ job queue and runners â€“â€“â€“
	log.Println("Initializing job queue with database persistence...")
	// Wire the host resolver before NewQueueWithDB so the DB-restore path
	// uses the full task-aware policy when assigning buckets to persisted
	// jobs.
	jobqueue.SetHostResolver(tasks.ResolveHost)
	jobqueue.SetResourceResolver(tasks.ResolveResources)
	queue := jobqueue.NewQueueWithDB(db)
	log.Printf("Job queue initialized. Current jobs: %d", len(queue.GetJobs()))
	tasks.ApplyHostLimits(queue, currentConfig)
	currentRunners = runners.New(queue)

	// â€“â€“â€“ auth service â€“â€“â€“
	authService := auth.NewAuthService(db, currentConfig.JWTSecret)
	if err := authService.CreateDefaultUser(); err != nil {
		log.Printf("Failed to create default user: %v", err)
	}

	// â€“â€“â€“ create dependencies struct â€“â€“â€“
	storageReg, storageErrs := storage.BuildRegistry(currentConfig.Roots)
	for _, err := range storageErrs {
		log.Printf("Warning: storage backend init error: %v", err)
	}
	tasks.SetStorageRegistry(storageReg)
	deps = &Dependencies{
		Queue:   queue,
		DB:      db,
		Auth:    authService,
		Storage: storageReg,
	}

	// Background idle scheduler (mode/config-gated; dormant when off). Reads
	// deps.Queue on every tick, so it follows database switches transparently.
	startAutoScheduler(deps)

	// â€“â€“â€“ embedding vector index (best-effort, non-fatal) â€“â€“â€“
	log.Printf("Building embedding search indexâ€¦")
	if model, n, err := tasks.RebuildActiveIndex(db, indexProgressFn()); err == nil {
		log.Printf("embedding index loaded: %d vectors (model %s)", n, model)
	} else {
		log.Printf("embedding index unavailable (model %s), using brute-force: %v", model, err)
	}
	buildFaceIndexAtStartup(db)

	// Initialize renderer auth middleware
	renderer.AuthMiddleware = func(next http.Handler, role renderer.AuthRole) http.Handler {
		return authMiddleware(deps, next, role)
	}

	// â€“â€“â€“ routes â€“â€“â€“
	mux := http.NewServeMux()
	mux.HandleFunc("/", renderer.ApplyMiddlewares(homeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/jobs", renderer.ApplyMiddlewares(jobsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/jobs/list", renderer.ApplyMiddlewares(jobsListHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/job/{id}", renderer.ApplyMiddlewares(detailHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/job/{id}/cancel", renderer.ApplyMiddlewares(cancelHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/job/{id}/pause", renderer.ApplyMiddlewares(pauseJobHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/job/{id}/resume", renderer.ApplyMiddlewares(resumeJobHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scheduler", renderer.ApplyMiddlewares(schedulerStatusHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scheduler/mode", renderer.ApplyMiddlewares(schedulerModeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/scheduler/run", renderer.ApplyMiddlewares(schedulerRunHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/job/{id}/copy", renderer.ApplyMiddlewares(copyHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/job/{id}/remove", renderer.ApplyMiddlewares(removeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/jobs/clear", renderer.ApplyMiddlewares(clearNonRunningJobsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/stream", streamHandler())
	mux.HandleFunc("/health", healthHandler(deps))
	mux.HandleFunc("/create", renderer.ApplyMiddlewares(createJobHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media", renderer.ApplyMiddlewares(mediaHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/api", renderer.ApplyMiddlewares(mediaAPIHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/file", renderer.ApplyMiddlewares(mediaFileHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/thumbnail", renderer.ApplyMiddlewares(mediaThumbnailHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/hls", renderer.ApplyMiddlewares(hlsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/hls/", renderer.ApplyMiddlewares(hlsSegmentHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/suggest", renderer.ApplyMiddlewares(mediaSuggestHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/tag", renderer.ApplyMiddlewares(mediaTagHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/media/has-tag", renderer.ApplyMiddlewares(mediaHasTagHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/swipe", renderer.ApplyMiddlewares(swipeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/swipe/api", renderer.ApplyMiddlewares(swipeAPIHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/swipe/manifest.json", swipeManifestHandler())
	mux.HandleFunc("/api/prompts/describe", renderer.ApplyMiddlewares(describePromptHandler, renderer.RoleAdmin))
	mux.HandleFunc("/config", renderer.ApplyMiddlewares(configHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/stats", renderer.ApplyMiddlewares(statsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/ollama/models", renderer.ApplyMiddlewares(ollamaModelsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/tasks", renderer.ApplyMiddlewares(tasksHandler(deps), renderer.RoleAdmin))
	RegisterDepsRoutes(mux)
	registerSetupRoutes(mux, deps)
	RegisterVizRoutes(mux, deps)
	RegisterFacesRoutes(mux, deps)
	mux.HandleFunc("/open", renderer.ApplyMiddlewares(openPathHandler(), renderer.RoleAdmin))
	mux.HandleFunc("/editor", renderer.ApplyMiddlewares(editorHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflow", renderer.ApplyMiddlewares(workflowHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows", renderer.ApplyMiddlewares(workflowsListHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows/create", renderer.ApplyMiddlewares(workflowCreateHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows/{id}", renderer.ApplyMiddlewares(workflowDetailHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/workflows/{id}/run", renderer.ApplyMiddlewares(workflowRunHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/db/query", renderer.ApplyMiddlewares(dbQueryHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/config", renderer.ApplyMiddlewares(configGetAPIHandler(deps), renderer.RoleAdmin))

	// Library stats API (stats_api.go) â€” powers the home page coverage cards
	mux.HandleFunc("/api/stats", renderer.ApplyMiddlewares(statsAPIHandler(deps), renderer.RoleAdmin))

	// Embeddings index + library data API (index_api.go / library_api.go)
	mux.HandleFunc("/api/index/status", renderer.ApplyMiddlewares(indexStatusHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/index/models", renderer.ApplyMiddlewares(indexModelsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/index/rebuild", renderer.ApplyMiddlewares(indexRebuildHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/index/missing", renderer.ApplyMiddlewares(indexMissingHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/embeddings", renderer.ApplyMiddlewares(embeddingsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/embeddings/prune", renderer.ApplyMiddlewares(embeddingsPruneHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/embeddings/all", renderer.ApplyMiddlewares(embeddingsWipeHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/transcript", renderer.ApplyMiddlewares(mediaTranscriptHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/rating", renderer.ApplyMiddlewares(mediaRatingHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/list", renderer.ApplyMiddlewares(tagsListHandler(deps), renderer.RoleAdmin))

	// Auth routes
	mux.HandleFunc("/login", renderer.ApplyMiddlewares(loginPageHandler(deps), renderer.RolePublic))
	mux.HandleFunc("/auth/login", renderer.ApplyMiddlewares(loginAPIHandler(deps), renderer.RolePublic))
	mux.HandleFunc("/auth/logout", renderer.ApplyMiddlewares(logoutHandler(deps), renderer.RolePublic))
	mux.HandleFunc("/auth/status", renderer.ApplyMiddlewares(authStatusHandler(deps), renderer.RolePublic))
	mux.HandleFunc("/auth/users", renderer.ApplyMiddlewares(userManagementHandler(deps), renderer.RolePublic))
	mux.HandleFunc("/auth/keys", renderer.ApplyMiddlewares(apiKeysHandler(deps), renderer.RoleAdmin))

	// File upload
	mux.HandleFunc("/api/upload", renderer.ApplyMiddlewares(uploadHandler(deps), renderer.RoleAdmin))

	// ---- Loki Web Client API ----
	mux.HandleFunc("/api/media", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			lokiMediaHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))
	mux.HandleFunc("/api/media/search", renderer.ApplyMiddlewares(lokiMediaSearchHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/query", renderer.ApplyMiddlewares(lokiMediaQueryHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/metadata", renderer.ApplyMiddlewares(lokiMediaMetadataHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/tags", renderer.ApplyMiddlewares(lokiMediaTagsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/description", renderer.ApplyMiddlewares(lokiUpdateDescriptionHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/preview", renderer.ApplyMiddlewares(lokiMediaPreviewHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/delete", renderer.ApplyMiddlewares(lokiMediaDeleteHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/gif-metadata", renderer.ApplyMiddlewares(lokiGifMetadataHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/similar", renderer.ApplyMiddlewares(lokiSimilarHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/search/visual", renderer.ApplyMiddlewares(lokiVisualSearchHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/media/search/image", renderer.ApplyMiddlewares(lokiImageSearchHandler(deps), renderer.RoleAdmin))

	mux.HandleFunc("/api/taxonomy", renderer.ApplyMiddlewares(lokiTaxonomyHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/taxonomy/categories", renderer.ApplyMiddlewares(lokiCategoriesHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/taxonomy/tags", renderer.ApplyMiddlewares(lokiTaxonomyTagsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/taxonomy/category-count", renderer.ApplyMiddlewares(lokiCategoryCountHandler(deps), renderer.RoleAdmin))

	mux.HandleFunc("/api/tags", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			lokiCreateTagHandler(deps)(w, r)
		case http.MethodDelete:
			lokiDeleteTagHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/rename", renderer.ApplyMiddlewares(lokiRenameTagHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/move", renderer.ApplyMiddlewares(lokiMoveTagHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/order", renderer.ApplyMiddlewares(lokiOrderTagsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/weight", renderer.ApplyMiddlewares(lokiUpdateTagWeightHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/timestamp", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPut:
			lokiUpdateTimestampHandler(deps)(w, r)
		case http.MethodDelete:
			lokiRemoveTimestampHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/preview", renderer.ApplyMiddlewares(lokiTagPreviewHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/tags/count", renderer.ApplyMiddlewares(lokiTagCountHandler(deps), renderer.RoleAdmin))

	mux.HandleFunc("/api/categories", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			lokiCreateCategoryHandler(deps)(w, r)
		case http.MethodDelete:
			lokiDeleteCategoryHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))
	mux.HandleFunc("/api/categories/rename", renderer.ApplyMiddlewares(lokiRenameCategoryHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/categories/tag-view-mode", renderer.ApplyMiddlewares(lokiUpdateCategoryTagViewModeHandler(deps), renderer.RoleAdmin))

	mux.HandleFunc("/api/assignments", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			lokiCreateAssignmentHandler(deps)(w, r)
		case http.MethodDelete:
			lokiDeleteAssignmentHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))
	mux.HandleFunc("/api/assignments/weight", renderer.ApplyMiddlewares(lokiUpdateAssignmentWeightHandler(deps), renderer.RoleAdmin))

	mux.HandleFunc("/api/thumbnails", renderer.ApplyMiddlewares(lokiThumbnailsHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/thumbnails/regenerate", renderer.ApplyMiddlewares(lokiRegenerateThumbnailHandler(deps), renderer.RoleAdmin))

	// Filesystem browser (web mode)
	mux.HandleFunc("/api/fs/list", renderer.ApplyMiddlewares(fsListHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/fs/scan", renderer.ApplyMiddlewares(fsScanHandler(deps), renderer.RoleAdmin))

	mux.HandleFunc("/api/settings", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			lokiSettingsGetHandler(deps)(w, r)
		case http.MethodPut:
			lokiSettingsPutHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))

	mux.HandleFunc("/api/session/keys", renderer.ApplyMiddlewares(lokiSessionDeleteKeysHandler(deps), renderer.RoleAdmin))
	mux.HandleFunc("/api/session/", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			lokiSessionGetHandler(deps)(w, r)
		case http.MethodPut:
			lokiSessionPutHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))
	mux.HandleFunc("/api/session", renderer.ApplyMiddlewares(func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			lokiSessionGetAllHandler(deps)(w, r)
		case http.MethodPut:
			lokiSessionPutHandler(deps)(w, r)
		case http.MethodDelete:
			lokiSessionDeleteHandler(deps)(w, r)
		}
	}, renderer.RoleAdmin))

	mux.HandleFunc("/api/db/load", renderer.ApplyMiddlewares(lokiDBLoadHandler(deps), renderer.RoleAdmin))

	// Loki SPA - serve webpack bundle
	mux.HandleFunc("/app/", renderer.ApplyMiddlewares(lokiSPAHandler(spaFS), renderer.RoleAdmin))

	// Serve embedded static files
	mux.Handle("/static/",
		http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))

	// Verify bundled deps and rebuild model state. Both are non-fatal; the
	// server boots either way and surfaces problems via /api/deps/status.
	go func() {
		bundled.VerifyAll()
		models.RebuildState()
		legacy := filepath.Join(platform.GetDataDir(), "dependencies.json")
		if _, err := os.Stat(legacy); err == nil {
			if rerr := os.Rename(legacy, legacy+".bak"); rerr == nil {
				log.Printf("deps: legacy dependencies.json renamed to dependencies.json.bak")
			}
		}
	}()

	srv = &http.Server{
		Addr: appconfig.Get().ListenAddr(),
		// Activity tracking wraps everything: user-intent requests feed the
		// auto-scheduler's "app is in use" signal.
		Handler: withActivityTracking(mux),
	}

	// Set up signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// start HTTP server in background
	go func() {
		log.Printf("HTTP server starting on %s", appconfig.Get().LocalBaseURL())
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("lowkeymediaserver: %v", err)
		}
	}()

	// Menu-bar UI when built with cgo (the released .app); headless
	// signal-wait otherwise. Opens the browser and blocks until shutdown.
	runDarwinUI(sigChan)
}

// onExit is called when the application is shutting down.
func onExit() {
	log.Println("Shutting down Lowkey Media Server...")

	// Shutdown runners first to stop processing new jobs
	if currentRunners != nil {
		log.Println("Shutting down job runners...")
		currentRunners.Shutdown()
		log.Println("Job runners shut down successfully")
	}

	// Shutdown stream connections
	log.Println("Shutting down stream connections...")
	stream.Shutdown()

	// Save all jobs to database before shutting down
	if deps != nil && deps.Queue != nil {
		log.Println("Saving job queue to database...")
		if err := deps.Queue.SaveAllJobsToDB(); err != nil {
			log.Printf("Error saving jobs to database: %v", err)
		} else {
			log.Println("Job queue saved successfully")
		}
	}

	// Shutdown HTTP server
	log.Println("Shutting down HTTP server...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("HTTP server shutdown error: %v", err)
	} else {
		log.Println("HTTP server shutdown complete")
	}

	log.Println("Lowkey Media Server shutdown complete")
}
