# Web Filesystem Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow web mode users to browse the server's filesystem and load media directories, with configurable root path constraints.

**Architecture:** Two new Go API endpoints (`/api/fs/list` for browsing, `/api/fs/scan` for loading) plus a React file browser modal. The platform abstraction layer maps existing IPC channels to these endpoints so the XState state machine needs no code changes.

**Tech Stack:** Go (net/http, filepath, os), React, TypeScript, XState, CSS

**Spec:** `docs/superpowers/specs/2026-03-20-web-fs-browser-design.md`

---

### Task 1: Add RootPaths to Config

**Files:**
- Modify: `media-server/appconfig/config.go:16-46` (Config struct)
- Modify: `media-server/appconfig/config.go:75-95` (defaultConfig)
- Test: `media-server/appconfig/config_test.go`

- [ ] **Step 1: Write test for RootPaths config field**

In `media-server/appconfig/config_test.go`, add a test that verifies RootPaths marshals/unmarshals correctly:

```go
func TestConfigRootPaths(t *testing.T) {
	c := Config{
		DBPath:    "/tmp/test.db",
		JWTSecret: "test-secret",
		RootPaths: []string{"/mnt/media", "/home/user/photos"},
	}
	data, err := json.Marshal(c)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var c2 Config
	if err := json.Unmarshal(data, &c2); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if len(c2.RootPaths) != 2 || c2.RootPaths[0] != "/mnt/media" || c2.RootPaths[1] != "/home/user/photos" {
		t.Fatalf("unexpected RootPaths: %v", c2.RootPaths)
	}
}

func TestConfigRootPathsDefaultEmpty(t *testing.T) {
	c := defaultConfig()
	if c.RootPaths == nil {
		t.Fatal("RootPaths should not be nil, should be empty slice")
	}
	if len(c.RootPaths) != 0 {
		t.Fatalf("expected empty RootPaths, got: %v", c.RootPaths)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test -run TestConfigRootPaths -v ./appconfig/`
Expected: FAIL — `RootPaths` field does not exist

- [ ] **Step 3: Add RootPaths field to Config struct and defaultConfig**

In `media-server/appconfig/config.go`, add to the Config struct (after JWTSecret):

```go
// Allowed root paths for web filesystem browsing (empty = unrestricted)
RootPaths []string `json:"rootPaths"`
```

In `defaultConfig()`, add to the return struct (after JWTSecret line):

```go
RootPaths: []string{},
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd media-server && go test -run TestConfigRootPaths -v ./appconfig/`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/appconfig/config.go media-server/appconfig/config_test.go
git commit -m "feat: add RootPaths config field for web filesystem browsing"
```

---

### Task 2: Path jail validation helper

**Files:**
- Create: `media-server/fsbrowser.go`
- Test: `media-server/fsbrowser_test.go`

- [ ] **Step 1: Write tests for path validation**

Create `media-server/fsbrowser_test.go`:

```go
package main

import (
	"testing"
)

func TestValidatePathWithinRoots(t *testing.T) {
	roots := []string{"/mnt/media", "/home/user/photos"}

	tests := []struct {
		name    string
		path    string
		roots   []string
		wantErr bool
	}{
		{"valid path under root", "/mnt/media/vacation", roots, false},
		{"valid path exact root", "/mnt/media", roots, false},
		{"valid second root", "/home/user/photos/2024", roots, false},
		{"path outside roots", "/etc/passwd", roots, true},
		{"traversal attack", "/mnt/media/../../../etc/passwd", roots, true},
		{"empty roots allows all", "/any/path", []string{}, false},
		{"empty path with roots", "", roots, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validatePathWithinRoots(tt.path, tt.roots)
			if (err != nil) != tt.wantErr {
				t.Errorf("validatePathWithinRoots(%q, %v) error = %v, wantErr %v",
					tt.path, tt.roots, err, tt.wantErr)
			}
		})
	}
}

func TestMediaExtensionFilter(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"photo.jpg", true},
		{"photo.JPG", true},
		{"video.mp4", true},
		{"song.mp3", true},
		{"doc.pdf", false},
		{"readme.txt", false},
		{"image.webp", true},
		{"clip.mkv", true},
		{"audio.flac", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isMediaFile(tt.name); got != tt.want {
				t.Errorf("isMediaFile(%q) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test -run "TestValidatePath|TestMediaExtension" -v .`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement path validation and media filter**

Create `media-server/fsbrowser.go`:

```go
package main

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

// mediaExtRegex matches supported media file extensions (case-insensitive).
// Shared between /api/fs/list and /api/fs/scan to avoid drift.
var mediaExtRegex = regexp.MustCompile(
	`(?i)\.(jpg|jpeg|jfif|webp|png|webm|mp4|mov|mpeg|gif|mkv|m4v|mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape)$`,
)

// isMediaFile returns true if the filename has a supported media extension.
func isMediaFile(name string) bool {
	return mediaExtRegex.MatchString(name)
}

// validatePathWithinRoots checks that path is under one of the allowed roots.
// If roots is empty, all paths are allowed.
// An empty path is always allowed (used to request root listing).
func validatePathWithinRoots(path string, roots []string) error {
	if len(roots) == 0 || path == "" {
		return nil
	}

	cleaned := filepath.Clean(path)

	// Resolve symlinks to prevent escaping via symlink
	resolved, err := filepath.EvalSymlinks(cleaned)
	if err != nil {
		// If the path doesn't exist yet, use the cleaned version
		resolved = cleaned
	}

	for _, root := range roots {
		cleanRoot := filepath.Clean(root)
		rel, err := filepath.Rel(cleanRoot, resolved)
		if err != nil {
			continue
		}
		// If rel starts with "..", the path is outside the root
		if !strings.HasPrefix(rel, "..") {
			return nil
		}
	}

	return fmt.Errorf("path %q is not within any allowed root", path)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test -run "TestValidatePath|TestMediaExtension" -v .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/fsbrowser.go media-server/fsbrowser_test.go
git commit -m "feat: add path jail validation and media extension filter"
```

---

### Task 3: `/api/fs/list` endpoint

**Files:**
- Modify: `media-server/fsbrowser.go`
- Modify: `media-server/fsbrowser_test.go`

- [ ] **Step 1: Write tests for fs list handler**

Add to `media-server/fsbrowser_test.go`:

```go
import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestFsListHandler_EmptyPathNoRoots(t *testing.T) {
	deps := &Dependencies{DB: setupTestDB(t)}
	handler := fsListHandler(deps)

	body, _ := json.Marshal(map[string]string{"path": ""})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/list", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp fsListResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Should return filesystem roots
	if len(resp.Entries) == 0 {
		t.Fatal("expected at least one filesystem root entry")
	}
}

func TestFsListHandler_BrowseDirectory(t *testing.T) {
	// Create a temp directory with some files
	tmpDir := t.TempDir()
	os.MkdirAll(filepath.Join(tmpDir, "subdir"), 0755)
	os.WriteFile(filepath.Join(tmpDir, "photo.jpg"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "readme.txt"), []byte("text"), 0644)

	deps := &Dependencies{DB: setupTestDB(t)}
	handler := fsListHandler(deps)

	body, _ := json.Marshal(map[string]string{"path": tmpDir})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/list", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp fsListResponse
	json.Unmarshal(rr.Body.Bytes(), &resp)

	// Should see subdir and photo.jpg, but NOT readme.txt
	names := map[string]bool{}
	for _, e := range resp.Entries {
		names[e.Name] = true
	}
	if !names["subdir"] {
		t.Error("expected subdir in entries")
	}
	if !names["photo.jpg"] {
		t.Error("expected photo.jpg in entries")
	}
	if names["readme.txt"] {
		t.Error("readme.txt should be filtered out (not a media file)")
	}
}

func TestFsListHandler_RootPathJail(t *testing.T) {
	tmpDir := t.TempDir()
	allowedDir := filepath.Join(tmpDir, "allowed")
	os.MkdirAll(allowedDir, 0755)

	deps := &Dependencies{DB: setupTestDB(t)}
	// Temporarily override config to set roots
	origGet := getRootPaths
	getRootPaths = func() []string { return []string{allowedDir} }
	defer func() { getRootPaths = origGet }()

	handler := fsListHandler(deps)

	// Trying to browse parent should fail
	body, _ := json.Marshal(map[string]string{"path": tmpDir})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/list", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test -run "TestFsListHandler" -v .`
Expected: FAIL — `fsListHandler`, `fsListResponse`, `getRootPaths` not defined

- [ ] **Step 3: Implement fs list handler**

Add to `media-server/fsbrowser.go`:

```go
import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"

	"loki/appconfig"
)

// getRootPaths is a function variable so tests can override it.
var getRootPaths = func() []string {
	cfg := appconfig.Get()
	return cfg.RootPaths
}

type fsEntry struct {
	Name    string  `json:"name"`
	Path    string  `json:"path"`
	IsDir   bool    `json:"isDir"`
	MtimeMs float64 `json:"mtimeMs"`
}

type fsListResponse struct {
	Entries []fsEntry `json:"entries"`
	Parent  *string   `json:"parent"`
	Roots   []string  `json:"roots"`
}

// getFilesystemRoots returns root paths for the current OS.
func getFilesystemRoots() []fsEntry {
	if runtime.GOOS == "windows" {
		var roots []fsEntry
		for c := 'A'; c <= 'Z'; c++ {
			drive := string(c) + ":\\"
			if _, err := os.Stat(drive); err == nil {
				roots = append(roots, fsEntry{
					Name:  drive,
					Path:  drive,
					IsDir: true,
				})
			}
		}
		return roots
	}
	return []fsEntry{{Name: "/", Path: "/", IsDir: true}}
}

func fsListHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path string `json:"path"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		roots := getRootPaths()

		// Empty path: return roots
		if req.Path == "" {
			var entries []fsEntry
			if len(roots) > 0 {
				for _, root := range roots {
					info, err := os.Stat(root)
					if err != nil {
						continue
					}
					entries = append(entries, fsEntry{
						Name:    filepath.Base(root),
						Path:    root,
						IsDir:   true,
						MtimeMs: float64(info.ModTime().UnixMilli()),
					})
				}
			} else {
				entries = getFilesystemRoots()
			}
			writeJSON(w, fsListResponse{
				Entries: entries,
				Parent:  nil,
				Roots:   roots,
			})
			return
		}

		// Validate path is within roots
		if err := validatePathWithinRoots(req.Path, roots); err != nil {
			httpError(w, err.Error(), http.StatusForbidden)
			return
		}

		dirEntries, err := os.ReadDir(req.Path)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		var entries []fsEntry
		for _, de := range dirEntries {
			info, err := de.Info()
			if err != nil {
				continue
			}
			if de.IsDir() {
				entries = append(entries, fsEntry{
					Name:    de.Name(),
					Path:    filepath.Join(req.Path, de.Name()),
					IsDir:   true,
					MtimeMs: float64(info.ModTime().UnixMilli()),
				})
			} else if isMediaFile(de.Name()) {
				entries = append(entries, fsEntry{
					Name:    de.Name(),
					Path:    filepath.Join(req.Path, de.Name()),
					IsDir:   false,
					MtimeMs: float64(info.ModTime().UnixMilli()),
				})
			}
		}

		// Sort: directories first, then alphabetical
		sort.Slice(entries, func(i, j int) bool {
			if entries[i].IsDir != entries[j].IsDir {
				return entries[i].IsDir
			}
			return strings.ToLower(entries[i].Name) < strings.ToLower(entries[j].Name)
		})

		// Calculate parent path (nil if at a root boundary)
		var parent *string
		cleanPath := filepath.Clean(req.Path)
		parentPath := filepath.Dir(cleanPath)
		if len(roots) > 0 {
			// Check if we're at a root — don't allow navigating above
			isAtRoot := false
			for _, root := range roots {
				if filepath.Clean(root) == cleanPath {
					isAtRoot = true
					break
				}
			}
			if !isAtRoot {
				parent = &parentPath
			}
		} else {
			// Unrestricted: parent is nil only at filesystem root
			if parentPath != cleanPath {
				parent = &parentPath
			}
		}

		writeJSON(w, fsListResponse{
			Entries: entries,
			Parent:  parent,
			Roots:   roots,
		})
	}
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test -run "TestFsListHandler" -v .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/fsbrowser.go media-server/fsbrowser_test.go
git commit -m "feat: add /api/fs/list endpoint for directory browsing"
```

---

### Task 4: `/api/fs/scan` endpoint

**Files:**
- Modify: `media-server/fsbrowser.go`
- Modify: `media-server/fsbrowser_test.go`

- [ ] **Step 1: Write tests for fs scan handler**

Add to `media-server/fsbrowser_test.go`:

```go
func TestFsScanHandler_ScanDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	// Create media files
	os.WriteFile(filepath.Join(tmpDir, "a.jpg"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "b.mp4"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "c.txt"), []byte("text"), 0644)
	// Create subdirectory with more media
	sub := filepath.Join(tmpDir, "sub")
	os.MkdirAll(sub, 0755)
	os.WriteFile(filepath.Join(sub, "d.png"), []byte("fake"), 0644)

	deps := &Dependencies{DB: setupTestDB(t)}
	handler := fsScanHandler(deps)

	// Non-recursive scan
	body, _ := json.Marshal(map[string]any{"path": tmpDir, "recursive": false})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/scan", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp fsScanResponse
	json.Unmarshal(rr.Body.Bytes(), &resp)

	// Should find a.jpg and b.mp4, NOT c.txt, NOT sub/d.png
	if len(resp.Library) != 2 {
		t.Fatalf("expected 2 files, got %d: %v", len(resp.Library), resp.Library)
	}
}

func TestFsScanHandler_Recursive(t *testing.T) {
	tmpDir := t.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "a.jpg"), []byte("fake"), 0644)
	sub := filepath.Join(tmpDir, "sub")
	os.MkdirAll(sub, 0755)
	os.WriteFile(filepath.Join(sub, "d.png"), []byte("fake"), 0644)

	deps := &Dependencies{DB: setupTestDB(t)}
	handler := fsScanHandler(deps)

	body, _ := json.Marshal(map[string]any{"path": tmpDir, "recursive": true})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/scan", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp fsScanResponse
	json.Unmarshal(rr.Body.Bytes(), &resp)

	// Should find a.jpg and sub/d.png
	if len(resp.Library) != 2 {
		t.Fatalf("expected 2 files, got %d: %v", len(resp.Library), resp.Library)
	}
}

func TestFsScanHandler_InsertsIntoDB(t *testing.T) {
	tmpDir := t.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "photo.jpg"), []byte("fake"), 0644)

	deps := &Dependencies{DB: setupTestDB(t)}
	handler := fsScanHandler(deps)

	body, _ := json.Marshal(map[string]any{"path": tmpDir, "recursive": false})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/scan", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	// Verify the file was inserted into the media table
	var count int
	deps.DB.QueryRow("SELECT COUNT(*) FROM media WHERE path = ?",
		filepath.Join(tmpDir, "photo.jpg")).Scan(&count)
	if count != 1 {
		t.Fatalf("expected 1 row in media table, got %d", count)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test -run "TestFsScanHandler" -v .`
Expected: FAIL — `fsScanHandler`, `fsScanResponse` not defined

- [ ] **Step 3: Implement fs scan handler**

Add to `media-server/fsbrowser.go`:

```go
import (
	"database/sql"
	"io/fs"
)

type fsScanFile struct {
	Path    string  `json:"path"`
	MtimeMs float64 `json:"mtimeMs"`
}

type fsScanResponse struct {
	Library []fsScanFile `json:"library"`
	Cursor  int          `json:"cursor"`
}

func fsScanHandler(deps *Dependencies) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Path      string `json:"path"`
			Recursive bool   `json:"recursive"`
		}
		if err := readJSON(r, &req); err != nil {
			httpError(w, "bad request", http.StatusBadRequest)
			return
		}

		roots := getRootPaths()
		if err := validatePathWithinRoots(req.Path, roots); err != nil {
			httpError(w, err.Error(), http.StatusForbidden)
			return
		}

		var files []fsScanFile

		if req.Recursive {
			filepath.WalkDir(req.Path, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return nil // skip errors
				}
				// Skip symlinks to directories to prevent loops
				// WalkDir reports symlinks with d.Type() == fs.ModeSymlink, not d.IsDir()
				if d.Type()&fs.ModeSymlink != 0 {
					if info, err := os.Stat(path); err == nil && info.IsDir() {
						return filepath.SkipDir
					}
				}
				if !d.IsDir() && isMediaFile(d.Name()) {
					info, err := d.Info()
					if err != nil {
						return nil
					}
					files = append(files, fsScanFile{
						Path:    path,
						MtimeMs: float64(info.ModTime().UnixMilli()),
					})
				}
				return nil
			})
		} else {
			dirEntries, err := os.ReadDir(req.Path)
			if err != nil {
				httpError(w, err.Error(), http.StatusInternalServerError)
				return
			}
			for _, de := range dirEntries {
				if !de.IsDir() && isMediaFile(de.Name()) {
					info, err := de.Info()
					if err != nil {
						continue
					}
					files = append(files, fsScanFile{
						Path:    filepath.Join(req.Path, de.Name()),
						MtimeMs: float64(info.ModTime().UnixMilli()),
					})
				}
			}
		}

		// Insert files into media table
		insertBulkMediaPaths(deps.DB, files)

		if files == nil {
			files = []fsScanFile{}
		}

		writeJSON(w, fsScanResponse{
			Library: files,
			Cursor:  0,
		})
	}
}

// insertBulkMediaPaths inserts file paths into the media table.
func insertBulkMediaPaths(db *sql.DB, files []fsScanFile) {
	if len(files) == 0 {
		return
	}
	tx, err := db.Begin()
	if err != nil {
		return
	}
	stmt, err := tx.Prepare("INSERT INTO media (path) VALUES (?) ON CONFLICT(path) DO NOTHING")
	if err != nil {
		tx.Rollback()
		return
	}
	defer stmt.Close()

	for _, f := range files {
		stmt.Exec(f.Path)
	}
	tx.Commit()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd media-server && go test -run "TestFsScanHandler" -v .`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/fsbrowser.go media-server/fsbrowser_test.go
git commit -m "feat: add /api/fs/scan endpoint for directory scanning"
```

---

### Task 5: Register routes in main.go and platform variants

**Files:**
- Modify: `media-server/main.go:2823-2826` (after thumbnails routes)
- Modify: `media-server/main_darwin.go:2407-2410` (after thumbnails routes)
- Modify: `media-server/main_linux.go:2407-2410` (after thumbnails routes)

Routes are duplicated across all three platform files. The same two lines must be added in each.

- [ ] **Step 1: Add route registrations to all three files**

In each of `media-server/main.go`, `media-server/main_darwin.go`, and `media-server/main_linux.go`, find the thumbnails route block (after `/api/thumbnails/regenerate`) and add:

```go
// Filesystem browser (web mode)
mux.HandleFunc("/api/fs/list", renderer.ApplyMiddlewares(fsListHandler(deps), renderer.RoleAdmin))
mux.HandleFunc("/api/fs/scan", renderer.ApplyMiddlewares(fsScanHandler(deps), renderer.RoleAdmin))
```

- [ ] **Step 2: Verify the server compiles**

Run: `cd media-server && go build .`
Expected: SUCCESS (no errors)

- [ ] **Step 3: Run all tests to verify nothing broke**

Run: `cd media-server && go test -v .`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add media-server/main.go media-server/main_darwin.go media-server/main_linux.go
git commit -m "feat: register /api/fs/list and /api/fs/scan routes"
```

---

### Task 6: Platform layer — wire up web mode

**Files:**
- Modify: `src/renderer/platform.ts:8-14` (capabilities)
- Modify: `src/renderer/platform.ts:340-345` (stubbedChannels)
- Modify: `src/renderer/platform.ts:85-221` (channelToEndpoint)

- [ ] **Step 1: Enable fileSystemAccess in web mode**

In `src/renderer/platform.ts`, change the capabilities object (line 8-14). Change `fileSystemAccess` to always be `true`:

```typescript
export const capabilities = {
  fileSystemAccess: true,
  clipboard: isElectron,
  windowControls: isElectron,
  autoUpdate: isElectron,
  shutdown: isElectron,
};
```

- [ ] **Step 2: Remove select-directory and load-files from stubbed channels**

In `src/renderer/platform.ts`, update the `stubbedChannels` array (line 340-345) to remove `'select-directory'` and `'load-files'`:

```typescript
const stubbedChannels = [
  'select-file', 'select-db', 'select-new-path',
  'refresh-library', 'copy-file-into-clipboard',
  'check-for-updates', 'update-elo',
  'load-duplicates-by-path', 'merge-duplicates-by-path',
];
```

- [ ] **Step 3: Add load-files to channelToEndpoint mapping**

In `src/renderer/platform.ts`, add to the `channelToEndpoint` map (inside the `map` object, after the `'load-db'` entry around line 218):

```typescript
'load-files': {
  url: '/api/fs/scan',
  method: 'POST',
  argsToBody: (args) => ({ path: args[0], recursive: args[2] ?? false }),
},
```

- [ ] **Step 4: Add select-directory handling**

The `select-directory` channel cannot go through `channelToEndpoint` because it needs to open a modal, not make an API call. Add special handling in the web-mode `invoke` function (around line 347, before the `stubbedChannels` check):

```typescript
invoke = async (channel, args) => {
  // Special handling: select-directory opens file browser modal
  if (channel === 'select-directory') {
    const { openFileBrowser } = await import('./components/controls/file-browser-modal');
    return openFileBrowser();
  }

  if (stubbedChannels.includes(channel)) {
    console.warn(`[platform] Stubbed in web mode: ${channel}`);
    return undefined;
  }
  // ... rest of existing invoke logic
```

- [ ] **Step 5: Commit**

```bash
git add src/renderer/platform.ts
git commit -m "feat: wire up filesystem browsing in web mode platform layer"
```

---

### Task 7: File browser modal component

**Files:**
- Create: `src/renderer/components/controls/file-browser-modal.tsx`
- Create: `src/renderer/components/controls/file-browser-modal.css`

- [ ] **Step 1: Create the CSS file**

Create `src/renderer/components/controls/file-browser-modal.css`:

```css
.file-browser-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.file-browser-modal {
  background: var(--bg-secondary, #1e1e1e);
  border: 1px solid var(--border-color, #333);
  border-radius: 8px;
  width: 600px;
  max-width: 90vw;
  max-height: 70vh;
  display: flex;
  flex-direction: column;
  color: var(--text-primary, #e0e0e0);
}

.file-browser-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-color, #333);
}

.file-browser-header h3 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.file-browser-breadcrumb {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 8px 16px;
  font-size: 12px;
  overflow-x: auto;
  white-space: nowrap;
  border-bottom: 1px solid var(--border-color, #333);
}

.file-browser-breadcrumb span {
  cursor: pointer;
  opacity: 0.7;
}

.file-browser-breadcrumb span:hover {
  opacity: 1;
  text-decoration: underline;
}

.file-browser-breadcrumb .separator {
  opacity: 0.4;
  cursor: default;
}

.file-browser-breadcrumb .separator:hover {
  text-decoration: none;
}

.file-browser-entries {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}

.file-browser-entry {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 16px;
  cursor: pointer;
  font-size: 13px;
}

.file-browser-entry:hover {
  background: var(--bg-hover, #2a2a2a);
}

.file-browser-entry .icon {
  font-size: 16px;
  width: 20px;
  text-align: center;
  flex-shrink: 0;
}

.file-browser-entry .name {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-browser-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid var(--border-color, #333);
}

.file-browser-footer button {
  padding: 6px 16px;
  border-radius: 4px;
  border: 1px solid var(--border-color, #333);
  background: var(--bg-secondary, #1e1e1e);
  color: var(--text-primary, #e0e0e0);
  cursor: pointer;
  font-size: 13px;
}

.file-browser-footer button:hover {
  background: var(--bg-hover, #2a2a2a);
}

.file-browser-footer button.primary {
  background: var(--accent-color, #4a9eff);
  border-color: var(--accent-color, #4a9eff);
  color: #fff;
}

.file-browser-footer button.primary:hover {
  opacity: 0.9;
}

.file-browser-loading {
  padding: 24px;
  text-align: center;
  opacity: 0.6;
}

.file-browser-empty {
  padding: 24px;
  text-align: center;
  opacity: 0.5;
}
```

- [ ] **Step 2: Create the modal component**

Create `src/renderer/components/controls/file-browser-modal.tsx`:

```tsx
import { useState, useEffect, useCallback, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import './file-browser-modal.css';

type FsEntry = {
  name: string;
  path: string;
  isDir: boolean;
  mtimeMs: number;
};

type FsListResponse = {
  entries: FsEntry[];
  parent: string | null;
  roots: string[];
};

// Promise resolve/reject stored here so platform.ts can await the modal
let resolveModal: ((path: string) => void) | null = null;
let rejectModal: (() => void) | null = null;
let mountContainer: HTMLDivElement | null = null;
let root: ReturnType<typeof createRoot> | null = null;

function FileBrowserModal() {
  const [currentPath, setCurrentPath] = useState('');
  const [entries, setEntries] = useState<FsEntry[]>([]);
  const [parent, setParent] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const fetchEntries = useCallback(async (path: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/fs/list', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ path }),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Error ${res.status}`);
      }
      const data: FsListResponse = await res.json();
      setEntries(data.entries || []);
      setParent(data.parent);
      setCurrentPath(path);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchEntries('');
  }, [fetchEntries]);

  const handleEntryClick = (entry: FsEntry) => {
    if (entry.isDir) {
      fetchEntries(entry.path);
    }
  };

  const handleOpen = () => {
    if (resolveModal && currentPath) {
      resolveModal(currentPath);
      cleanup();
    }
  };

  const handleCancel = () => {
    if (rejectModal) {
      rejectModal();
    }
    cleanup();
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      handleCancel();
    }
  };

  // Build breadcrumb segments from currentPath
  const breadcrumbs = currentPath
    ? currentPath.replace(/\\/g, '/').split('/').filter(Boolean)
    : [];

  // Reconstruct path up to each breadcrumb index
  const breadcrumbPath = (index: number) => {
    const isWindows = currentPath.includes('\\') || /^[A-Z]:/.test(currentPath);
    const sep = isWindows ? '\\' : '/';
    const segments = breadcrumbs.slice(0, index + 1);
    if (isWindows) {
      // "C:" alone is not a valid dir path on Windows; needs trailing backslash
      const p = segments.join(sep);
      return /^[A-Z]:$/.test(p) ? p + sep : p;
    }
    return sep + segments.join(sep);
  };

  return (
    <div className="file-browser-overlay" onClick={handleOverlayClick}>
      <div className="file-browser-modal" ref={ref}>
        <div className="file-browser-header">
          <h3>Browse Directory</h3>
        </div>

        {currentPath && (
          <div className="file-browser-breadcrumb">
            <span onClick={() => fetchEntries('')}>Root</span>
            {breadcrumbs.map((seg, i) => (
              <span key={i}>
                <span className="separator"> / </span>
                <span onClick={() => fetchEntries(breadcrumbPath(i))}>{seg}</span>
              </span>
            ))}
          </div>
        )}

        <div className="file-browser-entries">
          {loading && <div className="file-browser-loading">Loading...</div>}
          {error && <div className="file-browser-empty">{error}</div>}
          {!loading && !error && entries.length === 0 && (
            <div className="file-browser-empty">Empty directory</div>
          )}
          {!loading && !error && parent !== null && (
            <div className="file-browser-entry" onClick={() => fetchEntries(parent!)}>
              <span className="icon">..</span>
              <span className="name">(parent directory)</span>
            </div>
          )}
          {!loading &&
            !error &&
            entries.map((entry) => (
              <div
                key={entry.path}
                className="file-browser-entry"
                onClick={() => handleEntryClick(entry)}
              >
                <span className="icon">{entry.isDir ? '\uD83D\uDCC1' : '\uD83D\uDCC4'}</span>
                <span className="name">{entry.name}</span>
              </div>
            ))}
        </div>

        <div className="file-browser-footer">
          <button onClick={handleCancel}>Cancel</button>
          <button className="primary" onClick={handleOpen} disabled={!currentPath}>
            Open
          </button>
        </div>
      </div>
    </div>
  );
}

function cleanup() {
  resolveModal = null;
  rejectModal = null;
  if (root) {
    root.unmount();
    root = null;
  }
  if (mountContainer) {
    mountContainer.remove();
    mountContainer = null;
  }
}

/**
 * Opens the file browser modal and returns a promise that resolves
 * with the selected directory path (bare string) or rejects on cancel.
 */
export function openFileBrowser(): Promise<string> {
  // Clean up any previous instance
  cleanup();

  return new Promise<string>((resolve, reject) => {
    resolveModal = resolve;
    rejectModal = reject;

    mountContainer = document.createElement('div');
    mountContainer.id = 'file-browser-mount';
    document.body.appendChild(mountContainer);

    root = createRoot(mountContainer);
    root.render(<FileBrowserModal />);
  });
}
```

- [ ] **Step 3: Verify the app compiles**

Run: `cd /c/Users/steph/dev/loki && npx webpack --config .erb/configs/webpack.config.renderer.dev.ts 2>&1 | head -20`
Expected: No compilation errors related to file-browser-modal

- [ ] **Step 4: Commit**

```bash
git add src/renderer/components/controls/file-browser-modal.tsx src/renderer/components/controls/file-browser-modal.css
git commit -m "feat: add file browser modal component for web mode"
```

---

### Task 8: Integration test — end-to-end verification

- [ ] **Step 1: Run all Go tests**

Run: `cd media-server && go test -v .`
Expected: All tests PASS

- [ ] **Step 2: Run all Go tests including subpackages**

Run: `cd media-server && go test -v ./...`
Expected: All tests PASS

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd /c/Users/steph/dev/loki && npx tsc --noEmit 2>&1 | grep -i error | head -20`
Expected: No new errors (pre-existing type errors may appear per MEMORY.md)

- [ ] **Step 4: Manual smoke test**

Start the media server and open the web UI in a browser:
1. Click the folder icon in the command palette
2. The file browser modal should appear showing filesystem roots (or configured roots)
3. Navigate into a directory containing media files — should see folders and media files, no .txt or other non-media files
4. Click "Open" on a directory — the app should load media from that directory
5. Verify the media thumbnails/previews load correctly

- [ ] **Step 5: Test root path jail**

Add a `rootPaths` entry to config.json (e.g., `"rootPaths": ["C:\\Users\\steph\\Pictures"]`):
1. Restart server
2. Open file browser — should only show the configured root
3. Cannot navigate above it (no parent link at root level)
4. Navigating subdirectories works normally

- [ ] **Step 6: Final commit if any fixes needed**

```bash
git add -A
git commit -m "fix: address integration test findings"
```
