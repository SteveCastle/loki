package main

import (
	"database/sql"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/stevecastle/shrike/storage"
)

type fsEntry struct {
	Name    string  `json:"name"`
	Path    string  `json:"path"`
	IsDir   bool    `json:"isDir"`
	MtimeMs float64 `json:"mtimeMs"`
	Size    int64   `json:"size"`
	Type    string  `json:"type,omitempty"`
}

type fsListResponse struct {
	Entries []fsEntry `json:"entries"`
	Parent  *string   `json:"parent"`
	Roots   []string  `json:"roots"`
}

// computeParent returns the parent of a path, handling both local and s3:// paths.
func computeParent(p string) string {
	if strings.HasPrefix(p, "s3://") {
		trimmed := strings.TrimSuffix(p, "/")
		idx := strings.LastIndex(trimmed, "/")
		if idx <= len("s3://") {
			return p // already at bucket root
		}
		return trimmed[:idx+1]
	}
	return filepath.Dir(filepath.Clean(p))
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

		// Empty path: return all configured roots
		if req.Path == "" {
			allRoots := deps.Storage.AllRoots()
			entries := make([]fsEntry, 0, len(allRoots))
			for _, root := range allRoots {
				entries = append(entries, fsEntry{
					Name:  root.Name,
					Path:  root.Path,
					IsDir: true,
					Type:  root.Type,
				})
			}
			writeJSON(w, fsListResponse{
				Entries: entries,
				Parent:  nil,
				Roots:   []string{},
			})
			return
		}

		// Find the backend that owns this path
		backend := deps.Storage.BackendFor(req.Path)
		if backend == nil {
			httpError(w, "path is not within any configured storage root", http.StatusForbidden)
			return
		}

		storageEntries, err := backend.List(r.Context(), req.Path)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		entries := make([]fsEntry, 0, len(storageEntries))
		for _, e := range storageEntries {
			entries = append(entries, fsEntry{
				Name:    e.Name,
				Path:    e.Path,
				IsDir:   e.IsDir,
				MtimeMs: e.MtimeMs,
				Size:    e.Size,
			})
		}

		// Calculate parent — nil if we're at a root
		var parent *string
		isRoot := false
		for _, root := range deps.Storage.AllRoots() {
			if req.Path == root.Path {
				isRoot = true
				break
			}
		}
		if !isRoot {
			p := computeParent(req.Path)
			if p != req.Path {
				parent = &p
			}
		}

		writeJSON(w, fsListResponse{
			Entries: entries,
			Parent:  parent,
			Roots:   []string{},
		})
	}
}

type fsScanFile struct {
	Path    string  `json:"path"`
	MtimeMs float64 `json:"mtimeMs"`
	Size    int64   `json:"size"`
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

		backend := deps.Storage.BackendFor(req.Path)
		if backend == nil {
			httpError(w, "path is not within any configured storage root", http.StatusForbidden)
			return
		}

		// If the path points to a file, scan its parent directory instead
		// and remember the selected file so we can set the cursor to it.
		scanPath := req.Path
		selectedFile := ""
		if strings.HasPrefix(req.Path, "s3://") {
			// S3 directories from the picker carry a trailing "/"; a file
			// does not. Confirm it's a real object before treating it as the
			// selected file (a bare prefix without "/" is still a folder).
			if !strings.HasSuffix(req.Path, "/") {
				if ok, _ := backend.Exists(r.Context(), req.Path); ok {
					selectedFile = req.Path
					scanPath = computeParent(req.Path)
				}
			}
		} else if info, err := os.Stat(req.Path); err == nil && !info.IsDir() {
			selectedFile = req.Path
			scanPath = filepath.Dir(req.Path)
		}

		storageFiles, err := backend.Scan(r.Context(), scanPath, req.Recursive)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		files := make([]fsScanFile, 0, len(storageFiles))
		for _, f := range storageFiles {
			files = append(files, fsScanFile{Path: f.Path, MtimeMs: f.MtimeMs, Size: f.Size})
		}

		// Importing scanned paths into the library is a WRITE — view-only
		// public visitors browse the listing without mutating the library.
		if isAdminRequest(deps, r) {
			insertBulkMediaPaths(deps.DB, files)
		}

		cursor := 0
		if selectedFile != "" {
			// S3 keys are already canonical; filepath.Clean would mangle the
			// s3:// scheme, so match those directly.
			isS3 := strings.HasPrefix(selectedFile, "s3://")
			cleanSelected := selectedFile
			if !isS3 {
				cleanSelected = filepath.Clean(selectedFile)
			}
			for i, f := range files {
				fp := f.Path
				if !isS3 {
					fp = filepath.Clean(fp)
				}
				if fp == cleanSelected {
					cursor = i
					break
				}
			}
		}

		writeJSON(w, fsScanResponse{
			Library: files,
			Cursor:  cursor,
		})
	}
}

func insertBulkMediaPaths(db *sql.DB, files []fsScanFile) {
	if len(files) == 0 {
		return
	}
	tx, err := db.Begin()
	if err != nil {
		return
	}
	// Backfill size on conflict: rows ingested before sizes flowed through the
	// storage layer (S3 roots especially) sit at NULL/0 and self-heal on browse.
	stmt, err := tx.Prepare(`INSERT INTO media (path, size) VALUES (?, ?)
		ON CONFLICT(path) DO UPDATE SET size = excluded.size
		WHERE excluded.size > 0 AND (media.size IS NULL OR media.size = 0)`)
	if err != nil {
		tx.Rollback()
		return
	}
	defer stmt.Close()

	for _, f := range files {
		stmt.Exec(f.Path, f.Size)
	}
	tx.Commit()
}

// isMediaFile is kept for use by other files in the main package (e.g., thumbnail.go).
func isMediaFile(name string) bool {
	return storage.IsMediaFile(name)
}
