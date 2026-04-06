package main

import (
	"database/sql"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"

	"github.com/stevecastle/shrike/appconfig"
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
						Name:    root,
						Path:    root,
						IsDir:   true,
						MtimeMs: float64(info.ModTime().UnixMilli()),
					})
				}
			} else {
				entries = getFilesystemRoots()
			}
			if entries == nil {
				entries = []fsEntry{}
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

		if entries == nil {
			entries = []fsEntry{}
		}

		// Calculate parent path (nil if at a root boundary)
		var parent *string
		cleanPath := filepath.Clean(req.Path)
		parentPath := filepath.Dir(cleanPath)
		if len(roots) > 0 {
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

		// If the path points to a file, scan its parent directory instead
		// and remember the selected file so we can set the cursor to it.
		scanPath := req.Path
		selectedFile := ""
		if info, err := os.Stat(req.Path); err == nil && !info.IsDir() {
			selectedFile = req.Path
			scanPath = filepath.Dir(req.Path)
		}

		var files []fsScanFile

		if req.Recursive {
			filepath.WalkDir(scanPath, func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					return nil
				}
				// Skip symlinks to directories to prevent loops
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
			dirEntries, err := os.ReadDir(scanPath)
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
						Path:    filepath.Join(scanPath, de.Name()),
						MtimeMs: float64(info.ModTime().UnixMilli()),
					})
				}
			}
		}

		insertBulkMediaPaths(deps.DB, files)

		if files == nil {
			files = []fsScanFile{}
		}

		cursor := 0
		if selectedFile != "" {
			cleanSelected := filepath.Clean(selectedFile)
			for i, f := range files {
				if filepath.Clean(f.Path) == cleanSelected {
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
