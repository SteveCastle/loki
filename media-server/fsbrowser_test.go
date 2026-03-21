package main

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
	if len(resp.Entries) == 0 {
		t.Fatal("expected at least one filesystem root entry")
	}
}

func TestFsListHandler_BrowseDirectory(t *testing.T) {
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
	origGet := getRootPaths
	getRootPaths = func() []string { return []string{allowedDir} }
	defer func() { getRootPaths = origGet }()

	handler := fsListHandler(deps)

	body, _ := json.Marshal(map[string]string{"path": tmpDir})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/list", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}

// Suppress unused import warnings for runtime on non-Windows builds
var _ = runtime.GOOS
