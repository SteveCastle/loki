package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stevecastle/shrike/storage"
)

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
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry(nil)}
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
	// With no configured roots, we get an empty list
	if resp.Entries == nil {
		t.Fatal("entries should not be nil")
	}
}

func TestFsListHandler_BrowseDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	os.MkdirAll(filepath.Join(tmpDir, "subdir"), 0755)
	os.WriteFile(filepath.Join(tmpDir, "photo.jpg"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "readme.txt"), []byte("text"), 0644)

	b := storage.NewLocalBackend(tmpDir, "test")
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry([]storage.Backend{b})}
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

	b := storage.NewLocalBackend(allowedDir, "allowed")
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry([]storage.Backend{b})}
	handler := fsListHandler(deps)

	// Try to browse tmpDir which is outside the allowed root
	body, _ := json.Marshal(map[string]string{"path": tmpDir})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/list", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestFsScanHandler_ScanDirectory(t *testing.T) {
	tmpDir := t.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "a.jpg"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "b.mp4"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(tmpDir, "c.txt"), []byte("text"), 0644)
	sub := filepath.Join(tmpDir, "sub")
	os.MkdirAll(sub, 0755)
	os.WriteFile(filepath.Join(sub, "d.png"), []byte("fake"), 0644)

	b := storage.NewLocalBackend(tmpDir, "test")
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry([]storage.Backend{b})}
	handler := fsScanHandler(deps)

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

	b := storage.NewLocalBackend(tmpDir, "test")
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry([]storage.Backend{b})}
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

	if len(resp.Library) != 2 {
		t.Fatalf("expected 2 files, got %d: %v", len(resp.Library), resp.Library)
	}
}

func TestFsScanHandler_InsertsIntoDB(t *testing.T) {
	tmpDir := t.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "photo.jpg"), []byte("fake"), 0644)

	b := storage.NewLocalBackend(tmpDir, "test")
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry([]storage.Backend{b})}
	handler := fsScanHandler(deps)

	body, _ := json.Marshal(map[string]any{"path": tmpDir, "recursive": false})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/scan", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	var count int
	deps.DB.QueryRow("SELECT COUNT(*) FROM media WHERE path = ?",
		filepath.Join(tmpDir, "photo.jpg")).Scan(&count)
	if count != 1 {
		t.Fatalf("expected 1 row in media table, got %d", count)
	}
}

func TestComputeParent_LocalPath(t *testing.T) {
	got := computeParent("/mnt/photos/vacation")
	want := filepath.Clean("/mnt/photos")
	if got != want {
		t.Errorf("computeParent local = %q, want %q", got, want)
	}
}

func TestComputeParent_S3Path(t *testing.T) {
	got := computeParent("s3://bucket/media/photos/")
	if got != "s3://bucket/media/" {
		t.Errorf("computeParent s3 = %q, want 's3://bucket/media/'", got)
	}
}
