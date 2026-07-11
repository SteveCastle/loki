package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stevecastle/shrike/auth"
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

	db := setupTestDB(t)
	// Auth tables + a user so the test can make an ADMIN request: importing
	// scanned paths into the library only happens for admins — anonymous
	// (public view-only) scans return the listing without mutating the DB.
	for _, stmt := range []string{
		`CREATE TABLE users (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			username TEXT UNIQUE NOT NULL,
			password_hash TEXT NOT NULL,
			created_at INTEGER
		)`,
		`CREATE TABLE api_keys (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id INTEGER NOT NULL,
			name TEXT NOT NULL,
			key_hash TEXT UNIQUE NOT NULL,
			prefix TEXT NOT NULL,
			created_at INTEGER,
			last_used_at INTEGER
		)`,
	} {
		if _, err := db.Exec(stmt); err != nil {
			t.Fatalf("create auth tables: %v", err)
		}
	}
	svc := auth.NewAuthService(db, "test-secret")
	if err := svc.Register("steve", "pw"); err != nil {
		t.Fatalf("register: %v", err)
	}
	token, err := svc.Login("steve", "pw")
	if err != nil {
		t.Fatalf("login: %v", err)
	}

	b := storage.NewLocalBackend(tmpDir, "test")
	deps := &Dependencies{DB: db, Auth: svc, Storage: storage.NewRegistry([]storage.Backend{b})}
	handler := fsScanHandler(deps)

	scan := func(authed bool) *httptest.ResponseRecorder {
		body, _ := json.Marshal(map[string]any{"path": tmpDir, "recursive": false})
		req := httptest.NewRequest(http.MethodPost, "/api/fs/scan", bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if authed {
			req.Header.Set("Authorization", "Bearer "+token)
		}
		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		return rr
	}

	// Anonymous: 200 with the listing, but no library import.
	if rr := scan(false); rr.Code != http.StatusOK {
		t.Fatalf("anonymous scan: expected 200, got %d", rr.Code)
	}
	var count int
	deps.DB.QueryRow("SELECT COUNT(*) FROM media WHERE path = ?",
		filepath.Join(tmpDir, "photo.jpg")).Scan(&count)
	if count != 0 {
		t.Fatalf("anonymous scan must not import: got %d rows", count)
	}

	// Admin: 200 and the path lands in the media table.
	if rr := scan(true); rr.Code != http.StatusOK {
		t.Fatalf("admin scan: expected 200, got %d", rr.Code)
	}
	deps.DB.QueryRow("SELECT COUNT(*) FROM media WHERE path = ?",
		filepath.Join(tmpDir, "photo.jpg")).Scan(&count)
	if count != 1 {
		t.Fatalf("expected 1 row in media table after admin scan, got %d", count)
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

// scanFakeS3 is a minimal storage.Backend for exercising the s3 file-pick
// path: Scan returns the objects directly under a prefix; Exists reports a
// non-"/" key present in the object set.
type scanFakeS3 struct{ objs map[string]bool }

func (f *scanFakeS3) List(ctx context.Context, path string) ([]storage.Entry, error) {
	return nil, nil
}
func (f *scanFakeS3) Scan(ctx context.Context, path string, recursive bool) ([]storage.FileInfo, error) {
	var out []storage.FileInfo
	for k := range f.objs {
		if strings.HasPrefix(k, path) && k != path {
			rest := strings.TrimPrefix(k, path)
			if recursive || !strings.Contains(rest, "/") {
				out = append(out, storage.FileInfo{Path: k})
			}
		}
	}
	return out, nil
}
func (f *scanFakeS3) Download(ctx context.Context, p string) (io.ReadCloser, error) {
	return io.NopCloser(bytes.NewReader(nil)), nil
}
func (f *scanFakeS3) Upload(ctx context.Context, p string, r io.Reader, ct string) error {
	return nil
}
func (f *scanFakeS3) MediaURL(p string) (string, error) { return p, nil }
func (f *scanFakeS3) Exists(ctx context.Context, p string) (bool, error) {
	return f.objs[p], nil
}
func (f *scanFakeS3) Contains(p string) bool { return strings.HasPrefix(p, "s3://b/") }
func (f *scanFakeS3) Root() storage.Entry {
	return storage.Entry{Name: "b", Path: "s3://b/", IsDir: true, Type: "s3"}
}

// TestFsScanHandler_S3FilePicksParent: picking a specific s3 FILE must scan
// its parent prefix and set the cursor to that file (the reported bug: a
// file path was scanned as a prefix -> empty library).
func TestFsScanHandler_S3FilePicksParent(t *testing.T) {
	b := &scanFakeS3{objs: map[string]bool{
		"s3://b/photos/a.jpg": true,
		"s3://b/photos/b.jpg": true,
		"s3://b/photos/c.jpg": true,
	}}
	deps := &Dependencies{DB: setupTestDB(t), Storage: storage.NewRegistry([]storage.Backend{b})}
	handler := fsScanHandler(deps)

	body, _ := json.Marshal(map[string]any{"path": "s3://b/photos/b.jpg", "recursive": false})
	req := httptest.NewRequest(http.MethodPost, "/api/fs/scan", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
	var resp fsScanResponse
	json.Unmarshal(rr.Body.Bytes(), &resp)
	if len(resp.Library) != 3 {
		t.Fatalf("expected the parent folder's 3 files, got %d: %v", len(resp.Library), resp.Library)
	}
	if resp.Library[resp.Cursor].Path != "s3://b/photos/b.jpg" {
		t.Fatalf("cursor points at %q, want the picked file s3://b/photos/b.jpg", resp.Library[resp.Cursor].Path)
	}
}
