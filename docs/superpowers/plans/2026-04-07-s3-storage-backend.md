# S3 Storage Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add S3-compatible object storage as an alternative storage backend alongside local filesystem, with presigned URL serving and thumbnail generation support.

**Architecture:** A `storage.Backend` interface with `LocalBackend` and `S3Backend` implementations, managed by a `Registry`. Each configured root maps to one backend. Existing handlers (`fsbrowser`, `mediaFileHandler`, `thumbnailHandler`) resolve the backend via the registry instead of accessing the filesystem directly.

**Tech Stack:** Go (AWS SDK v2 for S3), React/TypeScript (minimal frontend changes)

**Spec:** `docs/superpowers/specs/2026-04-07-s3-storage-backend-design.md`

---

### Task 1: Add AWS SDK v2 Dependency

**Files:**
- Modify: `media-server/go.mod`

- [ ] **Step 1: Add AWS SDK v2 packages**

```bash
cd media-server && go get github.com/aws/aws-sdk-go-v2 github.com/aws/aws-sdk-go-v2/config github.com/aws/aws-sdk-go-v2/credentials github.com/aws/aws-sdk-go-v2/service/s3 github.com/aws/aws-sdk-go-v2/feature/s3/manager
```

- [ ] **Step 2: Verify build**

Run: `cd media-server && go build ./...`
Expected: clean build, no errors

- [ ] **Step 3: Commit**

```bash
git add media-server/go.mod media-server/go.sum
git commit -m "chore: add AWS SDK v2 dependency for S3 storage backend"
```

---

### Task 2: Update Config — StorageRoot Type and Migration

**Files:**
- Modify: `media-server/appconfig/config.go:16-49` (Config struct)
- Modify: `media-server/appconfig/config.go:78-100` (defaultConfig)
- Modify: `media-server/appconfig/config.go:156-247` (Load — migration logic)
- Test: `media-server/appconfig/config_test.go`

- [ ] **Step 1: Write test for StorageRoot config and migration**

Add to `media-server/appconfig/config_test.go`:

```go
func TestStorageRootMigrationFromRootPaths(t *testing.T) {
	// Simulate old config with rootPaths but no roots
	dir := t.TempDir()
	oldConfig := map[string]any{
		"dbPath":    filepath.Join(dir, "media.db"),
		"rootPaths": []string{"/mnt/photos", "/mnt/videos"},
		"jwtSecret": "test-secret",
	}
	data, _ := json.Marshal(oldConfig)
	configPath := filepath.Join(dir, "config.json")
	os.WriteFile(configPath, data, 0644)

	// Override getConfigPath for test
	origGetConfigPath := getConfigPath
	getConfigPath = func() (string, error) { return configPath, nil }
	defer func() { getConfigPath = origGetConfigPath }()

	cfg, _, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if len(cfg.Roots) != 2 {
		t.Fatalf("Expected 2 roots, got %d", len(cfg.Roots))
	}
	if cfg.Roots[0].Type != "local" {
		t.Errorf("Expected type 'local', got %q", cfg.Roots[0].Type)
	}
	if cfg.Roots[0].Path != "/mnt/photos" {
		t.Errorf("Expected path '/mnt/photos', got %q", cfg.Roots[0].Path)
	}
	if cfg.Roots[0].Label != "/mnt/photos" {
		t.Errorf("Expected label '/mnt/photos', got %q", cfg.Roots[0].Label)
	}
	if cfg.Roots[1].Path != "/mnt/videos" {
		t.Errorf("Expected path '/mnt/videos', got %q", cfg.Roots[1].Path)
	}
}

func TestStorageRootS3Config(t *testing.T) {
	dir := t.TempDir()
	s3Config := map[string]any{
		"dbPath":    filepath.Join(dir, "media.db"),
		"jwtSecret": "test-secret",
		"roots": []map[string]any{
			{
				"type":  "s3",
				"label": "Cloud Archive",
				"endpoint": "https://s3.us-east-1.amazonaws.com",
				"region": "us-east-1",
				"bucket": "my-bucket",
				"prefix": "media/",
				"accessKey": "AKIATEST",
				"secretKey": "secret123",
			},
		},
	}
	data, _ := json.Marshal(s3Config)
	configPath := filepath.Join(dir, "config.json")
	os.WriteFile(configPath, data, 0644)

	origGetConfigPath := getConfigPath
	getConfigPath = func() (string, error) { return configPath, nil }
	defer func() { getConfigPath = origGetConfigPath }()

	cfg, _, err := Load()
	if err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if len(cfg.Roots) != 1 {
		t.Fatalf("Expected 1 root, got %d", len(cfg.Roots))
	}
	r := cfg.Roots[0]
	if r.Type != "s3" || r.Bucket != "my-bucket" || r.Prefix != "media/" {
		t.Errorf("S3 root not parsed correctly: %+v", r)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd media-server && go test ./appconfig/ -run "TestStorageRoot" -v`
Expected: FAIL — `StorageRoot` type not defined, `Roots` field doesn't exist

- [ ] **Step 3: Add StorageRoot type and update Config struct**

In `media-server/appconfig/config.go`, add the `StorageRoot` type before the `Config` struct (before line 16) and add `Roots` field to `Config`:

```go
// StorageRoot defines a single storage root — either a local filesystem path or an S3 bucket.
type StorageRoot struct {
	Type     string `json:"type"`               // "local" or "s3"
	Path     string `json:"path,omitempty"`      // local filesystem path
	Label    string `json:"label"`               // display name in UI

	// S3-only fields
	Endpoint        string `json:"endpoint,omitempty"`
	Region          string `json:"region,omitempty"`
	Bucket          string `json:"bucket,omitempty"`
	Prefix          string `json:"prefix,omitempty"`
	AccessKey       string `json:"accessKey,omitempty"`
	SecretKey       string `json:"secretKey,omitempty"`
	ThumbnailPrefix string `json:"thumbnailPrefix,omitempty"` // default: "_thumbnails"
}
```

Add to the `Config` struct (replacing `RootPaths`):

```go
	// Storage roots — each root is either a local path or an S3 location
	Roots []StorageRoot `json:"roots"`

	// Deprecated: use Roots instead. Kept for migration on load.
	RootPaths []string `json:"rootPaths,omitempty"`
```

- [ ] **Step 4: Make getConfigPath overridable for tests**

Change the `getConfigPath` function from a regular function to a function variable (like `getRootPaths` in `fsbrowser.go`):

```go
// getConfigPath returns the full path to the config.json file.
var getConfigPath = func() (string, error) {
	configDir := DefaultConfigDir()
	return filepath.Join(configDir, "config.json"), nil
}
```

- [ ] **Step 5: Update defaultConfig and Load with migration**

In `defaultConfig()`, change `RootPaths: []string{}` to `Roots: []StorageRoot{}`.

In `Load()`, add migration after the JSON unmarshal (after line 195 where `json.Unmarshal` happens, before the field-level defaults merge):

```go
	// Migrate legacy rootPaths to roots
	if len(c.RootPaths) > 0 && len(c.Roots) == 0 {
		for _, p := range c.RootPaths {
			c.Roots = append(c.Roots, StorageRoot{
				Type:  "local",
				Path:  p,
				Label: p,
			})
		}
		c.RootPaths = nil
		needsSave = true
	}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd media-server && go test ./appconfig/ -run "TestStorageRoot" -v`
Expected: PASS

- [ ] **Step 7: Run all tests**

Run: `cd media-server && go test ./...`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add media-server/appconfig/config.go media-server/appconfig/config_test.go
git commit -m "feat: add StorageRoot config type with rootPaths migration"
```

---

### Task 3: Storage Backend Interface and LocalBackend

**Files:**
- Create: `media-server/storage/storage.go` (interface + Entry/FileInfo types)
- Create: `media-server/storage/local.go` (LocalBackend)
- Create: `media-server/storage/registry.go` (Registry)
- Create: `media-server/storage/local_test.go`
- Create: `media-server/storage/registry_test.go`

- [ ] **Step 1: Write failing test for LocalBackend.List**

Create `media-server/storage/local_test.go`:

```go
package storage

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestLocalBackendList(t *testing.T) {
	dir := t.TempDir()
	// Create test files
	os.MkdirAll(filepath.Join(dir, "subdir"), 0755)
	os.WriteFile(filepath.Join(dir, "image.jpg"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(dir, "video.mp4"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(dir, "readme.txt"), []byte("fake"), 0644) // non-media

	b := NewLocalBackend(dir, dir)
	entries, err := b.List(context.Background(), dir)
	if err != nil {
		t.Fatalf("List error: %v", err)
	}

	// Should have subdir + 2 media files (txt filtered out)
	names := map[string]bool{}
	for _, e := range entries {
		names[e.Name] = true
	}
	if !names["subdir"] {
		t.Error("missing subdir")
	}
	if !names["image.jpg"] {
		t.Error("missing image.jpg")
	}
	if !names["video.mp4"] {
		t.Error("missing video.mp4")
	}
	if names["readme.txt"] {
		t.Error("readme.txt should be filtered out")
	}
}

func TestLocalBackendScan(t *testing.T) {
	dir := t.TempDir()
	os.MkdirAll(filepath.Join(dir, "sub"), 0755)
	os.WriteFile(filepath.Join(dir, "a.jpg"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(dir, "sub", "b.png"), []byte("fake"), 0644)
	os.WriteFile(filepath.Join(dir, "sub", "c.txt"), []byte("fake"), 0644)

	b := NewLocalBackend(dir, dir)

	// Non-recursive
	files, err := b.Scan(context.Background(), dir, false)
	if err != nil {
		t.Fatalf("Scan error: %v", err)
	}
	if len(files) != 1 {
		t.Errorf("non-recursive scan: got %d files, want 1", len(files))
	}

	// Recursive
	files, err = b.Scan(context.Background(), dir, true)
	if err != nil {
		t.Fatalf("Scan error: %v", err)
	}
	if len(files) != 2 {
		t.Errorf("recursive scan: got %d files, want 2", len(files))
	}
}

func TestLocalBackendContains(t *testing.T) {
	b := NewLocalBackend("/mnt/photos", "/mnt/photos")

	if !b.Contains("/mnt/photos/vacation/img.jpg") {
		t.Error("should contain path under root")
	}
	if b.Contains("/mnt/videos/clip.mp4") {
		t.Error("should not contain path outside root")
	}
}

func TestLocalBackendMediaURL(t *testing.T) {
	b := NewLocalBackend("/mnt/photos", "/mnt/photos")
	url, err := b.MediaURL("/mnt/photos/img.jpg")
	if err != nil {
		t.Fatal(err)
	}
	expected := "/media/file?path=%2Fmnt%2Fphotos%2Fimg.jpg"
	if url != expected {
		t.Errorf("MediaURL = %q, want %q", url, expected)
	}
}

func TestLocalBackendExists(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "exists.jpg"), []byte("x"), 0644)

	b := NewLocalBackend(dir, dir)
	ctx := context.Background()

	exists, err := b.Exists(ctx, filepath.Join(dir, "exists.jpg"))
	if err != nil || !exists {
		t.Error("should exist")
	}

	exists, err = b.Exists(ctx, filepath.Join(dir, "nope.jpg"))
	if err != nil || exists {
		t.Error("should not exist")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./storage/ -v`
Expected: FAIL — package doesn't exist

- [ ] **Step 3: Create the interface**

Create `media-server/storage/storage.go`:

```go
package storage

import (
	"context"
	"io"
	"regexp"
)

// Entry represents a file or directory in a storage backend.
type Entry struct {
	Name    string  `json:"name"`
	Path    string  `json:"path"`
	IsDir   bool    `json:"isDir"`
	MtimeMs float64 `json:"mtimeMs"`
	Type    string  `json:"type,omitempty"` // "local" or "s3" — included in root listings
}

// FileInfo represents a media file found during scanning.
type FileInfo struct {
	Path    string  `json:"path"`
	MtimeMs float64 `json:"mtimeMs"`
}

// Backend defines the interface for storage operations.
type Backend interface {
	// List returns entries in a directory (one level deep).
	List(ctx context.Context, path string) ([]Entry, error)

	// Scan returns all media files under path, optionally recursive.
	Scan(ctx context.Context, path string, recursive bool) ([]FileInfo, error)

	// Download streams a file for local processing (e.g., ffmpeg).
	Download(ctx context.Context, path string) (io.ReadCloser, error)

	// Upload writes content to the backend. No-op for local.
	Upload(ctx context.Context, path string, r io.Reader, contentType string) error

	// MediaURL returns a URL the client can use to fetch the file.
	MediaURL(path string) (string, error)

	// Exists checks whether a file exists at the given path.
	Exists(ctx context.Context, path string) (bool, error)

	// Contains reports whether this backend owns the given path.
	Contains(path string) bool

	// Root returns the root entry for this backend.
	Root() Entry
}

// MediaExtRegex matches supported media file extensions.
var MediaExtRegex = regexp.MustCompile(
	`(?i)\.(jpg|jpeg|jfif|webp|png|webm|mp4|mov|mpeg|gif|mkv|m4v|mp3|wav|flac|aac|ogg|m4a|opus|wma|aiff|ape)$`,
)

// IsMediaFile returns true if the filename has a supported media extension.
func IsMediaFile(name string) bool {
	return MediaExtRegex.MatchString(name)
}
```

- [ ] **Step 4: Create LocalBackend**

Create `media-server/storage/local.go`:

```go
package storage

import (
	"context"
	"fmt"
	"io"
	"io/fs"
	"net/url"
	"os"
	"path/filepath"
	"strings"
)

// LocalBackend implements Backend for local filesystem storage.
type LocalBackend struct {
	rootPath string
	label    string
}

// NewLocalBackend creates a backend rooted at the given filesystem path.
func NewLocalBackend(rootPath, label string) *LocalBackend {
	return &LocalBackend{rootPath: filepath.Clean(rootPath), label: label}
}

func (b *LocalBackend) Root() Entry {
	return Entry{
		Name:  b.label,
		Path:  b.rootPath,
		IsDir: true,
		Type:  "local",
	}
}

func (b *LocalBackend) Contains(path string) bool {
	cleaned := filepath.Clean(path)
	rel, err := filepath.Rel(b.rootPath, cleaned)
	if err != nil {
		return false
	}
	return !strings.HasPrefix(rel, "..")
}

func (b *LocalBackend) List(ctx context.Context, path string) ([]Entry, error) {
	dirEntries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}

	var entries []Entry
	for _, de := range dirEntries {
		info, err := de.Info()
		if err != nil {
			continue
		}
		if de.IsDir() {
			entries = append(entries, Entry{
				Name:    de.Name(),
				Path:    filepath.Join(path, de.Name()),
				IsDir:   true,
				MtimeMs: float64(info.ModTime().UnixMilli()),
			})
		} else if IsMediaFile(de.Name()) {
			entries = append(entries, Entry{
				Name:    de.Name(),
				Path:    filepath.Join(path, de.Name()),
				IsDir:   false,
				MtimeMs: float64(info.ModTime().UnixMilli()),
			})
		}
	}
	return entries, nil
}

func (b *LocalBackend) Scan(ctx context.Context, path string, recursive bool) ([]FileInfo, error) {
	var files []FileInfo

	if recursive {
		filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil
			}
			if d.Type()&fs.ModeSymlink != 0 {
				if info, err := os.Stat(p); err == nil && info.IsDir() {
					return filepath.SkipDir
				}
			}
			if !d.IsDir() && IsMediaFile(d.Name()) {
				info, err := d.Info()
				if err != nil {
					return nil
				}
				files = append(files, FileInfo{
					Path:    p,
					MtimeMs: float64(info.ModTime().UnixMilli()),
				})
			}
			return nil
		})
	} else {
		dirEntries, err := os.ReadDir(path)
		if err != nil {
			return nil, err
		}
		for _, de := range dirEntries {
			if !de.IsDir() && IsMediaFile(de.Name()) {
				info, err := de.Info()
				if err != nil {
					continue
				}
				files = append(files, FileInfo{
					Path:    filepath.Join(path, de.Name()),
					MtimeMs: float64(info.ModTime().UnixMilli()),
				})
			}
		}
	}
	return files, nil
}

func (b *LocalBackend) Download(ctx context.Context, path string) (io.ReadCloser, error) {
	return os.Open(path)
}

func (b *LocalBackend) Upload(ctx context.Context, path string, r io.Reader, contentType string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = io.Copy(f, r)
	return err
}

func (b *LocalBackend) MediaURL(path string) (string, error) {
	return fmt.Sprintf("/media/file?path=%s", url.QueryEscape(path)), nil
}

func (b *LocalBackend) Exists(ctx context.Context, path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
```

- [ ] **Step 5: Create Registry**

Create `media-server/storage/registry.go`:

```go
package storage

import "sync"

// Registry holds all configured storage backends and routes paths to the correct one.
type Registry struct {
	mu       sync.RWMutex
	backends []Backend
}

// NewRegistry creates a registry from a list of backends.
func NewRegistry(backends []Backend) *Registry {
	return &Registry{backends: backends}
}

// BackendFor returns the backend that owns the given path, or nil.
func (r *Registry) BackendFor(path string) Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, b := range r.backends {
		if b.Contains(path) {
			return b
		}
	}
	return nil
}

// AllRoots returns the root entry for every configured backend.
func (r *Registry) AllRoots() []Entry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	roots := make([]Entry, len(r.backends))
	for i, b := range r.backends {
		roots[i] = b.Root()
	}
	return roots
}

// Replace swaps the backend list (called when config changes).
func (r *Registry) Replace(backends []Backend) {
	r.mu.Lock()
	r.backends = backends
	r.mu.Unlock()
}
```

- [ ] **Step 6: Write Registry test**

Create `media-server/storage/registry_test.go`:

```go
package storage

import (
	"testing"
)

func TestRegistryBackendFor(t *testing.T) {
	b1 := NewLocalBackend("/mnt/photos", "Photos")
	b2 := NewLocalBackend("/mnt/videos", "Videos")
	reg := NewRegistry([]Backend{b1, b2})

	if got := reg.BackendFor("/mnt/photos/img.jpg"); got != b1 {
		t.Error("expected photos backend")
	}
	if got := reg.BackendFor("/mnt/videos/clip.mp4"); got != b2 {
		t.Error("expected videos backend")
	}
	if got := reg.BackendFor("/other/file.jpg"); got != nil {
		t.Error("expected nil for unknown path")
	}
}

func TestRegistryAllRoots(t *testing.T) {
	b1 := NewLocalBackend("/mnt/photos", "Photos")
	b2 := NewLocalBackend("/mnt/videos", "Videos")
	reg := NewRegistry([]Backend{b1, b2})

	roots := reg.AllRoots()
	if len(roots) != 2 {
		t.Fatalf("expected 2 roots, got %d", len(roots))
	}
	if roots[0].Name != "Photos" {
		t.Errorf("root[0].Name = %q, want 'Photos'", roots[0].Name)
	}
	if roots[1].Name != "Videos" {
		t.Errorf("root[1].Name = %q, want 'Videos'", roots[1].Name)
	}
}
```

- [ ] **Step 7: Run tests**

Run: `cd media-server && go test ./storage/ -v`
Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add media-server/storage/
git commit -m "feat: add storage.Backend interface, LocalBackend, and Registry"
```

---

### Task 4: S3Backend Implementation

**Files:**
- Create: `media-server/storage/s3.go`
- Create: `media-server/storage/s3_test.go`

- [ ] **Step 1: Write test for S3Backend.Contains and path helpers**

Create `media-server/storage/s3_test.go`:

```go
package storage

import (
	"testing"
)

func TestS3BackendContains(t *testing.T) {
	b := &S3Backend{
		bucket: "my-bucket",
		prefix: "media/",
	}

	if !b.Contains("s3://my-bucket/media/photo.jpg") {
		t.Error("should contain path under bucket/prefix")
	}
	if !b.Contains("s3://my-bucket/media/sub/photo.jpg") {
		t.Error("should contain nested path under bucket/prefix")
	}
	if b.Contains("s3://my-bucket/other/photo.jpg") {
		t.Error("should not contain path outside prefix")
	}
	if b.Contains("s3://other-bucket/media/photo.jpg") {
		t.Error("should not contain path in different bucket")
	}
	if b.Contains("/local/path/photo.jpg") {
		t.Error("should not contain local paths")
	}
}

func TestS3PathToKey(t *testing.T) {
	b := &S3Backend{bucket: "my-bucket", prefix: "media/"}

	key := b.pathToKey("s3://my-bucket/media/photos/img.jpg")
	if key != "media/photos/img.jpg" {
		t.Errorf("pathToKey = %q, want 'media/photos/img.jpg'", key)
	}
}

func TestS3KeyToPath(t *testing.T) {
	b := &S3Backend{bucket: "my-bucket", prefix: "media/"}

	path := b.keyToPath("media/photos/img.jpg")
	if path != "s3://my-bucket/media/photos/img.jpg" {
		t.Errorf("keyToPath = %q, want 's3://my-bucket/media/photos/img.jpg'", path)
	}
}

func TestS3ThumbnailPath(t *testing.T) {
	b := &S3Backend{
		bucket:          "my-bucket",
		prefix:          "media/",
		thumbnailPrefix: "_thumbnails",
	}

	tp := b.ThumbnailPath("abcdef123.png")
	expected := "s3://my-bucket/_thumbnails/abcdef123.png"
	if tp != expected {
		t.Errorf("ThumbnailPath = %q, want %q", tp, expected)
	}
}

func TestS3Root(t *testing.T) {
	b := &S3Backend{
		bucket: "my-bucket",
		prefix: "media/",
		label:  "Cloud Archive",
	}
	root := b.Root()
	if root.Name != "Cloud Archive" {
		t.Errorf("Name = %q, want 'Cloud Archive'", root.Name)
	}
	if root.Path != "s3://my-bucket/media/" {
		t.Errorf("Path = %q, want 's3://my-bucket/media/'", root.Path)
	}
	if root.Type != "s3" {
		t.Errorf("Type = %q, want 's3'", root.Type)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./storage/ -run "TestS3" -v`
Expected: FAIL — S3Backend not defined

- [ ] **Step 3: Implement S3Backend**

Create `media-server/storage/s3.go`:

```go
package storage

import (
	"context"
	"fmt"
	"io"
	"path"
	"strings"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	awsconfig "github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

// S3Config holds the configuration needed to create an S3Backend.
type S3Config struct {
	Label           string
	Endpoint        string
	Region          string
	Bucket          string
	Prefix          string
	AccessKey       string
	SecretKey       string
	ThumbnailPrefix string
}

// S3Backend implements Backend for S3-compatible object storage.
type S3Backend struct {
	client          *s3.Client
	presignClient   *s3.PresignClient
	bucket          string
	prefix          string
	label           string
	thumbnailPrefix string
}

// NewS3Backend creates a new S3 storage backend.
func NewS3Backend(ctx context.Context, cfg S3Config) (*S3Backend, error) {
	if cfg.ThumbnailPrefix == "" {
		cfg.ThumbnailPrefix = "_thumbnails"
	}

	creds := credentials.NewStaticCredentialsProvider(cfg.AccessKey, cfg.SecretKey, "")

	optFns := []func(*awsconfig.LoadOptions) error{
		awsconfig.WithCredentialsProvider(creds),
		awsconfig.WithRegion(cfg.Region),
	}

	awsCfg, err := awsconfig.LoadDefaultConfig(ctx, optFns...)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS config: %w", err)
	}

	s3Opts := []func(*s3.Options){}
	if cfg.Endpoint != "" {
		s3Opts = append(s3Opts, func(o *s3.Options) {
			o.BaseEndpoint = aws.String(cfg.Endpoint)
			o.UsePathStyle = true // Required for MinIO and most S3-compatible services
		})
	}

	client := s3.NewFromConfig(awsCfg, s3Opts...)
	presignClient := s3.NewPresignClient(client)

	return &S3Backend{
		client:          client,
		presignClient:   presignClient,
		bucket:          cfg.Bucket,
		prefix:          cfg.Prefix,
		label:           cfg.Label,
		thumbnailPrefix: cfg.ThumbnailPrefix,
	}, nil
}

// pathToKey converts an s3://bucket/key path to just the key.
func (b *S3Backend) pathToKey(p string) string {
	prefix := fmt.Sprintf("s3://%s/", b.bucket)
	return strings.TrimPrefix(p, prefix)
}

// keyToPath converts an S3 key to the canonical s3://bucket/key path.
func (b *S3Backend) keyToPath(key string) string {
	return fmt.Sprintf("s3://%s/%s", b.bucket, key)
}

// ThumbnailPath returns the S3 path for a thumbnail with the given filename.
func (b *S3Backend) ThumbnailPath(filename string) string {
	return fmt.Sprintf("s3://%s/%s/%s", b.bucket, b.thumbnailPrefix, filename)
}

func (b *S3Backend) Root() Entry {
	return Entry{
		Name:  b.label,
		Path:  fmt.Sprintf("s3://%s/%s", b.bucket, b.prefix),
		IsDir: true,
		Type:  "s3",
	}
}

func (b *S3Backend) Contains(p string) bool {
	expected := fmt.Sprintf("s3://%s/%s", b.bucket, b.prefix)
	return strings.HasPrefix(p, expected)
}

func (b *S3Backend) List(ctx context.Context, dirPath string) ([]Entry, error) {
	key := b.pathToKey(dirPath)
	if !strings.HasSuffix(key, "/") {
		key += "/"
	}

	delimiter := "/"
	input := &s3.ListObjectsV2Input{
		Bucket:    aws.String(b.bucket),
		Prefix:    aws.String(key),
		Delimiter: aws.String(delimiter),
	}

	var entries []Entry
	paginator := s3.NewListObjectsV2Paginator(b.client, input)
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("S3 list failed: %w", err)
		}

		// Common prefixes are "directories"
		for _, cp := range page.CommonPrefixes {
			prefix := aws.ToString(cp.Prefix)
			name := strings.TrimSuffix(strings.TrimPrefix(prefix, key), "/")
			if name == "" {
				continue
			}
			entries = append(entries, Entry{
				Name:  name,
				Path:  b.keyToPath(prefix),
				IsDir: true,
			})
		}

		// Objects are "files"
		for _, obj := range page.Contents {
			objKey := aws.ToString(obj.Key)
			name := strings.TrimPrefix(objKey, key)
			if name == "" || strings.Contains(name, "/") {
				continue
			}
			if !IsMediaFile(name) {
				continue
			}
			var mtimeMs float64
			if obj.LastModified != nil {
				mtimeMs = float64(obj.LastModified.UnixMilli())
			}
			entries = append(entries, Entry{
				Name:    name,
				Path:    b.keyToPath(objKey),
				IsDir:   false,
				MtimeMs: mtimeMs,
			})
		}
	}
	return entries, nil
}

func (b *S3Backend) Scan(ctx context.Context, dirPath string, recursive bool) ([]FileInfo, error) {
	key := b.pathToKey(dirPath)
	if !strings.HasSuffix(key, "/") {
		key += "/"
	}

	input := &s3.ListObjectsV2Input{
		Bucket: aws.String(b.bucket),
		Prefix: aws.String(key),
	}
	if !recursive {
		input.Delimiter = aws.String("/")
	}

	var files []FileInfo
	paginator := s3.NewListObjectsV2Paginator(b.client, input)
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(ctx)
		if err != nil {
			return nil, fmt.Errorf("S3 scan failed: %w", err)
		}
		for _, obj := range page.Contents {
			objKey := aws.ToString(obj.Key)
			name := path.Base(objKey)
			if !IsMediaFile(name) {
				continue
			}
			var mtimeMs float64
			if obj.LastModified != nil {
				mtimeMs = float64(obj.LastModified.UnixMilli())
			}
			files = append(files, FileInfo{
				Path:    b.keyToPath(objKey),
				MtimeMs: mtimeMs,
			})
		}
	}
	return files, nil
}

func (b *S3Backend) Download(ctx context.Context, p string) (io.ReadCloser, error) {
	key := b.pathToKey(p)
	output, err := b.client.GetObject(ctx, &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		return nil, fmt.Errorf("S3 download failed for %s: %w", key, err)
	}
	return output.Body, nil
}

func (b *S3Backend) Upload(ctx context.Context, p string, r io.Reader, contentType string) error {
	key := b.pathToKey(p)
	_, err := b.client.PutObject(ctx, &s3.PutObjectInput{
		Bucket:      aws.String(b.bucket),
		Key:         aws.String(key),
		Body:        r,
		ContentType: aws.String(contentType),
	})
	if err != nil {
		return fmt.Errorf("S3 upload failed for %s: %w", key, err)
	}
	return nil
}

func (b *S3Backend) MediaURL(p string) (string, error) {
	key := b.pathToKey(p)
	presigned, err := b.presignClient.PresignGetObject(context.Background(), &s3.GetObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(key),
	}, func(o *s3.PresignOptions) {
		o.Expires = 1 * time.Hour
	})
	if err != nil {
		return "", fmt.Errorf("presign failed for %s: %w", key, err)
	}
	return presigned.URL, nil
}

func (b *S3Backend) Exists(ctx context.Context, p string) (bool, error) {
	key := b.pathToKey(p)
	_, err := b.client.HeadObject(ctx, &s3.HeadObjectInput{
		Bucket: aws.String(b.bucket),
		Key:    aws.String(key),
	})
	if err != nil {
		// Check if it's a "not found" error
		if strings.Contains(err.Error(), "NotFound") || strings.Contains(err.Error(), "404") {
			return false, nil
		}
		return false, err
	}
	return true, nil
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./storage/ -run "TestS3" -v`
Expected: PASS (unit tests that don't call S3 APIs)

Run: `cd media-server && go test ./storage/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/storage/s3.go media-server/storage/s3_test.go
git commit -m "feat: add S3Backend implementation with presigned URLs"
```

---

### Task 5: Build Registry from Config

**Files:**
- Create: `media-server/storage/build.go`
- Test: `media-server/storage/build_test.go`

- [ ] **Step 1: Write test for building registry from config**

Create `media-server/storage/build_test.go`:

```go
package storage

import (
	"testing"

	"github.com/stevecastle/shrike/appconfig"
)

func TestBuildRegistryFromConfig(t *testing.T) {
	roots := []appconfig.StorageRoot{
		{Type: "local", Path: "/mnt/photos", Label: "Photos"},
		{Type: "local", Path: "/mnt/videos", Label: "Videos"},
	}

	reg, errs := BuildRegistry(roots)
	if len(errs) != 0 {
		t.Fatalf("unexpected errors: %v", errs)
	}

	allRoots := reg.AllRoots()
	if len(allRoots) != 2 {
		t.Fatalf("expected 2 roots, got %d", len(allRoots))
	}
	if allRoots[0].Name != "Photos" {
		t.Errorf("root[0].Name = %q, want 'Photos'", allRoots[0].Name)
	}
}

func TestBuildRegistrySkipsBadS3(t *testing.T) {
	roots := []appconfig.StorageRoot{
		{Type: "local", Path: "/mnt/photos", Label: "Photos"},
		{Type: "s3", Label: "Bad S3"}, // missing bucket — should error
	}

	reg, errs := BuildRegistry(roots)
	if len(errs) != 1 {
		t.Fatalf("expected 1 error, got %d: %v", len(errs), errs)
	}

	allRoots := reg.AllRoots()
	if len(allRoots) != 1 {
		t.Fatalf("expected 1 root (local only), got %d", len(allRoots))
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd media-server && go test ./storage/ -run "TestBuildRegistry" -v`
Expected: FAIL — BuildRegistry not defined

- [ ] **Step 3: Implement BuildRegistry**

Create `media-server/storage/build.go`:

```go
package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/stevecastle/shrike/appconfig"
)

// BuildRegistry creates a Registry from the config's storage roots.
// Returns the registry and a slice of non-fatal errors (one per failed S3 backend).
// Local backends never fail. Failed S3 backends are skipped.
func BuildRegistry(roots []appconfig.StorageRoot) (*Registry, []error) {
	var backends []Backend
	var errs []error

	for _, root := range roots {
		switch root.Type {
		case "local", "":
			backends = append(backends, NewLocalBackend(root.Path, root.Label))
		case "s3":
			if root.Bucket == "" {
				errs = append(errs, fmt.Errorf("S3 root %q: bucket is required", root.Label))
				continue
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			b, err := NewS3Backend(ctx, S3Config{
				Label:           root.Label,
				Endpoint:        root.Endpoint,
				Region:          root.Region,
				Bucket:          root.Bucket,
				Prefix:          root.Prefix,
				AccessKey:       root.AccessKey,
				SecretKey:       root.SecretKey,
				ThumbnailPrefix: root.ThumbnailPrefix,
			})
			cancel()
			if err != nil {
				errs = append(errs, fmt.Errorf("S3 root %q: %w", root.Label, err))
				continue
			}
			backends = append(backends, b)
		default:
			errs = append(errs, fmt.Errorf("unknown storage type %q for root %q", root.Type, root.Label))
		}
	}

	return NewRegistry(backends), errs
}
```

- [ ] **Step 4: Run tests**

Run: `cd media-server && go test ./storage/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add media-server/storage/build.go media-server/storage/build_test.go
git commit -m "feat: add BuildRegistry to construct backends from config"
```

---

### Task 6: Wire Registry into Server Startup

**Files:**
- Modify: `media-server/main.go` (startup, Dependencies struct, route registration)
- Modify: `media-server/main_linux.go` (same changes)
- Modify: `media-server/main_darwin.go` (same changes)

- [ ] **Step 1: Add Registry to Dependencies struct**

Find the `Dependencies` struct in each platform file and add the registry field. Search for `type Dependencies struct` in `main.go`, `main_linux.go`, `main_darwin.go`:

```go
import "github.com/stevecastle/shrike/storage"
```

Add to `Dependencies`:

```go
	Storage *storage.Registry
```

- [ ] **Step 2: Build registry at startup**

In each platform's `main()` or startup function, after `appconfig.Load()` and before route registration, add:

```go
	storageReg, storageErrs := storage.BuildRegistry(cfg.Roots)
	for _, err := range storageErrs {
		log.Printf("Warning: storage backend init error: %v", err)
	}
	deps.Storage = storageReg
```

- [ ] **Step 3: Rebuild registry on config change**

In `configHandler()` (main.go line ~2040, after `currentConfig = newCfg`), add:

```go
			// Rebuild storage backends from new config
			newReg, regErrs := storage.BuildRegistry(newCfg.Roots)
			for _, err := range regErrs {
				log.Printf("Warning: storage backend init error: %v", err)
			}
			deps.Storage.Replace(newReg.AllBackends())
```

Also add an `AllBackends()` method to Registry in `media-server/storage/registry.go`:

```go
// AllBackends returns the underlying backend slice (used for Replace).
func (r *Registry) AllBackends() []Backend {
	r.mu.RLock()
	defer r.mu.RUnlock()
	cp := make([]Backend, len(r.backends))
	copy(cp, r.backends)
	return cp
}
```

- [ ] **Step 4: Build and verify**

Run: `cd media-server && go build ./...`
Expected: clean build

- [ ] **Step 5: Commit**

```bash
git add media-server/main.go media-server/main_linux.go media-server/main_darwin.go media-server/storage/registry.go
git commit -m "feat: wire storage registry into server startup and config reload"
```

---

### Task 7: Update fsbrowser Handlers to Use Registry

**Files:**
- Modify: `media-server/fsbrowser.go`

- [ ] **Step 1: Update fsListHandler to use registry**

Replace the body of `fsListHandler` to use the storage registry. The handler receives `deps *Dependencies` which now has `deps.Storage`:

```go
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
			entries := make([]fsEntry, len(allRoots))
			for i, r := range allRoots {
				entries[i] = fsEntry{
					Name:  r.Name,
					Path:  r.Path,
					IsDir: true,
				}
			}
			if entries == nil {
				entries = []fsEntry{}
			}
			// Return empty roots array — roots are now in entries
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

		var entries []fsEntry
		for _, e := range storageEntries {
			entries = append(entries, fsEntry{
				Name:    e.Name,
				Path:    e.Path,
				IsDir:   e.IsDir,
				MtimeMs: e.MtimeMs,
			})
		}
		if entries == nil {
			entries = []fsEntry{}
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
			// For local paths, parent is filepath.Dir
			// For s3:// paths, parent is trimming the last path segment
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
```

Add helper function:

```go
// computeParent returns the parent of a path, handling both local and s3:// paths.
func computeParent(p string) string {
	if strings.HasPrefix(p, "s3://") {
		// s3://bucket/prefix/sub/ → s3://bucket/prefix/
		trimmed := strings.TrimSuffix(p, "/")
		idx := strings.LastIndex(trimmed, "/")
		if idx <= len("s3://") {
			return p // already at bucket root
		}
		return trimmed[:idx+1]
	}
	return filepath.Dir(filepath.Clean(p))
}
```

- [ ] **Step 2: Update fsScanHandler to use registry**

Replace the scanning logic in `fsScanHandler`:

```go
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

		// If the path points to a file, scan its parent directory instead
		// and remember the selected file so we can set the cursor to it.
		scanPath := req.Path
		selectedFile := ""

		backend := deps.Storage.BackendFor(req.Path)
		if backend == nil {
			httpError(w, "path is not within any configured storage root", http.StatusForbidden)
			return
		}

		// Check if path is a file (for local backend)
		if info, err := os.Stat(req.Path); err == nil && !info.IsDir() {
			selectedFile = req.Path
			scanPath = filepath.Dir(req.Path)
		}

		storageFiles, err := backend.Scan(r.Context(), scanPath, req.Recursive)
		if err != nil {
			httpError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		files := make([]fsScanFile, len(storageFiles))
		for i, f := range storageFiles {
			files[i] = fsScanFile{Path: f.Path, MtimeMs: f.MtimeMs}
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
```

- [ ] **Step 3: Remove old validatePathWithinRoots and getRootPaths**

These are no longer needed — the registry handles validation. Remove the `validatePathWithinRoots` function, `getRootPaths` variable, and `getFilesystemRoots` function from `fsbrowser.go`. If they are used in tests, update the tests too.

- [ ] **Step 4: Build and verify**

Run: `cd media-server && go build ./...`
Expected: clean build

- [ ] **Step 5: Run all tests**

Run: `cd media-server && go test ./...`
Expected: all PASS (some fsbrowser tests may need updating if they reference removed functions)

- [ ] **Step 6: Commit**

```bash
git add media-server/fsbrowser.go media-server/fsbrowser_test.go
git commit -m "refactor: use storage registry in fsbrowser handlers"
```

---

### Task 8: Update mediaFileHandler for S3 Redirect

**Files:**
- Modify: `media-server/main.go:2076-2177` (mediaFileHandler)
- Modify: `media-server/main_linux.go` (same handler)
- Modify: `media-server/main_darwin.go` (same handler)

- [ ] **Step 1: Add S3 path handling to mediaFileHandler**

Insert after the remote URL proxy check (after line 2120 in main.go) and before the local file handling, add S3 detection:

```go
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
```

Also remove the absolute path check for S3 paths — update the validation block:

```go
		// If local path, enforce absolute path to avoid traversal via relative inputs
		if !strings.HasPrefix(filePath, "http://") && !strings.HasPrefix(filePath, "https://") && !strings.HasPrefix(filePath, "s3://") {
			if !filepath.IsAbs(filePath) {
				http.Error(w, "Path must be absolute", http.StatusBadRequest)
				return
			}
		}
```

- [ ] **Step 2: Apply same changes to main_linux.go and main_darwin.go**

The `mediaFileHandler` is duplicated across all three platform files. Apply the same two changes (S3 redirect block and updated path validation) to `main_linux.go` and `main_darwin.go`.

- [ ] **Step 3: Build and verify**

Run: `cd media-server && go build ./...`
Expected: clean build

- [ ] **Step 4: Commit**

```bash
git add media-server/main.go media-server/main_linux.go media-server/main_darwin.go
git commit -m "feat: redirect S3 paths to presigned URLs in media file handler"
```

---

### Task 9: Update Thumbnail Generation for S3

**Files:**
- Modify: `media-server/thumbnail.go`
- Modify: `media-server/loki_api.go:361-420` (lokiMediaPreviewHandler)

- [ ] **Step 1: Add S3 thumbnail path computation**

In `media-server/thumbnail.go`, add a function that computes thumbnail path for S3 sources. Add after `getThumbnailPath` (line 116):

```go
// getS3ThumbnailPath computes the thumbnail S3 path for an S3 media file.
// Uses the backend's thumbnail prefix with the same hash scheme as local.
func getS3ThumbnailPath(mediaPath string, backend *storage.S3Backend, cache string, timeStamp int) string {
	hashInput := mediaPath
	if timeStamp > 0 {
		hashInput += fmt.Sprintf("%d", timeStamp)
	}
	fileName := createHash(hashInput)
	if getFileType(mediaPath) == "video" {
		fileName += ".mp4"
	} else {
		fileName += ".png"
	}
	return backend.ThumbnailPath(fileName)
}
```

Add import for `"github.com/stevecastle/shrike/storage"`.

- [ ] **Step 2: Add S3 thumbnail generation function**

Add after `generateThumbnailThrottled`:

```go
// generateS3ThumbnailThrottled generates a thumbnail for an S3-hosted media file.
// Downloads source to temp, runs ffmpeg, uploads result back to S3.
func generateS3ThumbnailThrottled(ctx context.Context, mediaPath string, backend *storage.S3Backend, cache string, timeStamp int) (string, error) {
	key := thumbKey(mediaPath, cache, timeStamp)

	inflightMu.Lock()
	if ch, ok := inflight[key]; ok {
		inflightMu.Unlock()
		<-ch
		thumbPath := getS3ThumbnailPath(mediaPath, backend, cache, timeStamp)
		exists, _ := backend.Exists(ctx, thumbPath)
		if exists {
			return thumbPath, nil
		}
		return "", fmt.Errorf("in-flight S3 thumbnail generation failed for %s", mediaPath)
	}
	done := make(chan struct{})
	inflight[key] = done
	inflightMu.Unlock()

	defer func() {
		inflightMu.Lock()
		delete(inflight, key)
		close(done)
		inflightMu.Unlock()
	}()

	thumbSem <- struct{}{}
	defer func() { <-thumbSem }()

	// Download source to temp file
	reader, err := backend.Download(ctx, mediaPath)
	if err != nil {
		return "", fmt.Errorf("failed to download %s: %w", mediaPath, err)
	}
	defer reader.Close()

	ext := filepath.Ext(mediaPath)
	tmpSource, err := os.CreateTemp("", "loki-thumb-src-*"+ext)
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmpSource.Name())
	if _, err := io.Copy(tmpSource, reader); err != nil {
		tmpSource.Close()
		return "", fmt.Errorf("failed to write temp source: %w", err)
	}
	tmpSource.Close()

	// Generate thumbnail to temp output
	tmpOutput, err := os.CreateTemp("", "loki-thumb-out-*")
	if err != nil {
		return "", fmt.Errorf("failed to create temp output: %w", err)
	}
	tmpOutputPath := tmpOutput.Name()
	tmpOutput.Close()
	defer os.Remove(tmpOutputPath)

	// Use the local generateThumbnail with temp paths
	// We need to call ffmpeg directly with the temp source
	ffmpegPath := depspkg.GetFFmpegPath()
	if ffmpegPath == "" {
		return "", fmt.Errorf("ffmpeg not found")
	}

	fileType := getFileType(mediaPath)
	switch fileType {
	case "image":
		if err := generateImageThumbnail(ffmpegPath, tmpSource.Name(), tmpOutputPath, cache); err != nil {
			return "", err
		}
	case "video":
		if err := generateVideoThumbnail(ffmpegPath, tmpSource.Name(), tmpOutputPath, timeStamp); err != nil {
			return "", err
		}
	default:
		return "", fmt.Errorf("unsupported file type: %s", ext)
	}

	// Upload result to S3
	thumbPath := getS3ThumbnailPath(mediaPath, backend, cache, timeStamp)
	outputFile, err := os.Open(tmpOutputPath)
	if err != nil {
		return "", fmt.Errorf("failed to open generated thumbnail: %w", err)
	}
	defer outputFile.Close()

	contentType := "image/png"
	if fileType == "video" {
		contentType = "video/mp4"
	}
	if err := backend.Upload(ctx, thumbPath, outputFile, contentType); err != nil {
		return "", fmt.Errorf("failed to upload thumbnail: %w", err)
	}

	return thumbPath, nil
}
```

Add imports: `"io"`, `"context"`.

- [ ] **Step 3: Update lokiMediaPreviewHandler for S3 paths**

In `media-server/loki_api.go`, update `lokiMediaPreviewHandler` to handle S3 paths. Replace the thumbnail existence check and generation block (lines 390-418):

```go
		if thumbPath.Valid && thumbPath.String != "" {
			// Check if the thumbnail file actually exists
			if strings.HasPrefix(thumbPath.String, "s3://") {
				backend := deps.Storage.BackendFor(thumbPath.String)
				if backend != nil {
					exists, _ := backend.Exists(r.Context(), thumbPath.String)
					if exists {
						writeJSON(w, thumbPath.String)
						return
					}
				}
			} else if _, err := os.Stat(thumbPath.String); err == nil {
				writeJSON(w, thumbPath.String)
				return
			}
		}

		// Generate thumbnail
		if strings.HasPrefix(req.Path, "s3://") {
			backend := deps.Storage.BackendFor(req.Path)
			if backend == nil {
				writeJSON(w, nil)
				return
			}
			s3b, ok := backend.(*storage.S3Backend)
			if !ok {
				writeJSON(w, nil)
				return
			}
			generated, err := generateS3ThumbnailThrottled(r.Context(), req.Path, s3b, cache, req.TimeStamp)
			if err != nil {
				log.Printf("S3 thumbnail generation failed for %s: %v", req.Path, err)
				writeJSON(w, nil)
				return
			}
			deps.DB.Exec(
				fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
				generated, req.Path,
			)
			log.Printf("Generated S3 thumbnail for %s → %s", req.Path, generated)
			writeJSON(w, generated)
			return
		}

		// Local thumbnail generation (existing code)
		dbPath := currentConfig.DBPath
		if dbPath == "" {
			writeJSON(w, nil)
			return
		}
		basePath := filepath.Dir(dbPath)
		generated, err := generateThumbnailThrottled(req.Path, basePath, cache, req.TimeStamp)
		if err != nil {
			log.Printf("Thumbnail generation failed for %s: %v", req.Path, err)
			writeJSON(w, nil)
			return
		}
		deps.DB.Exec(
			fmt.Sprintf("UPDATE media SET %s = ? WHERE path = ?", cache),
			generated, req.Path,
		)
		log.Printf("Generated thumbnail for %s → %s", req.Path, generated)
		writeJSON(w, generated)
```

Add import for `"github.com/stevecastle/shrike/storage"`.

- [ ] **Step 4: Build and verify**

Run: `cd media-server && go build ./...`
Expected: clean build

- [ ] **Step 5: Commit**

```bash
git add media-server/thumbnail.go media-server/loki_api.go
git commit -m "feat: add S3 thumbnail generation with download/upload cycle"
```

---

### Task 10: Update Config API for StorageRoot

**Files:**
- Modify: `media-server/main.go:1930-1946` (updateConfigRequest)
- Modify: `media-server/main.go:2013-2014` (config POST handler)
- Modify: `media-server/main_linux.go` (same)
- Modify: `media-server/main_darwin.go` (same)

- [ ] **Step 1: Update updateConfigRequest**

Replace `RootPaths []string` with `Roots` in the `updateConfigRequest` struct:

```go
type updateConfigRequest struct {
	DBPath                 string                 `json:"dbPath"`
	DownloadPath           string                 `json:"downloadPath"`
	OllamaBaseURL          string                 `json:"ollamaBaseUrl"`
	OllamaModel            string                 `json:"ollamaModel"`
	DescribePrompt         string                 `json:"describePrompt"`
	AutotagPrompt          string                 `json:"autotagPrompt"`
	OnnxModelPath          string                 `json:"onnxModelPath"`
	OnnxLabelsPath         string                 `json:"onnxLabelsPath"`
	OnnxConfigPath         string                 `json:"onnxConfigPath"`
	OnnxORTSharedLibPath   string                 `json:"onnxOrtSharedLibPath"`
	OnnxGeneralThreshold   float64                `json:"onnxGeneralThreshold"`
	OnnxCharacterThreshold float64                `json:"onnxCharacterThreshold"`
	FasterWhisperPath      string                 `json:"fasterWhisperPath"`
	DiscordToken           string                 `json:"discordToken"`
	Roots                  []appconfig.StorageRoot `json:"roots"`
}
```

- [ ] **Step 2: Update config POST handler**

Replace the `RootPaths` assignment (line ~2013-2014) with:

```go
			if req.Roots != nil {
				newCfg.Roots = req.Roots
			}
```

- [ ] **Step 3: Apply same changes to main_linux.go and main_darwin.go**

- [ ] **Step 4: Build and verify**

Run: `cd media-server && go build ./...`
Expected: clean build

- [ ] **Step 5: Commit**

```bash
git add media-server/main.go media-server/main_linux.go media-server/main_darwin.go
git commit -m "feat: update config API to accept StorageRoot array"
```

---

### Task 11: Frontend — Update File Browser for S3 Roots

**Files:**
- Modify: `src/renderer/components/controls/file-browser-modal.tsx`

- [ ] **Step 1: Update FsEntry type to include type indicator**

Update the type at line 5:

```typescript
type FsEntry = {
  name: string;
  path: string;
  isDir: boolean;
  mtimeMs: number;
  type?: 'local' | 's3';
};
```

- [ ] **Step 2: Show storage type indicator on root entries**

In the entry rendering (line ~153-163), add a type badge when showing root-level entries:

```typescript
            entries.map((entry) => (
              <div
                key={entry.path}
                className={`file-browser-entry${!entry.isDir && selectedFile === entry.path ? ' selected' : ''}`}
                onClick={() => handleEntryClick(entry)}
                onDoubleClick={() => handleEntryDoubleClick(entry)}
              >
                <span className="icon">{entry.isDir ? '\uD83D\uDCC1' : '\uD83D\uDCC4'}</span>
                <span className="name">{entry.name}</span>
                {entry.type === 's3' && <span className="badge-s3">S3</span>}
              </div>
            ))
```

- [ ] **Step 3: Add CSS for S3 badge**

In `src/renderer/components/controls/file-browser-modal.css`, add:

```css
.badge-s3 {
  display: inline-block;
  margin-left: 8px;
  padding: 1px 6px;
  font-size: 10px;
  font-weight: 600;
  color: #ff9900;
  border: 1px solid #ff9900;
  border-radius: 3px;
  text-transform: uppercase;
}
```

- [ ] **Step 4: Verify build**

Run: `cd /c/Users/steph/dev/loki && npm run build:renderer`
Expected: clean build

- [ ] **Step 5: Commit**

```bash
git add src/renderer/components/controls/file-browser-modal.tsx src/renderer/components/controls/file-browser-modal.css
git commit -m "feat: show S3 badge on storage roots in file browser"
```

---

### Task 12: Final Integration Test and Cleanup

**Files:**
- Modify: `media-server/fsbrowser_test.go` (update for new API)

- [ ] **Step 1: Update fsbrowser tests**

Update any tests in `fsbrowser_test.go` that directly call `validatePathWithinRoots` or `getRootPaths` to work with the registry-based approach instead. Tests that create a `Dependencies` struct need to include a `Storage` field with a `storage.Registry`.

- [ ] **Step 2: Run full test suite**

Run: `cd media-server && go test ./...`
Expected: all PASS

- [ ] **Step 3: Run frontend build**

Run: `cd /c/Users/steph/dev/loki && npm run build:web`
Expected: clean build

- [ ] **Step 4: Run full build**

Run: `cd /c/Users/steph/dev/loki && npm run build:server`
Expected: clean build, server binary produced

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "test: update fsbrowser tests for storage registry"
```
