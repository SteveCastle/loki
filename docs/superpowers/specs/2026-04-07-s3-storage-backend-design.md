# S3-Compatible Object Storage Backend

## Summary

Add S3-compatible object storage as a storage layer alongside the existing local filesystem. Each configured root can independently be local or S3. Media files and thumbnails on S3 are served via presigned URLs (no proxy). SQLite database remains on local disk.

## Configuration

### StorageRoot Schema

The `RootPaths []string` field in `appconfig.Config` is replaced by `Roots []StorageRoot`:

```go
type StorageRoot struct {
    Type     string `json:"type"`     // "local" or "s3"
    Path     string `json:"path"`     // local: filesystem path; s3: unused (bucket+prefix define location)
    Label    string `json:"label"`    // display name in UI

    // S3-only fields
    Endpoint        string `json:"endpoint,omitempty"`        // e.g. "https://s3.amazonaws.com", MinIO URL
    Region          string `json:"region,omitempty"`
    Bucket          string `json:"bucket,omitempty"`
    Prefix          string `json:"prefix,omitempty"`          // key prefix within bucket
    AccessKey       string `json:"accessKey,omitempty"`
    SecretKey       string `json:"secretKey,omitempty"`
    ThumbnailPrefix string `json:"thumbnailPrefix,omitempty"` // default: "_thumbnails"
}
```

### Backward Compatibility

On config load, if `rootPaths` is present but `roots` is absent, each string is migrated to `StorageRoot{Type: "local", Path: p, Label: p}`. The old `rootPaths` field is removed on next save.

### Example Config

```json
{
  "dbPath": "/data/media.db",
  "roots": [
    {
      "type": "local",
      "path": "/mnt/photos",
      "label": "Local Photos"
    },
    {
      "type": "s3",
      "label": "Cloud Archive",
      "endpoint": "https://s3.us-east-1.amazonaws.com",
      "region": "us-east-1",
      "bucket": "my-media-bucket",
      "prefix": "archive/",
      "accessKey": "AKIA...",
      "secretKey": "...",
      "thumbnailPrefix": "_thumbnails"
    }
  ]
}
```

## Storage Backend Interface

New package: `media-server/storage/`

### Core Interface

```go
type Entry struct {
    Name    string
    Path    string  // local absolute path or "s3://{bucket}/{key}"
    IsDir   bool
    MtimeMs float64
}

type FileInfo struct {
    Path    string
    MtimeMs float64
}

type Backend interface {
    // List returns entries in a directory (one level).
    List(ctx context.Context, path string) ([]Entry, error)

    // Scan returns all media files under path, optionally recursive.
    Scan(ctx context.Context, path string, recursive bool) ([]FileInfo, error)

    // Download streams a file for local processing (e.g., ffmpeg).
    Download(ctx context.Context, path string) (io.ReadCloser, error)

    // Upload writes content to the backend (thumbnails). No-op for local.
    Upload(ctx context.Context, path string, r io.Reader, contentType string) error

    // MediaURL returns a URL the client can use to fetch the file.
    // Local: "/media/file?path=..." server-relative URL.
    // S3: presigned GetObject URL (1-hour expiry).
    MediaURL(path string) (string, error)

    // Exists checks whether a file exists at the given path.
    // Local: os.Stat. S3: HeadObject.
    Exists(ctx context.Context, path string) (bool, error)

    // Contains reports whether this backend owns the given path.
    Contains(path string) bool

    // Root returns the root Entry for this backend (for root listing).
    Root() Entry
}
```

### Registry

```go
type Registry struct {
    backends []Backend
}

func (r *Registry) BackendFor(path string) Backend
func (r *Registry) AllRoots() []Entry
```

Built at startup from `config.Roots`. Rebuilt when config changes via `POST /config`.

### LocalBackend

Wraps existing `os.ReadDir`, `filepath.WalkDir`, `os.Stat` logic. `MediaURL` returns `/media/file?path={encoded}`. `Upload`/`Download` are direct file I/O. `Contains` checks path is within the root's `Path` (same logic as current `validatePathWithinRoots`).

### S3Backend

Uses AWS SDK v2 (`github.com/aws/aws-sdk-go-v2`). Initialized with static credentials from the `StorageRoot` config.

- **List**: `ListObjectsV2` with delimiter `/` to simulate directory listing. Common prefixes become directory entries.
- **Scan**: `ListObjectsV2` paginator, no delimiter for recursive. With delimiter for non-recursive. Filters by media extension regex.
- **Download**: `GetObject`, returns the response body.
- **Upload**: `PutObject` with content type.
- **MediaURL**: `s3.NewPresignClient` → `PresignGetObject` with 1-hour expiry.
- **Contains**: Path starts with `s3://{bucket}/{prefix}`.

S3 paths are stored in the database as `s3://{bucket}/{key}` to distinguish them from local paths.

## File Browsing

### /api/fs/list

- Empty path: Returns `registry.AllRoots()` — each root with its label and type.
- Non-empty path: `registry.BackendFor(path).List(ctx, path)`.
- Path validation: `BackendFor(path) != nil` replaces `validatePathWithinRoots`.
- Response adds a `type` field to entries so the frontend can show an indicator.

### /api/fs/scan

- `registry.BackendFor(path).Scan(ctx, path, recursive)`.
- S3 paths stored in DB as `s3://bucket/key`.
- `insertBulkMediaPaths` unchanged — paths are just strings regardless of backend.

## Media Serving

### /media/file Handler

```
GET /media/file?path=...
```

- Resolve backend: `registry.BackendFor(path)`.
- **Local**: Existing `http.ServeFile` behavior, unchanged.
- **S3**: Call `backend.MediaURL(path)` → presigned URL. Return **HTTP 302 redirect**. Browser fetches directly from S3.
- **No backend found**: 404.

The React `mediaUrl` function in `platform.ts` is unchanged — it still builds `/media/file?path=...`. The 302 redirect is transparent to `<img>` and `<video>` tags.

## Thumbnail Generation

### Flow for S3 Media

1. `/api/media/preview` handler receives request for S3 path.
2. Check DB for existing thumbnail path. If exists and backend confirms it exists (`Download` succeeds or `HEAD` check), return it.
3. If missing, generate:
   a. `backend.Download(ctx, sourcePath)` → temp file on local disk.
   b. Run ffmpeg against temp file (existing `generateThumbnailThrottled` logic).
   c. Compute thumbnail S3 path: `s3://{bucket}/{thumbnailPrefix}/{hash}.{ext}` where hash is SHA256 of `sourcePath + timestamp`.
   d. `backend.Upload(ctx, thumbnailPath, file, contentType)`.
   e. Clean up temp files.
   f. Store `thumbnailPath` in DB.
   g. Return `thumbnailPath` to client.
4. Client loads thumbnail via `/media/file?path=s3://...` → presigned redirect.

### Existing Local Behavior

Unchanged. Local backend `Upload` writes to disk as before. `MediaURL` returns the local server URL.

### Thumbnail Existence Check

For S3 thumbnails, checking file existence via `os.Stat` won't work. The thumbnail handler needs to go through the backend:
- Local: `os.Stat` (existing).
- S3: `HeadObject` call (add a `Stat(ctx, path) (bool, error)` method to the interface, or catch errors on `Download`).

Add `Exists(ctx context.Context, path string) (bool, error)` to the Backend interface for this purpose.

## Frontend Changes

### File Browser Modal

- Root listing now includes `type` and `label` from the API response.
- Display a small badge or icon indicating "local" vs "S3" next to each root.
- Browsing behavior is identical — click to navigate, select file/directory.

### Config UI

The config page (Go template at `/config`) needs updated form fields:
- Roots displayed as a list of cards, each with type selector (local/S3).
- Local root: just a path field.
- S3 root: endpoint, region, bucket, prefix, access key, secret key, thumbnail prefix fields.
- Add/remove root buttons.

### No Other Frontend Changes

`mediaUrl`, `fetchMediaPreview`, thumbnail loading, tag loading — all unchanged. The `/media/file` redirect and DB path storage handle everything transparently.

## Error Handling

- S3 connection failures: Return HTTP 502 with message indicating which root is unreachable.
- Invalid credentials: Surface on config save with a test connection (optional `POST /api/storage/test`).
- Presigned URL expiry: 1-hour default. If a page is open longer, stale URLs return 403 from S3 — the browser will re-request from the server, getting a fresh presigned URL.
- Download failures during thumbnail gen: Log error, return null to client (same as current ffmpeg failure handling).

## Testing

- Unit tests for `S3Backend` using a mock S3 client interface.
- Unit tests for `Registry.BackendFor` path routing.
- Integration test for `LocalBackend` (wraps existing fsbrowser tests).
- Config migration test: `rootPaths` → `roots` conversion.

## Dependencies

- `github.com/aws/aws-sdk-go-v2` and sub-packages (`config`, `credentials`, `service/s3`, `feature/s3/manager`).
- No new frontend dependencies.

## Out of Scope

- Multi-user bucket isolation (all users share the same S3 roots).
- S3 upload of new media from the UI (browsing/viewing only + thumbnail generation).
- Database on S3 (stays local per design decision).
- Server-side caching/CDN layer for presigned URLs.
