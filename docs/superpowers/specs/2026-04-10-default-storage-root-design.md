# Default Storage Root — Design Spec

**Date:** 2026-04-10
**Status:** Approved

## Problem

The `DownloadPath` config field is a local filesystem path. When the server runs in Docker or the cloud with S3 storage, uploads and downloads have nowhere to go — there's no local media directory. The storage root system already supports S3 via the `Backend` interface (including `Upload`), but new media always targets `DownloadPath`.

## Design

Designate one storage root as the **default root** — the destination for all new media entering the system (uploads, downloads, ingests). This unifies the download path concept with the existing roots system.

### Config: StorageRoot.Default field

Add `Default bool` to `StorageRoot`:

```json
{
  "roots": [
    {"type": "local", "path": "/mnt/photos", "label": "Photos"},
    {"type": "s3", "label": "Cloud Media", "bucket": "media", "default": true, ...}
  ]
}
```

**Resolution rules:**
1. If exactly one root has `"default": true`, it is the default root.
2. If no root is marked default, the first root in the array is the default.
3. If multiple roots are marked default, the first one marked wins.
4. If there are no roots at all but `DownloadPath` is set, auto-create a local root from `DownloadPath` marked as default. This preserves backwards compatibility for existing configs.

### Config: DownloadPath deprecation

`DownloadPath` remains in the `Config` struct and JSON for backwards compatibility. On load:
- If roots exist and one is default, `DownloadPath` is ignored for upload/download routing.
- If no roots exist and `DownloadPath` is non-empty, it is migrated into a local root marked default (same pattern as the existing `RootPaths` migration).

### Environment variables

**LOWKEY_ROOTS JSON:** Set `"default": true` on one entry:
```
LOWKEY_ROOTS='[{"type":"s3","label":"Media","bucket":"media","default":true,...}]'
```

**LOWKEY_ROOT_N shorthand:** Add `LOWKEY_DEFAULT_ROOT` — the label or 1-based index of the root to mark default:
```
LOWKEY_ROOT_1=/mnt/photos:Photos
LOWKEY_ROOT_2=/mnt/videos:Videos
LOWKEY_DEFAULT_ROOT=2
```

If `LOWKEY_DEFAULT_ROOT` is not set when using numbered roots, root 1 is the default.

### Registry: DefaultBackend method

Add to `Registry`:

```go
func (r *Registry) DefaultBackend() Backend
```

Returns the backend for the root marked default. Falls back to the first backend. Returns nil if the registry is empty.

### Upload handler

Currently in `main.go` (Windows only):
```go
uploadDir := filepath.Join(cfg.DownloadPath, "uploads")
dst, err := os.Create(destPath)
io.Copy(dst, file)
```

Changes to:
```go
backend := deps.Storage.DefaultBackend()
destPath := "uploads/" + filename
backend.Upload(ctx, destPath, file, contentType)
```

This works for both local and S3 backends — no filesystem assumptions. The uploaded path is relative to the root, not absolute.

The ingest job created after upload uses the path within the default backend, which `BackendFor` will resolve correctly.

### CLI-based ingest tasks (yt-dlp, gallery-dl, discord)

These tools shell out to CLI programs that must write to local disk. They cannot stream to S3 directly.

**New flow:**
1. Create a staging directory: `platform.GetTempDir()/staging/<job-id>/`
2. Run the CLI tool with output directed to the staging directory (same as today, but using a temp dir instead of `DownloadPath`)
3. After the tool completes, iterate over output files:
   - Call `DefaultBackend().Upload(ctx, path, reader, contentType)` for each file
   - The destination path within the root preserves the relative structure from staging
4. Clean up the staging directory
5. The ingest step uses the final paths in the default backend for database insertion

When the default backend is local, step 3 is effectively a move/copy within the filesystem. When it's S3, files get uploaded then the local staging dir is cleaned.

**Staging directory location:**
- Docker: `/data/staging/<job-id>/` (inside the data volume, survives restarts)
- Native Linux: `platform.GetTempDir()/staging/<job-id>/`
- Native Windows: `platform.GetTempDir()/staging/<job-id>/`

### Files to modify

| File | Change |
|------|--------|
| `appconfig/config.go` | Add `Default bool` to `StorageRoot`, add `LOWKEY_DEFAULT_ROOT` env support, add `DownloadPath` migration logic |
| `appconfig/config_test.go` | Tests for default root resolution, migration, env vars |
| `storage/registry.go` | Add `DefaultBackend()` method |
| `storage/registry_test.go` | Test `DefaultBackend` |
| `storage/build.go` | Pass `Default` field through when building backends |
| `main.go` | Update upload handler to use `DefaultBackend().Upload` |
| `main_linux.go` | Add upload handler (currently missing on Linux), using `DefaultBackend().Upload` |
| `main_darwin.go` | Add upload handler (currently missing on macOS), using `DefaultBackend().Upload` |
| `tasks/ingest_youtube.go` | Use staging dir + upload to default backend |
| `tasks/ingest_gallery.go` | Use staging dir + upload to default backend |
| `tasks/ingest_discord.go` | Use staging dir + upload to default backend |
| `renderer/templates/config.go.html` | Add default root toggle to UI |
| `docker-compose.yml` | Update examples |
| `docs/server/index.html` | Update Docker docs |

### What doesn't change

- `Backend` interface — already has `Upload`, `Download`, `Contains`, etc.
- `Registry.BackendFor` — still routes reads by path matching
- Existing local-only setups — `DownloadPath` auto-migrates to a default local root
- Read paths for media serving — unchanged, still resolved via `BackendFor`
