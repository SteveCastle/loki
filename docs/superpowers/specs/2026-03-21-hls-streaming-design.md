# HLS Streaming Support

## Summary

Add HLS (HTTP Live Streaming) support to Loki so videos can be streamed as adaptive-bitrate HLS rather than served as raw files. This includes a new Go media server task to generate HLS segments via ffmpeg, new HTTP endpoints to serve manifests and segments, and hls.js integration in the React video player. A toggle allows falling back to direct source video playback.

## Server Side

### New Task: `hls`

Registered in `tasks/registry.go` via `RegisterTask("hls", "HLS Transcode", hlsTask)`.

**Inputs:**
- `Input`: file path (single file) or newline-separated paths. Also supports query-based file selection via `extractQueryFromJob` (consistent with existing `ffmpegTask` pattern).
- `Arguments`: optional flags
  - `--preset passthrough` (default) — remux only, no transcode
  - `--preset adaptive` — generate multiple quality renditions
  - `--presets 480p,720p,1080p` — select specific renditions (only with `--preset adaptive`)

**Behavior:**
1. Parse input file path(s) or query
2. Probe source with ffprobe (via `deps.GetFFprobePath()`) to get resolution, codec, bitrate
3. Detect audio-only streams — if no video stream, generate audio-only HLS (skip video presets, use `-c:a aac` only)
4. Compute output directory: `<cache_dir>/hls/<sha256_of_filepath>/`
5. If output already exists and source file hasn't changed (mtime check), skip generation
6. Generate HLS segments per selected preset
7. Write master playlist pointing to all generated renditions
8. Report progress via `PushJobStdout`
9. Respect `j.Ctx` for cancellation — pass context to ffmpeg subprocesses, check `ctx.Done()` between files

**Binary resolution:** Use `deps.GetExec(ctx, "ffmpeg", "ffmpeg", ...)` for ffmpeg and `deps.GetFFprobePath()` for ffprobe, consistent with existing code in `tasks/ffmpeg.go` and `thumbnail.go`.

### Preset Tiers

| Name        | Resolution | Video Bitrate | Audio      |
|-------------|-----------|---------------|------------|
| 480p        | 854x480   | 1 Mbps        | 128k AAC   |
| 720p        | 1280x720  | 3 Mbps        | 192k AAC   |
| 1080p       | 1920x1080 | 8 Mbps        | 256k AAC   |
| passthrough | original  | original      | original   |

Only renditions at or below the source resolution are generated. Passthrough is always included regardless of mode.

### FFmpeg Commands

All commands include `-y` to overwrite without prompting (consistent with existing ffmpeg usage in `thumbnail.go`).

**Passthrough (remux):**
```
ffmpeg -y -i <input> -c copy -f hls \
  -hls_time 6 -hls_segment_type mpegts \
  -hls_playlist_type vod \
  -hls_segment_filename <outdir>/passthrough/segment_%03d.ts \
  <outdir>/passthrough/stream.m3u8
```

Note: Passthrough remux requires source codecs compatible with mpegts (H.264, H.265, AAC, MP3). If the source uses incompatible codecs (e.g., VP9 in WebM), passthrough will fail and the task should fall back to transcoding the passthrough tier using libx264/aac at the source resolution and bitrate.

**Transcoded rendition (e.g., 720p):**
```
ffmpeg -y -i <input> \
  -vf "scale=1280:-2:force_original_aspect_ratio=decrease" \
  -c:v libx264 -b:v 3000k -preset medium \
  -c:a aac -b:a 192k \
  -f hls -hls_time 6 -hls_segment_type mpegts \
  -hls_playlist_type vod \
  -hls_segment_filename <outdir>/720p/segment_%03d.ts \
  <outdir>/720p/stream.m3u8
```

The scale filter uses `-2` for height to maintain aspect ratio and ensure even dimensions (required by libx264), consistent with the `force_original_aspect_ratio=decrease` pattern used in `thumbnail.go`.

### Output Structure

```
<cache_dir>/hls/<sha256_of_filepath>/
  master.m3u8
  .meta                  # JSON: source mtime, generation timestamp, presets
  passthrough/
    stream.m3u8
    segment_000.ts
    segment_001.ts
    ...
  720p/                  # only if adaptive
    stream.m3u8
    segment_000.ts
    ...
  480p/                  # only if adaptive and source >= 480p
    stream.m3u8
    ...
```

### URL Scheme and Playlist Resolution

HLS content is served under a path-based URL scheme so that relative URLs in playlists resolve correctly for hls.js:

```
/media/hls/<hash>/master.m3u8
/media/hls/<hash>/passthrough/stream.m3u8
/media/hls/<hash>/passthrough/segment_000.ts
/media/hls/<hash>/720p/stream.m3u8
/media/hls/<hash>/720p/segment_000.ts
```

The master playlist uses relative paths (e.g., `720p/stream.m3u8`) which resolve naturally against the master playlist URL. Sub-playlists use relative segment filenames (e.g., `segment_000.ts`).

### New HTTP Endpoints

**`GET /media/hls?path=<encoded_path>`**
- Computes the hash for the path and redirects to `/media/hls/<hash>/master.m3u8`
- If HLS cache doesn't exist for this path, generates passthrough on-the-fly (blocks until ready)
- Uses inflight deduplication (similar to `thumbnail.go` pattern) to prevent duplicate ffmpeg processes for concurrent requests to the same file
- Content-Type: `application/vnd.apple.mpegurl`
- CORS headers for web mode

**`GET /media/hls/<hash>/<path...>`**
- Serves files from the HLS cache directory using `http.ServeFile` (which handles Range requests automatically)
- Path parameter is validated: only serves `.m3u8` and `.ts` files, rejects path traversal attempts (no `..` components, filename must match `^(master|stream)\.m3u8$` or `^segment_\d+\.ts$`)
- Content-Type: `video/MP2T` for .ts, `application/vnd.apple.mpegurl` for .m3u8

Both endpoints are registered in all platform main files (`main.go`, `main_linux.go`, `main_darwin.go`) alongside existing `/media/file`.

### Cache Management

**Cache directory:** Reuses the existing cache directory pattern (same parent as thumbnail cache). The `<sha256_of_filepath>` hashing ensures unique, collision-free directories per source file.

**Cache invalidation:** Compare source file mtime against a stored `.meta` JSON file in the HLS output directory. If source is newer, regenerate.

**Cache cleanup endpoint:** `DELETE /media/hls?path=<encoded_path>` removes the HLS cache for a specific file. `DELETE /media/hls` with no path clears the entire HLS cache. This gives the user manual control over disk usage.

**Cache size:** HLS segments can be large (passthrough is roughly equal to source file size). No automatic eviction is implemented in v1 — the user manages cache via the cleanup endpoint. The `.meta` file records generation timestamp for future LRU eviction if needed.

## Client Side

### New Dependency: hls.js

Added to root `package.json` (`C:\Users\steph\dev\loki\package.json`). hls.js is the standard HLS playback library for browsers. It handles:
- Manifest parsing and segment loading
- Adaptive bitrate switching
- Full seek support in VOD streams
- Buffer management

### Changes to video.tsx

**New prop:** `useHLS?: boolean` — when true, use hls.js to play from HLS manifest instead of direct source.

**Logic:**
1. If `useHLS` is true and `Hls.isSupported()`:
   - Create `Hls` instance, load manifest from `hlsUrl(path)`, attach to `<video>` element
   - Clean up Hls instance on unmount or path change
2. If `useHLS` is true but `Hls.isSupported()` is false (Safari):
   - Set `src` to the HLS URL directly since Safari supports HLS natively
3. If `useHLS` is false:
   - Use current behavior: `src={mediaUrl(path)}` for direct playback

**Error fallback:** On hls.js fatal error, destroy the Hls instance and fall back to direct source playback via `mediaUrl(path)`. This ensures playback continues even if HLS generation failed or the format is unsupported.

**Seeking:** Works automatically with VOD playlists. The `#EXT-X-PLAYLIST-TYPE:VOD` directive tells hls.js that all segments are available from the start, enabling unrestricted seeking. The existing `currentTime` manipulation in video.tsx continues to work — hls.js seeks to the correct segment transparently.

**Existing functionality preserved:** Volume control, looping, play/pause, timeupdate events, error fallback to Image — all remain unchanged. hls.js just replaces how media data reaches the `<video>` element.

### New Platform Function

In `platform.ts`, following the existing `mediaUrl` pattern which branches on `isElectron`:

```typescript
// Web mode (inside webPlatform):
export const hlsUrl = (path: string): string =>
  `/media/hls?path=${encodeURIComponent(path)}`;

// Electron mode (inside electronPlatform):
export const hlsUrl = (path: string): string =>
  `http://localhost:${serverPort}/media/hls?path=${encodeURIComponent(path)}`;
```

### HLS Toggle

Added to the existing settings system:

- `src/settings.ts`: Add `useHLS: boolean` to the `Settings` type and `'useHLS'` to the `SettingKey` union
- Default value: `false` (direct playback by default, user opts into HLS)
- The video player reads this setting and passes it as the `useHLS` prop

This toggle is useful for:
- Local playback where direct file access is faster
- Debugging
- Formats that play fine natively without transcoding

## Seeking Support

Full seek is guaranteed by:
1. **VOD playlist type** — `#EXT-X-PLAYLIST-TYPE:VOD` signals all segments exist
2. **Segment index** — each segment's offset and duration are in the playlist
3. **hls.js seek handling** — on seek, hls.js calculates the target segment, fetches it, and resumes playback from the correct position within the segment

## Error Handling

- If HLS generation fails (corrupt file, unsupported codec), the task reports the error via stdout and the endpoint returns a 500
- Passthrough remux failure (incompatible codecs) triggers automatic fallback to transcode
- The client falls back to direct source playback on HLS error (same pattern as the existing Image fallback on video error)
- If ffmpeg is not available, task logs error and fails gracefully
- Audio-only files are handled by skipping video presets

## Files Modified

### Go (media-server/)
- `tasks/hls.go` — new file, HLS generation task
- `tasks/registry.go` — register `hls` task
- `main.go` — add `/media/hls` endpoint and handler (Windows)
- `main_linux.go` — add `/media/hls` endpoint and handler (Linux)
- `main_darwin.go` — add `/media/hls` endpoint and handler (macOS)

### TypeScript (src/renderer/)
- `components/media-viewers/video.tsx` — hls.js integration, `useHLS` prop
- `platform.ts` — add `hlsUrl` function (both Electron and web mode)
- `src/settings.ts` — add `useHLS` setting to `Settings` type, `SettingKey` union, and defaults

### Config
- `package.json` (root) — add `hls.js` dependency
