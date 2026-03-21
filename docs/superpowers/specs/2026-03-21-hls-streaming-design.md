# HLS Streaming Support

## Summary

Add HLS (HTTP Live Streaming) support to Loki so videos can be streamed as adaptive-bitrate HLS rather than served as raw files. This includes a new Go media server task to generate HLS segments via ffmpeg, new HTTP endpoints to serve manifests and segments, and hls.js integration in the React video player. A toggle allows falling back to direct source video playback.

## Server Side

### New Task: `hls`

Registered in `tasks/registry.go` via `RegisterTask("hls", "HLS Transcode", hlsTask)`.

**Inputs:**
- `Input`: file path (single file) or newline-separated paths
- `Arguments`: optional flags
  - `--preset passthrough` (default) — remux only, no transcode
  - `--preset adaptive` — generate multiple quality renditions
  - `--presets 480p,720p,1080p` — select specific renditions (only with `--preset adaptive`)

**Behavior:**
1. Parse input file path(s)
2. Probe source with ffprobe to get resolution, codec, bitrate
3. Compute output directory: `<cache_dir>/hls/<sha256_of_filepath>/`
4. If output already exists and source file hasn't changed (mtime check), skip generation
5. Generate HLS segments per selected preset
6. Write master playlist pointing to all generated renditions
7. Report progress via `PushJobStdout`

### Preset Tiers

| Name        | Resolution | Video Bitrate | Audio      |
|-------------|-----------|---------------|------------|
| 480p        | 854x480   | 1 Mbps        | 128k AAC   |
| 720p        | 1280x720  | 3 Mbps        | 192k AAC   |
| 1080p       | 1920x1080 | 8 Mbps        | 256k AAC   |
| passthrough | original  | original      | original   |

Only renditions at or below the source resolution are generated. Passthrough is always included regardless of mode.

### FFmpeg Commands

**Passthrough (remux):**
```
ffmpeg -i <input> -c copy -f hls \
  -hls_time 6 -hls_segment_type mpegts \
  -hls_playlist_type vod \
  -hls_segment_filename <outdir>/passthrough/segment_%03d.ts \
  <outdir>/passthrough/stream.m3u8
```

**Transcoded rendition (e.g., 720p):**
```
ffmpeg -i <input> \
  -vf scale=1280:720 -c:v libx264 -b:v 3000k -preset medium \
  -c:a aac -b:a 192k \
  -f hls -hls_time 6 -hls_segment_type mpegts \
  -hls_playlist_type vod \
  -hls_segment_filename <outdir>/720p/segment_%03d.ts \
  <outdir>/720p/stream.m3u8
```

### Output Structure

```
<cache_dir>/hls/<sha256_of_filepath>/
  master.m3u8
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

### Master Playlist Format

```m3u8
#EXTM3U
#EXT-X-VERSION:3

#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1280x720
720p/stream.m3u8

#EXT-X-STREAM-INF:BANDWIDTH=8000000,RESOLUTION=1920x1080
passthrough/stream.m3u8
```

### New HTTP Endpoints

**`GET /media/hls?path=<encoded_path>`**
- Returns the master.m3u8 playlist
- If HLS cache doesn't exist for this path, generates passthrough on-the-fly (blocks until ready)
- Content-Type: `application/vnd.apple.mpegurl`
- CORS headers for web mode

**`GET /media/hls/segment?path=<encoded_path>&preset=<preset>&file=<filename>`**
- Serves individual .ts segments and sub-playlist .m3u8 files
- Content-Type: `video/MP2T` for .ts, `application/vnd.apple.mpegurl` for .m3u8
- Supports HTTP Range requests for seeking

Both endpoints are registered in `main.go` alongside existing `/media/file`.

### Cache Directory

Reuses the existing cache directory pattern (same parent as thumbnail cache). The `<sha256_of_filepath>` hashing ensures unique, collision-free directories per source file.

**Cache invalidation:** Compare source file mtime against a stored `.meta` JSON file in the HLS output directory. If source is newer, regenerate.

## Client Side

### New Dependency: hls.js

Added to `package.json`. hls.js is the standard HLS playback library for browsers. It handles:
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
2. If `useHLS` is false, or HLS is not supported (Safari plays HLS natively):
   - Fall back to current behavior: `src={mediaUrl(path)}` for direct playback
   - On Safari, can set `src` to the HLS URL directly since Safari supports HLS natively

**Seeking:** Works automatically with VOD playlists. The `#EXT-X-PLAYLIST-TYPE:VOD` directive tells hls.js that all segments are available from the start, enabling unrestricted seeking. The existing `currentTime` manipulation in video.tsx continues to work — hls.js seeks to the correct segment transparently.

**Existing functionality preserved:** Volume control, looping, play/pause, timeupdate events, error fallback to Image — all remain unchanged. hls.js just replaces how media data reaches the `<video>` element.

### New Platform Function

In `platform.ts`:

```typescript
export const hlsUrl = (path: string): string => {
  // Web mode
  return `/media/hls?path=${encodeURIComponent(path)}`;
  // Electron mode: use server URL similarly
};
```

### HLS Toggle

A setting (in the existing settings system) to disable HLS globally and use source video directly. When disabled, the video player uses the current `mediaUrl(path)` approach with no hls.js involvement.

This toggle is useful for:
- Local playback where direct file access is faster
- Debugging
- Formats that play fine natively without transcoding

## Seeking Support

Full seek is guaranteed by:
1. **VOD playlist type** — `#EXT-X-PLAYLIST-TYPE:VOD` signals all segments exist
2. **Segment index** — each segment's offset and duration are in the playlist
3. **hls.js seek handling** — on seek, hls.js calculates the target segment, fetches it, and resumes playback from the correct position within the segment
4. **Range request support** — segment endpoint supports HTTP Range for partial segment fetches

## Error Handling

- If HLS generation fails (corrupt file, unsupported codec), the task reports the error via stdout and the endpoint returns a 500
- The client falls back to direct source playback on HLS error (same pattern as the existing Image fallback on video error)
- If ffmpeg is not available, task logs error and fails gracefully

## Files Modified

### Go (media-server/)
- `tasks/hls.go` — new file, HLS generation task
- `tasks/registry.go` — register `hls` task
- `main.go` — add `/media/hls` and `/media/hls/segment` endpoints

### TypeScript (src/renderer/)
- `components/media-viewers/video.tsx` — hls.js integration
- `platform.ts` — add `hlsUrl` function

### Config
- `package.json` — add `hls.js` dependency
- Settings type — add HLS toggle
