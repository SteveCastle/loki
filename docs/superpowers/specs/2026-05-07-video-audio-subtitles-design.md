# Video Audio Track Selection + Subtitle Sidecars — Design

**Date:** 2026-05-07
**Scope:** Electron desktop app only (`src/main`, `src/renderer`). Web-mode (Go server) is out of scope and may be added later by exposing the same handlers as HTTP endpoints.

## Goal

Extend the existing volume popover in the video controls so that when a video is loaded:

- If it has more than one audio track, the user can pick which track to listen to.
- If a sidecar subtitle file (`<basename>.srt` or `<basename>.vtt`) exists in the same directory, the user can toggle subtitle rendering on/off.
- The on/off subtitle preference is remembered globally across videos.

## Non-goals

- Language-coded sidecars such as `movie.en.srt` — only exact basename match.
- ASS/SSA subtitles — only `.srt` and `.vtt`.
- Multiple subtitles per video / a subtitle language picker.
- Video-track (multi-stream) picker — only audio.
- Web-mode (Go server) support.
- Subtitle styling overrides — use browser defaults.
- Persisting which audio track was chosen across videos. Different videos have different tracks, so we default each new video to track 1.

## Architecture

Three new modules, each with one clear job:

1. **`src/main/subtitles.ts`** — main-process IPC handler `find-subtitle`. Given a video path, looks for `<basename>.srt` then `<basename>.vtt` in the same directory. Returns `{ ext: 'srt' | 'vtt', content: string } | null`. Read errors are logged and surfaced as `null` so the renderer treats them like "not found".

2. **`src/renderer/components/media-viewers/subtitle-loader.ts`** — pure helper. Given raw text and an extension, returns a VTT blob URL. SRT→VTT is a small string transform: replace `,` with `.` in cue timestamps and prepend a `WEBVTT\n\n` header. VTT passes through unmodified. The blob URL is later attached to the `<video>` via a `<track>` element.

3. **`src/renderer/components/media-viewers/audio-track-controls.tsx`** — new popover content rendered next to the existing `volumeControlHover` in `video-controls.tsx`. Shows:
   - The volume slider (existing).
   - An audio track dropdown row, only when `availableAudioTracks.length >= 2`.
   - A subtitles on/off toggle row, only when `availableSubtitle !== null`.

## Data flow

`video.tsx` already mounts the `<video>` element. Two new effects on it:

- On `loadedmetadata`, read `videoEl.audioTracks` (a `TextTrackList`-like object) and dispatch `SET_AVAILABLE_AUDIO_TRACKS` with `[{ id, label, language, enabled }]`. This list is reset to `[]` when `path` changes.
- On `loadedmetadata`, call `invoke('find-subtitle', path)`. If the result is non-null, run `subtitle-loader` to build a blob URL and dispatch `SET_AVAILABLE_SUBTITLE` with `{ blobUrl, label }`. The previous blob URL (if any) is `URL.revokeObjectURL`'d when the path changes.

The `<video>` element gets a `<track kind="subtitles" src={blobUrl} default={settings.subtitlesEnabled}>` child whenever a subtitle is available. When the user toggles the subtitle setting, we set the `<track>` element's underlying `track.mode` to `"showing"` or `"hidden"`. We address the track via a React ref to the `<track>` element rather than indexing into `videoEl.textTracks`, so the lookup is unaffected by any in-container text tracks the file may also expose.

When the user picks an audio track, we dispatch `SET_AUDIO_TRACK(index)`. `video.tsx` reacts by setting `videoEl.audioTracks[i].enabled = true` and the others `false`. We do not persist this — each new path starts at index 0.

## State changes

Three additions to the XState context in `src/renderer/state.tsx`:

- `availableAudioTracks: { id: string, label: string, language: string, enabled: boolean }[]` — populated on `SET_AVAILABLE_AUDIO_TRACKS`, cleared on path change.
- `availableSubtitle: { blobUrl: string, label: string } | null` — same lifecycle.
- `settings.subtitlesEnabled: boolean` — initial value `false`, persisted via the existing `store` (electron-store) on every change. Toggle dispatches the existing `CHANGE_SETTING` event.

Three new events:

- `SET_AVAILABLE_AUDIO_TRACKS(tracks)` — overwrites context.
- `SET_AVAILABLE_SUBTITLE(subtitle | null)` — overwrites context.
- `SET_AUDIO_TRACK(index)` — fires a side effect that toggles `audioTracks[i].enabled` flags on the underlying `<video>` element. The renderer re-emits `SET_AVAILABLE_AUDIO_TRACKS` afterward so the dropdown's "selected" highlight stays in sync with reality.

## Chromium flag

Without `--enable-blink-features=AudioVideoTracks`, `HTMLMediaElement.audioTracks` is empty for everyone. Adding it to `app.commandLine.appendSwitch(...)` in `src/main/main.ts` (early, before `app.ready`) is required for the audio-track row to ever appear.

If the flag fails or the file format isn't supported (e.g., MKV in Chromium), `audioTracks.length` stays at zero, the dropdown row is suppressed, and the rest of the popover behaves identically to today.

## UI

The existing volume hover layout becomes:

```
Volume     [════●═══] 45%       (existing)
─────────────────────────────
Audio      ▼ Track 2            (only if audioTracks ≥ 2)
─────────────────────────────
Subtitles  [ on ]               (only if a sidecar exists)
```

When neither extra row applies, the popover is visually identical to today.

## Error handling

- Sidecar not found: `find-subtitle` returns `null`. Subtitle row is hidden.
- Sidecar exists but unreadable: log in main, return `null`. Same as not found.
- SRT parse produces no cues: still wraps as a valid VTT (just an empty body); browser handles it. No need to special-case.
- `audioTracks` is empty: row hidden, current behavior preserved.
- `audioTracks` indexes out of range when `SET_AUDIO_TRACK` fires (e.g., user spams clicks during reload): no-op, log a warning.

## Testing

Two new test files, both pure-function focused following the existing convention:

- **`src/__tests__/subtitle-loader.test.ts`** — covers SRT→VTT conversion (timestamp comma-to-period, header prepend, multi-cue file, empty input), VTT passthrough, and malformed input that should still produce a parseable VTT.
- **`src/__tests__/subtitles.test.ts`** — covers `find-subtitle`'s lookup logic by stubbing `fs.promises.readFile` / `fs.promises.access`. Verifies `.srt` precedence, `.vtt` fallback, missing-both → null, read-error → null. The IPC handler is split from the lookup helper so the lookup is testable without an Electron context.

Component-level rendering tests for the popover are deferred. The data-flow contracts (IPC handler, reducer, blob-URL helper) carry the correctness weight, and the existing codebase keeps React component tests minimal.

## Implementation order

1. `subtitle-loader.ts` and its test (TDD: RED → GREEN).
2. `subtitles.ts` (lookup helper + IPC handler) and its test.
3. Wire `find-subtitle` IPC into the renderer side of `platform.ts`.
4. Add the Chromium flag in `src/main/main.ts`.
5. Add context fields and events to `state.tsx`. Add `subtitlesEnabled` to `settings`.
6. Wire `loadedmetadata` effects in `video.tsx` (audio tracks + subtitle fetch + `<track>` child + cleanup of blob URLs on path change).
7. Build `audio-track-controls.tsx` and integrate it into the volume popover in `video-controls.tsx`.
8. Manual verification: a multi-audio-track MP4, a video with a `.srt` sidecar, a video with a `.vtt` sidecar, a video with neither, a video that the audioTracks API doesn't enumerate (graceful no-op).
