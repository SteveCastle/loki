# Video Audio Track Selection + Subtitle Sidecars Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the volume popover in the video controls so the user can pick an audio track when a video has more than one, and toggle a sidecar `.srt`/`.vtt` subtitle file on/off (preference persisted globally).

**Architecture:** Three new modules — a pure SRT→VTT helper in the renderer, a main-process IPC for finding sidecar subtitles, and a popover component. The XState context gains four fields (`availableAudioTracks`, `availableSubtitle`, `selectedAudioTrackIndex`, plus `settings.subtitlesEnabled`). Audio track changes flow through the state machine and are reflected back into `videoEl.audioTracks[i].enabled` via a `useEffect` in `video.tsx`. Subtitle blob URLs are attached to `<video>` via a `<track>` child whose `mode` is toggled directly via a React ref.

**Tech Stack:** Electron 39, React 18, XState 4, TypeScript, ts-jest, electron-store. Existing platform abstraction (`src/renderer/platform.ts`) carries the new IPC.

**Spec:** `docs/superpowers/specs/2026-05-07-video-audio-subtitles-design.md`

---

## File map

**New files**

- `src/renderer/components/media-viewers/subtitle-loader.ts` — pure SRT→VTT + blob URL helper.
- `src/__tests__/subtitle-loader.test.ts` — unit tests for the above.
- `src/main/subtitles.ts` — sidecar lookup helper + IPC handler registration.
- `src/__tests__/subtitles.test.ts` — unit tests for the lookup helper.
- `src/renderer/components/controls/audio-track-controls.tsx` — popover content (volume + audio track + subtitles).
- `src/renderer/components/controls/audio-track-controls.css` — styles for the new popover.

**Modified files**

- `src/main/preload.ts` — add `'find-subtitle'` to `Channels`.
- `src/main/main.ts` — register subtitle handler, add `--enable-blink-features=AudioVideoTracks` Electron flag.
- `src/renderer/platform.ts` — add `findSubtitle` export, wire to IPC in Electron mode and a stub in web mode.
- `src/renderer/preload.d.ts` — declare `findSubtitle` on `window.electron` if needed.
- `src/settings.ts` — add `subtitlesEnabled` to `Settings` and `SettingKey`.
- `src/renderer/state.tsx` — add context fields, events, settings load.
- `src/renderer/components/media-viewers/video.tsx` — wire `loadedmetadata` effects, `<track>` child, audio track sync.
- `src/renderer/components/controls/video-controls.tsx` — replace inline volume slider with `AudioTrackControls`.

---

### Task 1: SRT→VTT helper (pure)

**Files:**
- Create: `src/renderer/components/media-viewers/subtitle-loader.ts`
- Test: `src/__tests__/subtitle-loader.test.ts`

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/subtitle-loader.test.ts`:

```ts
/**
 * Tests for the SRT→VTT conversion helper. The helper produces a string
 * (callers wrap it in a Blob URL). Conversion rules:
 *   - SRT cue timestamps use commas as the millisecond separator;
 *     VTT uses periods. Replace `,` with `.` only inside timestamp lines.
 *   - SRT files have no header. Prepend `WEBVTT\n\n`.
 *   - VTT input passes through unchanged except for an enforced trailing
 *     newline so callers can always concatenate cues safely.
 *   - Empty / malformed input still produces a parseable VTT (header only).
 */
import { srtToVtt, toVttString } from '../renderer/components/media-viewers/subtitle-loader';

describe('srtToVtt', () => {
  it('prepends WEBVTT header', () => {
    expect(srtToVtt('1\n00:00:01,000 --> 00:00:02,000\nHello\n')).toMatch(/^WEBVTT\n\n/);
  });

  it('replaces comma millisecond separator with period in timestamps', () => {
    const out = srtToVtt('1\n00:00:01,500 --> 00:00:02,750\nHi\n');
    expect(out).toContain('00:00:01.500 --> 00:00:02.750');
  });

  it('does not replace commas in cue text', () => {
    const out = srtToVtt('1\n00:00:01,000 --> 00:00:02,000\nHello, world\n');
    expect(out).toContain('Hello, world');
  });

  it('preserves multiple cues', () => {
    const srt = [
      '1',
      '00:00:01,000 --> 00:00:02,000',
      'First',
      '',
      '2',
      '00:00:03,000 --> 00:00:04,000',
      'Second',
      '',
    ].join('\n');
    const out = srtToVtt(srt);
    expect(out).toContain('First');
    expect(out).toContain('Second');
    expect(out).toContain('00:00:03.000 --> 00:00:04.000');
  });

  it('handles empty input as an empty VTT', () => {
    expect(srtToVtt('')).toBe('WEBVTT\n\n');
  });
});

describe('toVttString', () => {
  it('passes VTT input through with header preserved', () => {
    const vtt = 'WEBVTT\n\n1\n00:00:01.000 --> 00:00:02.000\nHi\n';
    expect(toVttString(vtt, 'vtt')).toContain('WEBVTT');
    expect(toVttString(vtt, 'vtt')).toContain('00:00:01.000 --> 00:00:02.000');
  });

  it('converts SRT input via srtToVtt', () => {
    expect(toVttString('1\n00:00:01,000 --> 00:00:02,000\nHi\n', 'srt')).toMatch(
      /^WEBVTT\n\n/
    );
  });

  it('adds a WEBVTT header to VTT input that is missing one', () => {
    const noHeader = '1\n00:00:01.000 --> 00:00:02.000\nHi\n';
    expect(toVttString(noHeader, 'vtt')).toMatch(/^WEBVTT\n\n/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest subtitle-loader --no-coverage`
Expected: FAIL with `Cannot find module '../renderer/components/media-viewers/subtitle-loader'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/renderer/components/media-viewers/subtitle-loader.ts`:

```ts
/**
 * Converts a sidecar subtitle file's text into a WebVTT string suitable
 * for use with a `<track>` element.
 *
 * Why: HTML5 `<track>` only accepts WebVTT. SRT, the most common sidecar
 * format, differs in two trivial ways — no header, and timestamps use
 * `,` rather than `.` as the millisecond separator. Translating in the
 * renderer avoids a main-process round trip per video.
 */

export type SubtitleExt = 'srt' | 'vtt';

const TIMESTAMP_LINE = /^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}/;

export function srtToVtt(srt: string): string {
  const converted = srt
    .split('\n')
    .map((line) =>
      TIMESTAMP_LINE.test(line) ? line.replace(/,(\d{3})/g, '.$1') : line
    )
    .join('\n');
  return `WEBVTT\n\n${converted}`;
}

export function toVttString(content: string, ext: SubtitleExt): string {
  if (ext === 'srt') return srtToVtt(content);
  // VTT: ensure header is present.
  if (!/^WEBVTT/.test(content)) return `WEBVTT\n\n${content}`;
  return content;
}

/**
 * Builds a `blob:` URL for a VTT string. Returned URL must be revoked
 * via `URL.revokeObjectURL` once the consumer is done with it.
 */
export function vttBlobUrl(vtt: string): string {
  const blob = new Blob([vtt], { type: 'text/vtt' });
  return URL.createObjectURL(blob);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest subtitle-loader --no-coverage`
Expected: PASS, all 8 cases.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/components/media-viewers/subtitle-loader.ts src/__tests__/subtitle-loader.test.ts
git commit -m "feat(subtitles): SRT/VTT to WebVTT helper"
```

---

### Task 2: Sidecar lookup helper + IPC handler

**Files:**
- Create: `src/main/subtitles.ts`
- Test: `src/__tests__/subtitles.test.ts`

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/subtitles.test.ts`:

```ts
/**
 * Tests the sidecar lookup helper. We separate the lookup from the IPC
 * handler so the lookup can be tested without an Electron context.
 *
 * Lookup precedence: `<basename>.srt` first, then `<basename>.vtt`.
 * Returns null when neither exists or when both are unreadable.
 */
import { findSidecarSubtitle } from '../main/subtitles';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

describe('findSidecarSubtitle', () => {
  let tmp: string;

  beforeEach(() => {
    tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'subtitle-test-'));
  });

  afterEach(() => {
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it('returns null when no sidecar exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    expect(await findSidecarSubtitle(video)).toBeNull();
  });

  it('returns srt content when only .srt exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.srt'), 'srt-content');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'srt-content' });
  });

  it('returns vtt content when only .vtt exists', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.vtt'), 'vtt-content');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'vtt', content: 'vtt-content' });
  });

  it('prefers .srt when both .srt and .vtt exist', async () => {
    const video = path.join(tmp, 'movie.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.srt'), 'srt-content');
    fs.writeFileSync(path.join(tmp, 'movie.vtt'), 'vtt-content');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'srt-content' });
  });

  it('handles paths with multiple dots in basename', async () => {
    const video = path.join(tmp, 'movie.s01e02.mp4');
    fs.writeFileSync(video, '');
    fs.writeFileSync(path.join(tmp, 'movie.s01e02.srt'), 'ok');
    const out = await findSidecarSubtitle(video);
    expect(out).toEqual({ ext: 'srt', content: 'ok' });
  });

  it('returns null when video path is empty', async () => {
    expect(await findSidecarSubtitle('')).toBeNull();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest subtitles --no-coverage`
Expected: FAIL with `Cannot find module '../main/subtitles'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/main/subtitles.ts`:

```ts
/**
 * Sidecar subtitle file lookup + IPC handler.
 *
 * For a given video path we check for `<basename>.srt` first, then
 * `<basename>.vtt` in the same directory. Exact basename only — no
 * language-suffix matching by design.
 */
import { ipcMain } from 'electron';
import * as fs from 'fs';
import * as path from 'path';

export type SubtitleSidecar = {
  ext: 'srt' | 'vtt';
  content: string;
};

const EXTS: Array<'srt' | 'vtt'> = ['srt', 'vtt'];

export async function findSidecarSubtitle(
  videoPath: string
): Promise<SubtitleSidecar | null> {
  if (!videoPath) return null;
  const dir = path.dirname(videoPath);
  const ext = path.extname(videoPath);
  const base = path.basename(videoPath, ext);

  for (const candidate of EXTS) {
    const p = path.join(dir, `${base}.${candidate}`);
    try {
      const content = await fs.promises.readFile(p, 'utf-8');
      return { ext: candidate, content };
    } catch (err: any) {
      if (err && err.code === 'ENOENT') continue;
      // Other read errors (permission, etc.): log and treat as not found.
      console.warn('[subtitles] failed to read', p, err);
    }
  }
  return null;
}

export function registerSubtitleHandlers(): void {
  ipcMain.handle('find-subtitle', async (_event, args: unknown[]) => {
    const videoPath = Array.isArray(args) ? (args[0] as string) : '';
    return findSidecarSubtitle(videoPath);
  });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest subtitles --no-coverage`
Expected: PASS, 6 cases.

- [ ] **Step 5: Commit**

```bash
git add src/main/subtitles.ts src/__tests__/subtitles.test.ts
git commit -m "feat(subtitles): main-process sidecar lookup IPC"
```

---

### Task 3: Wire find-subtitle channel into preload + platform

**Files:**
- Modify: `src/main/preload.ts` (Channels union)
- Modify: `src/renderer/platform.ts` (add `findSubtitle` export, wire Electron + web)

- [ ] **Step 1: Add channel name to preload Channels union**

In `src/main/preload.ts`, the `Channels` union (currently lines 15-68 ends with `'consolidate-category-files'`). Add `'find-subtitle'`:

```ts
  | 'consolidate-category-files'
  | 'find-subtitle';
```

- [ ] **Step 2: Add findSubtitle export to platform.ts**

After the existing `getGifMetadata` declaration (search for `export let getGifMetadata`), add:

```ts
export let findSubtitle: (
  videoPath: string
) => Promise<{ ext: 'srt' | 'vtt'; content: string } | null>;
```

In the `if (isElectron) { ... }` block (search for `getGifMetadata = window.electron.getGifMetadata;`), add right after that line:

```ts
  findSubtitle = (videoPath) =>
    window.electron.ipcRenderer.invoke('find-subtitle', [videoPath]);
```

In the `else` (web mode) block, add at the end of the assignments — search for `getGifMetadata = ...` in the web mode branch and add right after it:

```ts
  findSubtitle = async () => null;
```

(Web mode does not have filesystem access; subtitle sidecars are an Electron-only feature.)

- [ ] **Step 3: Verify TypeScript compiles**

Run: `npx tsc --noEmit -p . 2>&1 | grep -E "subtitle|find-subtitle" | head -5`
Expected: no output (no new errors related to this work). The codebase has pre-existing type errors per CLAUDE.md — those are not our concern.

- [ ] **Step 4: Commit**

```bash
git add src/main/preload.ts src/renderer/platform.ts
git commit -m "feat(subtitles): wire find-subtitle through platform layer"
```

---

### Task 4: Register subtitle handler in main + Chromium audio-tracks flag

**Files:**
- Modify: `src/main/main.ts`

- [ ] **Step 1: Import subtitle handler**

Near the top of `src/main/main.ts`, alongside the existing imports for sessionStore (currently lines 20-23):

```ts
import {
  registerSessionStoreHandlers,
  setupSessionStoreLifecycle,
} from './sessionStore';
import { registerSubtitleHandlers } from './subtitles';
```

- [ ] **Step 2: Enable AudioVideoTracks blink feature**

Right after the `protocol.registerSchemesAsPrivileged([...])` block ends at `}]);` (around line 48), add:

```ts
// Expose HTMLMediaElement.audioTracks / videoTracks to renderer code.
// Without this flag Chromium leaves these collections empty even on
// MP4 files that contain multiple audio streams.
app.commandLine.appendSwitch('enable-blink-features', 'AudioVideoTracks');
```

- [ ] **Step 3: Register the IPC handler**

Find the line `registerSessionStoreHandlers();` (around line 130). Add immediately after it:

```ts
registerSubtitleHandlers();
```

- [ ] **Step 4: Build to confirm main process is happy**

Run: `npm run build:main`
Expected: build succeeds without errors.

- [ ] **Step 5: Commit**

```bash
git add src/main/main.ts
git commit -m "feat(subtitles): register IPC, enable AudioVideoTracks flag"
```

---

### Task 5: Settings + state context additions

**Files:**
- Modify: `src/settings.ts`
- Modify: `src/renderer/state.tsx`

- [ ] **Step 1: Add subtitlesEnabled to Settings and SettingKey**

In `src/settings.ts`:

In `SettingKey` (currently ends at `'useHLS'` around line 56), add a new entry:

```ts
  | 'useHLS'
  | 'subtitlesEnabled';
```

In `Settings` type (currently ends with `useHLS: boolean;` around line 90), add:

```ts
  useHLS: boolean;
  subtitlesEnabled: boolean;
};
```

- [ ] **Step 2: Wire subtitlesEnabled through getInitialContext**

In `src/renderer/state.tsx`, find the `store.getMany([...])` call inside `getInitialContext` (search for `['useHLS', false],` around line 447). Add a new entry right after it:

```ts
    ['useHLS', false],
    ['subtitlesEnabled', false],
```

In the same file, find the `useHLS: batched['useHLS'] as boolean,` line (around line 578) inside the returned `settings` object. Add after it:

```ts
      useHLS: batched['useHLS'] as boolean,
      subtitlesEnabled: batched['subtitlesEnabled'] as boolean,
```

- [ ] **Step 3: Add new context fields**

In `src/renderer/state.tsx`, locate the `LibraryState` type definition. Find the `videoPlayer: { ... }` block (around lines 84-93) and add three new fields right after it (before `dbQuery: { tags: string[] };`):

```ts
  // Audio tracks discovered on the currently-loaded <video> element. Reset
  // to [] when the path changes; populated on `loadedmetadata`. The list
  // is sourced from HTMLMediaElement.audioTracks (gated by the
  // --enable-blink-features=AudioVideoTracks Chromium flag).
  availableAudioTracks: Array<{
    id: string;
    label: string;
    language: string;
  }>;
  // Index into availableAudioTracks of the user-selected (or default)
  // track. Always 0 on a fresh path; never persisted.
  selectedAudioTrackIndex: number;
  // Sidecar subtitle for the current path. The blob URL is owned by the
  // renderer and revoked when the path changes or the file unloads.
  availableSubtitle: {
    blobUrl: string;
    label: string;
  } | null;
```

- [ ] **Step 4: Initialize new context fields in getInitialContext**

In `src/renderer/state.tsx`, find the `return { ... }` of `getInitialContext` (around line 502). Add the new fields anywhere convenient, for example right before `videoPlayer:`:

```ts
    masonryDimensionsCache: {},
    availableAudioTracks: [],
    selectedAudioTrackIndex: 0,
    availableSubtitle: null,
    videoPlayer: {
```

(The exact line above will vary — pick the line just before `videoPlayer:`.)

- [ ] **Step 5: Add three new event handlers in the top-level `on:` block**

In `src/renderer/state.tsx`, find the existing `CHANGE_SETTING` handler (search for `CHANGE_SETTING: {` around line 701). The handler lives inside a top-level `on:` block. Add the three new handlers in the same block, right after the closing brace of `CHANGE_SETTING`'s actions:

```ts
          SET_AVAILABLE_AUDIO_TRACKS: {
            actions: assign<LibraryState, AnyEventObject>({
              availableAudioTracks: (_context, event) => event.tracks,
              // Reset selection when the track list changes (new video load).
              selectedAudioTrackIndex: () => 0,
            }),
          },
          SET_AUDIO_TRACK: {
            actions: assign<LibraryState, AnyEventObject>({
              selectedAudioTrackIndex: (_context, event) => event.index,
            }),
          },
          SET_AVAILABLE_SUBTITLE: {
            actions: assign<LibraryState, AnyEventObject>({
              availableSubtitle: (_context, event) => event.subtitle,
            }),
          },
```

- [ ] **Step 6: Run existing tests to confirm no regressions**

Run: `npx jest --no-coverage`
Expected: all suites pass (the new ones and the pre-existing ones).

- [ ] **Step 7: Commit**

```bash
git add src/settings.ts src/renderer/state.tsx
git commit -m "feat(state): audio tracks, subtitle, subtitlesEnabled context"
```

---

### Task 6: Wire video.tsx to populate state and apply selection

**Files:**
- Modify: `src/renderer/components/media-viewers/video.tsx`

- [ ] **Step 1: Import the helpers and add a track ref**

At the top of `src/renderer/components/media-viewers/video.tsx`, alongside the existing platform import (currently at line 9):

```ts
import { mediaUrl, hlsUrl, fetchMediaPreview as platformFetchMediaPreview, findSubtitle } from '../../platform';
import { toVttString, vttBlobUrl } from './subtitle-loader';
```

Inside the `Video` component, alongside the existing refs (search for `const prevTimeRef = useRef<number>(0);`), add:

```ts
  const prevTimeRef = useRef<number>(0);
  const trackRef = useRef<HTMLTrackElement>(null);
```

- [ ] **Step 2: Subscribe to the new context fields**

Below the existing `useSelector` calls (after `eventId`), add:

```ts
  const availableSubtitle = useSelector(
    libraryService,
    (state) => state.context.availableSubtitle
  );
  const subtitlesEnabled = useSelector(
    libraryService,
    (state) => state.context.settings.subtitlesEnabled
  );
  const selectedAudioTrackIndex = useSelector(
    libraryService,
    (state) => state.context.selectedAudioTrackIndex
  );
```

- [ ] **Step 3: Effect — enumerate audio tracks on metadata load**

Add a new effect after the existing `timeupdate` effect:

```ts
  useEffect(() => {
    const video = mediaRef?.current;
    if (!video) return;
    const handleMetadata = () => {
      const list = (video as any).audioTracks as
        | { length: number; [i: number]: any }
        | undefined;
      if (!list || typeof list.length !== 'number') {
        libraryService.send({ type: 'SET_AVAILABLE_AUDIO_TRACKS', tracks: [] });
        return;
      }
      const tracks: Array<{ id: string; label: string; language: string }> = [];
      for (let i = 0; i < list.length; i++) {
        const t = list[i];
        tracks.push({
          id: String(t.id ?? i),
          label: typeof t.label === 'string' && t.label ? t.label : `Track ${i + 1}`,
          language: typeof t.language === 'string' ? t.language : '',
        });
      }
      libraryService.send({ type: 'SET_AVAILABLE_AUDIO_TRACKS', tracks });
    };
    video.addEventListener('loadedmetadata', handleMetadata);
    // Some browsers fire only `loadeddata` reliably for already-cached files.
    video.addEventListener('loadeddata', handleMetadata);
    return () => {
      video.removeEventListener('loadedmetadata', handleMetadata);
      video.removeEventListener('loadeddata', handleMetadata);
    };
  }, [path, libraryService]);
```

- [ ] **Step 4: Effect — apply selectedAudioTrackIndex to videoEl**

Right after the previous effect:

```ts
  useEffect(() => {
    const video = mediaRef?.current;
    if (!video) return;
    const list = (video as any).audioTracks as
      | { length: number; [i: number]: { enabled: boolean } }
      | undefined;
    if (!list || list.length < 2) return;
    for (let i = 0; i < list.length; i++) {
      list[i].enabled = i === selectedAudioTrackIndex;
    }
  }, [selectedAudioTrackIndex]);
```

- [ ] **Step 5: Effect — fetch sidecar subtitle on path change**

Right after the previous effect:

```ts
  useEffect(() => {
    let revoked: string | null = null;
    let cancelled = false;
    findSubtitle(path)
      .then((sidecar) => {
        if (cancelled) return;
        if (!sidecar) {
          libraryService.send({ type: 'SET_AVAILABLE_SUBTITLE', subtitle: null });
          return;
        }
        const vtt = toVttString(sidecar.content, sidecar.ext);
        const url = vttBlobUrl(vtt);
        revoked = url;
        libraryService.send({
          type: 'SET_AVAILABLE_SUBTITLE',
          subtitle: { blobUrl: url, label: sidecar.ext.toUpperCase() },
        });
      })
      .catch(() => {
        if (cancelled) return;
        libraryService.send({ type: 'SET_AVAILABLE_SUBTITLE', subtitle: null });
      });
    return () => {
      cancelled = true;
      if (revoked) URL.revokeObjectURL(revoked);
      libraryService.send({ type: 'SET_AVAILABLE_SUBTITLE', subtitle: null });
    };
  }, [path, libraryService]);
```

- [ ] **Step 6: Effect — toggle the `<track>` mode based on subtitlesEnabled**

Right after the previous effect:

```ts
  useEffect(() => {
    const trackEl = trackRef.current;
    if (!trackEl || !trackEl.track) return;
    trackEl.track.mode = subtitlesEnabled ? 'showing' : 'hidden';
  }, [subtitlesEnabled, availableSubtitle]);
```

- [ ] **Step 7: Render the `<track>` child inside the two `<video>` returns**

There are two `<video>` JSX blocks in this file:
- The non-cache return (around lines 442-486 in the original)
- The cache return (around lines 521-551)

In **both** blocks, replace the self-closing `<video ... />` with an opening tag, a `<track>` child, and a closing `</video>`. For example, in the non-cache block change:

```tsx
        <video
          // ...existing props...
          autoPlay
          loop={!hlsActive}
        />
```

to:

```tsx
        <video
          // ...existing props...
          autoPlay
          loop={!hlsActive}
        >
          {availableSubtitle && (
            <track
              ref={trackRef}
              kind="subtitles"
              src={availableSubtitle.blobUrl}
              label={availableSubtitle.label}
              default={subtitlesEnabled}
            />
          )}
        </video>
```

Repeat for the second `<video>` block.

- [ ] **Step 8: Run the full Jest suite**

Run: `npx jest --no-coverage`
Expected: all suites pass; no regressions.

- [ ] **Step 9: Commit**

```bash
git add src/renderer/components/media-viewers/video.tsx
git commit -m "feat(video): populate audio tracks + subtitle, render <track>"
```

---

### Task 7: AudioTrackControls component

**Files:**
- Create: `src/renderer/components/controls/audio-track-controls.tsx`
- Create: `src/renderer/components/controls/audio-track-controls.css`

- [ ] **Step 1: Write the component**

Create `src/renderer/components/controls/audio-track-controls.tsx`:

```tsx
import React, { useContext, useCallback } from 'react';
import { useSelector } from '@xstate/react';
import { GlobalStateContext } from '../../state';
import { clampVolume } from '../../../settings';
import './audio-track-controls.css';

/**
 * Content of the audio popover. Wraps the volume slider and conditionally
 * adds an audio-track picker (when 2+ tracks) and a subtitle on/off toggle
 * (when a sidecar exists). Designed to be rendered inside the existing
 * volumeControlHover container in video-controls.tsx.
 */
export default function AudioTrackControls() {
  const { libraryService } = useContext(GlobalStateContext);

  const volume = useSelector(
    libraryService,
    (state: any) => state.context.settings.volume
  );
  const subtitlesEnabled = useSelector(
    libraryService,
    (state: any) => state.context.settings.subtitlesEnabled
  );
  const tracks = useSelector(
    libraryService,
    (state) => state.context.availableAudioTracks
  );
  const selectedIndex = useSelector(
    libraryService,
    (state) => state.context.selectedAudioTrackIndex
  );
  const subtitle = useSelector(
    libraryService,
    (state) => state.context.availableSubtitle
  );

  const setSetting = useCallback(
    (key: string, value: unknown) => {
      libraryService.send('CHANGE_SETTING', { data: { [key]: value } });
    },
    [libraryService]
  );

  const showTracks = tracks.length >= 2;
  const showSubtitleToggle = subtitle !== null;

  return (
    <div className="audioTrackControls">
      <div className="row volumeRow">
        <span className="rowLabel">Volume</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={volume}
          onChange={(e) => setSetting('volume', clampVolume(parseFloat(e.target.value)))}
          className="volumeSliderHover"
          aria-label="Volume"
        />
        <span className="rowValue">{Math.round(volume * 100)}%</span>
      </div>

      {showTracks && (
        <div className="row trackRow">
          <span className="rowLabel">Audio</span>
          <select
            value={selectedIndex}
            onChange={(e) =>
              libraryService.send({
                type: 'SET_AUDIO_TRACK',
                index: Number(e.target.value),
              })
            }
            aria-label="Audio track"
          >
            {tracks.map((t, i) => {
              const langSuffix = t.language ? ` (${t.language})` : '';
              return (
                <option key={t.id} value={i}>
                  {t.label}
                  {langSuffix}
                </option>
              );
            })}
          </select>
        </div>
      )}

      {showSubtitleToggle && (
        <div className="row subtitleRow">
          <span className="rowLabel">Subtitles</span>
          <button
            className={`toggle ${subtitlesEnabled ? 'on' : 'off'}`}
            onClick={() => setSetting('subtitlesEnabled', !subtitlesEnabled)}
            aria-pressed={subtitlesEnabled}
          >
            {subtitlesEnabled ? 'On' : 'Off'}
          </button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Add basic CSS**

Create `src/renderer/components/controls/audio-track-controls.css`:

```css
.audioTrackControls {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  min-width: 220px;
}

.audioTrackControls .row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.audioTrackControls .rowLabel {
  flex: 0 0 70px;
  color: #ccc;
  font-size: 12px;
}

.audioTrackControls .rowValue {
  flex: 0 0 40px;
  text-align: right;
  color: #aaa;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.audioTrackControls .volumeSliderHover {
  flex: 1 1 auto;
  min-width: 0;
}

.audioTrackControls .trackRow select {
  flex: 1 1 auto;
  background: #222;
  color: #ddd;
  border: 1px solid #444;
  border-radius: 3px;
  padding: 2px 4px;
  font-size: 12px;
}

.audioTrackControls .toggle {
  flex: 0 0 auto;
  background: #333;
  color: #ccc;
  border: 1px solid #555;
  border-radius: 3px;
  padding: 2px 10px;
  cursor: pointer;
  font-size: 12px;
}

.audioTrackControls .toggle.on {
  background: #2a5;
  color: #fff;
  border-color: #2a5;
}
```

- [ ] **Step 3: Commit**

```bash
git add src/renderer/components/controls/audio-track-controls.tsx src/renderer/components/controls/audio-track-controls.css
git commit -m "feat(controls): AudioTrackControls popover content"
```

---

### Task 8: Replace the inline volume slider in video-controls.tsx

**Files:**
- Modify: `src/renderer/components/controls/video-controls.tsx`

- [ ] **Step 1: Import the new component**

At the top of `src/renderer/components/controls/video-controls.tsx`, add:

```ts
import AudioTrackControls from './audio-track-controls';
```

- [ ] **Step 2: Replace the inline volume slider in the popover**

Find the `{showVolumeControl && playSound && (` block (currently around lines 462-479 in the original file). Replace its entire contents — the `<div className="volumeControlHover">...</div>` — with:

```tsx
          {showVolumeControl && playSound && (
            <div className="volumeControlHover">
              <AudioTrackControls />
            </div>
          )}
```

The `volumeControlHover` div stays because it carries the existing positioning CSS; only the inner volume label + range input is replaced by the new component.

- [ ] **Step 3: Remove now-unused imports/locals**

The `clampVolume` import at the top of `video-controls.tsx` is no longer used by this file (the new component imports it instead). Remove the import line:

```ts
// Before:
import { clampVolume } from '../../../settings';
// After: delete this line
```

- [ ] **Step 4: Build the renderer to confirm no breakage**

Run: `npm run build:renderer`
Expected: build succeeds.

- [ ] **Step 5: Commit**

```bash
git add src/renderer/components/controls/video-controls.tsx
git commit -m "feat(controls): mount AudioTrackControls in volume popover"
```

---

### Task 9: Final verification + version bump

**Files:**
- Modify: `package.json`, `package-lock.json`, `release/app/package.json`, `release/app/package-lock.json`, `docs/index.html`, `src/renderer/components/controls/command-palette.tsx`

- [ ] **Step 1: Run full test suite**

Run: `npx jest --no-coverage`
Expected: all suites pass; subtitle-loader and subtitles tests are present and green.

- [ ] **Step 2: Bump version 2.9.2 → 2.10.0**

Replace `2.9.2` with `2.10.0` in:
- `package.json` (one entry)
- `package-lock.json` (two entries — top-level `version` and `packages[""].version`)
- `release/app/package.json`
- `release/app/package-lock.json` (two entries)
- `docs/index.html` (four entries — version label and three release URLs)
- `src/renderer/components/controls/command-palette.tsx` (one fallback string)

Verify with `git grep '2\.9\.2'` returning no matches and `git grep '2\.10\.0'` returning the expected ones.

- [ ] **Step 3: Run full Jest suite a final time**

Run: `npx jest --no-coverage`
Expected: all green.

- [ ] **Step 4: Commit and push**

```bash
git add -A
git commit -m "feat(video): audio track + sidecar subtitle support, bump 2.10.0"
git push
```

- [ ] **Step 5: Manual UI verification (cannot automate)**

After the next desktop run:
- Open an MP4 with multiple audio tracks → audio popover shows the track dropdown, picking a track switches audio.
- Open a video with `<basename>.srt` next to it → subtitle toggle appears; turning it on renders subtitles overlaid on the video; preference persists across video changes.
- Open a video with `<basename>.vtt` → same as above.
- Open a video with neither → popover looks identical to today (volume slider only).
- Open a single-audio-track MP4 → no track dropdown, popover compact.

Report any deviation; otherwise the feature is complete.

---

## Self-review notes

- Spec coverage: every requirement in the design doc is touched by exactly one task. Audio-tracks Chromium flag → Task 4. SRT/VTT helper → Task 1. Sidecar lookup → Task 2. Renderer wiring → Task 3. State changes → Task 5. Video element behaviour → Task 6. Popover UI → Tasks 7+8. Manual verification → Task 9.
- Type consistency: `availableAudioTracks` shape (id/label/language) is the same in Task 5 (context type), Task 6 (population), and Task 7 (consumer). `availableSubtitle` shape (blobUrl/label) is identical in Tasks 5, 6, 7. `SubtitleSidecar` (ext/content) flows from Task 2 IPC into Task 3 platform typing into Task 6 consumer.
- No placeholders: every TDD step has the actual code; every modify step has the exact lines to add and where; every test step has the exact command and expected outcome.
- Out-of-scope items from the spec (language-coded sidecars, ASS/SSA, multi-subtitle picker, video-track picker, web-mode) are absent from the plan as intended.
