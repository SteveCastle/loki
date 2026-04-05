# Web Mode for Loki

**Date:** 2026-03-15
**Status:** Draft

## Problem

Loki is an Electron desktop app for browsing and organizing a media library. The React UI is tightly coupled to Electron via IPC calls (`window.electron.ipcRenderer.invoke`), fire-and-forget messages (`sendMessage`), event listeners (`on`), a custom `gsm://` protocol for file serving, and direct preload APIs (`store`, `sessionStore`, `transcript`, `url`). Users want to browse their library from any device over the web using the same experience, served by the existing Go media-server.

## Goals

- Serve the Loki React UI from the Go media-server as a web app
- Full feature parity with the Electron app, minus OS-level features (window controls, clipboard, file system scanning)
- Reuse the same React components and single webpack build
- Do not regress the Electron app
- Implement core endpoints first; stub the rest with 501 responses

## Non-Goals

- Separate webpack build or entry point for web
- Multi-user or authentication (beyond what the Go server already has)
- Mobile-specific responsive design
- Replacing the Electron app

## Architecture

Three layers of change:

1. **Platform service** (`src/renderer/platform.ts`) — abstracts Electron IPC vs HTTP
2. **Component refactor** — replace all direct `window.electron.*` calls with platform service calls
3. **Go server REST API + static serving** — new endpoints mirroring IPC events, plus serving the webpack bundle

### Runtime Platform Detection

A single webpack build produces one bundle. At import time, `platform.ts` checks for `window.electron`:

```typescript
export const isElectron = typeof window !== 'undefined'
  && typeof window.electron !== 'undefined';
```

All downstream code uses the platform service. The platform never changes at runtime, so this check runs once.

## Platform Service (`src/renderer/platform.ts`)

### Key Design Decision: Argument Convention

The Electron preload's `ipcRenderer.invoke(channel, args: unknown[])` always takes a **single array** as the second argument. Components call it as:

```typescript
window.electron.ipcRenderer.invoke('create-tag', [newLabel, categoryLabel, 0]);
```

The platform service preserves this convention exactly:

```typescript
platform.invoke('create-tag', [newLabel, categoryLabel, 0]);
```

This means `invoke` takes `(channel: string, args?: unknown[])` — matching the preload signature. The `argsToBody` transformers receive this single array and map positional elements to named JSON fields for the REST API.

For the typed wrapper functions (e.g., `platform.loadMediaFromDB`), the signatures match the preload's exported functions exactly. In web mode, they call `authFetch` directly with properly constructed bodies — they do **not** go through `invoke`/`channelToEndpoint`.

### Exports

```typescript
// Detection
export const isElectron: boolean;

// Capabilities — components check these to hide Electron-only UI
export const capabilities: {
  fileSystemAccess: boolean;  // select-file, select-directory, load-files
  clipboard: boolean;         // copy-file-into-clipboard
  windowControls: boolean;    // minimize, toggle-fullscreen, set-always-on-top
  autoUpdate: boolean;        // check-for-updates
  shutdown: boolean;          // app shutdown
};

// Media URLs
export function mediaUrl(path: string, version?: string): string;

// Generic IPC-equivalent (request/response) — single array arg, matching preload
export function invoke(channel: string, args?: unknown[]): Promise<any>;

// Fire-and-forget messages (sendMessage equivalent)
export function send(channel: string, args?: unknown[]): void;

// Event listeners (ipcRenderer.on equivalent)
export function on(channel: string, callback: (...args: unknown[]) => void): () => void;

// App arguments
export const appArgs: { dbPath?: string; filePath?: string; appUserData?: string };

// Config storage (synchronous)
export const store: {
  get(key: string, defaultValue?: any): any;
  set(key: string, value: any): void;
  getMany(pairs: [string, any][]): Record<string, any>;
};

// Session storage (async)
export const sessionStore: {
  get(key: string): Promise<any>;
  set(key: string, value: any): Promise<void>;
  getAll(): Promise<any>;
  setMany(entries: Record<string, any>): Promise<void>;
  clear(): Promise<void>;
  clearKeys(keys: string[]): Promise<void>;
  flush(): Promise<void>;
};

// Transcript API (stubbed in web mode initially)
export const transcript: {
  loadTranscript(filePath: string): Promise<any>;
  modifyTranscript(input: any): Promise<void>;
};

// Typed wrappers — signatures match preload exactly
export function loadMediaFromDB(
  tags: string[], mode?: string
): Promise<Item[]>;

export function loadMediaByDescriptionSearch(
  description: string, tags?: string[], filteringMode?: string
): Promise<Item[]>;

export function fetchMediaPreview(
  tag: string, cache: string, timeStamp: number
): Promise<any>;

export function fetchTagPreview(tag: string): Promise<any>;

export function fetchTagCount(tag: string): Promise<number>;

export function listThumbnails(filePath: string): Promise<{
  cache: string; path: string; exists: boolean; size: number;
}[]>;

export function regenerateThumbnail(
  filePath: string, cache: string, timeStamp?: number
): Promise<string>;

export function loadDuplicatesByPath(path: string): Promise<any>;
export function mergeDuplicatesByPath(path: string): Promise<any>;

export function getGifMetadata(filePath: string): Promise<{
  frameCount: number; duration: number;
} | null>;
```

### Electron Implementation

Delegates directly to `window.electron.*`:

```typescript
if (isElectron) {
  invoke = (channel, args) => window.electron.ipcRenderer.invoke(channel, args);
  send = (channel, args) => window.electron.ipcRenderer.sendMessage(channel, args);
  on = (channel, cb) => window.electron.ipcRenderer.on(channel, cb);
  mediaUrl = (path, version) => window.electron.url.format({
    protocol: 'gsm',
    pathname: path,
    search: version ? `?v=${version}` : undefined,
  });
  appArgs = window.appArgs ?? {};
  store = window.electron.store;
  sessionStore = window.electron.sessionStore;
  transcript = window.electron.transcript;
  // Typed wrappers delegate to preload functions
  loadMediaFromDB = window.electron.loadMediaFromDB;
  loadMediaByDescriptionSearch = window.electron.loadMediaByDescriptionSearch;
  fetchMediaPreview = window.electron.fetchMediaPreview;
  fetchTagPreview = window.electron.fetchTagPreview;
  fetchTagCount = window.electron.fetchTagCount;
  listThumbnails = window.electron.listThumbnails;
  regenerateThumbnail = window.electron.regenerateThumbnail;
  loadDuplicatesByPath = window.electron.loadDuplicatesByPath;
  mergeDuplicatesByPath = window.electron.mergeDuplicatesByPath;
  getGifMetadata = window.electron.getGifMetadata;
}
```

### Web Implementation

```typescript
if (!isElectron) {
  // Auth: include JWT cookie on all requests
  const authFetch = (url: string, opts: RequestInit = {}) =>
    fetch(url, { ...opts, credentials: 'include' });

  const jsonPost = (url: string, body: any) =>
    authFetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(handleResponse);

  const jsonPut = (url: string, body: any) =>
    authFetch(url, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(handleResponse);

  const jsonDelete = (url: string, body: any) =>
    authFetch(url, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(handleResponse);

  async function handleResponse(res: Response) {
    if (res.status === 401) { window.location.href = '/login'; return; }
    if (res.status === 501) throw new Error('Not available in web mode');
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const text = await res.text();
    return text ? JSON.parse(text) : undefined;
  }

  // Args come from URL query params
  const params = new URLSearchParams(window.location.search);
  appArgs = {
    dbPath: params.get('db') ?? undefined,
    filePath: params.get('file') ?? undefined,
  };

  // invoke: matches preload signature — args is a single array
  invoke = async (channel, args) => {
    const mapping = channelToEndpoint(channel);
    if (!mapping) throw new Error(`Not implemented in web mode: ${channel}`);
    const body = mapping.argsToBody(args ?? []);
    if (mapping.method === 'GET') {
      const qs = body ? '?' + new URLSearchParams(body).toString() : '';
      return authFetch(mapping.url + qs).then(handleResponse);
    }
    if (mapping.method === 'PUT') return jsonPut(mapping.url, body);
    if (mapping.method === 'DELETE') return jsonDelete(mapping.url, body);
    return jsonPost(mapping.url, body);
  };

  send = (channel, args) => {
    // Window control channels are no-ops in web
    if (['shutdown', 'minimize', 'toggle-fullscreen',
         'set-always-on-top'].includes(channel)) return;
    // open-external: open URL in new tab
    if (channel === 'open-external' && args?.[0]) {
      window.open(String(args[0]), '_blank');
      return;
    }
    // Fallback: fire-and-forget via invoke
    invoke(channel, args).catch(() => {});
  };

  on = (_channel, _cb) => {
    // Electron-only event channels are no-ops in web mode
    return () => {};
  };

  mediaUrl = (path, version) => {
    const params = new URLSearchParams({ path });
    if (version) params.set('v', version);
    return `/media/file?${params}`;
  };

  // Store: server-backed settings with in-memory cache
  let _settingsCache: Record<string, any> = {};
  const _settingsLoaded = authFetch('/api/settings')
    .then(r => r.json())
    .then(data => { _settingsCache = data; })
    .catch(() => {});

  store = {
    get: (key, defaultValue) => {
      const val = _settingsCache[key];
      return val !== undefined ? val : defaultValue;
    },
    set: (key, value) => {
      _settingsCache[key] = value;
      authFetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key, value }),
      });
    },
    getMany: (pairs) => {
      const result: Record<string, any> = {};
      pairs.forEach(([k, def]) => {
        const val = _settingsCache[k];
        result[k] = val !== undefined ? val : def;
      });
      return result;
    },
  };

  // Session store: server-backed with in-memory cache
  let _sessionCache: Record<string, any> = {};
  const _sessionLoaded = authFetch('/api/session')
    .then(r => r.json())
    .then(data => { _sessionCache = data; })
    .catch(() => {});

  sessionStore = {
    get: async (key) => {
      await _sessionLoaded;
      return _sessionCache[key];
    },
    set: async (key, value) => {
      _sessionCache[key] = value;
      await authFetch(`/api/session/${key}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value),
      });
    },
    getAll: async () => {
      await _sessionLoaded;
      return { ..._sessionCache };
    },
    setMany: async (entries) => {
      Object.assign(_sessionCache, entries);
      await authFetch('/api/session', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entries),
      });
    },
    clear: async () => {
      _sessionCache = {};
      await authFetch('/api/session', { method: 'DELETE' });
    },
    clearKeys: async (keys) => {
      keys.forEach(k => delete _sessionCache[k]);
      await authFetch('/api/session/keys', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keys }),
      });
    },
    flush: async () => {
      // No-op: web writes are immediate (no debounce)
    },
  };

  // Transcript: stubbed initially
  transcript = {
    loadTranscript: async () => { throw new Error('Transcripts not yet available in web mode'); },
    modifyTranscript: async () => { throw new Error('Transcripts not yet available in web mode'); },
  };

  // Typed wrappers call authFetch directly — do NOT go through invoke/channelToEndpoint
  loadMediaFromDB = (tags, mode = 'EXCLUSIVE') =>
    jsonPost('/api/media', { tags, mode });

  loadMediaByDescriptionSearch = (description, tags, filteringMode) =>
    jsonPost('/api/media/search', { description, tags, filteringMode });

  fetchMediaPreview = (tag, cache, timeStamp) =>
    jsonPost('/api/media/preview', { path: tag, cache, timeStamp });

  fetchTagPreview = (tag) =>
    jsonPost('/api/tags/preview', { label: tag });

  fetchTagCount = (tag) =>
    jsonPost('/api/tags/count', { label: tag }).then(r => r.count);

  listThumbnails = (filePath) =>
    jsonPost('/api/thumbnails', { path: filePath });

  regenerateThumbnail = (filePath, cache, timeStamp) =>
    jsonPost('/api/thumbnails/regenerate', { path: filePath, cache, timeStamp });

  loadDuplicatesByPath = async () => {
    throw new Error('Duplicates not yet available in web mode');
  };

  mergeDuplicatesByPath = async () => {
    throw new Error('Duplicates not yet available in web mode');
  };

  getGifMetadata = (filePath) =>
    jsonPost('/api/media/gif-metadata', { path: filePath });
}
```

### Channel-to-Endpoint Mapping

Used only by `invoke()` for raw IPC channel calls from components. Each `argsToBody` receives the single array that was passed as the second arg to `invoke`.

```typescript
type EndpointMapping = {
  url: string;
  method: string;
  argsToBody: (args: unknown[]) => any;
};

function channelToEndpoint(channel: string): EndpointMapping | null {
  const map: Record<string, EndpointMapping> = {
    // Media
    'load-file-metadata': {
      url: '/api/media/metadata', method: 'POST',
      argsToBody: (args) => ({ path: args[0] }),
    },
    'load-tags-by-media-path': {
      url: '/api/media/tags', method: 'POST',
      argsToBody: (args) => ({ path: args[0] }),
    },
    'update-description': {
      url: '/api/media/description', method: 'PUT',
      argsToBody: (args) => ({ path: args[0], description: args[1] }),
    },
    'delete-file': {
      url: '/api/media/delete', method: 'DELETE',
      argsToBody: (args) => ({ path: args[0] }),
    },

    // Taxonomy
    'load-taxonomy': {
      url: '/api/taxonomy', method: 'GET',
      argsToBody: () => null,
    },
    'create-tag': {
      url: '/api/tags', method: 'POST',
      argsToBody: (args) => ({ label: args[0], categoryLabel: args[1], weight: args[2] }),
    },
    'rename-tag': {
      url: '/api/tags/rename', method: 'PUT',
      argsToBody: (args) => ({ label: args[0], newLabel: args[1] }),
    },
    'move-tag': {
      url: '/api/tags/move', method: 'PUT',
      argsToBody: (args) => ({ label: args[0], categoryLabel: args[1] }),
    },
    'delete-tag': {
      url: '/api/tags', method: 'DELETE',
      argsToBody: (args) => ({ label: args[0] }),
    },
    'order-tags': {
      url: '/api/tags/order', method: 'PUT',
      argsToBody: (args) => ({ labels: args[0] }),
    },
    'update-tag-weight': {
      url: '/api/tags/weight', method: 'PUT',
      argsToBody: (args) => ({ label: args[0], weight: args[1] }),
    },
    'update-timestamp': {
      url: '/api/tags/timestamp', method: 'PUT',
      argsToBody: (args) => ({
        mediaPath: args[0], tagLabel: args[1],
        categoryLabel: args[2], timestamp: args[3],
      }),
    },
    'remove-timestamp': {
      url: '/api/tags/timestamp', method: 'DELETE',
      argsToBody: (args) => ({
        mediaPath: args[0], tagLabel: args[1], categoryLabel: args[2],
      }),
    },

    // Categories
    'create-category': {
      url: '/api/categories', method: 'POST',
      argsToBody: (args) => ({ label: args[0], weight: args[1] }),
    },
    'rename-category': {
      url: '/api/categories/rename', method: 'PUT',
      argsToBody: (args) => ({ label: args[0], newLabel: args[1] }),
    },
    'delete-category': {
      url: '/api/categories', method: 'DELETE',
      argsToBody: (args) => ({ label: args[0] }),
    },

    // Assignments
    'create-assignment': {
      url: '/api/assignments', method: 'POST',
      argsToBody: (args) => ({
        mediaPath: args[0], tagLabel: args[1],
        categoryLabel: args[2], weight: args[3],
      }),
    },
    'delete-assignment': {
      url: '/api/assignments', method: 'DELETE',
      argsToBody: (args) => ({
        mediaPath: args[0], tagLabel: args[1], categoryLabel: args[2],
      }),
    },
    'update-assignment-weight': {
      url: '/api/assignments/weight', method: 'PUT',
      argsToBody: (args) => ({
        mediaPath: args[0], tagLabel: args[1],
        categoryLabel: args[2], weight: args[3],
      }),
    },

    // Database
    'load-db': {
      url: '/api/db/load', method: 'POST',
      argsToBody: (args) => ({ path: args[0] }),
    },
  };
  return map[channel] ?? null;
}
```

**Note:** The typed wrapper functions (`loadMediaFromDB`, `fetchMediaPreview`, etc.) do NOT go through `channelToEndpoint`. They call `authFetch` directly with correctly shaped bodies. `channelToEndpoint` is only used by `invoke()` for the raw IPC channel calls that remain in components like `hotkey-controller.tsx`, `taxonomy.tsx`, `tags.tsx`, etc.

The `argsToBody` transformers must be verified against the actual call sites during implementation. Each component that calls `invoke('channel', [...])` passes args in a specific positional order — the transformers must match. The implementation plan will include a verification step for each channel.

## Exhaustive IPC Channel Audit

Every `window.electron` usage in the renderer, with disposition:

### ipcRenderer.invoke() — 25 unique channels used in renderer

| Channel | Files | Disposition |
|---------|-------|-------------|
| `load-db` | state.tsx | Implemented: `/api/db/load` |
| `select-db` | state.tsx | Stubbed (file dialog) |
| `select-file` | state.tsx | Stubbed (file dialog) |
| `select-directory` | state.tsx | Stubbed (file dialog) |
| `load-files` | state.tsx | Stubbed (FS scanning) |
| `refresh-library` | state.tsx | Stubbed (FS scanning) |
| `select-new-path` | state.tsx | Stubbed (file dialog) |
| `delete-file` | state.tsx | Implemented: `/api/media/delete` |
| `load-taxonomy` | taxonomy.tsx | Implemented: `/api/taxonomy` |
| `load-file-metadata` | file-metadata.tsx | Implemented: `/api/media/metadata` |
| `update-description` | description.tsx | Implemented: `/api/media/description` |
| `create-tag` | new-tag-modal.tsx | Implemented: `/api/tags` |
| `rename-tag` | new-tag-modal.tsx | Implemented: `/api/tags/rename` |
| `move-tag` | category.tsx | Implemented: `/api/tags/move` |
| `delete-tag` | confirm-delete-tag.tsx | Implemented: `/api/tags` DELETE |
| `order-tags` | new-category-modal.tsx | Implemented: `/api/tags/order` |
| `update-tag-weight` | tag.tsx | Implemented: `/api/tags/weight` |
| `update-timestamp` | tags.tsx | Implemented: `/api/tags/timestamp` |
| `remove-timestamp` | tags.tsx | Implemented: `/api/tags/timestamp` DELETE |
| `create-category` | new-category-modal.tsx | Implemented: `/api/categories` |
| `rename-category` | new-category-modal.tsx | Implemented: `/api/categories/rename` |
| `delete-category` | confirm-delete-category.tsx | Implemented: `/api/categories` DELETE |
| `create-assignment` | useTagDrop.tsx, hotkey-controller.tsx, file-metadata.tsx | Implemented: `/api/assignments` |
| `delete-assignment` | tags.tsx | Implemented: `/api/assignments` DELETE |
| `update-assignment-weight` | useTagDrop.tsx, hotkey-controller.tsx | Implemented: `/api/assignments/weight` |
| `copy-file-into-clipboard` | hotkey-controller.tsx | Stubbed (clipboard) |
| `check-for-updates` | command-palette.tsx | Stubbed (auto-update) |
| `update-elo` | BattleMode.tsx | Stubbed (battle mode) |

### Channels in preload Channels type but not used via invoke in renderer

| Channel | Notes |
|---------|-------|
| `add-media` | Defined in type, not called in renderer |
| `generate-transcript` | Defined in type; generate-transcript.tsx uses direct HTTP to Go server, not IPC |
| `load-tags-by-media-path` | Called via state.tsx context, not direct invoke — handled by typed wrapper |

### ipcRenderer.sendMessage() — 4 channels

| Channel | Files | Disposition |
|---------|-------|-------------|
| `open-external` | command-palette.tsx | Web: `window.open(url, '_blank')` |
| `shutdown` | command-palette.tsx, setup-wizard.tsx | Electron-only no-op |
| `minimize` | command-palette.tsx, hotkey-controller.tsx, setup-wizard.tsx | Electron-only no-op |
| `toggle-fullscreen` | command-palette.tsx, setup-wizard.tsx | Electron-only no-op |

### ipcRenderer.on() — 2 channels

| Channel | Files | Disposition |
|---------|-------|-------------|
| `load-files-batch` | state.tsx | Electron-only (FS scanning); no-op |
| `load-files-done` | state.tsx | Electron-only (FS scanning); no-op |

### Direct preload typed functions

| Function | Files | Disposition |
|----------|-------|-------------|
| `loadMediaFromDB(tags, mode)` | state.tsx | Platform typed wrapper → `/api/media` |
| `loadMediaByDescriptionSearch(desc, tags?, filterMode?)` | state.tsx | Platform typed wrapper → `/api/media/search` |
| `fetchMediaPreview(tag, cache, timeStamp)` | video.tsx, animated-gif.tsx, image.tsx | Platform typed wrapper → `/api/media/preview` |
| `fetchTagPreview(tag)` | tag.tsx | Platform typed wrapper → `/api/tags/preview` |
| `fetchTagCount(tag)` | tag-count.tsx | Platform typed wrapper → `/api/tags/count` |
| `listThumbnails(filePath)` | thumbnails.tsx | Platform typed wrapper → `/api/thumbnails` |
| `regenerateThumbnail(filePath, cache, timeStamp?)` | thumbnails.tsx | Platform typed wrapper → `/api/thumbnails/regenerate` |
| `loadDuplicatesByPath(path)` | duplicates.tsx | Stubbed (duplicate management) |
| `mergeDuplicatesByPath(path)` | duplicates.tsx | Stubbed (duplicate management) |
| `getGifMetadata(filePath)` | animated-gif.tsx | Platform typed wrapper → `/api/media/gif-metadata` |

### Other preload APIs

| API | Files | Disposition |
|-----|-------|-------------|
| `transcript.loadTranscript(filePath)` | transcript.tsx | Stubbed initially |
| `transcript.modifyTranscript(input)` | cue.tsx | Stubbed initially |
| `transcript.checkIfWhisperIsInstalled()` | (not used in renderer) | Omitted |
| `url.format(...)` | image.tsx, video.tsx, audio.tsx, animated-gif.tsx, tag.tsx | `platform.mediaUrl()` |
| `store.get(key, default)` | layout.tsx, metadata.tsx, state.tsx | `platform.store.get()` |
| `store.set(key, value)` | state.tsx, layout.tsx, metadata.tsx | `platform.store.set()` |
| `store.getMany(pairs)` | state.tsx | `platform.store.getMany()` |
| `sessionStore.*` | useSessionStore.ts | `platform.sessionStore.*` |
| `window.appArgs` | state.tsx | `platform.appArgs` |

## Complete File List Requiring Changes

| # | File | Changes |
|---|------|---------|
| 1 | `src/renderer/state.tsx` | invoke, send, store, sessionStore, appArgs, loadMediaFromDB, loadMediaByDescriptionSearch |
| 2 | `src/renderer/hooks/useSessionStore.ts` | sessionStore |
| 3 | `src/renderer/hooks/useTagDrop.tsx` | invoke |
| 4 | `src/renderer/components/media-viewers/image.tsx` | url.format → mediaUrl, fetchMediaPreview |
| 5 | `src/renderer/components/media-viewers/video.tsx` | url.format → mediaUrl, fetchMediaPreview |
| 6 | `src/renderer/components/media-viewers/audio.tsx` | url.format → mediaUrl |
| 7 | `src/renderer/components/media-viewers/animated-gif.tsx` | url.format → mediaUrl, fetchMediaPreview, getGifMetadata |
| 8 | `src/renderer/components/controls/hotkey-controller.tsx` | invoke, send |
| 9 | `src/renderer/components/controls/command-palette.tsx` | invoke, send |
| 10 | `src/renderer/components/controls/setup-wizard.tsx` | send |
| 11 | `src/renderer/components/layout/layout.tsx` | store |
| 12 | `src/renderer/components/metadata/metadata.tsx` | store |
| 13 | `src/renderer/components/metadata/description.tsx` | invoke |
| 14 | `src/renderer/components/metadata/file-metadata.tsx` | invoke |
| 15 | `src/renderer/components/metadata/tags.tsx` | invoke |
| 16 | `src/renderer/components/metadata/thumbnails.tsx` | listThumbnails, regenerateThumbnail |
| 17 | `src/renderer/components/metadata/transcript.tsx` | transcript.loadTranscript |
| 18 | `src/renderer/components/metadata/cue.tsx` | transcript.modifyTranscript |
| 19 | `src/renderer/components/metadata/duplicates.tsx` | loadDuplicatesByPath, mergeDuplicatesByPath |
| 20 | `src/renderer/components/metadata/generate-transcript.tsx` | hardcoded `localhost:8090` → relative URL |
| 21 | `src/renderer/components/taxonomy/taxonomy.tsx` | invoke |
| 22 | `src/renderer/components/taxonomy/tag.tsx` | invoke, url.format → mediaUrl, fetchTagPreview |
| 23 | `src/renderer/components/taxonomy/tag-count.tsx` | fetchTagCount |
| 24 | `src/renderer/components/taxonomy/category.tsx` | invoke |
| 25 | `src/renderer/components/taxonomy/new-tag-modal.tsx` | invoke |
| 26 | `src/renderer/components/taxonomy/new-category-modal.tsx` | invoke |
| 27 | `src/renderer/components/taxonomy/confirm-delete-tag.tsx` | invoke |
| 28 | `src/renderer/components/taxonomy/confirm-delete-category.tsx` | invoke |
| 29 | `src/renderer/components/elo/BattleMode.tsx` | invoke (stubbed) |

### Electron-Only Feature Gating

Components check `capabilities` and hide/disable UI that doesn't apply in web mode:

```typescript
import { capabilities } from '../platform';

// In setup wizard — hide file browsing
{capabilities.fileSystemAccess && <button onClick={selectDirectory}>Browse...</button>}

// In title bar — hide window controls
{capabilities.windowControls && <WindowControls />}
```

## Authentication

The Go media-server already has JWT authentication with cookie-based sessions:

- Login: `POST /auth/login` returns JWT token and sets `auth_token` HttpOnly cookie (365-day expiry)
- Status: `GET /auth/status` checks Bearer token or cookie
- All protected routes check JWT via `authMiddleware`
- CORS allows `http://localhost:1212` with credentials

**Web mode auth flow:**
1. The Loki SPA routes (`/app/*`) are protected by the existing auth middleware
2. If not authenticated, the middleware redirects to `/login`
3. After login, the `auth_token` cookie is set; all `authFetch` calls include it via `credentials: 'include'`
4. 401 responses redirect to `/login`
5. New `/api/*` endpoints are registered as admin-role routes, inheriting existing JWT protection

No new auth code is needed.

## Go Server REST API

### New Endpoints (Initial Set)

All endpoints are prefixed with `/api/` under a new namespace. The Go server already has `/api/stats` and `/api/upload` — no conflicts with the proposed routes.

Handlers follow the existing pattern in `main.go`: factory functions receiving `*Dependencies`.

#### Media

```
POST /api/media
  Body: { tags: string[], mode: string }
  Returns: Item[]
  Notes: Mirrors loadMediaFromDB — queries media_tag_by_category + media tables

POST /api/media/search
  Body: { description: string, tags?: string[], filteringMode?: string }
  Returns: Item[]
  Notes: Mirrors loadMediaByDescriptionSearch — LIKE query on media.description

POST /api/media/metadata
  Body: { path: string }
  Returns: { width: number, height: number, duration?: number }
  Notes: Reads file metadata (ffprobe or image size detection)

POST /api/media/tags
  Body: { path: string }
  Returns: Tag[]
  Notes: Queries media_tag_by_category for given media_path

PUT  /api/media/description
  Body: { path: string, description: string }
  Returns: {}
  Notes: UPDATE media SET description = ? WHERE path = ?

POST /api/media/preview
  Body: { path: string, cache: string, timeStamp: number }
  Returns: Item[] (nearby media items for preview)

POST /api/media/gif-metadata
  Body: { path: string }
  Returns: { frameCount: number, duration: number }

DELETE /api/media/delete
  Body: { path: string }
  Returns: {}
  Notes: Deletes media record and optionally the file
```

#### Taxonomy

```
GET  /api/taxonomy
  Returns: { categories: Category[], tags: Tag[] }
  Notes: Full taxonomy tree — all categories with their tags

POST /api/tags
  Body: { label: string, categoryLabel: string, weight: number }
  Returns: { label: string }

PUT  /api/tags/rename
  Body: { label: string, newLabel: string }
  Returns: {}

PUT  /api/tags/move
  Body: { label: string, categoryLabel: string }
  Returns: {}

DELETE /api/tags
  Body: { label: string }
  Returns: {}

PUT  /api/tags/order
  Body: { labels: string[] }
  Returns: {}

PUT  /api/tags/weight
  Body: { label: string, weight: number }
  Returns: {}

PUT  /api/tags/timestamp
  Body: { mediaPath: string, tagLabel: string, categoryLabel: string, timestamp: number }
  Returns: {}

DELETE /api/tags/timestamp
  Body: { mediaPath: string, tagLabel: string, categoryLabel: string }
  Returns: {}

POST /api/tags/preview
  Body: { label: string }
  Returns: string[] (file paths)

POST /api/tags/count
  Body: { label: string }
  Returns: { count: number }
```

Note: The Go DB schema uses `label TEXT` primary keys for tags and categories (not integer IDs).

#### Categories

```
POST /api/categories
  Body: { label: string, weight: number }
  Returns: { label: string }

PUT  /api/categories/rename
  Body: { label: string, newLabel: string }
  Returns: {}

DELETE /api/categories
  Body: { label: string }
  Returns: {}
```

#### Assignments

```
POST /api/assignments
  Body: { mediaPath: string, tagLabel: string, categoryLabel: string, weight: number }
  Returns: {}

DELETE /api/assignments
  Body: { mediaPath: string, tagLabel: string, categoryLabel: string }
  Returns: {}

PUT  /api/assignments/weight
  Body: { mediaPath: string, tagLabel: string, categoryLabel: string, weight: number }
  Returns: {}
```

#### Thumbnails

```
POST /api/thumbnails
  Body: { path: string }
  Returns: { cache: string, path: string, exists: boolean, size: number }[]

POST /api/thumbnails/regenerate
  Body: { path: string, cache: string, timeStamp?: number }
  Returns: { path: string }
```

#### Settings & Session

```
GET  /api/settings
  Returns: Record<string, any>
  Notes: Server-side key-value store (new table or JSON file)

PUT  /api/settings
  Body: { key: string, value: any }
  Returns: {}

GET  /api/session
  Returns: Record<string, any>

GET  /api/session/{key}
  Returns: any

PUT  /api/session/{key}
  Body: any (JSON value)
  Returns: {}

PUT  /api/session
  Body: Record<string, any> (bulk update)
  Returns: {}

DELETE /api/session
  Returns: {}

DELETE /api/session/keys
  Body: { keys: string[] }
  Returns: {}
```

#### Database

```
POST /api/db/load
  Body: { path: string }
  Returns: {}
  Notes: Switch active database file
```

### Stubbed Channels

These return 501 when called via `invoke` in web mode:

- `update-elo` — battle mode
- `load-duplicates-by-path`, `merge-duplicates-by-path` — duplicate management
- `load-files`, `refresh-library` — file system scanning
- `copy-file-into-clipboard` — clipboard
- `select-db`, `select-file`, `select-directory`, `select-new-path` — file dialogs
- `check-for-updates` — auto-update

Transcript operations are stubbed at the typed wrapper level (not via invoke).

Window control channels (`shutdown`, `minimize`, `toggle-fullscreen`, `set-always-on-top`) are no-ops in `send()`.

### Media File Serving

The existing `/media/file?path=...` endpoint already supports:
- Range requests via `http.ServeFile` (HTTP 206 for video seeking)
- Content-type detection by extension
- ETag caching with `Cache-Control: public, max-age=3600`
- Path traversal protection (absolute paths only)
- Files up to 2GB

Used directly by `platform.mediaUrl()` in web mode. No changes needed.

### Static File Serving & SPA

```go
// Loki SPA static assets
mux.Handle("/app/static/", http.StripPrefix("/app/static/",
    http.FileServer(http.Dir("loki-static"))))

// SPA catch-all: any /app/* path serves index.html
mux.HandleFunc("/app/", spaHandler)  // catch-all via prefix matching
```

The `spaHandler` serves `loki-static/index.html` for any path under `/app/`. This coexists with the existing Go web UI at `/`.

The SPA is accessed at `http://server:8090/app/`.

## Error Handling (Web Mode)

1. **Network failures**: `fetch` rejects → `invoke` throws. Components handle via React Query error states or try/catch in XState actions.
2. **Auth failures (401)**: Redirect to `/login`.
3. **Server errors (500)**: `invoke` throws with status code. Surfaces via existing toast system.
4. **Not implemented (501)**: Stubbed channels throw explicitly. Components gate features via `capabilities`.
5. **Settings cache race**: `store.get()` returns `defaultValue` if the settings cache hasn't loaded yet. The XState boot sequence awaits `_settingsLoaded` before proceeding to ensure cache is warm.

No retry logic — the Go server runs on the same machine.

## Build & Deployment

### Build Steps

```bash
# 1. Build renderer (same as today)
npm run build:renderer

# 2. Copy to Go server static directory
cp -r release/app/dist/renderer/* media-server/loki-static/

# 3. Build Go server
cd media-server && go build -o loki-server .
```

A new npm script `build:web` automates steps 1-2.

### index.html for Web

A separate `index.html` in `media-server/loki-static/` with paths adjusted for `/app/static/`:

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Loki</title>
  <link rel="stylesheet" href="/app/static/renderer.css">
</head>
<body>
  <div id="root"></div>
  <script src="/app/static/renderer.js"></script>
</body>
</html>
```

### Development Workflow

```bash
# Terminal 1: Webpack dev server with HMR (port 1212)
npm run start:renderer

# Terminal 2: Go server proxying /app/* to webpack dev server
cd media-server && go run . --dev-proxy=http://localhost:1212
```

In dev mode (`--dev-proxy` flag), the Go server:
- Serves `/api/*` and `/media/file` requests normally
- Proxies `/app/*` requests to the webpack dev server for HMR
- All other routes serve the existing Go web UI

## Testing Strategy

- **Platform service unit tests**: Mock both Electron and web paths, verify correct dispatch
- **Electron regression**: Run existing app after refactor — must work identically
- **Go API integration tests**: `go test` for each new endpoint against a test SQLite DB
- **Web e2e smoke test**: Start Go server, open `/app/` in browser, verify library loads and media displays
- **argsToBody verification**: During implementation, verify each channelToEndpoint transformer against the actual IPC call site args

## Migration Safety

The refactor is additive:

1. `platform.ts` is a new file
2. Components change import source but not behavior
3. Electron code paths are preserved exactly via delegation
4. Go server gets new routes alongside existing ones — no conflicts
5. No webpack config changes
6. No preload.ts changes

If `window.electron` exists, behavior is identical to today. The web path is entirely new code.
