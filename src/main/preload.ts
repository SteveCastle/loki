import { contextBridge, ipcRenderer, IpcRendererEvent, webUtils } from 'electron';
import * as url from 'url';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { isValidFilePath } from './file-handling';
// Defer transcript module loading until used to speed cold start
let transcriptModule: typeof import('./transcript') | null = null;
async function ensureTranscriptModule() {
  if (!transcriptModule) {
    transcriptModule = await import('./transcript');
  }
  return transcriptModule;
}
import { FilterModeOption } from 'settings';

export type Channels =
  | 'shutdown'
  | 'select-file'
  | 'select-directory'
  | 'load-files'
  | 'load-files-batch'
  | 'load-files-done'
  | 'refresh-library'
  | 'select-new-path'
  | 'select-db'
  | 'load-db'
  | 'open-external'
  | 'show-item-in-folder'
  | 'toggle-fullscreen'
  | 'set-always-on-top'
  | 'add-media'
  | 'record-battle'
  | 'update-description'
  | 'copy-file-into-clipboard'
  | 'load-categories'
  | 'load-category-tags'
  | 'load-all-tags'
  | 'get-tag'
  | 'get-tag-count'
  | 'get-category-count'
  | 'load-file-metadata'
  | 'load-gif-metadata'
  | 'load-tags-by-media-path'
  | 'create-tag'
  // Job-related IPC handlers removed - now handled by external job runner service
  | 'create-category'
  | 'rename-category'
  | 'delete-category'
  | 'rename-tag'
  | 'move-tag'
  | 'order-tags'
  | 'update-tag-description'
  | 'update-category-description'
  | 'update-category-tag-view-mode'
  | 'create-assignment'
  | 'fetch-tag-preview'
  | 'fetch-media-preview'
  | 'update-tag-weight'
  | 'delete-assignment'
  | 'delete-tag'
  | 'update-assignment-weight'
  | 'update-timestamp'
  | 'remove-timestamp'
  | 'generate-transcript'
  | 'modify-transcript'
  | 'delete-file'
  | 'forget-media'
  | 'merge-item-metadata'
  | 'move-media'
  | 'import-files'
  | 'minimize'
  | 'load-duplicates-by-path'
  | 'merge-duplicates-by-path'
  | 'check-for-updates'
  | 'apply-elo-ordering'
  | 'consolidate-tag-files'
  | 'consolidate-category-files'
  | 'log-event'
  | 'find-subtitle'
  | 'open-studio'
  | 'studio-media-saved'
  | 'open-path'
  | 'startup-first-media';

// Renderer -> main error/diagnostics channel. Fire-and-forget; persisted to
// <userData>/app-log.jsonl alongside main-process errors.
export interface RendererLogEntry {
  level?: 'error' | 'warn' | 'info';
  scope: string;
  message: string;
  data?: unknown;
  error?: unknown;
}

const loadMediaFromDB = async (
  tags: string[],
  mode: FilterModeOption = 'EXCLUSIVE'
) => {
  const files = await ipcRenderer.invoke('load-media-by-tags', [tags, mode]);
  return files;
};

const loadMediaByDescriptionSearch = async (
  description: string,
  tags?: string[],
  filteringMode?: string
) => {
  const files = await ipcRenderer.invoke('load-media-by-description-search', [
    description,
    tags,
    filteringMode,
  ]);
  return files;
};

const loadMediaByQuery = async (predicates: unknown[], mode: string) => {
  const files = await ipcRenderer.invoke('load-media-by-query', [predicates, mode]);
  return files;
};

const fetchTagPreview = async (tag: string) => {
  const results = await ipcRenderer.invoke('fetch-tag-preview', [tag]);
  if (!results) return null;
  return results;
};

const fetchTagCount = async (tag: string) => {
  const count = await ipcRenderer.invoke('get-tag-count', [tag]);
  return count;
};

const fetchMediaPreview = async (
  tag: string,
  cache: string,
  timeStamp: number
) => {
  const results = await ipcRenderer.invoke('fetch-media-preview', [
    tag,
    cache,
    timeStamp,
  ]);
  if (!results) return null;
  return results;
};

const listThumbnails = async (filePath: string) => {
  const results = await ipcRenderer.invoke('list-thumbnails', [filePath]);
  return results as {
    cache: 'thumbnail_path_100' | 'thumbnail_path_600' | 'thumbnail_path_1200';
    path: string;
    exists: boolean;
    size: number;
  }[];
};

const regenerateThumbnail = async (
  filePath: string,
  cache: 'thumbnail_path_100' | 'thumbnail_path_600' | 'thumbnail_path_1200',
  timeStamp?: number
) => {
  const result = await ipcRenderer.invoke('regenerate-thumbnail', [
    filePath,
    cache,
    timeStamp || 0,
  ]);
  return result as string;
};

const loadDuplicatesByPath = async (path: string) => {
  const files = await ipcRenderer.invoke('load-duplicates-by-path', [path]);
  return files;
};

const mergeDuplicatesByPath = async (path: string) => {
  const result = await ipcRenderer.invoke('merge-duplicates-by-path', [path]);
  return result as {
    mergedInto: string;
    deleted: string[];
    copiedTags: number;
  };
};

const getGifMetadata = async (filePath: string) => {
  const result = await ipcRenderer.invoke('load-gif-metadata', [filePath]);
  return result as { frameCount: number; duration: number } | null;
};

// Base URL of the local Lowkey Media Server. The server's port is
// configurable (config.json "port" / LOWKEY_PORT env), so discover it the
// same way lokictl does: LOWKEY_PORT env > the server's own config.json >
// the server's compiled-in default (10111, "L0K1"). Resolved once at preload
// time — the renderer reads window.electron.mediaServerBase synchronously.
const DEFAULT_MEDIA_SERVER_PORT = 10111;

function mediaServerConfigPath(): string {
  // Mirrors the Go server's platform.GetDataDir() per OS
  // (AppName "lowkey-media-viewer" / AppDisplayName "Lowkey Media Viewer").
  switch (process.platform) {
    case 'win32':
      return process.env.APPDATA
        ? path.join(process.env.APPDATA, 'Lowkey Media Viewer', 'config.json')
        : path.join(os.homedir(), '.lowkey-media-viewer', 'config.json');
    case 'darwin':
      return path.join(
        os.homedir(),
        'Library',
        'Application Support',
        'Lowkey Media Viewer',
        'config.json'
      );
    default:
      return path.join(
        process.env.XDG_DATA_HOME ||
          path.join(os.homedir(), '.local', 'share'),
        'lowkey-media-viewer',
        'config.json'
      );
  }
}

function detectMediaServerBase(): string {
  let port = 0;
  const envPort = parseInt(process.env.LOWKEY_PORT || '', 10);
  if (envPort > 0 && envPort <= 65535) {
    port = envPort;
  } else {
    try {
      const cfg = JSON.parse(fs.readFileSync(mediaServerConfigPath(), 'utf8'));
      if (
        typeof cfg.port === 'number' &&
        cfg.port > 0 &&
        cfg.port <= 65535
      ) {
        port = cfg.port;
      }
    } catch {
      // no server config readable — fall through to the default
    }
  }
  return `http://localhost:${port || DEFAULT_MEDIA_SERVER_PORT}`;
}

const captureRegion = async (rect: {
  x: number;
  y: number;
  width: number;
  height: number;
}): Promise<Uint8Array | null> => {
  const png = await ipcRenderer.invoke('capture-region', [rect]);
  return png ? new Uint8Array(png) : null;
};

contextBridge.exposeInMainWorld('electron', {
  getPathForFile: (file: File) => webUtils.getPathForFile(file),
  // Local media-server base URL with the configured port baked in.
  mediaServerBase: detectMediaServerBase(),
  // Whether the media server appears to be INSTALLED on this machine: its
  // config.json exists (the Go server writes it on first run), or the user
  // points at one explicitly via LOWKEY_PORT. Lets the renderer tell
  // "not installed" apart from "installed but not running / unreachable".
  mediaServerConfigured: (() => {
    if (process.env.LOWKEY_PORT) return true;
    try {
      return fs.existsSync(mediaServerConfigPath());
    } catch {
      return false;
    }
  })(),
  // Forward renderer errors/load failures to the main-process file logger.
  logEvent: (entry: RendererLogEntry) => {
    try {
      ipcRenderer.send('log-event', entry);
    } catch {
      // never let logging throw
    }
  },
  loadMediaFromDB,
  loadMediaByDescriptionSearch,
  loadMediaByQuery,
  fetchTagPreview,
  fetchTagCount,
  fetchMediaPreview,
  listThumbnails,
  regenerateThumbnail,
  loadDuplicatesByPath,
  mergeDuplicatesByPath,
  getGifMetadata,
  captureRegion,
  async loadTranscript(filePath: string) {
    const mod = await ensureTranscriptModule();
    return mod.loadTranscript(filePath);
  },
  async modifyTranscript(input: any) {
    const mod = await ensureTranscriptModule();
    return mod.modifyTranscript(input);
  },
  async deleteTranscriptCue(input: any) {
    const mod = await ensureTranscriptModule();
    return mod.deleteTranscriptCue(input);
  },
  async insertTranscriptCue(input: any) {
    const mod = await ensureTranscriptModule();
    return mod.insertTranscriptCue(input);
  },
  userHome: path.join(process.env.HOME || '', '.lowkey', 'dream.sqlite'),
  // Config store (synchronous, for settings/config that rarely change)
  store: {
    get(key: string, defaultValue: any) {
      return ipcRenderer.sendSync('electron-store-get', key, defaultValue);
    },
    set(property: string, val: any) {
      ipcRenderer.send('electron-store-set', property, val);
    },
    getMany(pairs: [string, any][]) {
      return ipcRenderer.sendSync('electron-store-get-many', pairs);
    },
  },
  // Session store (async, for frequently-changing ephemeral data like library state)
  sessionStore: {
    async get(key: 'library' | 'cursor' | 'query' | 'previous') {
      return ipcRenderer.invoke('session-store-get', key);
    },
    async getAll() {
      return ipcRenderer.invoke('session-store-get-all');
    },
    async set(key: 'library' | 'cursor' | 'query' | 'previous', value: any) {
      return ipcRenderer.invoke('session-store-set', key, value);
    },
    async setMany(updates: Record<string, any>) {
      return ipcRenderer.invoke('session-store-set-many', updates);
    },
    async clear() {
      return ipcRenderer.invoke('session-store-clear');
    },
    async clearKeys(keys: Array<'library' | 'cursor' | 'query' | 'previous'>) {
      return ipcRenderer.invoke('session-store-clear-keys', keys);
    },
    async flush() {
      return ipcRenderer.invoke('session-store-flush');
    },
  },
  url: {
    format: url.format,
  },
  ipcRenderer: {
    sendMessage(channel: Channels, args: unknown[]) {
      ipcRenderer.send(channel, args);
    },
    invoke(channel: Channels, args: unknown[]) {
      return ipcRenderer.invoke(channel, args);
    },
    on(channel: Channels, func: (...args: unknown[]) => void) {
      const subscription = (_event: IpcRendererEvent, ...args: unknown[]) =>
        func(...args);
      ipcRenderer.on(channel, subscription);

      return () => ipcRenderer.removeListener(channel, subscription);
    },
    once(channel: Channels, func: (...args: unknown[]) => void) {
      ipcRenderer.once(channel, (_event, ...args) => func(...args));
    },
    removeListener(channel: Channels, func: (...args: unknown[]) => void) {
      ipcRenderer.removeListener(channel, func);
    },
  },
  transcript: {
    async loadTranscript(filePath: string) {
      const mod = await ensureTranscriptModule();
      return mod.loadTranscript(filePath);
    },
    async modifyTranscript(input: any) {
      const mod = await ensureTranscriptModule();
      return mod.modifyTranscript(input);
    },
    async deleteTranscriptCue(input: any) {
      const mod = await ensureTranscriptModule();
      return mod.deleteTranscriptCue(input);
    },
    async insertTranscriptCue(input: any) {
      const mod = await ensureTranscriptModule();
      return mod.insertTranscriptCue(input);
    },
    async checkIfWhisperIsInstalled() {
      const mod = await ensureTranscriptModule();
      return mod.checkIfWhisperIsInstalled();
    },
  },
});

// Media types worth pulling into Chromium's cache before the renderer boots
// (see warmInitialMedia). Videos are excluded on purpose: the <video> element
// range-requests what it needs, and prefetching a multi-GB file would fight it
// for bandwidth instead of helping.
const WARMABLE_IMAGE_RE = /\.(jpe?g|jfif|pjpe?g|pjp|png|gif|webp|avif|bmp|svg)$/i;

// Kick off the fetch for the file the user opened *before* the renderer bundle
// has even parsed. The gsm:// handler lives in the main process, which is busy
// with DB init and the directory scan by the time React mounts, so starting the
// read here overlaps disk I/O + decode with the whole boot sequence. The <img>
// React renders later hits a warm cache entry instead of a cold file read.
function warmInitialMedia(filePath: string, bootId: string) {
  const trace = (outcome: string, data?: Record<string, unknown>) =>
    ipcRenderer.send('log-event', {
      level: 'info',
      scope: 'startup',
      message: 'media-warm',
      data: {
        bootId,
        outcome,
        at: Math.round(performance.now()),
        ...(data ?? {}),
      },
    });

  if (!filePath) return trace('no-file');
  if (!WARMABLE_IMAGE_RE.test(filePath)) return trace('not-warmable');
  try {
    const href = url.format({ protocol: 'gsm', pathname: filePath });
    const started = performance.now();
    // A detached Image() — not fetch() — because it populates the renderer's
    // *decoded* image cache, which is exactly what the <img> React mounts later
    // reads from. It never enters the document, so it renders nothing itself.
    // Failures are irrelevant: the real element re-requests and surfaces its
    // own error state.
    const warm = new Image();
    // Both outcomes are logged: the whole point of this optimisation is that
    // the decode finishes before React mounts the real element, and the only
    // way to know whether it did is to record when it landed.
    warm.onload = () =>
      trace('loaded', {
        elapsedMs: Math.round(performance.now() - started),
        width: warm.naturalWidth,
        height: warm.naturalHeight,
      });
    warm.onerror = () =>
      trace('failed', { elapsedMs: Math.round(performance.now() - started) });
    warm.src = href;
    trace('started');
  } catch (err) {
    // Never let a warm-up break startup — but do say so, otherwise a silently
    // dead optimisation looks exactly like a working one.
    trace('threw', { error: String((err as Error)?.message ?? err) });
  }
}

// Get the electron main process args from ipc and expose to mainWorld.
//
// Synchronous by design: the renderer reads `window.appArgs` at module-eval
// time (see platform.ts / getInitialContext), so an async exposure raced the
// bundle — losing that race meant the double-clicked file silently didn't open.
// One sendSync at document-start costs ~1ms and removes the race entirely.
function loadMainArgs() {
  let argv: string[] = process.argv;
  let macPath = '';
  let appUserData = '';
  let bootId = 'unknown';
  let appVersion = 'unknown';
  const askedAt = performance.now();
  try {
    const boot = ipcRenderer.sendSync('get-boot-args') as {
      argv: string[];
      macPath: string;
      appUserData: string;
      bootId: string;
      appVersion: string;
    };
    argv = boot?.argv ?? process.argv;
    macPath = boot?.macPath ?? '';
    appUserData = boot?.appUserData ?? '';
    bootId = boot?.bootId ?? 'unknown';
    appVersion = boot?.appVersion ?? 'unknown';
  } catch {
    // Fall through with defaults; the renderer's own error paths cover it.
  }
  const filePath = isValidFilePath(argv[1]) ? argv[1] : macPath;
  contextBridge.exposeInMainWorld('appArgs', {
    filePath,
    appUserData,
    dbPath: path.join(appUserData, 'dream.sqlite'),
    allArgs: process.argv,
    // Carried through so the renderer's marks join the main process timeline.
    bootId,
    appVersion,
  });
  // The earliest timestamp anything in the renderer process can take, and the
  // cost of the one synchronous IPC the preload makes.
  ipcRenderer.send('log-event', {
    level: 'info',
    scope: 'startup',
    message: 'preload',
    data: {
      bootId,
      at: Math.round(performance.now()),
      bootArgsMs: Math.round(performance.now() - askedAt),
      resolvedFile: filePath,
      // If this is false on a launch that was supposed to open a file, the
      // argv plumbing is broken and nothing downstream will make sense.
      hasFile: !!filePath,
    },
  });
  warmInitialMedia(filePath, bootId);
}
loadMainArgs();
