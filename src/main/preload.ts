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
  | 'import-files'
  | 'minimize'
  | 'load-duplicates-by-path'
  | 'merge-duplicates-by-path'
  | 'check-for-updates'
  | 'apply-elo-ordering'
  | 'consolidate-tag-files'
  | 'consolidate-category-files'
  | 'log-event'
  | 'find-subtitle';

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

// Get the electron main process args from ipc and expose to mainWorld.
async function loadMainArgs() {
  const mainProcessArgs = await ipcRenderer.invoke('get-main-args');
  const appUserData = await ipcRenderer.invoke('get-user-data-path');
  const macPath = await ipcRenderer.invoke('get-mac-path');
  const filePath = isValidFilePath(mainProcessArgs[1])
    ? mainProcessArgs[1]
    : macPath;
  contextBridge.exposeInMainWorld('appArgs', {
    filePath,
    appUserData,
    dbPath: path.join(appUserData, 'dream.sqlite'),
    allArgs: process.argv,
  });
}
loadMainArgs();
