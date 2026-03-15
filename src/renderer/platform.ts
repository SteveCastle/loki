// Platform abstraction layer
// Detects Electron vs Web and routes calls accordingly

export const isElectron =
  typeof window !== 'undefined' &&
  typeof (window as any).electron !== 'undefined';

export const capabilities = {
  fileSystemAccess: isElectron,
  clipboard: isElectron,
  windowControls: isElectron,
  autoUpdate: isElectron,
  shutdown: isElectron,
};

// ---- Types ----

type ThumbnailCache =
  | 'thumbnail_path_100'
  | 'thumbnail_path_600'
  | 'thumbnail_path_1200';
type ImageCache = 'thumbnail_path_1200' | 'thumbnail_path_600' | false;

interface ThumbnailInfo {
  cache: ThumbnailCache;
  path: string;
  exists: boolean;
  size: number;
}

interface GifMetadata {
  frameCount: number;
  duration: number;
}

// ---- Helpers for web mode ----

function authFetch(url: string, opts: RequestInit = {}): Promise<Response> {
  return fetch(url, { ...opts, credentials: 'include' });
}

async function handleResponse(res: Response): Promise<any> {
  if (res.status === 401) {
    window.location.href = '/login';
    return;
  }
  if (res.status === 501) throw new Error('Not available in web mode');
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const text = await res.text();
  return text ? JSON.parse(text) : undefined;
}

function jsonPost(url: string, body: any): Promise<any> {
  return authFetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handleResponse);
}

function jsonPut(url: string, body: any): Promise<any> {
  return authFetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handleResponse);
}

function jsonDelete(url: string, body: any): Promise<any> {
  return authFetch(url, {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handleResponse);
}

// ---- Channel-to-endpoint mapping (web mode only) ----

interface EndpointMapping {
  url: string;
  method: string;
  argsToBody: (args: unknown[]) => any;
}

function channelToEndpoint(channel: string): EndpointMapping | null {
  const map: Record<string, EndpointMapping> = {
    'load-file-metadata': {
      url: '/api/media/metadata',
      method: 'POST',
      argsToBody: (args) => ({ path: args[0] }),
    },
    'load-tags-by-media-path': {
      url: '/api/media/tags',
      method: 'POST',
      argsToBody: (args) => ({ path: args[0] }),
    },
    'update-description': {
      url: '/api/media/description',
      method: 'PUT',
      argsToBody: (args) => ({ path: args[0], description: args[1] }),
    },
    'delete-file': {
      url: '/api/media/delete',
      method: 'DELETE',
      argsToBody: (args) => ({ path: args[0] }),
    },
    'load-taxonomy': {
      url: '/api/taxonomy',
      method: 'GET',
      argsToBody: () => null,
    },
    'create-tag': {
      url: '/api/tags',
      method: 'POST',
      argsToBody: (args) => ({
        label: args[0],
        categoryLabel: args[1],
        weight: args[2],
      }),
    },
    'rename-tag': {
      url: '/api/tags/rename',
      method: 'PUT',
      argsToBody: (args) => ({ label: args[0], newLabel: args[1] }),
    },
    'move-tag': {
      url: '/api/tags/move',
      method: 'PUT',
      argsToBody: (args) => ({ label: args[0], categoryLabel: args[1] }),
    },
    'delete-tag': {
      url: '/api/tags',
      method: 'DELETE',
      argsToBody: (args) => ({ label: args[0] }),
    },
    'order-tags': {
      url: '/api/tags/order',
      method: 'PUT',
      argsToBody: (args) => ({ labels: args[0] }),
    },
    'update-tag-weight': {
      url: '/api/tags/weight',
      method: 'PUT',
      argsToBody: (args) => ({ label: args[0], weight: args[1] }),
    },
    // update-timestamp args: [mediaPath, tagLabel, oldTimestamp, newTimestamp]
    'update-timestamp': {
      url: '/api/tags/timestamp',
      method: 'PUT',
      argsToBody: (args) => ({
        mediaPath: args[0],
        tagLabel: args[1],
        oldTimestamp: args[2],
        newTimestamp: args[3],
      }),
    },
    // remove-timestamp args: [mediaPath, tagLabel, timestamp]
    'remove-timestamp': {
      url: '/api/tags/timestamp',
      method: 'DELETE',
      argsToBody: (args) => ({
        mediaPath: args[0],
        tagLabel: args[1],
        timestamp: args[2],
      }),
    },
    'create-category': {
      url: '/api/categories',
      method: 'POST',
      argsToBody: (args) => ({ label: args[0], weight: args[1] }),
    },
    'rename-category': {
      url: '/api/categories/rename',
      method: 'PUT',
      argsToBody: (args) => ({ label: args[0], newLabel: args[1] }),
    },
    'delete-category': {
      url: '/api/categories',
      method: 'DELETE',
      argsToBody: (args) => ({ label: args[0] }),
    },
    // create-assignment args: [mediaPaths: string[], tagLabel, categoryLabel, timeStamp, applyTagPreview]
    'create-assignment': {
      url: '/api/assignments',
      method: 'POST',
      argsToBody: (args) => ({
        mediaPaths: args[0],
        tagLabel: args[1],
        categoryLabel: args[2],
        timeStamp: args[3],
        applyTagPreview: args[4],
      }),
    },
    // delete-assignment args: [mediaPath: string, tag: {tag_label, time_stamp}]
    'delete-assignment': {
      url: '/api/assignments',
      method: 'DELETE',
      argsToBody: (args) => ({
        mediaPath: args[0],
        tag: args[1], // object with tag_label and time_stamp
      }),
    },
    // update-assignment-weight args: [mediaPath, tagLabel, weight, mediaTimeStamp?]
    'update-assignment-weight': {
      url: '/api/assignments/weight',
      method: 'PUT',
      argsToBody: (args) => ({
        mediaPath: args[0],
        tagLabel: args[1],
        weight: args[2],
        mediaTimeStamp: args[3],
      }),
    },
    'load-db': {
      url: '/api/db/load',
      method: 'POST',
      argsToBody: (args) => ({ path: args[0] }),
    },
  };
  return map[channel] ?? null;
}

// ---- Core platform functions ----

export let invoke: (channel: string, args?: unknown[]) => Promise<any>;
export let send: (channel: string, args?: unknown[]) => void;
export let on: (
  channel: string,
  callback: (...args: unknown[]) => void
) => () => void;
export let mediaUrl: (path: string, version?: string) => string;

export let appArgs: {
  dbPath?: string;
  filePath?: string;
  appUserData?: string;
};

export let store: {
  get(key: string, defaultValue?: any): any;
  set(key: string, value: any): void;
  getMany(pairs: [string, any][]): Record<string, any>;
};

export let sessionStore: {
  get(key: string): Promise<any>;
  set(key: string, value: any): Promise<void>;
  getAll(): Promise<any>;
  setMany(entries: Record<string, any>): Promise<void>;
  clear(): Promise<void>;
  clearKeys(keys: string[]): Promise<void>;
  flush(): Promise<void>;
};

export let transcript: {
  loadTranscript(filePath: string): Promise<any>;
  modifyTranscript(input: any): Promise<any>;
};

// ---- Typed wrappers ----

export let loadMediaFromDB: (
  tags: string[],
  mode?: string
) => Promise<any[]>;

export let loadMediaByDescriptionSearch: (
  description: string,
  tags?: string[],
  filteringMode?: string
) => Promise<any[]>;

export let fetchMediaPreview: (
  path: string,
  cache: ImageCache,
  timeStamp?: number
) => Promise<string | null>;

export let fetchTagPreview: (tag: string) => Promise<string | null>;

export let fetchTagCount: (tag: string) => Promise<number>;

export let listThumbnails: (
  filePath: string
) => Promise<ThumbnailInfo[]>;

export let regenerateThumbnail: (
  filePath: string,
  cache: ThumbnailCache,
  timeStamp?: number
) => Promise<string>;

export let loadDuplicatesByPath: (path: string) => Promise<any>;

export let mergeDuplicatesByPath: (path: string) => Promise<any>;

export let getGifMetadata: (
  filePath: string
) => Promise<GifMetadata | null>;

// ---- Platform initialization ----

if (isElectron) {
  // Electron mode: delegate to window.electron.*
  invoke = (channel, args) =>
    window.electron.ipcRenderer.invoke(channel as any, args ?? []);
  send = (channel, args) =>
    window.electron.ipcRenderer.sendMessage(channel as any, args ?? []);
  on = (channel, cb) => window.electron.ipcRenderer.on(channel, cb) ?? (() => {});
  mediaUrl = (path, version) =>
    window.electron.url.format({
      protocol: 'gsm',
      pathname: path,
      search: version ? `?v=${version}` : undefined,
    });
  appArgs = (window as any).appArgs ?? {};
  store = window.electron.store;
  sessionStore = window.electron.sessionStore;
  transcript = window.electron.transcript;
  loadMediaFromDB = window.electron.loadMediaFromDB as any;
  loadMediaByDescriptionSearch = window.electron.loadMediaByDescriptionSearch;
  fetchMediaPreview = window.electron.fetchMediaPreview;
  fetchTagPreview = window.electron.fetchTagPreview;
  fetchTagCount = window.electron.fetchTagCount;
  listThumbnails = window.electron.listThumbnails;
  regenerateThumbnail = window.electron.regenerateThumbnail;
  loadDuplicatesByPath = window.electron.loadDuplicatesByPath;
  mergeDuplicatesByPath = window.electron.mergeDuplicatesByPath;
  getGifMetadata = window.electron.getGifMetadata;
} else {
  // Web mode

  const urlParams = new URLSearchParams(window.location.search);
  appArgs = {
    dbPath: urlParams.get('db') ?? undefined,
    filePath: urlParams.get('file') ?? undefined,
  };

  invoke = async (channel, args) => {
    const mapping = channelToEndpoint(channel);
    if (!mapping) throw new Error(`Not implemented in web mode: ${channel}`);
    const body = mapping.argsToBody(args ?? []);
    if (mapping.method === 'GET') {
      return authFetch(mapping.url).then(handleResponse);
    }
    if (mapping.method === 'PUT') return jsonPut(mapping.url, body);
    if (mapping.method === 'DELETE') return jsonDelete(mapping.url, body);
    return jsonPost(mapping.url, body);
  };

  send = (channel, args) => {
    if (
      ['shutdown', 'minimize', 'toggle-fullscreen', 'set-always-on-top'].includes(
        channel
      )
    )
      return;
    if (channel === 'open-external' && args?.[0]) {
      window.open(String(args[0]), '_blank');
      return;
    }
    invoke(channel, args).catch(() => {});
  };

  on = () => () => {};

  mediaUrl = (path, version) => {
    const p = new URLSearchParams({ path });
    if (version) p.set('v', version);
    return `/media/file?${p}`;
  };

  // Settings store: server-backed with in-memory cache
  let _settingsCache: Record<string, any> = {};
  const _settingsLoaded = authFetch('/api/settings')
    .then((r) => r.json())
    .then((data) => {
      _settingsCache = data;
    })
    .catch(() => {});

  // Expose the settings loaded promise for boot sequence
  (window as any).__settingsLoaded = _settingsLoaded;

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
      }).catch(() => {});
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
    .then((r) => r.json())
    .then((data) => {
      _sessionCache = data;
    })
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
      keys.forEach((k) => delete _sessionCache[k]);
      await authFetch('/api/session/keys', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keys }),
      });
    },
    flush: async () => {
      // No-op: web writes are immediate
    },
  };

  transcript = {
    loadTranscript: async () => {
      throw new Error('Transcripts not yet available in web mode');
    },
    modifyTranscript: async () => {
      throw new Error('Transcripts not yet available in web mode');
    },
  };

  loadMediaFromDB = (tags, mode = 'EXCLUSIVE') =>
    jsonPost('/api/media', { tags, mode });

  loadMediaByDescriptionSearch = (description, tags, filteringMode) =>
    jsonPost('/api/media/search', { description, tags, filteringMode });

  fetchMediaPreview = (path, cache, timeStamp) =>
    jsonPost('/api/media/preview', { path, cache, timeStamp });

  fetchTagPreview = (tag) =>
    jsonPost('/api/tags/preview', { label: tag });

  fetchTagCount = (tag) =>
    jsonPost('/api/tags/count', { label: tag }).then((r: any) => r.count);

  listThumbnails = (filePath) =>
    jsonPost('/api/thumbnails', { path: filePath });

  regenerateThumbnail = (filePath, cache, timeStamp) =>
    jsonPost('/api/thumbnails/regenerate', {
      path: filePath,
      cache,
      timeStamp,
    });

  loadDuplicatesByPath = async () => {
    throw new Error('Duplicates not yet available in web mode');
  };

  mergeDuplicatesByPath = async () => {
    throw new Error('Duplicates not yet available in web mode');
  };

  getGifMetadata = (filePath) =>
    jsonPost('/api/media/gif-metadata', { path: filePath });
}
