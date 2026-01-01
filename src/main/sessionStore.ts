/**
 * Session Store - High-performance async storage for ephemeral session data
 *
 * This module handles storage of frequently-changing session state separately
 * from application configuration. It uses async file operations and debounced
 * writes to optimize performance for large data like library state.
 *
 * Key features:
 * - Async file I/O (non-blocking)
 * - Debounced writes to batch rapid updates
 * - In-memory cache for fast reads
 * - Separate file from electron-store config
 * - Atomic writes using temp file + rename pattern
 */

import { app, ipcMain } from 'electron';
import fs from 'fs';
import path from 'path';

// Session data types
export interface SessionLibraryData {
  library: Array<{
    path: string;
    tagLabel?: string;
    mtimeMs: number;
    weight?: number;
    timeStamp?: number;
    elo?: number | null;
    description?: string;
    height?: number | null;
    width?: number | null;
  }>;
  initialFile: string;
}

export interface SessionCursorData {
  cursor: number;
  scrollPosition?: number;
}

export interface SessionQueryData {
  dbQuery: {
    tags: string[];
  };
  mostRecentTag: string;
  mostRecentCategory: string;
  textFilter: string;
}

export interface SessionPreviousData {
  previousLibrary: SessionLibraryData['library'];
  previousCursor: number;
}

export interface SessionData {
  library: SessionLibraryData | null;
  cursor: SessionCursorData | null;
  query: SessionQueryData | null;
  previous: SessionPreviousData | null;
  version: number;
}

const DEFAULT_SESSION_DATA: SessionData = {
  library: null,
  cursor: null,
  query: null,
  previous: null,
  version: 1,
};

class SessionStore {
  private filePath: string;
  private cache: SessionData;
  private writeDebounceTimer: NodeJS.Timeout | null = null;
  private pendingWrites: Partial<SessionData> = {};
  private isInitialized = false;
  private initPromise: Promise<void> | null = null;
  private writeInProgress = false;
  private writeQueue: Array<() => void> = [];

  // Debounce times (ms) - cursor updates are very frequent, so use longer debounce
  private static readonly DEBOUNCE_CURSOR = 100;
  private static readonly DEBOUNCE_LIBRARY = 300;
  private static readonly DEBOUNCE_OTHER = 150;

  constructor() {
    const userDataPath = app.getPath('userData');
    this.filePath = path.join(userDataPath, 'session.json');
    this.cache = { ...DEFAULT_SESSION_DATA };
  }

  /**
   * Initialize the store by loading from disk
   */
  async init(): Promise<void> {
    if (this.isInitialized) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = this._loadFromDisk();
    await this.initPromise;
    this.isInitialized = true;
  }

  private async _loadFromDisk(): Promise<void> {
    try {
      if (fs.existsSync(this.filePath)) {
        const content = await fs.promises.readFile(this.filePath, 'utf-8');
        const parsed = JSON.parse(content);
        this.cache = {
          ...DEFAULT_SESSION_DATA,
          ...parsed,
        };
        console.log('[SessionStore] Loaded session data from disk');
      } else {
        console.log('[SessionStore] No existing session file, using defaults');
        this.cache = { ...DEFAULT_SESSION_DATA };
      }
    } catch (error) {
      console.error('[SessionStore] Failed to load session data:', error);
      this.cache = { ...DEFAULT_SESSION_DATA };
    }
  }

  /**
   * Get a specific key from the session store
   */
  get<K extends keyof SessionData>(key: K): SessionData[K] {
    return this.cache[key];
  }

  /**
   * Get all session data
   */
  getAll(): SessionData {
    return { ...this.cache };
  }

  /**
   * Set a specific key in the session store (debounced write)
   */
  set<K extends keyof SessionData>(key: K, value: SessionData[K]): void {
    // Update cache immediately for fast reads
    this.cache[key] = value;
    this.pendingWrites[key] = value;

    // Determine debounce time based on key
    let debounceTime: number;
    switch (key) {
      case 'cursor':
        debounceTime = SessionStore.DEBOUNCE_CURSOR;
        break;
      case 'library':
        debounceTime = SessionStore.DEBOUNCE_LIBRARY;
        break;
      default:
        debounceTime = SessionStore.DEBOUNCE_OTHER;
    }

    this._scheduleWrite(debounceTime);
  }

  /**
   * Set multiple keys at once
   */
  setMany(updates: Partial<SessionData>): void {
    for (const [key, value] of Object.entries(updates)) {
      this.cache[key as keyof SessionData] = value as any;
      this.pendingWrites[key as keyof SessionData] = value as any;
    }
    this._scheduleWrite(SessionStore.DEBOUNCE_OTHER);
  }

  /**
   * Clear all session data
   */
  clear(): void {
    this.cache = { ...DEFAULT_SESSION_DATA };
    this.pendingWrites = { ...DEFAULT_SESSION_DATA };
    this._scheduleWrite(0); // Immediate write for clear
  }

  /**
   * Clear specific keys
   */
  clearKeys(keys: Array<keyof SessionData>): void {
    for (const key of keys) {
      // Use type-safe approach for clearing keys
      if (key === 'library') {
        this.cache.library = null;
        this.pendingWrites.library = null;
      } else if (key === 'cursor') {
        this.cache.cursor = null;
        this.pendingWrites.cursor = null;
      } else if (key === 'query') {
        this.cache.query = null;
        this.pendingWrites.query = null;
      } else if (key === 'previous') {
        this.cache.previous = null;
        this.pendingWrites.previous = null;
      }
    }
    this._scheduleWrite(SessionStore.DEBOUNCE_OTHER);
  }

  /**
   * Force an immediate write to disk (used on app quit)
   */
  async flush(): Promise<void> {
    if (this.writeDebounceTimer) {
      clearTimeout(this.writeDebounceTimer);
      this.writeDebounceTimer = null;
    }
    await this._writeToDisk();
  }

  private _scheduleWrite(debounceMs: number): void {
    if (this.writeDebounceTimer) {
      clearTimeout(this.writeDebounceTimer);
    }
    this.writeDebounceTimer = setTimeout(() => {
      this._writeToDisk();
    }, debounceMs);
  }

  private async _writeToDisk(): Promise<void> {
    // If a write is in progress, queue this write
    if (this.writeInProgress) {
      return new Promise<void>((resolve) => {
        this.writeQueue.push(resolve);
      });
    }

    this.writeInProgress = true;
    this.pendingWrites = {};

    try {
      const tempPath = this.filePath + '.tmp';
      const data = JSON.stringify(this.cache);

      // Write to temp file first, then rename (atomic operation)
      await fs.promises.writeFile(tempPath, data, 'utf-8');
      await fs.promises.rename(tempPath, this.filePath);
    } catch (error) {
      console.error('[SessionStore] Failed to write session data:', error);
    } finally {
      this.writeInProgress = false;

      // Process queued writes
      if (this.writeQueue.length > 0) {
        const callbacks = this.writeQueue.splice(0);
        callbacks.forEach((cb) => cb());

        // If there are pending writes, schedule another write
        if (Object.keys(this.pendingWrites).length > 0) {
          this._scheduleWrite(SessionStore.DEBOUNCE_OTHER);
        }
      }
    }
  }
}

// Singleton instance
let sessionStore: SessionStore | null = null;

export function getSessionStore(): SessionStore {
  if (!sessionStore) {
    sessionStore = new SessionStore();
  }
  return sessionStore;
}

/**
 * Register IPC handlers for session store
 */
export function registerSessionStoreHandlers(): void {
  const store = getSessionStore();

  // Async get operations
  ipcMain.handle('session-store-get', async (_, key: keyof SessionData) => {
    await store.init();
    return store.get(key);
  });

  ipcMain.handle('session-store-get-all', async () => {
    await store.init();
    return store.getAll();
  });

  // Async set operations
  ipcMain.handle(
    'session-store-set',
    async (_, key: keyof SessionData, value: any) => {
      await store.init();
      store.set(key, value);
    }
  );

  ipcMain.handle(
    'session-store-set-many',
    async (_, updates: Partial<SessionData>) => {
      await store.init();
      store.setMany(updates);
    }
  );

  // Clear operations
  ipcMain.handle('session-store-clear', async () => {
    await store.init();
    store.clear();
  });

  ipcMain.handle(
    'session-store-clear-keys',
    async (_, keys: Array<keyof SessionData>) => {
      await store.init();
      store.clearKeys(keys);
    }
  );

  // Flush (used on beforeunload)
  ipcMain.handle('session-store-flush', async () => {
    await store.init();
    await store.flush();
  });
}

/**
 * Flush session store on app quit
 */
export function setupSessionStoreLifecycle(): void {
  app.on('before-quit', async () => {
    const store = getSessionStore();
    await store.flush();
  });
}
