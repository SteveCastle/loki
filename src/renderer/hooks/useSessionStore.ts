/**
 * Session Store Hook - High-performance access to session storage
 *
 * This hook provides a cached, debounced interface to the session store
 * for storing frequently-changing ephemeral data like library state,
 * cursor position, and query state.
 *
 * Key features:
 * - In-memory cache for instant synchronous reads
 * - Debounced async writes for performance
 * - Type-safe API
 * - Batch write support
 */

import { Item } from '../state';

// Session data types (mirroring main process types)
export interface SessionLibraryData {
  library: Item[];
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
  previousLibrary: Item[];
  previousCursor: number;
}

export interface SessionData {
  library: SessionLibraryData | null;
  cursor: SessionCursorData | null;
  query: SessionQueryData | null;
  previous: SessionPreviousData | null;
}

type SessionKey = keyof SessionData;

// In-memory cache for synchronous reads
const cache: SessionData = {
  library: null,
  cursor: null,
  query: null,
  previous: null,
};

// Track if cache has been initialized
let cacheInitialized = false;
let initPromise: Promise<void> | null = null;

// Debounce timers for each key
const debounceTimers: Partial<Record<SessionKey, NodeJS.Timeout>> = {};

// Debounce durations (ms)
const DEBOUNCE_TIMES: Record<SessionKey, number> = {
  cursor: 100, // Very frequent updates
  library: 300, // Large data, less frequent
  query: 150, // Medium frequency
  previous: 300, // Large data, infrequent
};

/**
 * Initialize the session store cache from disk
 * Call this early in app startup
 */
export async function initSessionStore(): Promise<SessionData> {
  if (cacheInitialized) {
    return cache;
  }

  if (initPromise) {
    await initPromise;
    return cache;
  }

  initPromise = (async () => {
    try {
      const data = await window.electron.sessionStore.getAll();
      if (data) {
        cache.library = data.library ?? null;
        cache.cursor = data.cursor ?? null;
        cache.query = data.query ?? null;
        cache.previous = data.previous ?? null;
      }
      cacheInitialized = true;
    } catch (error) {
      console.error('[SessionStore] Failed to initialize cache:', error);
      cacheInitialized = true; // Mark as initialized to prevent infinite retries
    }
  })();

  await initPromise;
  return cache;
}

/**
 * Get a value from the session store (synchronous, uses cache)
 * Returns null if not yet initialized
 */
export function getSessionValue<K extends SessionKey>(key: K): SessionData[K] {
  return cache[key];
}

/**
 * Get all session data (synchronous, uses cache)
 */
export function getSessionData(): SessionData {
  return { ...cache };
}

/**
 * Check if session has persisted library data
 */
export function hasPersistedLibrary(): boolean {
  const data = cache.library;
  return !!(
    data &&
    data.library &&
    Array.isArray(data.library) &&
    data.library.length > 0
  );
}

/**
 * Check if session has persisted text filter
 */
export function hasPersistedTextFilter(): boolean {
  const data = cache.query;
  return !!(data && data.textFilter && data.textFilter.length > 0);
}

/**
 * Check if session has persisted tags
 */
export function hasPersistedTags(): boolean {
  const data = cache.query;
  return !!(
    data &&
    data.dbQuery &&
    data.dbQuery.tags &&
    data.dbQuery.tags.length > 0
  );
}

/**
 * Set a value in the session store (debounced async write)
 */
export function setSessionValue<K extends SessionKey>(
  key: K,
  value: SessionData[K]
): void {
  // Update cache immediately for fast reads
  cache[key] = value;

  // Clear existing timer for this key
  if (debounceTimers[key]) {
    clearTimeout(debounceTimers[key]);
  }

  // Schedule debounced write
  debounceTimers[key] = setTimeout(() => {
    window.electron.sessionStore.set(key, value).catch((error) => {
      console.error(`[SessionStore] Failed to write ${key}:`, error);
    });
    delete debounceTimers[key];
  }, DEBOUNCE_TIMES[key]);
}

/**
 * Set multiple values at once (debounced)
 */
export function setSessionValues(updates: Partial<SessionData>): void {
  // Update cache immediately
  for (const [key, value] of Object.entries(updates)) {
    cache[key as SessionKey] = value as any;
  }

  // Use the longest debounce time for batched updates
  const maxDebounce = Math.max(
    ...Object.keys(updates).map((k) => DEBOUNCE_TIMES[k as SessionKey] || 150)
  );

  // Clear any existing timers for the keys being updated
  for (const key of Object.keys(updates)) {
    if (debounceTimers[key as SessionKey]) {
      clearTimeout(debounceTimers[key as SessionKey]);
      delete debounceTimers[key as SessionKey];
    }
  }

  // Use a combined timer key
  const batchKey = 'batch' as any;
  if (debounceTimers[batchKey]) {
    clearTimeout(debounceTimers[batchKey]);
  }

  debounceTimers[batchKey] = setTimeout(() => {
    window.electron.sessionStore.setMany(updates).catch((error) => {
      console.error('[SessionStore] Failed to write batch:', error);
    });
    delete debounceTimers[batchKey];
  }, maxDebounce);
}

/**
 * Clear all session data
 */
export function clearSession(): void {
  cache.library = null;
  cache.cursor = null;
  cache.query = null;
  cache.previous = null;

  // Clear all pending timers
  for (const timer of Object.values(debounceTimers)) {
    if (timer) clearTimeout(timer);
  }

  window.electron.sessionStore.clear().catch((error) => {
    console.error('[SessionStore] Failed to clear:', error);
  });
}

/**
 * Clear specific session keys
 */
export function clearSessionKeys(keys: SessionKey[]): void {
  for (const key of keys) {
    cache[key] = null;
    if (debounceTimers[key]) {
      clearTimeout(debounceTimers[key]);
      delete debounceTimers[key];
    }
  }

  window.electron.sessionStore.clearKeys(keys).catch((error) => {
    console.error('[SessionStore] Failed to clear keys:', error);
  });
}

/**
 * Force flush all pending writes (call on beforeunload)
 */
export async function flushSession(): Promise<void> {
  // Clear all debounce timers and write immediately
  for (const [key, timer] of Object.entries(debounceTimers)) {
    if (timer) clearTimeout(timer);
    delete debounceTimers[key as SessionKey];
  }

  // Write current cache state
  await window.electron.sessionStore.setMany(cache);
  await window.electron.sessionStore.flush();
}

// Convenience functions for specific data types

/**
 * Update just the cursor position (high-frequency operation)
 */
export function updateCursor(cursor: number, scrollPosition?: number): void {
  setSessionValue('cursor', { cursor, scrollPosition });
}

/**
 * Update the library data
 */
export function updateLibrary(library: Item[], initialFile: string): void {
  setSessionValue('library', { library, initialFile });
}

/**
 * Update query state
 */
export function updateQuery(
  dbQuery: { tags: string[] },
  mostRecentTag: string,
  mostRecentCategory: string,
  textFilter: string
): void {
  setSessionValue('query', {
    dbQuery,
    mostRecentTag,
    mostRecentCategory,
    textFilter,
  });
}

/**
 * Update previous library state (for back navigation)
 */
export function updatePrevious(
  previousLibrary: Item[],
  previousCursor: number
): void {
  setSessionValue('previous', { previousLibrary, previousCursor });
}
