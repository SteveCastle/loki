// tag-search-service — a MODULE-LEVEL singleton wrapping ONE shared tag-search
// worker (plus a synchronous Fuse fallback) for the whole app.
//
// Both the taxonomy sidebar (via useTagSearch) and the command palette (via
// useTagSearch) route through this single index, which is warmed at startup by
// useWarmTagSearch so the first search from either surface is instant.
//
// The worker is constructed lazily on first use and lives for the app lifetime
// — it is NEVER terminated. Under environments without Web Worker support
// (e.g. jsdom in tests) construction is guarded and we fall back to a
// synchronous Fuse search with the SAME options as the worker.
import Fuse from 'fuse.js';

export interface TagConcept {
  label: string;
  category: string;
  weight: number;
  description?: string;
}

// Keep these Fuse options in sync with tag-search.worker.ts and the historical
// taxonomy fallback. (Not `as const` — Fuse's option types are mutable, and the
// readonly form trips ts-jest's stricter type check.)
const FUSE_OPTIONS = {
  keys: [
    { name: 'label', weight: 2 },
    { name: 'category', weight: 1 },
  ],
  threshold: 0.4,
  ignoreLocation: true,
  minMatchCharLength: 1,
};

// Don't dispatch a fuzzy search until the query is at least this long. A 1-char
// query clears Fuse's threshold against ~every entry in a 175K-tag library, so
// it both pegs the search and floods the result list with noise. Two characters
// is the standard type-ahead floor and removes that pathological case.
const MIN_QUERY_LENGTH = 2;

type ResultCb = (items: TagConcept[]) => void;

// Module-level singleton state.
let worker: Worker | null = null;
// True once we've attempted (and failed) to construct a worker, so we know to
// use the synchronous Fuse fallback instead.
let useFallback = false;
// Whether worker setup (or fallback) has been attempted at least once.
let initialized = false;
// The raw source array last handed to indexTags, kept by reference so indexing
// is a no-op when the same reference comes back. All consumers (the startup
// warmer, the taxonomy sidebar, the command palette) read the SAME React Query
// cache entry, so they pass an identical reference — meaning the full library is
// structured-cloned to the worker exactly once, not re-cloned on each surface's
// first search.
let indexedSource: TagConcept[] | null = null;
// The cleaned, indexed tag array (label-less rows dropped). Backs the fallback
// Fuse instance and isWarm().
let indexedTags: TagConcept[] | null = null;
// Synchronous Fuse instance — only built when running without a worker.
let fallbackFuse: Fuse<TagConcept> | null = null;

// Pending search callbacks keyed by request id so concurrent callers (taxonomy
// + palette) don't clobber each other.
let seq = 0;
const pending: Record<number, ResultCb> = {};

function buildFallbackFuse(tags: TagConcept[]): void {
  fallbackFuse = new Fuse(tags, FUSE_OPTIONS);
}

// Lazily set up the shared worker (or mark the fallback). Idempotent.
function ensureInit(): void {
  if (initialized) return;
  initialized = true;

  if (typeof Worker === 'undefined') {
    useFallback = true;
    return;
  }
  try {
    // Lazy require: the factory carries the `import.meta` worker URL, which the
    // CommonJS test transform can't parse. We only reach here when a real Worker
    // exists (never under jsdom), so the test graph never loads that module.
    // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
    const { createTagSearchWorker } = require('./tag-search-worker-factory');
    worker = createTagSearchWorker();
  } catch {
    worker = null;
    useFallback = true;
    return;
  }
  if (worker) {
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg?.type === 'result') {
        const cb = pending[msg.id];
        if (cb) {
          delete pending[msg.id];
          cb((msg.items as TagConcept[]) ?? []);
        }
      }
    };
  }
}

/**
 * (Re)index the shared tag set. Idempotent on the SOURCE reference: passing the
 * same array again is a no-op, so consumers can hand over the raw, shared React
 * Query array and the full library is cloned to the worker only when the data
 * actually changes — not once per consumer. Returns whether a (re)index ran.
 *
 * The defensive label filter (Fuse and the row renderers assume a non-empty
 * label) lives here so callers don't each spin up a divergent filtered array,
 * which is what previously defeated the reference guard.
 */
export function indexTags(tags: TagConcept[]): boolean {
  ensureInit();
  if (tags === indexedSource) return false;
  indexedSource = tags;
  const clean = tags.filter(
    (t) => t && typeof t.label === 'string' && t.label.length > 0
  );
  indexedTags = clean;
  if (worker) {
    worker.postMessage({ type: 'index', tags: clean });
  } else if (useFallback) {
    buildFallbackFuse(clean);
  }
  return true;
}

/**
 * Run a search. The callback fires with the ranked results for THIS call.
 * Multiple concurrent callers are routed by an internal incrementing id so
 * they don't clobber each other. A query shorter than MIN_QUERY_LENGTH (empty
 * included) resolves to [] synchronously without touching the index.
 */
export function searchTags(
  query: string,
  limit: number,
  cb: ResultCb
): void {
  ensureInit();
  if (!query || query.length < MIN_QUERY_LENGTH) {
    cb([]);
    return;
  }
  const id = (seq += 1);
  if (worker) {
    pending[id] = cb;
    worker.postMessage({ type: 'search', id, query, limit });
    return;
  }
  // Synchronous fallback.
  if (!fallbackFuse) {
    buildFallbackFuse(indexedTags ?? []);
  }
  const items = fallbackFuse
    ? fallbackFuse.search(query, { limit }).map((r) => r.item)
    : [];
  cb(items);
}

/**
 * True once the worker (or fallback) is set up and tags have been indexed.
 */
export function isWarm(): boolean {
  return initialized && indexedTags !== null;
}
