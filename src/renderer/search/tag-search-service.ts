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
// taxonomy fallback.
const FUSE_OPTIONS = {
  keys: [
    { name: 'label', weight: 2 },
    { name: 'category', weight: 1 },
  ],
  threshold: 0.4,
  ignoreLocation: true,
  minMatchCharLength: 1,
} as const;

type ResultCb = (items: TagConcept[]) => void;

// Module-level singleton state.
let worker: Worker | null = null;
// True once we've attempted (and failed) to construct a worker, so we know to
// use the synchronous Fuse fallback instead.
let useFallback = false;
// Whether worker setup (or fallback) has been attempted at least once.
let initialized = false;
// The last indexed tag array, kept by reference so indexTags can no-op when the
// same reference is passed again.
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
    worker = new Worker(
      new URL('../components/taxonomy/tag-search.worker.ts', import.meta.url)
    );
  } catch {
    worker = null;
    useFallback = true;
    return;
  }
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

/**
 * (Re)index the shared tag set. Idempotent: if the same array reference is
 * passed again it is ignored. Posts the index to the worker, or rebuilds the
 * synchronous Fuse instance when running without a worker.
 */
export function indexTags(tags: TagConcept[]): void {
  ensureInit();
  if (tags === indexedTags) return;
  indexedTags = tags;
  if (worker) {
    worker.postMessage({ type: 'index', tags });
  } else if (useFallback) {
    buildFallbackFuse(tags);
  }
}

/**
 * Run a search. The callback fires with the ranked results for THIS call.
 * Multiple concurrent callers are routed by an internal incrementing id so
 * they don't clobber each other. An empty query resolves to [] synchronously.
 */
export function searchTags(
  query: string,
  limit: number,
  cb: ResultCb
): void {
  ensureInit();
  if (!query) {
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
