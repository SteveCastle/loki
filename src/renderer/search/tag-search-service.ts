// tag-search-service — a MODULE-LEVEL singleton wrapping ONE shared tag-search
// worker (plus a synchronous Fuse fallback) for the whole app.
//
// The worker is shared; the INDEXES inside it are not. Each index is keyed by a
// scope (see tag-scopes.ts) because the app's two search surfaces want
// different slices of the tag table:
//
//   'curated' — the command palette. Everything except the autotagger's
//               "Suggested" bucket: ~5.6K entries.
//   'all'     — the taxonomy sidebar. The complete table: ~189K entries.
//
// They used to share one index of everything, which is what made the palette
// laggy: every keystroke ran a fuzzy match over 189K entries, and the palette
// couldn't open without that index existing. Splitting them means the palette's
// index is ~34x smaller and is built from a ~34x smaller fetch.
//
// The worker is constructed lazily on first use and lives for the app lifetime
// — it is NEVER terminated. Under environments without Web Worker support
// (e.g. jsdom in tests) construction is guarded and we fall back to a
// synchronous Fuse search with the SAME options as the worker.
import Fuse from 'fuse.js';
import { DEFAULT_TAG_SCOPE, type TagScope } from './tag-scopes';

export interface TagConcept {
  label: string;
  category: string;
  weight: number;
  description?: string;
}

// Keep these Fuse options in sync with tag-search.worker.ts. (Not `as const` —
// Fuse's option types are mutable, and the readonly form trips ts-jest's
// stricter type check.)
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
// query clears Fuse's threshold against ~every entry in a large tag library, so
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

// Per-scope state. `sources` holds the raw array last handed to indexTags for a
// scope, kept by reference so re-indexing the same data is a no-op. Every
// consumer of a given scope reads the SAME React Query cache entry, so they
// pass an identical reference — meaning each scope is structured-cloned to the
// worker exactly once, not re-cloned on each surface's first search.
const sources = new Map<TagScope, TagConcept[]>();
// The cleaned, indexed arrays (label-less rows dropped). Back the fallback Fuse
// instances and isWarm().
const indexed = new Map<TagScope, TagConcept[]>();
// Synchronous Fuse instances — only built when running without a worker.
const fallbackFuses = new Map<TagScope, Fuse<TagConcept>>();

// Pending search callbacks keyed by request id so concurrent callers (taxonomy
// + palette) don't clobber each other.
let seq = 0;
const pending: Record<number, ResultCb> = {};

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
 * (Re)index one scope's tag set. Idempotent on the SOURCE reference: passing the
 * same array again is a no-op, so consumers can hand over the raw, shared React
 * Query array and a scope is cloned to the worker only when its data actually
 * changes — not once per consumer. Returns whether a (re)index ran.
 *
 * The defensive label filter (Fuse and the row renderers assume a non-empty
 * label) lives here so callers don't each spin up a divergent filtered array,
 * which is what previously defeated the reference guard.
 */
export function indexTags(
  tags: TagConcept[],
  scope: TagScope = DEFAULT_TAG_SCOPE
): boolean {
  ensureInit();
  if (tags === sources.get(scope)) return false;
  sources.set(scope, tags);
  const clean = tags.filter(
    (t) => t && typeof t.label === 'string' && t.label.length > 0
  );
  indexed.set(scope, clean);
  if (worker) {
    worker.postMessage({ type: 'index', scope, tags: clean });
  } else if (useFallback) {
    fallbackFuses.set(scope, new Fuse(clean, FUSE_OPTIONS));
  }
  return true;
}

/**
 * Run a search against one scope. The callback fires with the ranked results
 * for THIS call. Multiple concurrent callers are routed by an internal
 * incrementing id so they don't clobber each other. A query shorter than
 * MIN_QUERY_LENGTH (empty included) resolves to [] synchronously without
 * touching the index.
 */
export function searchTags(
  query: string,
  limit: number,
  cb: ResultCb,
  scope: TagScope = DEFAULT_TAG_SCOPE
): void {
  ensureInit();
  if (!query || query.length < MIN_QUERY_LENGTH) {
    cb([]);
    return;
  }
  const id = (seq += 1);
  if (worker) {
    pending[id] = cb;
    worker.postMessage({ type: 'search', scope, id, query, limit });
    return;
  }
  // Synchronous fallback.
  let fuse = fallbackFuses.get(scope);
  if (!fuse) {
    fuse = new Fuse(indexed.get(scope) ?? [], FUSE_OPTIONS);
    fallbackFuses.set(scope, fuse);
  }
  cb(fuse.search(query, { limit }).map((r) => r.item));
}

/**
 * True once the worker (or fallback) is set up and the scope has been indexed.
 */
export function isWarm(scope: TagScope = DEFAULT_TAG_SCOPE): boolean {
  return initialized && indexed.has(scope);
}
