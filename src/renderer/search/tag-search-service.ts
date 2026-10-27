// tag-search-service — a MODULE-LEVEL singleton wrapping ONE shared tag-search
// worker (plus a synchronous fallback) for the whole app.
//
// The worker is shared; the INDEXES inside it are not. Each index is keyed by a
// scope (see tag-scopes.ts) because the app's two search surfaces want
// different slices of the tag table:
//
//   'curated' — the command palette. Everything except the autotagger's
//               "Suggested" bucket: ~5.6K entries.
//   'all'     — the taxonomy sidebar. The complete table: ~189K entries.
//
// Matching is tag-matcher.ts (tiered exact/prefix/substring ranking with a
// typo-tolerant fallback). It replaced Fuse.js, whose per-keystroke Bitap scan
// of the full table took 0.25–1s; the matcher answers the same queries in
// ~5–20ms.
//
// The worker is constructed lazily on first use and lives for the app lifetime
// — it is NEVER terminated. Under environments without Web Worker support
// (e.g. jsdom in tests) construction is guarded and the same matcher runs
// synchronously instead.
import {
  buildIndex,
  matchTags,
  packTags,
  type TagMatchIndex,
} from './tag-matcher';
import { DEFAULT_TAG_SCOPE, type TagScope } from './tag-scopes';

export interface TagConcept {
  label: string;
  category: string;
  weight: number;
  description?: string;
}

// Don't dispatch a search until the query is at least this long. A 1-char
// query matches a large fraction of a big tag library, flooding the result list
// with noise. Two characters is the standard type-ahead floor.
const MIN_QUERY_LENGTH = 2;

type ResultCb = (items: TagConcept[]) => void;

type PendingSearch = {
  scope: TagScope;
  query: string;
  limit: number;
  cb: ResultCb;
};

// Module-level singleton state.
let worker: Worker | null = null;
// Whether worker setup (or fallback) has been attempted at least once.
let initialized = false;

// Per-scope state. `sources` holds the raw array last handed to indexTags for a
// scope, kept by reference so re-indexing the same data is a no-op. Every
// consumer of a given scope reads the SAME React Query cache entry, so they
// pass an identical reference — meaning each scope is sent to the worker
// exactly once, not again on each surface's first search.
const sources = new Map<TagScope, TagConcept[]>();
// The cleaned, indexed arrays (label-less rows dropped). The worker answers
// with indices into these, mapped back to the caller's own objects here.
const indexed = new Map<TagScope, TagConcept[]>();
// Bumped on every (re)index of a scope. Worker results carry the generation
// they were computed against, so indices into a replaced array are never used.
const generations = new Map<TagScope, number>();
// In-thread indexes — only built when running without a worker.
const fallbackIndexes = new Map<TagScope, TagMatchIndex>();

// Pending searches keyed by request id so concurrent callers (taxonomy +
// palette) don't clobber each other.
let seq = 0;
const pending = new Map<number, PendingSearch>();
// Searches issued before their scope was first indexed; flushed by indexTags.
const awaitingIndex = new Map<number, PendingSearch>();

function postSearch(id: number, req: PendingSearch): void {
  worker?.postMessage({
    type: 'search',
    scope: req.scope,
    id,
    query: req.query,
    limit: req.limit,
  });
}

function runInThread(req: PendingSearch): TagConcept[] {
  const index = fallbackIndexes.get(req.scope);
  const items = indexed.get(req.scope) ?? [];
  if (!index) return [];
  return matchTags(index, req.query, req.limit).map((i) => items[i]);
}

function dispatch(id: number, req: PendingSearch): void {
  if (worker) {
    pending.set(id, req);
    postSearch(id, req);
  } else {
    req.cb(runInThread(req));
  }
}

// Lazily set up the shared worker (or fall back to in-thread matching).
// Idempotent.
function ensureInit(): void {
  if (initialized) return;
  initialized = true;

  if (typeof Worker === 'undefined') return;
  try {
    // Lazy require: the factory carries the `import.meta` worker URL, which the
    // CommonJS test transform can't parse. We only reach here when a real Worker
    // exists (never under jsdom), so the test graph never loads that module.
    // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
    const { createTagSearchWorker } = require('./tag-search-worker-factory');
    worker = createTagSearchWorker();
  } catch {
    worker = null;
    return;
  }
  if (worker) {
    worker.onmessage = (e: MessageEvent) => {
      const msg = e.data;
      if (msg?.type !== 'result') return;
      const req = pending.get(msg.id);
      if (!req) return;
      if (msg.gen !== generations.get(req.scope)) {
        // The scope was re-indexed after this search was posted, so these
        // indices point into the old array. Ask again against the new index.
        postSearch(msg.id, req);
        return;
      }
      pending.delete(msg.id);
      const items = indexed.get(req.scope) ?? [];
      req.cb((msg.idx as number[]).map((i) => items[i]).filter(Boolean));
    };
  }
}

/**
 * (Re)index one scope's tag set. Idempotent on the SOURCE reference: passing the
 * same array again is a no-op, so consumers can hand over the raw, shared React
 * Query array and a scope is sent to the worker only when its data actually
 * changes — not once per consumer. Returns whether a (re)index ran.
 *
 * The defensive label filter (the matcher and the row renderers assume a
 * non-empty label) lives here so callers don't each spin up a divergent filtered
 * array, which is what previously defeated the reference guard.
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
  const gen = (generations.get(scope) ?? 0) + 1;
  generations.set(scope, gen);
  // Packed — one joined label string plus a category-id array — because the
  // structured clone runs on THIS thread, and 190K tag objects took ~180ms to
  // clone where the packed form takes ~2ms.
  const packed = packTags(clean);
  if (worker) {
    worker.postMessage({ type: 'index', scope, gen, packed }, [
      packed.categoryIds.buffer,
    ]);
  } else {
    fallbackIndexes.set(scope, buildIndex(packed));
  }
  awaitingIndex.forEach((req, id) => {
    if (req.scope !== scope) return;
    awaitingIndex.delete(id);
    dispatch(id, req);
  });
  return true;
}

/**
 * Run a search against one scope. The callback fires with the ranked results
 * for THIS call — as soon as the scope is indexed, if it isn't yet. A query
 * shorter than MIN_QUERY_LENGTH (empty included) resolves to [] synchronously
 * without touching the index.
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
  const req: PendingSearch = { scope, query, limit, cb };
  if (!generations.has(scope)) {
    awaitingIndex.set(id, req);
    return;
  }
  dispatch(id, req);
}

/**
 * Drop every still-outstanding search issued with `cb`. A consumer searching
 * per keystroke only cares about its newest request; without this each
 * superseded request still resolves (and re-renders) in turn.
 */
export function cancelTagSearches(cb: ResultCb): void {
  pending.forEach((req, id) => {
    if (req.cb === cb) pending.delete(id);
  });
  awaitingIndex.forEach((req, id) => {
    if (req.cb === cb) awaitingIndex.delete(id);
  });
}

/**
 * True once the scope has been indexed.
 */
export function isWarm(scope: TagScope = DEFAULT_TAG_SCOPE): boolean {
  return initialized && indexed.has(scope);
}
