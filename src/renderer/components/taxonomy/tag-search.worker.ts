/**
 * Tag search worker.
 *
 * Runs Fuse.js indexing and fuzzy matching off the renderer's main thread so
 * a large tag library (tens of thousands of entries) can't block the search
 * input. The renderer posts a tag set whenever it changes ('index') and a query
 * per debounced keystroke ('search'); we reply with the ranked, capped match
 * list tagged with the request id so the renderer can discard stale responses.
 *
 * Indexes are keyed by SCOPE (see src/renderer/search/tag-scopes.ts). There used
 * to be exactly one index, which meant the command palette shared the taxonomy
 * sidebar's index of every tag in the library — and searching 189K entries per
 * keystroke is what made the palette feel laggy, even off-thread. The palette
 * now indexes only curated tags (~5.6K) while the sidebar keeps the full set,
 * and each lives in its own Fuse instance here.
 *
 * Keep the Fuse options in sync with the synchronous fallback in
 * tag-search-service.ts (used where Web Workers aren't available).
 */
import Fuse from 'fuse.js';

type Concept = {
  label: string;
  category: string;
  weight: number;
  description: string;
};

type IndexMessage = { type: 'index'; scope: string; tags: Concept[] };
type SearchMessage = {
  type: 'search';
  scope: string;
  id: number;
  query: string;
  limit: number;
};
type InMessage = IndexMessage | SearchMessage;

const ctx = self as unknown as Worker;

const indexes = new Map<string, Fuse<Concept>>();
// The most recent search per scope, retained so a (re)built index can re-run the
// outstanding query: the renderer may search before the tag data has arrived,
// and the data can change underneath an active search (e.g. after a mutation).
const lastSearch = new Map<string, SearchMessage>();

function buildFuse(tags: Concept[]): Fuse<Concept> {
  return new Fuse(tags, {
    keys: [
      { name: 'label', weight: 2 },
      { name: 'category', weight: 1 },
    ],
    threshold: 0.4,
    ignoreLocation: true,
    minMatchCharLength: 1,
  });
}

function runSearch(req: SearchMessage) {
  const fuse = indexes.get(req.scope);
  if (!fuse || !req.query) return;
  const items = fuse.search(req.query, { limit: req.limit }).map((r) => r.item);
  ctx.postMessage({ type: 'result', id: req.id, query: req.query, items });
}

ctx.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'index') {
    indexes.set(msg.scope, buildFuse(msg.tags || []));
    // Refresh whatever the user is currently looking at against the new index.
    const pending = lastSearch.get(msg.scope);
    if (pending) runSearch(pending);
    return;
  }
  if (msg.type === 'search') {
    lastSearch.set(msg.scope, msg);
    runSearch(msg);
  }
};
