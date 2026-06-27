/**
 * Tag search worker.
 *
 * Runs Fuse.js indexing and fuzzy matching off the renderer's main thread so
 * a large tag library (tens of thousands of entries) can't block the search
 * input. The renderer posts the full tag set whenever it changes ('index')
 * and a query per debounced keystroke ('search'); we reply with the ranked,
 * capped match list tagged with the request id so the renderer can discard
 * stale responses.
 *
 * Keep the Fuse options here in sync with the synchronous fallback in
 * taxonomy.tsx (used where Web Workers aren't available).
 */
import Fuse from 'fuse.js';

type Concept = {
  label: string;
  category: string;
  weight: number;
  description: string;
};

type IndexMessage = { type: 'index'; tags: Concept[] };
type SearchMessage = {
  type: 'search';
  id: number;
  query: string;
  limit: number;
};
type InMessage = IndexMessage | SearchMessage;

const ctx = self as unknown as Worker;

let fuse: Fuse<Concept> | null = null;
// The most recent search request, retained so a (re)built index can re-run the
// outstanding query: the renderer may search before the tag data has arrived,
// and the data can change underneath an active search (e.g. after a mutation).
let lastSearch: SearchMessage | null = null;

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
  if (!fuse || !req.query) return;
  const items = fuse.search(req.query, { limit: req.limit }).map((r) => r.item);
  ctx.postMessage({ type: 'result', id: req.id, query: req.query, items });
}

ctx.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'index') {
    fuse = buildFuse(msg.tags || []);
    // Refresh whatever the user is currently looking at against the new index.
    if (lastSearch) runSearch(lastSearch);
    return;
  }
  if (msg.type === 'search') {
    lastSearch = msg;
    runSearch(msg);
  }
};
