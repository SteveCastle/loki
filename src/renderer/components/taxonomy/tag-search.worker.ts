/**
 * Tag search worker.
 *
 * Runs index building and ranked matching (src/renderer/search/tag-matcher.ts)
 * off the renderer's main thread. The renderer posts a PACKED tag set whenever
 * it changes ('index') — labels as one joined string plus a category-id array,
 * which structured-clones in ~2ms where 190K tag objects took ~180ms of
 * main-thread time — and a query per keystroke ('search'). We reply with the
 * ranked, capped SOURCE INDICES, tagged with the request id and the index
 * generation they were computed against; the renderer maps them back to its own
 * tag objects and discards results from a superseded generation.
 *
 * Indexes are keyed by SCOPE (see src/renderer/search/tag-scopes.ts): the
 * command palette searches curated tags (~5.6K) while the taxonomy sidebar
 * searches the full set, each in its own index here.
 */
import { buildIndex, matchTags, type PackedTags, type TagMatchIndex } from '../../search/tag-matcher';

type IndexMessage = { type: 'index'; scope: string; gen: number; packed: PackedTags };
type SearchMessage = {
  type: 'search';
  scope: string;
  id: number;
  query: string;
  limit: number;
};
type InMessage = IndexMessage | SearchMessage;

const ctx = self as unknown as Worker;

const indexes = new Map<string, { gen: number; index: TagMatchIndex }>();

ctx.onmessage = (e: MessageEvent<InMessage>) => {
  const msg = e.data;
  if (msg.type === 'index') {
    indexes.set(msg.scope, { gen: msg.gen, index: buildIndex(msg.packed) });
    return;
  }
  if (msg.type === 'search') {
    const entry = indexes.get(msg.scope);
    // The service only searches a scope after posting its index, and messages
    // are ordered, so a missing entry means an empty scope.
    const idx = entry ? matchTags(entry.index, msg.query, msg.limit) : [];
    ctx.postMessage({ type: 'result', id: msg.id, gen: entry?.gen ?? -1, idx });
  }
};
