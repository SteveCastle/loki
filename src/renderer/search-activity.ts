// What the library search in flight is doing, for the progress UI (the
// QueryInput busy state, SearchStatus, the list overlay, pending tags).
//
// The machine runs a search in runningQuery / loadingFromDB and only swaps the
// library when it lands, so while one runs `context.query` is what was ASKED
// for and `context.settledQuery` is what the results on screen belong to.
// Everything here is derived from that pair.
import type { Predicate, Query } from './query/types';
import { predicateKey } from './query/types';
import { displayTagLabel } from './tag-display';

export const isSearchInFlight = (state: any): boolean =>
  state.matches({ library: 'runningQuery' }) ||
  state.matches({ library: 'loadingFromDB' });

// Identity of a predicate INCLUDING its modifiers: flipping exclude or join,
// or editing a blend, changes what the search returns, so the chip counts as
// pending until the search lands.
const signature = (p: Predicate) =>
  JSON.stringify([
    p.type,
    p.value,
    p.exclude,
    p.join ?? '',
    p.text ?? '',
    p.textWeight ?? '',
    p.nodes ?? [],
    p.blendMode ?? '',
  ]);

const pendingPredicates = (query: Query, settled: Query): Predicate[] => {
  const settledSigs = new Set(settled.predicates.map(signature));
  return query.predicates.filter((p) => !settledSigs.has(signature(p)));
};

/** Keys of the chips the results on screen don't reflect yet. */
export function pendingPredicateKeys(
  query: Query,
  settled: Query
): Set<string> {
  return new Set(pendingPredicates(query, settled).map(predicateKey));
}

// 'vector' = embedding search on the media server; 'scan' = substring match
// that reads every row (no index can serve `LIKE '%x%'`); 'filter' = indexed.
export type SearchKind = 'vector' | 'scan' | 'filter';

export interface SearchDescription {
  kind: SearchKind;
  // "Scanning file paths for" + subject → Scanning file paths for “zaush”…
  verb: string;
  subject?: string;
  // Shown once the search has been running a while: why it's slow and what
  // would make it faster.
  slowHint?: string;
}

const SCAN_HINT =
  'Substring search reads every item in the library. Add a tag or category to narrow it.';

const describePredicate = (p: Predicate): SearchDescription => {
  switch (p.type) {
    case 'visual':
      return { kind: 'vector', verb: 'Searching by meaning for', subject: p.value };
    case 'similar':
    case 'clip':
      return { kind: 'vector', verb: 'Ranking by visual similarity' };
    case 'face':
      return { kind: 'vector', verb: 'Finding this person' };
    case 'path':
      return {
        kind: 'scan',
        verb: 'Scanning file paths for',
        subject: p.value,
        slowHint: SCAN_HINT,
      };
    case 'description':
      return {
        kind: 'scan',
        verb: 'Searching descriptions for',
        subject: p.value,
        slowHint: SCAN_HINT,
      };
    case 'hash':
      return { kind: 'scan', verb: 'Matching file hash', slowHint: SCAN_HINT };
    case 'tag':
      return { kind: 'filter', verb: 'Loading', subject: displayTagLabel(p.value) };
    case 'category':
      return { kind: 'filter', verb: 'Loading category', subject: p.value };
    default:
      return { kind: 'filter', verb: 'Filtering' };
  }
};

const KIND_RANK: Record<SearchKind, number> = { filter: 0, scan: 1, vector: 2 };

/**
 * Describe the search in flight. Names the newest pending chip (what the
 * user just did), but the KIND is the heaviest part of the whole query: adding
 * a tag to a running path search still waits on the path scan.
 */
export function describeSearch(
  query: Query,
  settled: Query
): SearchDescription {
  const all = query.predicates;
  if (all.length === 0) return { kind: 'filter', verb: 'Loading library' };
  const pending = pendingPredicates(query, settled);
  const base: SearchDescription =
    pending.length > 0
      ? describePredicate(pending[pending.length - 1])
      : { kind: 'filter', verb: 'Updating results' };
  const heaviest = all
    .map(describePredicate)
    .reduce((a, b) => (KIND_RANK[b.kind] > KIND_RANK[a.kind] ? b : a));
  if (KIND_RANK[heaviest.kind] > KIND_RANK[base.kind]) {
    return { ...base, kind: heaviest.kind, slowHint: heaviest.slowHint };
  }
  return base;
}

/**
 * A tag that is selected in the query but whose results haven't landed yet:
 * the taxonomy shows it loading rather than done.
 */
export const isTagPending = (state: any, label: string): boolean =>
  isSearchInFlight(state) &&
  !(state.context.settledQuery as Query).predicates.some(
    (p) => p.type === 'tag' && !p.exclude && p.value === label
  );
