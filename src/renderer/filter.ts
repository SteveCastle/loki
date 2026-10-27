import { getMediaType } from 'file-types';
import { FilterOption, SortByOption } from 'settings';
import type { Item } from './state';
import { orderForBattle } from './battle-pairing';
import { seededShuffle } from './seeded-random';
import { sortByNaturalKey } from './natural-key';

// Single-entry memoization cache keyed by libraryLoadId + filters + sortBy
let lastCacheKey: string | null = null;
let lastResult: Item[] | null = null;
// The input behind lastResult, so stream order can extend it (below).
let lastLibrary: Item[] | null = null;
let lastFilters: FilterOption | null = null;
let lastSortBy: SortByOption | null = null;

// A removal-only change bigger than this re-derives the view from scratch:
// the incremental path's Set of removed items stops being the cheap option.
const MAX_INCREMENTAL_REMOVALS = 50000;

// Orders whose result depends only on fields an in-place library update never
// touches, so the same items in the same order give the same view. Elo is
// excluded (UPDATE_MEDIA_ELO mutates items in place), as are the seeded orders:
// shuffle and battle re-seed from the new libraryLoadId by design.
const STABLE_UNDER_IN_PLACE_UPDATE = new Set<string>([
  'name',
  'date',
  'weight',
  'similarity',
]);

/**
 * How `next` relates to `prev`: 'same' (identical items, identical order), the
 * set of items removed when `next` is `prev` with some items taken out and
 * nothing else changed, or null for anything else. One two-pointer walk, no
 * allocation beyond the removed set, and it bails as soon as `next` can't be
 * a subsequence of `prev`.
 */
function removalsBetween(prev: Item[], next: Item[]): 'same' | Set<Item> | null {
  if (next.length > prev.length) return null;
  const budget = prev.length - next.length;
  if (budget > MAX_INCREMENTAL_REMOVALS) return null;
  if (budget === 0) {
    for (let i = 0; i < prev.length; i++) if (prev[i] !== next[i]) return null;
    return 'same';
  }
  const removed = new Set<Item>();
  let i = 0;
  for (let j = 0; j < next.length; j++) {
    while (prev[i] !== next[j]) {
      removed.add(prev[i]);
      i++;
      if (removed.size > budget) return null;
    }
    i++;
  }
  for (; i < prev.length; i++) removed.add(prev[i]);
  return removed;
}

// True when `items` is already in the order `cmp` would sort it into — the
// usual case for server-ranked similarity results and pre-sorted scans, where
// re-sorting a million items is ~0.7s of wasted main-thread time.
function isSorted(items: Item[], cmp: (a: Item, b: Item) => number): boolean {
  for (let i = 1; i < items.length; i++) {
    if (cmp(items[i - 1], items[i]) > 0) return false;
  }
  return true;
}

const COMPARATORS: Partial<Record<string, (a: Item, b: Item) => number>> = {
  date: (a, b) => b.mtimeMs - a.mtimeMs,
  weight: (a, b) => (a.weight || 0) - (b.weight || 0),
  elo: (a, b) => (b.elo ?? 1500) - (a.elo ?? 1500),
  similarity: (a, b) => (b.score ?? 0) - (a.score ?? 0),
};

/** Whether the media-type filter setting lets `itemPath` into the view. */
export function passesTypeFilter(filters: FilterOption, itemPath: string) {
  if (filters === 'all') return true;
  const mediaType = getMediaType(itemPath);
  return (
    (filters === 'static' && mediaType === 'static') ||
    (filters === 'video' && mediaType === 'motion') ||
    (filters === 'audio' && mediaType === 'audio')
  );
}

function filter(
  libraryLoadId: string,
  textFilter: string,
  library: Item[],
  filters: FilterOption,
  sortBy: SortByOption
) {
  const cacheKey = `${libraryLoadId}::${filters}::${sortBy}`;
  if (lastCacheKey === cacheKey && lastResult) {
    return lastResult;
  }

  if (!textFilter && !filters && !sortBy) {
    lastCacheKey = cacheKey;
    lastResult = library;
    lastSortBy = null; // not a stream result: never extend it
    return library;
  }
  // A scan batch only appends to the library, and every batch mints a new
  // libraryLoadId, so stream order used to re-run the type filter over the
  // whole library per 500 files. When this library extends the one behind
  // the cached stream result, filter just the appended tail.
  if (
    sortBy === 'stream' &&
    lastSortBy === 'stream' &&
    lastFilters === filters &&
    lastLibrary &&
    lastResult &&
    library.length >= lastLibrary.length &&
    lastLibrary.length > 0 &&
    library[0] === lastLibrary[0] &&
    library[lastLibrary.length - 1] === lastLibrary[lastLibrary.length - 1]
  ) {
    const tail: Item[] = [];
    for (let i = lastLibrary.length; i < library.length; i++) {
      if (passesTypeFilter(filters, library[i].path)) tail.push(library[i]);
    }
    const result = tail.length ? lastResult.concat(tail) : lastResult;
    lastCacheKey = cacheKey;
    lastResult = result;
    lastLibrary = library;
    return result;
  }

  // Every single-item edit (delete, forget, merge, an elo vote) mints a new
  // libraryLoadId, which used to mean re-filtering and RE-SORTING the whole
  // library — seconds at a million items for a name sort. When the library is
  // the previous one with items removed, removing them from the previous view
  // preserves its order exactly (for shuffle too, so deleting a file no longer
  // reshuffles the grid). When it's the same items in the same order, an order
  // that can't have changed is reused as is. Battle always re-pairs.
  if (
    lastLibrary &&
    lastResult &&
    lastFilters === filters &&
    lastSortBy === sortBy &&
    sortBy !== 'battle' &&
    sortBy !== 'stream'
  ) {
    const diff = removalsBetween(lastLibrary, library);
    if (diff === 'same' && STABLE_UNDER_IN_PLACE_UPDATE.has(sortBy)) {
      lastCacheKey = cacheKey;
      lastLibrary = library;
      return lastResult;
    }
    if (diff instanceof Set) {
      const result = lastResult.filter((item) => !diff.has(item));
      lastCacheKey = cacheKey;
      lastResult = result;
      lastLibrary = library;
      return result;
    }
  }

  const filtered = library.filter((item) =>
    passesTypeFilter(filters, item.path)
  );
  lastLibrary = library;
  lastFilters = filters;
  lastSortBy = sortBy;

  if (sortBy === 'stream') {
    // Preserve insertion order during streaming
    lastCacheKey = cacheKey;
    lastResult = filtered;
    return filtered;
  }

  // Battle mode: the first two items are an informative pair (anchor that
  // needs rating signal + a close-rated opponent, occasionally a random one)
  // instead of two uniformly random items. Deterministic per libraryLoadId,
  // same as shuffle; each vote re-seeds via NEXT_BATTLE.
  if (sortBy === 'battle') {
    const result = orderForBattle(filtered, libraryLoadId);
    lastCacheKey = cacheKey;
    lastResult = result;
    return result;
  }

  // Deterministic shuffle based on libraryLoadId. This prevents re-shuffling on
  // every render; the order only changes when sort switches to shuffle and the
  // caller provides a new libraryLoadId (e.g., user re-applies shuffle).
  //
  // Seeded Fisher-Yates rather than sorting items by a per-item hash of the
  // seed+path: O(n) instead of O(n log n) with no per-item wrapper objects or
  // concatenated seed+path strings, and uniform over permutations instead of
  // only as uniform as the hash. The previous hash also folded the seed in as
  // a string prefix, which left consecutive seeds producing near-identical
  // orders — see seeded-random.ts.
  if (sortBy === 'shuffle') {
    const shuffled = seededShuffle(filtered, libraryLoadId);

    // Items with no elo rating come first, keeping shuffled order within each
    // group, so a partly battle-rated library still leads with unrated items.
    let result = shuffled;
    let rated = 0;
    for (const item of shuffled) if (item.elo) rated++;
    if (rated > 0 && rated < shuffled.length) {
      const unratedFirst: Item[] = [];
      for (const item of shuffled) if (!item.elo) unratedFirst.push(item);
      for (const item of shuffled) if (item.elo) unratedFirst.push(item);
      result = unratedFirst;
    }

    lastCacheKey = cacheKey;
    lastResult = result;
    return result;
  }

  let sortedLibrary = filtered;
  if (sortBy === 'name') {
    // Same order as naturalCompare on the lower-cased path, ~9x faster: the
    // keys are built once rather than re-derived inside every comparison.
    sortedLibrary = sortByNaturalKey(filtered, (item) =>
      item.path.toLowerCase()
    );
  } else {
    const cmp = COMPARATORS[sortBy];
    if (cmp && !isSorted(filtered, cmp)) filtered.sort(cmp);
  }
  lastCacheKey = cacheKey;
  lastResult = sortedLibrary;
  return sortedLibrary;
}

export default filter;
