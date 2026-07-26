import { getMediaType } from 'file-types';
import { FilterOption, SortByOption } from 'settings';
import naturalCompare from 'natural-compare';
import type { Item } from './state';
import { orderForBattle } from './battle-pairing';
import { seededShuffle } from './seeded-random';

// Single-entry memoization cache keyed by libraryLoadId + filters + sortBy
let lastCacheKey: string | null = null;
let lastResult: Item[] | null = null;

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
    return library;
  }
  const filtered = library.filter((item) => {
    const mediaType = getMediaType(item.path);
    if (filters === 'all') {
      return true;
    }
    if (filters === 'static' && mediaType === 'static') {
      return true;
    }
    if (filters === 'video' && mediaType === 'motion') {
      return true;
    }
    if (filters === 'audio' && mediaType === 'audio') {
      return true;
    }
    return false;
  });

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

  const sortedLibrary = filtered.sort((a, b) => {
    if (sortBy === 'name') {
      return naturalCompare(a.path.toLowerCase(), b.path.toLowerCase());
    } else if (sortBy === 'date') {
      return b.mtimeMs - a.mtimeMs;
    } else if (sortBy === 'weight') {
      return (a.weight || 0) - (b.weight || 0);
    } else if (sortBy === 'elo') {
      return (b.elo ?? 1500) - (a.elo ?? 1500);
    } else if (sortBy === 'similarity') {
      return (b.score ?? 0) - (a.score ?? 0);
    }

    return 0;
  });
  lastCacheKey = cacheKey;
  lastResult = sortedLibrary;
  return sortedLibrary;
}

export default filter;
