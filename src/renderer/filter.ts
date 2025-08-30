import { getMediaType } from 'file-types';
import { FilterOption, SortByOption } from 'settings';
import naturalCompare from 'natural-compare';
import type { Item } from './state';

function filter(
  libraryLoadId: string,
  textFilter: string,
  library: Item[],
  filters: FilterOption,
  sortBy: SortByOption
) {
  if (!textFilter && !filters && !sortBy) {
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
    return filtered;
  }

  // Deterministic shuffle based on libraryLoadId. This prevents re-shuffling on
  // every render; the order only changes when sort switches to shuffle and the
  // caller provides a new libraryLoadId (e.g., user re-applies shuffle).
  if (sortBy === 'shuffle') {
    const seed = libraryLoadId;
    const hash = (s: string) => {
      let h = 5381;
      for (let i = 0; i < s.length; i++) {
        h = ((h << 5) + h) ^ s.charCodeAt(i);
      }
      return h >>> 0;
    };

    const ranked = filtered.map((item) => ({
      item,
      rank: hash(`${seed}::${item.path}`),
    }));

    ranked.sort((a, b) => {
      const aNoElo = !a.item.elo;
      const bNoElo = !b.item.elo;
      if (aNoElo && !bNoElo) return -1;
      if (!aNoElo && bNoElo) return 1;
      return a.rank - b.rank;
    });

    return ranked.map((r) => r.item);
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
    }

    return 0;
  });
  return sortedLibrary;
}

export default filter;
