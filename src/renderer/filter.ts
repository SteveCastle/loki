import { getMediaType } from 'file-types';
import { FilterOption, SortByOption } from 'settings';
import shuffle from 'shuffle-array';

import naturalCompare from 'natural-compare';

const memoize = require('lodash.memoize');

export type Item = {
  path: string;
  mtimeMs: number;
  weight?: number;
  elo?: number;
};

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
  const sortedLibrary = library
    .filter((item) => {
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
      return false;
    })
    .filter((item) =>
      item.path.toLowerCase().includes(textFilter.toLowerCase())
    )
    .sort((a, b) => {
      if (sortBy === 'name') {
        return naturalCompare(a.path.toLowerCase(), b.path.toLowerCase());
      } else if (sortBy === 'date') {
        return b.mtimeMs - a.mtimeMs;
      } else if (sortBy === 'weight') {
        return (a.weight || 0) - (b.weight || 0);
      } else if (sortBy === 'elo') {
        return (b.elo || 1500) - (a.elo || 1500);
      }

      return 0;
    });

  if (sortBy === 'shuffle') {
    // shuffle then sort any items with no elo to the beginning
    shuffle(sortedLibrary).sort((a, b) => {
      if (!a.elo && !b.elo) {
        return 0;
      }
      if (!a.elo) {
        return -1;
      }
      if (!b.elo) {
        return 1;
      }
      return 0;
    });
  }
  return sortedLibrary;
}

export default memoize(
  filter,
  (
    libraryLoadId: string,
    textFilter: string,
    library: Item[],
    filters: FilterOption,
    sortBy: SortByOption
  ) => {
    return `${libraryLoadId}-${textFilter}-${sortBy}-${filters}`;
  }
);
