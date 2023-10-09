import { getMediaType } from 'file-types';
import { FilterOption, SortByOption } from 'settings';
import shuffle from 'shuffle-array';

import naturalCompare from 'natural-compare';

const memoize = require('lodash.memoize');

export type Item = {
  path: string;
  mtimeMs: number;
  weight?: number;
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
      }

      return 0;
    });

  if (sortBy === 'shuffle') {
    shuffle(sortedLibrary);
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
