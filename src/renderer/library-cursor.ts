// Cursor bookkeeping for removing one item from the in-memory library.
//
// Lives outside state.tsx because the machine defines the same transition in
// two states (loadedFromFS and loadedFromDB) and the index arithmetic is the
// part that's easy to get subtly wrong — off-by-one here means the viewer
// jumps two items forward, or lands past the end and shows nothing.

export interface LibraryEntry {
  path: string;
}

/** The library with the first entry matching `path` removed. */
export function libraryWithout<T extends LibraryEntry>(
  library: T[],
  path: string
): T[] {
  const index = library.findIndex((item) => item.path === path);
  if (index === -1) return library;
  const next = [...library];
  next.splice(index, 1);
  return next;
}

/**
 * Where the cursor lands after `path` is removed.
 *
 * Removing the item AT the cursor slides the NEXT item into that slot, so
 * holding the index is what "move to the next item" means — clamped to the
 * new last index when the removed item was at the end. A removal BEFORE the
 * cursor shifts everything down by one, so the cursor steps back to stay on
 * the item the user is actually looking at. A removal after it changes
 * nothing. Never returns a negative index: an emptied library reads as 0.
 */
export function cursorAfterRemoval<T extends LibraryEntry>(
  library: T[],
  cursor: number,
  path: string
): number {
  const index = library.findIndex((item) => item.path === path);
  if (index === -1) return cursor;
  const shifted = index < cursor ? cursor - 1 : cursor;
  const lastIndex = library.length - 2; // length after the removal, minus one
  return Math.max(0, Math.min(shifted, lastIndex));
}
