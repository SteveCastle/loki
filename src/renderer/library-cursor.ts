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

/**
 * Index of the first item whose path is `target`, compared the way the app
 * compares paths everywhere (normalized, case-insensitive) — or -1.
 *
 * Cursor lookups run over the whole view on every filter/sort change and
 * every job-output jump, and normalizing BOTH sides per item made that ~2M
 * path.normalize calls at a million items. Paths in a view almost always
 * match the target verbatim, so an exact pass runs first and the normalized
 * comparison (target normalized once) is only the fallback.
 */
export function findPathIndex<T extends LibraryEntry>(
  items: T[],
  target: string | undefined | null,
  normalize: (p: string) => string
): number {
  if (!target) return -1;
  for (let i = 0; i < items.length; i++) {
    if (items[i]?.path === target) return i;
  }
  const want = normalize(target).toLowerCase();
  for (let i = 0; i < items.length; i++) {
    const p = items[i]?.path;
    if (p && normalize(p).toLowerCase() === want) return i;
  }
  return -1;
}

/**
 * The library with every path in `paths` it doesn't already hold (compared
 * like findPathIndex) put in FRONT, made with `make` — or the same array when
 * there is nothing to add. One exact pass over the library, and a normalized
 * pass only while some path is still unmatched, so a job's output costs one
 * or two scans at a million items whatever the batch size.
 */
export function libraryWithPaths<T extends LibraryEntry>(
  library: T[],
  paths: string[],
  normalize: (p: string) => string,
  make: (p: string) => T
): T[] {
  const key = (p: string) => normalize(p).toLowerCase();
  const missing = new Map<string, string>();
  for (const p of paths) {
    if (p && !missing.has(key(p))) missing.set(key(p), p);
  }
  const exact = new Set(paths);
  for (const item of library) {
    if (missing.size === 0) break;
    if (item?.path && exact.has(item.path)) missing.delete(key(item.path));
  }
  for (const item of library) {
    if (missing.size === 0) break;
    if (item?.path) missing.delete(key(item.path));
  }
  if (missing.size === 0) return library;
  return [...Array.from(missing.values(), make), ...library];
}
