// Multi-item selection for the context palette.
//
// Shift+right-click opens the palette anchored on an item. While it is open:
//   - shift+right-click on another item adds the whole visual range from the
//     anchor to that item (in the list's current filtered/sorted order);
//   - ctrl/cmd+click (or ctrl+right-click) toggles just that one item.
// Tasks launched from the palette then act on the accumulated path list,
// submitted as a newline-separated list rather than a query.

export type ExtendMode = 'range' | 'single';

/**
 * extendSelection returns the new selection after a modifier-click on `path`
 * (at display index `idx`) while the palette is open.
 *
 * Range mode walks `items` — the SAME filtered/sorted array the list renders,
 * so indices line up with what the user sees — from the anchor to the clicked
 * item, inclusive in either direction, and appends everything not already
 * selected (existing order preserved, no duplicates). Without a usable anchor
 * (palette opened from the detail view or the library background) it degrades
 * to a single add. Single mode toggles: clicking an already-selected item
 * removes it, so mis-clicks are correctable without starting over.
 */
export function extendSelection(
  existing: string[],
  mode: ExtendMode,
  anchorIdx: number | null,
  idx: number | undefined,
  path: string,
  items: Array<{ path: string }>
): string[] {
  if (mode === 'single' || anchorIdx === null || typeof idx !== 'number') {
    if (mode === 'single' && existing.includes(path)) {
      return existing.filter((p) => p !== path);
    }
    return existing.includes(path) ? existing : [...existing, path];
  }
  const lo = Math.max(0, Math.min(anchorIdx, idx));
  const hi = Math.min(items.length - 1, Math.max(anchorIdx, idx));
  const have = new Set(existing);
  const added: string[] = [];
  for (let i = lo; i <= hi; i++) {
    const p = items[i]?.path;
    if (p && !have.has(p)) {
      have.add(p);
      added.push(p);
    }
  }
  return added.length ? [...existing, ...added] : existing;
}
