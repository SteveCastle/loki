// Incremental de-duplication for streamed folder scans.
//
// Each LOAD_FILES_BATCH (500 files) used to rebuild a Set of every library
// path, lower-cased, to drop duplicates: O(library) string allocation per
// batch, so a million-file scan did ~2,000 full passes. The index is kept
// alongside the exact library array it describes and extended in place
// while batches append; anything else (a different array) rebuilds it.

import type { Item } from './state';

let indexedLibrary: Item[] | null = null;
let index: Set<string> | null = null;

const key = (p: string) => p.toLowerCase();

/**
 * Returns `library` plus the incoming items whose paths (case-insensitively,
 * for Windows) are new — to it and within the batch itself.
 */
export function appendUnique(library: Item[], incoming: Item[]): Item[] {
  if (indexedLibrary !== library || !index) {
    index = new Set(library.map((item) => key(item.path)));
  }
  const newItems: Item[] = [];
  for (const f of incoming) {
    if (!f?.path) continue;
    const k = key(f.path);
    if (index.has(k)) continue;
    index.add(k);
    newItems.push(f);
  }
  const updated = newItems.length ? library.concat(newItems) : library;
  indexedLibrary = updated;
  return updated;
}

/** For tests. */
export function _resetScanIndex() {
  indexedLibrary = null;
  index = null;
}
