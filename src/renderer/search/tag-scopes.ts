// tag-scopes — which slice of the tag table a search surface works over.
//
// "Suggested" is the catch-all bucket the ONNX autotagger writes into, and on a
// real library it dwarfs everything else: 183,548 of 189,122 tags (97%) at the
// time of writing. Loading and Fuse-indexing that set costs hundreds of
// milliseconds of SQLite, an IPC clone of the whole table, and a fuzzy match
// over 189K entries on every keystroke.
//
// The command palette is the app's most frequently opened surface and doesn't
// need it — machine-suggested tags are noise there, and the palette already
// deprioritised them in its result ordering. The taxonomy sidebar DOES need the
// complete set, and pays for it lazily (its fetch is gated on the search input
// gaining focus).
//
// So the two surfaces index different scopes, and the search service keeps one
// Fuse index per scope rather than one shared index of everything.

export const SUGGESTED_CATEGORY = 'Suggested';

/** `all` = every tag. `curated` = everything except the autotagger's bucket. */
export type TagScope = 'all' | 'curated';

export const DEFAULT_TAG_SCOPE: TagScope = 'all';

/** Categories a scope leaves out. Empty for `all`. */
export function excludedCategories(scope: TagScope): string[] {
  return scope === 'curated' ? [SUGGESTED_CATEGORY] : [];
}

/**
 * Apply a scope client-side.
 *
 * The exclusion is also pushed down to SQL (Electron) and to the query string
 * (web), which is where the real saving is. This is the belt-and-braces pass so
 * the Fuse index is the right size even against a backend that ignored the
 * hint — an older media-server, say. Cheap: one linear scan.
 */
export function applyScope<T extends { category?: string }>(
  tags: T[],
  scope: TagScope
): T[] {
  const excluded = excludedCategories(scope);
  if (excluded.length === 0) return tags;
  return tags.filter((t) => !excluded.includes(t.category ?? ''));
}
