// Pure query-string builders for the context palette. Kept in their own module
// (no XState / platform imports) so they can be unit-tested directly without
// dragging in the renderer's state/platform module graph.

/** If initialFile is a media file path, return its parent directory; otherwise return as-is. */
export function getDirFromInitialFile(initialFile: string): string {
  const lastSegment = initialFile.split(/[/\\]/).pop() || '';
  if (lastSegment.includes('.')) {
    const sep = initialFile.includes('\\') ? '\\' : '/';
    const idx = initialFile.lastIndexOf(sep);
    let dir = idx > 0 ? initialFile.slice(0, idx) : initialFile;
    // Ensure Windows drive-letter roots keep their trailing separator (e.g. "D:" → "D:\")
    if (/^[A-Za-z]:$/.test(dir)) {
      dir += sep;
    }
    return dir;
  }
  return initialFile;
}

/**
 * Build the path query for a filesystem-loaded library context so it matches
 * the *current* list view.
 *
 * - Non-recursive: `pathdir:"dir"` matches files directly inside the directory
 *   only (server expands it to `LIKE 'dir/%' AND NOT LIKE 'dir/%/%'`).
 * - Recursive: the list view includes files from every subdirectory, so the
 *   query must match all paths *under* the directory. We emit a trailing-`*`
 *   wildcard, which the server's query lexer turns into a prefixed LIKE
 *   (`path:"dir/*"` → `m.path LIKE 'dir/%'`) — exactly the nested rows the
 *   `pathdir` form deliberately excludes via its `NOT LIKE 'dir/%/%'` clause.
 */
export function buildLibraryPathQuery(
  initialFile: string,
  recursive: boolean
): string {
  const dir = getDirFromInitialFile(initialFile);
  if (!recursive) {
    return `pathdir:"${dir}"`;
  }
  const sep = dir.includes('\\') ? '\\' : '/';
  const base = dir.endsWith(sep) ? dir : dir + sep;
  return `path:"${base}*"`;
}
