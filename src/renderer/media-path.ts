// Client-side mirror of the server's path-move rule (media/move.go's
// matchClause + rewriteExpr, and the same logic in src/main/media.ts's
// moveMedia). After a move is committed the renderer rewrites the paths it is
// already showing, and it has to pick EXACTLY the same items the database did
// — a looser rule here would rename an item on screen that still points at its
// old path in the library.

/**
 * The path `p` becomes after moving `from` to `to`, or `p` unchanged when it
 * isn't affected.
 *
 * Matching is segment-aligned: `/a/foo` covers `/a/foo` itself and anything
 * under `/a/foo/`, but never `/a/foobar`. Both separator styles are accepted
 * and the tail (its separator included) is preserved, so a Windows path keeps
 * its backslashes.
 */
/**
 * The containing folder of `p`, understanding BOTH separator styles.
 *
 * Node's path.dirname is platform-specific (the posix build returns "." for a
 * Windows path), but a library can hold paths from either OS — a web client
 * browsing a Windows server is the normal case. Returns `p` unchanged when it
 * has no separator at all.
 */
export function parentDir(p: string): string {
  const cut = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  if (cut <= 0) return p;
  return p.slice(0, cut);
}

/**
 * The from/to pair a "Find Media" pick should move: the file itself, or its
 * containing folder when the whole folder moved (updateAll).
 */
export function moveRange(
  oldPath: string,
  newPath: string,
  updateAll: boolean
): { from: string; to: string } {
  if (!updateAll) return { from: oldPath, to: newPath };
  return { from: parentDir(oldPath), to: parentDir(newPath) };
}

export function movedPath(p: string, from: string, to: string): string {
  if (p === from) return to;
  if (p.startsWith(`${from}/`) || p.startsWith(`${from}\\`)) {
    return to + p.slice(from.length);
  }
  return p;
}
