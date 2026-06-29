// Pure query-string builders for the context palette. Kept in their own module
// (no XState / platform imports) so they can be unit-tested directly without
// dragging in the renderer's state/platform module graph.

import type { Predicate } from '../../query/types';

// Wrap a value in double quotes so multi-word values (e.g. "Exchange Student")
// survive the server lexer's whitespace splitting. Embedded quotes are escaped.
export function quoteValue(value: string): string {
  return `"${value.replace(/"/g, '\\"')}"`;
}

// Map a unified predicate type to its legacy task-query prefix. The task query
// path (autotag/embed/metadata via getMediaPathsByQueryFast) uses the legacy
// lexer, which understands tag/category/path/description/hash, AND/OR/NOT and
// parentheses. visual/similar predicates have no legacy/SQL representation (they
// need the embedding backend), so they are omitted from task queries.
export const LEGACY_PREFIX: Partial<Record<Predicate['type'], string>> = {
  tag: 'tag:',
  category: 'category:',
  path: 'path:',
  description: 'description:',
  hash: 'hash:',
};

function legacyClause(p: Predicate): string | null {
  const prefix = LEGACY_PREFIX[p.type];
  if (!prefix || !p.value) return null;
  const clause = `${prefix}${quoteValue(p.value)}`;
  return p.exclude ? `NOT ${clause}` : clause;
}

// Serialize the unified query predicates into the legacy task-query string,
// mirroring BuildMediaQuery's LEFT-ASSOCIATIVE per-predicate join composition
// (parenthesized so the legacy parser's precedence matches the unified query).
// This is what makes context-menu task actions operate on EXACTLY the media the
// search input shows — not just the include-tags. Returns "" when no predicate
// is representable (e.g. filesystem browsing, or a visual-only query).
export function buildLegacyQuery(
  predicates: Predicate[],
  filteringMode: string
): string {
  const defaultJoin = filteringMode === 'OR' ? 'OR' : 'AND'; // EXCLUSIVE -> AND
  let expr = '';
  for (const p of predicates) {
    const clause = legacyClause(p);
    if (!clause) continue;
    if (!expr) {
      expr = clause;
      continue;
    }
    const join = p.join === 'AND' || p.join === 'OR' ? p.join : defaultJoin;
    expr = `(${expr} ${join} ${clause})`;
  }
  return expr;
}

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
