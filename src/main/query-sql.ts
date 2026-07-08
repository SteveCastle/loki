// src/main/query-sql.ts
import type { Predicate } from '../renderer/query/types';

export type FilteringMode = 'AND' | 'OR' | 'EXCLUSIVE';

// Columns returned for the library list. NOTE: `media.description` is
// intentionally NOT selected — it's a large text column the list view doesn't
// use (the detail/metadata view fetches it on demand). It is still available
// as a WHERE filter (see clauseFor 'description'). Ordering is done in the
// renderer, so queries emit no ORDER BY.
const BASE_COLUMNS = 'media.path, media.elo, media.height, media.width';

// Tag columns are NULL for category-driven / no-tag queries (a category spans
// many tags, so there's no single per-tag weight/timestamp to surface).
const NULL_TAG_COLS =
  'NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at';

// Columns when the query is DRIVEN from media_tag_by_category (mtcw), surfacing
// the per-assignment weight/tag/timestamp.
const DRIVEN_TAG_COLUMNS =
  'mtcw.media_path AS path, media.elo AS elo, media.height AS height, ' +
  'media.width AS width, mtcw.weight AS weight, mtcw.tag_label AS tag_label, ' +
  'mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at';

function clauseFor(p: Predicate, params: string[]): string {
  const like = `%${p.value}%`;
  switch (p.type) {
    case 'tag':
      params.push(p.value);
      return p.exclude
        ? '(NOT EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.tag_label = ?))'
        : '(EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.tag_label = ?))';
    case 'category':
      params.push(p.value);
      return p.exclude
        ? '(NOT EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.category_label = ?))'
        : '(EXISTS (SELECT 1 FROM media_tag_by_category mtc WHERE mtc.media_path = media.path AND mtc.category_label = ?))';
    case 'path':
      params.push(like);
      return p.exclude ? '(media.path NOT LIKE ?)' : '(media.path LIKE ?)';
    case 'description':
      params.push(like);
      return p.exclude ? '(media.description NOT LIKE ?)' : '(media.description LIKE ?)';
    case 'hash':
      params.push(like);
      return p.exclude ? '(media.hash NOT LIKE ?)' : '(media.hash LIKE ?)';
    case 'faces':
      // faces:ungrouped — media holding at least one detected face that isn't
      // assigned to any person yet. The face table is created by the media
      // server in the shared library DB; the predicate is only offered from
      // the People panel (which requires the server), so the table existing
      // here is a safe assumption. Unknown values match nothing.
      if (p.value === 'ungrouped') {
        return p.exclude
          ? '(NOT EXISTS (SELECT 1 FROM face f WHERE f.media_path = media.path AND COALESCE(f.person_id, 0) = 0))'
          : '(EXISTS (SELECT 1 FROM face f WHERE f.media_path = media.path AND COALESCE(f.person_id, 0) = 0))';
      }
      return p.exclude ? '(1=1)' : '(1=0)';
    case 'similar':
    case 'visual':
    case 'clip':
    case 'face':
      // Similarity search requires the embedding backend, which only exists in
      // the media-server (web mode). In Electron's local-SQLite path we cannot
      // resolve it, so treat it as no constraint and warn. The server path
      // (/api/media/query) handles these properly.
      console.warn(
        `Visual search ('${p.type}:') is only available in server/web mode; ignoring this predicate in local mode.`
      );
      return '(1=1)';
    default: {
      const _never: never = p.type as never;
      throw new Error(`Unknown predicate type: ${_never}`);
    }
  }
}

// AND-join clauses for the conjunct "rest" of an intersection fast path.
function andJoin(preds: Predicate[], params: string[]): string {
  return preds.map((p) => clauseFor(p, params)).join(' AND ');
}

// General, always-correct path: scan `media` and combine the predicate clauses
// LEFT-TO-RIGHT using the given connectors (connectors[i] joins valid[i+1] to
// the running expression), parenthesized so evaluation is left-associative and
// matches how the chips read. Surfaces tag columns via a LEFT JOIN on the first
// include-tag (if any). Used for mixed AND/OR and exclude/non-tag-only queries.
function mediaScan(
  valid: Predicate[],
  connectors: ('AND' | 'OR')[],
  isIncludeTag: (p: Predicate) => boolean
): { sql: string; params: string[] } {
  const primaryTag = valid.find(isIncludeTag)?.value;
  const params: string[] = [];
  let select: string;
  if (primaryTag) {
    select =
      `SELECT ${BASE_COLUMNS}, mtcw.weight AS weight, mtcw.tag_label AS tag_label, ` +
      `mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at ` +
      `FROM media LEFT JOIN media_tag_by_category mtcw ` +
      `ON mtcw.media_path = media.path AND mtcw.tag_label = ?`;
    params.push(primaryTag); // JOIN param comes first in the SQL
  } else {
    select = `SELECT ${BASE_COLUMNS}, ${NULL_TAG_COLS} FROM media`;
  }
  let expr = clauseFor(valid[0], params);
  for (let i = 1; i < valid.length; i += 1) {
    expr = `(${expr} ${connectors[i - 1]} ${clauseFor(valid[i], params)})`;
  }
  return { sql: `${select} WHERE ${expr}`, params };
}

export function buildMediaQuery(
  predicates: Predicate[],
  mode: FilteringMode
): { sql: string; params: string[] } {
  const valid = (predicates || []).filter((p) => p && p.value);

  if (valid.length === 0) {
    return {
      sql: `SELECT ${BASE_COLUMNS}, ${NULL_TAG_COLS} FROM media`,
      params: [],
    };
  }

  const defaultJoin: 'AND' | 'OR' = mode === 'OR' ? 'OR' : 'AND';
  const joinOf = (p: Predicate): 'AND' | 'OR' => p.join ?? defaultJoin;
  const isIncludeTag = (p: Predicate) => p.type === 'tag' && !p.exclude;
  const isIncludeCat = (p: Predicate) => p.type === 'category' && !p.exclude;

  // Predicates combine LEFT-TO-RIGHT: the first predicate is the base and each
  // subsequent predicate's join is the operator connecting it to the running
  // result (its chip badge; the first chip's badge is hidden). So [a, b(OR)]
  // reads "a OR b" (union) and [a, b(AND)] reads "a AND b" (intersection).
  // Homogeneous all-AND / all-OR get index-driven fast paths; mixed operators
  // fall back to a correct left-to-right media scan.
  const connectors = valid.slice(1).map(joinOf);
  const allOr = connectors.length > 0 && connectors.every((c) => c === 'OR');
  const allAnd = connectors.every((c) => c === 'AND'); // also true for one predicate

  // ---- Union (all-OR): "media with ANY of these" via an indexed IN lookup ----
  if (allOr && valid.every(isIncludeTag)) {
    const params = valid.map((p) => p.value);
    const ph = valid.map(() => '?').join(', ');
    return {
      sql:
        `SELECT ${DRIVEN_TAG_COLUMNS} FROM media_tag_by_category mtcw ` +
        `LEFT JOIN media ON media.path = mtcw.media_path ` +
        `WHERE mtcw.tag_label IN (${ph})`,
      params,
    };
  }
  if (allOr && valid.every(isIncludeCat)) {
    const params = valid.map((p) => p.value);
    const ph = valid.map(() => '?').join(', ');
    return {
      sql:
        `SELECT ${BASE_COLUMNS}, ${NULL_TAG_COLS} ` +
        `FROM (SELECT DISTINCT media_path FROM media_tag_by_category ` +
        `WHERE category_label IN (${ph})) cat ` +
        `JOIN media ON media.path = cat.media_path`,
      params,
    };
  }

  // ---- Intersection (single predicate or all-AND): drive from an indexed
  //      tag/category, with the rest as AND conjuncts ----
  if (allAnd) {
    const driveTag = valid.find(isIncludeTag);
    if (driveTag) {
      const params: string[] = [driveTag.value]; // driving join param first
      const rest = valid.filter((p) => p !== driveTag);
      const restWhere = andJoin(rest, params);
      const extra = restWhere ? ` AND ${restWhere}` : '';
      return {
        sql:
          `SELECT ${DRIVEN_TAG_COLUMNS} FROM media_tag_by_category mtcw ` +
          `LEFT JOIN media ON media.path = mtcw.media_path ` +
          `WHERE mtcw.tag_label = ?${extra}`,
        params,
      };
    }
    const driveCat = valid.find(isIncludeCat);
    if (driveCat) {
      const params: string[] = [driveCat.value];
      const rest = valid.filter((p) => p !== driveCat);
      const restWhere = andJoin(rest, params);
      const where = restWhere ? ` WHERE ${restWhere}` : '';
      return {
        sql:
          `SELECT ${BASE_COLUMNS}, ${NULL_TAG_COLS} ` +
          `FROM (SELECT DISTINCT media_path FROM media_tag_by_category ` +
          `WHERE category_label = ?) cat ` +
          `JOIN media ON media.path = cat.media_path${where}`,
        params,
      };
    }
    // No drivable tag/category (paths/hash/excludes only) → AND media scan.
    return mediaScan(valid, connectors, isIncludeTag);
  }

  // ---- Mixed AND/OR operators → correct left-to-right media scan ----
  return mediaScan(valid, connectors, isIncludeTag);
}
