// src/main/query-sql.ts
import type { Predicate } from '../renderer/query/types';

export type FilteringMode = 'AND' | 'OR' | 'EXCLUSIVE';

// Columns returned for the library list. NOTE: `media.description` is
// intentionally NOT selected — it's a large text column the list view doesn't
// use (the detail/metadata view fetches it on demand). It is still available
// as a WHERE filter (see clauseFor 'description'). Ordering is done in the
// renderer, so queries emit no ORDER BY.
const BASE_COLUMNS = 'media.path, media.elo, media.height, media.width';

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
    default: {
      const _never: never = p.type as never;
      throw new Error(`Unknown predicate type: ${_never}`);
    }
  }
}

// Faceted WHERE over a predicate set: AND-bucket clauses are required, the
// OR-bucket is grouped as "(a OR b ...)". Returns '' when there are none.
function facetedWhere(
  preds: Predicate[],
  joinOf: (p: Predicate) => 'AND' | 'OR',
  params: string[]
): string {
  const andClauses = preds.filter((p) => joinOf(p) !== 'OR').map((p) => clauseFor(p, params));
  const orClauses = preds.filter((p) => joinOf(p) === 'OR').map((p) => clauseFor(p, params));
  const pieces = [...andClauses];
  if (orClauses.length > 0) {
    pieces.push('(' + orClauses.join(' OR ') + ')');
  }
  return pieces.join(' AND ');
}

export function buildMediaQuery(
  predicates: Predicate[],
  mode: FilteringMode
): { sql: string; params: string[] } {
  const valid = (predicates || []).filter((p) => p && p.value);

  if (valid.length === 0) {
    return {
      sql:
        `SELECT ${BASE_COLUMNS}, NULL AS weight, NULL AS tag_label, ` +
        `NULL AS time_stamp, NULL AS created_at FROM media`,
      params: [],
    };
  }

  const defaultJoin: 'AND' | 'OR' = mode === 'OR' ? 'OR' : 'AND';
  const joinOf = (p: Predicate): 'AND' | 'OR' => p.join ?? defaultJoin;
  const isIncludeTag = (p: Predicate) => p.type === 'tag' && !p.exclude;

  // Fast path: when a REQUIRED include-tag exists, DRIVE the query from the
  // indexed media_tag_by_category lookup instead of scanning all of `media`.
  // "Required" means the tag is a conjunct: it's in the AND bucket, or it's
  // the only predicate (a lone OR-tag is still required). Driving an OR-tag
  // that has OR siblings would wrongly force its presence, so we don't.
  const driveTag =
    valid.find((p) => isIncludeTag(p) && joinOf(p) !== 'OR') ??
    (valid.length === 1 && isIncludeTag(valid[0]) ? valid[0] : undefined);

  if (driveTag) {
    const params: string[] = [driveTag.value]; // driving join param first
    const rest = valid.filter((p) => p !== driveTag);
    const restWhere = facetedWhere(rest, joinOf, params);
    const extra = restWhere ? ` AND ${restWhere}` : '';
    const sql =
      `SELECT mtcw.media_path AS path, ` +
      `media.elo AS elo, media.height AS height, media.width AS width, ` +
      `mtcw.weight AS weight, mtcw.tag_label AS tag_label, ` +
      `mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at ` +
      `FROM media_tag_by_category mtcw ` +
      `LEFT JOIN media ON media.path = mtcw.media_path ` +
      `WHERE mtcw.tag_label = ?${extra}`;
    return { sql, params };
  }

  // OR-set of include-tags: every predicate is an include-tag and none is a
  // required AND driver (so they're an OR bucket) — "media with ANY of these
  // tags" = tag_label IN (...). Drive from the indexed tag lookup instead of
  // scanning all of `media` with an OR of EXISTS subqueries.
  if (valid.every(isIncludeTag)) {
    const params = valid.map((p) => p.value);
    const placeholders = valid.map(() => '?').join(', ');
    const sql =
      `SELECT mtcw.media_path AS path, ` +
      `media.elo AS elo, media.height AS height, media.width AS width, ` +
      `mtcw.weight AS weight, mtcw.tag_label AS tag_label, ` +
      `mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at ` +
      `FROM media_tag_by_category mtcw ` +
      `LEFT JOIN media ON media.path = mtcw.media_path ` +
      `WHERE mtcw.tag_label IN (${placeholders})`;
    return { sql, params };
  }

  // Fallback: no drivable include-tag (exclude-only, or non-tag predicates,
  // or an OR bucket that mixes tags with non-tags). Scan media; LEFT JOIN the
  // first include-tag, if any, only to surface weight/tag/timestamp columns.
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
    select =
      `SELECT ${BASE_COLUMNS}, NULL AS weight, NULL AS tag_label, ` +
      `NULL AS time_stamp, NULL AS created_at FROM media`;
  }
  const restWhere = facetedWhere(valid, joinOf, params);
  const where = restWhere ? ` WHERE ${restWhere}` : '';
  return { sql: `${select}${where}`, params };
}
