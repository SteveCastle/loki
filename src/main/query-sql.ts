// src/main/query-sql.ts
import type { Predicate } from '../renderer/query/types';

export type FilteringMode = 'AND' | 'OR' | 'EXCLUSIVE';

const BASE_COLUMNS =
  'media.path, media.description, media.elo, media.height, media.width';

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

export function buildMediaQuery(
  predicates: Predicate[],
  mode: FilteringMode
): { sql: string; params: string[] } {
  const valid = (predicates || []).filter((p) => p && p.value);
  // First INCLUDE tag drives the LEFT JOIN that surfaces weight/tag/timestamp.
  const primaryTag = valid.find((p) => p.type === 'tag' && !p.exclude)?.value;

  const params: string[] = [];
  let select: string;
  let order = '';
  if (primaryTag) {
    select =
      `SELECT ${BASE_COLUMNS}, mtcw.weight AS weight, mtcw.tag_label AS tag_label, ` +
      `mtcw.time_stamp AS time_stamp, mtcw.created_at AS created_at ` +
      `FROM media LEFT JOIN media_tag_by_category mtcw ` +
      `ON mtcw.media_path = media.path AND mtcw.tag_label = ?`;
    params.push(primaryTag); // JOIN param comes first in the SQL
    order = ' ORDER BY mtcw.weight';
  } else {
    select =
      `SELECT ${BASE_COLUMNS}, NULL AS weight, NULL AS tag_label, ` +
      `NULL AS time_stamp, NULL AS created_at FROM media`;
  }

  if (valid.length === 0) {
    return { sql: select, params };
  }

  const defaultJoin: 'AND' | 'OR' = mode === 'OR' ? 'OR' : 'AND';
  const joinOf = (p: Predicate): 'AND' | 'OR' => p.join ?? defaultJoin;
  const andPreds = valid.filter((p) => joinOf(p) !== 'OR');
  const orPreds = valid.filter((p) => joinOf(p) === 'OR');
  const andClauses = andPreds.map((p) => clauseFor(p, params));
  const orClauses = orPreds.map((p) => clauseFor(p, params));
  const pieces = [...andClauses];
  if (orClauses.length > 0) {
    pieces.push('(' + orClauses.join(' OR ') + ')');
  }
  const where = pieces.length ? ` WHERE ${pieces.join(' AND ')}` : '';
  return { sql: `${select}${where}${order}`, params };
}
