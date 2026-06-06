// src/__tests__/query-sql.test.ts
import { buildMediaQuery } from '../main/query-sql';
import type { Predicate } from '../renderer/query/types';

const norm = (s: string) => s.replace(/\s+/g, ' ').trim();

describe('buildMediaQuery', () => {
  it('returns base query with no predicates', () => {
    const { sql, params } = buildMediaQuery([], 'AND');
    expect(norm(sql)).toBe(
      norm('SELECT media.path, media.description, media.elo, media.height, media.width, NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at FROM media')
    );
    expect(params).toEqual([]);
  });

  it('builds an include tag predicate with EXISTS', () => {
    const preds: Predicate[] = [{ type: 'tag', value: 'portrait', exclude: false }];
    const { sql, params } = buildMediaQuery(preds, 'AND');
    expect(sql).toContain('EXISTS');
    expect(sql).toContain('mtc.tag_label = ?');
    expect(params).toEqual(['portrait', 'portrait']);
  });

  it('builds exclude tag with NOT EXISTS', () => {
    const { sql } = buildMediaQuery([{ type: 'tag', value: 'x', exclude: true }], 'AND');
    expect(norm(sql)).toContain('NOT EXISTS');
  });

  it('uses category_label for category predicates', () => {
    const { sql, params } = buildMediaQuery([{ type: 'category', value: 'Studio', exclude: false }], 'AND');
    expect(sql).toContain('mtc.category_label = ?');
    expect(params).toEqual(['Studio']);
  });

  it('wraps path/description/hash values in % for LIKE', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'path', value: 'a', exclude: false },
        { type: 'description', value: 'b', exclude: true },
        { type: 'hash', value: 'c', exclude: false },
      ],
      'AND'
    );
    expect(sql).toContain('media.path LIKE ?');
    expect(sql).toContain('media.description NOT LIKE ?');
    expect(sql).toContain('media.hash LIKE ?');
    expect(params).toEqual(['%a%', '%b%', '%c%']);
  });

  it('joins clauses with OR in OR mode', () => {
    const { sql } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false },
        { type: 'tag', value: 'b', exclude: false },
      ],
      'OR'
    );
    expect(norm(sql)).toContain(') OR (');
  });

  it('joins clauses with AND in AND mode', () => {
    const { sql } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false },
        { type: 'tag', value: 'b', exclude: false },
      ],
      'AND'
    );
    expect(norm(sql)).toContain(') AND (');
  });

  it('treats EXCLUSIVE like AND for SQL joining', () => {
    const a = norm(buildMediaQuery(
      [{ type: 'tag', value: 'a', exclude: false }, { type: 'tag', value: 'b', exclude: false }],
      'EXCLUSIVE'
    ).sql);
    expect(a).toContain(') AND (');
  });

  it('faceted: per-predicate join buckets AND-required with OR-group', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false, join: 'AND' },
        { type: 'tag', value: 'b', exclude: false, join: 'OR' },
        { type: 'tag', value: 'c', exclude: false, join: 'OR' },
      ],
      'AND'
    );
    expect(norm(sql)).toContain(') AND ((');
    expect(norm(sql)).toContain(') OR (');
    expect(params).toEqual(['a', 'a', 'b', 'c']);
  });

  it('per-predicate join overrides the mode argument', () => {
    const { sql } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false, join: 'OR' },
        { type: 'tag', value: 'b', exclude: false, join: 'OR' },
      ],
      'AND'
    );
    expect(norm(sql)).toContain(') OR (');
  });

  it('LEFT JOINs media_tag_by_category for an include-tag query', () => {
    const { sql, params } = buildMediaQuery(
      [{ type: 'tag', value: 'cat', exclude: false }],
      'AND'
    );
    expect(sql).toContain('LEFT JOIN media_tag_by_category mtcw');
    expect(sql).toContain('mtcw.time_stamp AS time_stamp');
    expect(sql).toContain('ORDER BY mtcw.weight');
    expect(params[0]).toBe('cat'); // join param first
  });

  it('selects NULL tag columns and no join when there is no include tag', () => {
    const { sql } = buildMediaQuery(
      [{ type: 'path', value: 'x', exclude: false }],
      'AND'
    );
    expect(sql).toContain('NULL AS weight');
    expect(sql).not.toContain('LEFT JOIN');
  });
});
