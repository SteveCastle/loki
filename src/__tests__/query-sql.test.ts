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

  it('drives a single include-tag query from the indexed tag table (no media scan)', () => {
    const preds: Predicate[] = [{ type: 'tag', value: 'portrait', exclude: false }];
    const { sql, params } = buildMediaQuery(preds, 'AND');
    // Fast path: FROM media_tag_by_category filtered by tag_label, not a full
    // `media` scan with a correlated EXISTS subquery.
    expect(sql).toContain('FROM media_tag_by_category mtcw');
    expect(sql).toContain('WHERE mtcw.tag_label = ?');
    expect(sql).not.toContain('EXISTS');
    expect(sql).toContain('ORDER BY mtcw.weight');
    expect(params).toEqual(['portrait']);
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

  it('drives from the first AND-tag and adds other tags as conjunct EXISTS', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false },
        { type: 'tag', value: 'b', exclude: false },
      ],
      'AND'
    );
    expect(sql).toContain('FROM media_tag_by_category mtcw');
    expect(sql).toContain('WHERE mtcw.tag_label = ?');
    expect(sql).toContain('AND (EXISTS');
    expect(sql).toContain('mtc.tag_label = ?'); // EXISTS for tag b
    expect(params).toEqual(['a', 'b']); // drive tag a, then EXISTS b
  });

  it('treats EXCLUSIVE like AND (drives from first tag)', () => {
    const { sql, params } = buildMediaQuery(
      [{ type: 'tag', value: 'a', exclude: false }, { type: 'tag', value: 'b', exclude: false }],
      'EXCLUSIVE'
    );
    expect(sql).toContain('FROM media_tag_by_category mtcw');
    expect(sql).toContain('AND (EXISTS');
    expect(params).toEqual(['a', 'b']);
  });

  it('faceted: drives from the AND-tag and ORs the OR-bucket', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false, join: 'AND' },
        { type: 'tag', value: 'b', exclude: false, join: 'OR' },
        { type: 'tag', value: 'c', exclude: false, join: 'OR' },
      ],
      'AND'
    );
    expect(sql).toContain('FROM media_tag_by_category mtcw');
    expect(sql).toContain('WHERE mtcw.tag_label = ?');
    expect(norm(sql)).toContain(') OR ('); // OR-bucket for b/c
    expect(params).toEqual(['a', 'b', 'c']); // drive a, then EXISTS b, c
  });

  it('OR-only tags fall back to a media scan with an OR group', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false, join: 'OR' },
        { type: 'tag', value: 'b', exclude: false, join: 'OR' },
      ],
      'AND'
    );
    // Cannot drive from an optional OR-tag — fall back to FROM media + EXISTS.
    expect(sql).toContain('FROM media LEFT JOIN media_tag_by_category mtcw');
    expect(norm(sql)).toContain(') OR (');
  });

  it('exposes weight/tag/timestamp columns for an include-tag query', () => {
    const { sql, params } = buildMediaQuery(
      [{ type: 'tag', value: 'cat', exclude: false }],
      'AND'
    );
    expect(sql).toContain('mtcw.time_stamp AS time_stamp');
    expect(sql).toContain('mtcw.weight AS weight');
    expect(sql).toContain('ORDER BY mtcw.weight');
    expect(params[0]).toBe('cat');
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
