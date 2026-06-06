// src/__tests__/query-sql.test.ts
import { buildMediaQuery } from '../main/query-sql';
import type { Predicate } from '../renderer/query/types';

const norm = (s: string) => s.replace(/\s+/g, ' ').trim();

describe('buildMediaQuery', () => {
  it('returns base query with no predicates', () => {
    const { sql, params } = buildMediaQuery([], 'AND');
    expect(norm(sql)).toBe(
      norm('SELECT media.path, media.elo, media.height, media.width, NULL AS weight, NULL AS tag_label, NULL AS time_stamp, NULL AS created_at FROM media')
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
    // Ordering is done in the renderer; no SQL sort, and description is not
    // returned (only filtered).
    expect(sql).not.toContain('ORDER BY');
    expect(sql).not.toContain('description');
    expect(params).toEqual(['portrait']);
  });

  it('builds exclude tag with NOT EXISTS', () => {
    const { sql } = buildMediaQuery([{ type: 'tag', value: 'x', exclude: true }], 'AND');
    expect(norm(sql)).toContain('NOT EXISTS');
  });

  it('drives a single category from the indexed category table (DISTINCT, no media scan)', () => {
    const { sql, params } = buildMediaQuery([{ type: 'category', value: 'Studio', exclude: false }], 'AND');
    expect(sql).toContain('SELECT DISTINCT media_path FROM media_tag_by_category WHERE category_label = ?');
    expect(sql).toContain('JOIN media ON media.path = cat.media_path');
    expect(sql).toContain('NULL AS weight'); // a category has no single per-tag weight
    expect(params).toEqual(['Studio']);
  });

  it('drives an OR-set of categories from category_label IN (DISTINCT)', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'category', value: 'A', exclude: false },
        { type: 'category', value: 'B', exclude: false },
      ],
      'OR'
    );
    expect(sql).toContain('WHERE category_label IN (?, ?)');
    expect(sql).toContain('JOIN media ON media.path = cat.media_path');
    expect(params).toEqual(['A', 'B']);
  });

  it('a tag drives over a category (category becomes an EXISTS conjunct)', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'tag', value: 't', exclude: false },
        { type: 'category', value: 'C', exclude: false },
      ],
      'AND'
    );
    expect(sql).toContain('FROM media_tag_by_category mtcw'); // tag drive
    expect(sql).toContain('mtc.category_label = ?'); // category as EXISTS
    expect(params).toEqual(['t', 'C']);
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
    expect(sql).toContain('media.description NOT LIKE ?'); // filtered...
    expect(sql).not.toContain('AS description'); // ...but never returned
    expect(sql).toContain('media.hash LIKE ?');
    expect(params).toEqual(['%a%', '%b%', '%c%']);
  });

  it('drives an OR-set of include-tags from a tag_label IN lookup', () => {
    const { sql, params } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false },
        { type: 'tag', value: 'b', exclude: false },
      ],
      'OR'
    );
    // No media scan: indexed IN over the tag table.
    expect(sql).toContain('FROM media_tag_by_category mtcw');
    expect(sql).toContain('WHERE mtcw.tag_label IN (?, ?)');
    expect(sql).not.toContain('EXISTS');
    expect(sql).not.toContain('ORDER BY');
    expect(params).toEqual(['a', 'b']);
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

  it('an OR bucket mixing tags with a non-tag falls back to a media scan', () => {
    const { sql } = buildMediaQuery(
      [
        { type: 'tag', value: 'a', exclude: false, join: 'OR' },
        { type: 'path', value: 'p', exclude: false, join: 'OR' },
      ],
      'AND'
    );
    // Mixed OR bucket (tag OR path) can't be an indexed tag IN — media scan.
    expect(sql).toContain('FROM media');
    expect(norm(sql)).toContain(') OR (');
  });

  it('exposes weight/tag/timestamp columns for an include-tag query', () => {
    const { sql, params } = buildMediaQuery(
      [{ type: 'tag', value: 'cat', exclude: false }],
      'AND'
    );
    expect(sql).toContain('mtcw.time_stamp AS time_stamp');
    expect(sql).toContain('mtcw.weight AS weight');
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
