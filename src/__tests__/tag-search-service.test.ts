// Tests for the shared tag-search singleton. In jsdom there is no Web Worker,
// so the service exercises its synchronous Fuse fallback — which is exactly the
// path we want to assert the index-idempotency guard and the minimum-query-length
// gate on. Both guards exist to keep the 176K-tag library from (a) being
// re-cloned to the worker on every consumer's first search and (b) being fuzzy
// scanned for trivially-short, match-everything queries.
import {
  indexTags,
  searchTags,
  type TagConcept,
} from '../renderer/search/tag-search-service';

const TAGS: TagConcept[] = [
  { label: 'blonde_hair', category: 'Suggested', weight: 0 },
  { label: 'blue_eyes', category: 'Suggested', weight: 0 },
  { label: 'Some Artist', category: 'Artists', weight: 0 },
];

describe('tag-search-service', () => {
  it('rebuilds the index only when the source reference changes', () => {
    const ref = [...TAGS];
    // First time with a fresh reference builds the index.
    expect(indexTags(ref)).toBe(true);
    // Same reference again is a no-op — this is what stops every consumer
    // (warmer, taxonomy, palette) from re-cloning the full tag set to the
    // worker on its first search.
    expect(indexTags(ref)).toBe(false);
    // A genuinely new reference (e.g. after a mutation refetch) rebuilds.
    expect(indexTags([...TAGS])).toBe(true);
  });

  it('returns no matches below the minimum query length', (done) => {
    indexTags([...TAGS]);
    searchTags('b', 10, (items) => {
      expect(items).toEqual([]);
      done();
    });
  });

  it('searches once the query reaches the minimum length', (done) => {
    indexTags([...TAGS]);
    searchTags('blo', 10, (items) => {
      expect(items.length).toBeGreaterThan(0);
      expect(items.some((t) => t.label === 'blonde_hair')).toBe(true);
      done();
    });
  });
});
