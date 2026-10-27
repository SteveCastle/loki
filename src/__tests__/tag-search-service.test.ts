// Tests for the shared tag-search singleton. In jsdom there is no Web Worker,
// so the service exercises its synchronous Fuse fallback — which is exactly the
// path we want to assert the index-idempotency guard and the minimum-query-length
// gate on. Both guards exist to keep the 176K-tag library from (a) being
// re-cloned to the worker on every consumer's first search and (b) being fuzzy
// scanned for trivially-short, match-everything queries.
import {
  cancelTagSearches,
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

  // The service holds ONE index per scope, not one shared index. The command
  // palette ('curated') and the taxonomy sidebar ('all') must not see each
  // other's data — if they collapsed back into one index, the palette would
  // silently go back to fuzzy-matching all ~189K tags per keystroke.
  it('keeps each scope indexed separately', (done) => {
    const curated = TAGS.filter((t) => t.category !== 'Suggested');
    indexTags([...TAGS], 'all');
    indexTags(curated, 'curated');

    searchTags(
      'blo',
      10,
      (curatedItems) => {
        // 'blonde_hair' is a Suggested tag: present in 'all', absent here.
        expect(curatedItems.some((t) => t.label === 'blonde_hair')).toBe(false);
        searchTags(
          'blo',
          10,
          (allItems) => {
            expect(allItems.some((t) => t.label === 'blonde_hair')).toBe(true);
            done();
          },
          'all'
        );
      },
      'curated'
    );
  });

  it('tracks the idempotency guard per scope', () => {
    const ref = [...TAGS];
    expect(indexTags(ref, 'all')).toBe(true);
    expect(indexTags(ref, 'all')).toBe(false);
    // Same reference, different scope: must still index, or the second scope
    // would be left searching whatever the first one happened to load.
    expect(indexTags(ref, 'curated')).toBe(true);
  });

  // The palette can be typed into before its scope's tags have arrived. The
  // search must be held and answered once the index lands, not dropped.
  it('answers a search issued before its scope is indexed', () => {
    // 'curated' has no index yet in this fresh module only if no earlier test
    // indexed it, so isolate the module.
    jest.isolateModules(() => {
      // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
      const svc = require('../renderer/search/tag-search-service');
      const cb = jest.fn();
      svc.searchTags('blo', 10, cb, 'curated');
      expect(cb).not.toHaveBeenCalled();
      svc.indexTags([...TAGS], 'curated');
      expect(cb).toHaveBeenCalledTimes(1);
      expect(cb.mock.calls[0][0].map((t: TagConcept) => t.label)).toContain(
        'blonde_hair'
      );
    });
  });

  it('drops a cancelled search that was waiting on its index', () => {
    jest.isolateModules(() => {
      // eslint-disable-next-line global-require, @typescript-eslint/no-var-requires
      const svc = require('../renderer/search/tag-search-service');
      const cb = jest.fn();
      svc.searchTags('blo', 10, cb, 'curated');
      svc.cancelTagSearches(cb);
      svc.indexTags([...TAGS], 'curated');
      expect(cb).not.toHaveBeenCalled();
    });
  });

  it('returns the same tag objects it was given', (done) => {
    const ref = [...TAGS];
    indexTags(ref, 'all');
    searchTags(
      'blue',
      10,
      (items) => {
        expect(items[0]).toBe(ref[1]);
        done();
      },
      'all'
    );
    expect(typeof cancelTagSearches).toBe('function');
  });
});
