/**
 * Regression test for unified-query persistence: a restored session whose only
 * predicate is a NON-tag filter (path / category / description / hash) must
 * still be recognised as a filtered query at boot. Previously the boot routing
 * keyed solely on persisted tags, so a path-only query was misread as "no
 * filter", routed to the filesystem view, and the predicate was wiped.
 */
// Mock the platform layer so ts-jest doesn't type-check platform.ts (which
// carries pre-existing type errors the webpack transpile-only build ignores).
// useSessionStore only needs `sessionStore` from it at runtime.
jest.mock('../renderer/platform', () => ({
  sessionStore: {
    getAll: async () => null,
    set: async () => undefined,
    setMany: async () => undefined,
    flush: async () => undefined,
    clear: async () => undefined,
    clearKeys: async () => undefined,
  },
}));

import {
  setSessionValue,
  hasPersistedTags,
  hasPersistedQuery,
  clearSession,
} from '../renderer/hooks/useSessionStore';

const makeQuery = (predicates: any[], tags: string[] = []) =>
  ({
    dbQuery: { tags },
    query: { predicates },
    mostRecentTag: '',
    mostRecentCategory: '',
    textFilter: '',
  } as any);

describe('session restore routing guards', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    clearSession();
    jest.clearAllTimers();
    jest.useRealTimers();
  });

  it('recognises a path-only predicate query as a persisted filter', () => {
    setSessionValue(
      'query',
      makeQuery([
        { id: '1', type: 'path', value: 'Some/Folder', exclude: false, join: 'AND' },
      ])
    );
    // The legacy tags-only guard misses non-tag predicates...
    expect(hasPersistedTags()).toBe(false);
    // ...but the query guard catches them, so boot routes to the DB view.
    expect(hasPersistedQuery()).toBe(true);
  });

  it('returns false for an empty persisted query', () => {
    setSessionValue('query', makeQuery([]));
    expect(hasPersistedQuery()).toBe(false);
    expect(hasPersistedTags()).toBe(false);
  });

  it('still recognises tag queries via both guards', () => {
    setSessionValue(
      'query',
      makeQuery(
        [{ id: '1', type: 'tag', value: 'blonde', exclude: false, join: 'AND' }],
        ['blonde']
      )
    );
    expect(hasPersistedTags()).toBe(true);
    expect(hasPersistedQuery()).toBe(true);
  });

  it('returns false when nothing is persisted', () => {
    clearSession();
    expect(hasPersistedQuery()).toBe(false);
    expect(hasPersistedTags()).toBe(false);
  });
});
