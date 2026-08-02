// The session filter-state history: identity, dedupe-to-top, cap, and the
// empty-query exclusion (the empty state is the FS base, pinned separately).
import {
  MAX_QUERY_HISTORY,
  pushQueryHistory,
  queryStateKey,
  type QueryHistoryEntry,
} from '../renderer/query/history';
import type { Query } from '../renderer/query/types';

const tagQuery = (label: string): Query => ({
  predicates: [{ type: 'tag', value: label, exclude: false }],
});

describe('queryStateKey', () => {
  it('is stable for equivalent queries and differs across meaningful changes', () => {
    const a = tagQuery('cats');
    const b = tagQuery('cats');
    expect(queryStateKey(a)).toBe(queryStateKey(b));

    // Every semantic field participates in identity.
    expect(queryStateKey(tagQuery('dogs'))).not.toBe(queryStateKey(a));
    expect(
      queryStateKey({
        predicates: [{ type: 'tag', value: 'cats', exclude: true }],
      })
    ).not.toBe(queryStateKey(a));
    expect(
      queryStateKey({
        predicates: [{ type: 'tag', value: 'cats', exclude: false, join: 'OR' }],
      })
    ).not.toBe(queryStateKey(a));
    expect(
      queryStateKey({
        predicates: [
          {
            type: 'similar',
            value: 'a.jpg',
            exclude: false,
            nodes: [{ kind: 'text', value: 'night', weight: 0.5 }],
          },
        ],
      })
    ).not.toBe(
      queryStateKey({
        predicates: [
          {
            type: 'similar',
            value: 'a.jpg',
            exclude: false,
            nodes: [{ kind: 'text', value: 'night', weight: 0.7 }],
          },
        ],
      })
    );
  });

  it('is order-sensitive (predicate order changes left-to-right joins)', () => {
    const ab: Query = {
      predicates: [
        { type: 'tag', value: 'a', exclude: false },
        { type: 'tag', value: 'b', exclude: false },
      ],
    };
    const ba: Query = {
      predicates: [
        { type: 'tag', value: 'b', exclude: false },
        { type: 'tag', value: 'a', exclude: false },
      ],
    };
    expect(queryStateKey(ab)).not.toBe(queryStateKey(ba));
  });
});

describe('pushQueryHistory', () => {
  it('never records an empty query (the empty state is the base)', () => {
    const history: QueryHistoryEntry[] = [];
    expect(pushQueryHistory(history, { predicates: [] }, 10)).toBe(history);
  });

  it('prepends new states and snapshots them against later mutation', () => {
    const q = tagQuery('cats');
    const history = pushQueryHistory([], q, 3, 1000);
    expect(history).toHaveLength(1);
    expect(history[0]).toMatchObject({ count: 3, at: 1000 });
    // Mutating the live query must not rewrite the recorded entry.
    q.predicates[0].value = 'dogs';
    expect(history[0].query.predicates[0].value).toBe('cats');
  });

  it('re-running a recorded state moves it to the top and refreshes count/time', () => {
    let history = pushQueryHistory([], tagQuery('cats'), 3, 1000);
    history = pushQueryHistory(history, tagQuery('dogs'), 5, 2000);
    history = pushQueryHistory(history, tagQuery('cats'), 4, 3000);
    expect(history).toHaveLength(2);
    expect(history[0].query.predicates[0].value).toBe('cats');
    expect(history[0].count).toBe(4);
    expect(history[0].at).toBe(3000);
    expect(history[1].query.predicates[0].value).toBe('dogs');
  });

  it(`caps the history at ${MAX_QUERY_HISTORY} states, dropping the oldest`, () => {
    let history: QueryHistoryEntry[] = [];
    for (let i = 0; i < MAX_QUERY_HISTORY + 2; i += 1) {
      history = pushQueryHistory(history, tagQuery(`t${i}`), i, i);
    }
    expect(history).toHaveLength(MAX_QUERY_HISTORY);
    // Newest first; the two oldest (t0, t1) fell off.
    expect(history[0].query.predicates[0].value).toBe(
      `t${MAX_QUERY_HISTORY + 1}`
    );
    expect(
      history.some((e) => e.query.predicates[0].value === 't0')
    ).toBe(false);
    expect(
      history.some((e) => e.query.predicates[0].value === 't1')
    ).toBe(false);
  });
});
