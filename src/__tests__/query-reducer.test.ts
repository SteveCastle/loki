// src/__tests__/query-reducer.test.ts
import { addPredicate, removePredicate, toggleExclude, applyTagClick } from '../renderer/query/reducer';
import type { Query } from '../renderer/query/types';

const q = (preds: Query['predicates']): Query => ({ predicates: preds });

describe('query reducer', () => {
  it('adds a predicate', () => {
    expect(addPredicate(q([]), { type: 'tag', value: 'a', exclude: false }))
      .toEqual(q([{ type: 'tag', value: 'a', exclude: false }]));
  });

  it('does not duplicate an identical predicate', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(addPredicate(start, { type: 'tag', value: 'a', exclude: false })).toEqual(start);
  });

  it('removes a predicate by key', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }, { type: 'path', value: 'p', exclude: false }]);
    expect(removePredicate(start, '-tag:a'.replace('-', ''))).toBeDefined();
    expect(removePredicate(start, 'tag:a').predicates).toEqual([{ type: 'path', value: 'p', exclude: false }]);
  });

  it('toggles exclude on a predicate', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(toggleExclude(start, 'tag:a').predicates[0].exclude).toBe(true);
  });

  it('applyTagClick toggles a tag in non-exclusive modes', () => {
    expect(applyTagClick(q([]), 'a', 'AND').predicates).toEqual([{ type: 'tag', value: 'a', exclude: false }]);
    const has = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(applyTagClick(has, 'a', 'AND').predicates).toEqual([]);
  });

  it('applyTagClick replaces whole query in EXCLUSIVE mode', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }, { type: 'path', value: 'p', exclude: false }]);
    expect(applyTagClick(start, 'b', 'EXCLUSIVE').predicates).toEqual([{ type: 'tag', value: 'b', exclude: false }]);
  });

  it('applyTagClick in EXCLUSIVE toggles off when clicking the only active tag', () => {
    const start = q([{ type: 'tag', value: 'b', exclude: false }]);
    expect(applyTagClick(start, 'b', 'EXCLUSIVE').predicates).toEqual([]);
  });
});
