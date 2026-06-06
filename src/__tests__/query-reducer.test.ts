// src/__tests__/query-reducer.test.ts
import { addPredicate, removePredicate, toggleExclude, applyTagClick, setPredicateJoin } from '../renderer/query/reducer';
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
    expect(applyTagClick(q([]), 'a', 'AND').predicates).toEqual([{ type: 'tag', value: 'a', exclude: false, join: 'AND' }]);
    const has = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(applyTagClick(has, 'a', 'AND').predicates).toEqual([]);
  });

  it('applyTagClick replaces whole query in EXCLUSIVE mode', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }, { type: 'path', value: 'p', exclude: false }]);
    expect(applyTagClick(start, 'b', 'EXCLUSIVE').predicates).toEqual([{ type: 'tag', value: 'b', exclude: false, join: 'AND' }]);
  });

  it('applyTagClick in EXCLUSIVE toggles off when clicking the only active tag', () => {
    const start = q([{ type: 'tag', value: 'b', exclude: false }]);
    expect(applyTagClick(start, 'b', 'EXCLUSIVE').predicates).toEqual([]);
  });

  it('applyTagClick in EXCLUSIVE clears when clicking an active tag among several', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }, { type: 'tag', value: 'b', exclude: false }]);
    expect(applyTagClick(start, 'a', 'EXCLUSIVE').predicates).toEqual([]);
  });

  it('applyTagClick sets join to OR in OR mode', () => {
    expect(applyTagClick(q([]), 'a', 'OR').predicates).toEqual([
      { type: 'tag', value: 'a', exclude: false, join: 'OR' },
    ]);
  });

  it('setPredicateJoin changes a predicate join by key', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false, join: 'AND' }]);
    expect(setPredicateJoin(start, 'tag:a', 'OR').predicates[0].join).toBe('OR');
  });
});
