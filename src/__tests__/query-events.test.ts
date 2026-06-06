import { applyTagClick, addPredicate } from '../renderer/query/reducer';

// Documents the contract the machine relies on.
it('tag click contract produces a tag predicate', () => {
  expect(applyTagClick({ predicates: [] }, 'x', 'AND').predicates[0]).toEqual({
    type: 'tag',
    value: 'x',
    exclude: false,
    join: 'AND',
  });
});
it('add predicate contract', () => {
  expect(
    addPredicate({ predicates: [] }, { type: 'path', value: 'p', exclude: false })
      .predicates
  ).toHaveLength(1);
});
