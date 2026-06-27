// src/__tests__/query-parse.test.ts
import { parseQuery, serializeQuery } from '../renderer/query/parse';
import type { Predicate } from '../renderer/query/types';

describe('parseQuery', () => {
  it('parses tag, category, path, description, hash prefixes', () => {
    expect(parseQuery('#portrait in:Studio path:2023/ description:sunset hash:abc'))
      .toEqual<Predicate[]>([
        { type: 'tag', value: 'portrait', exclude: false },
        { type: 'category', value: 'Studio', exclude: false },
        { type: 'path', value: '2023/', exclude: false },
        { type: 'description', value: 'sunset', exclude: false },
        { type: 'hash', value: 'abc', exclude: false },
      ]);
  });

  it('honors leading - as exclude on any prefix', () => {
    expect(parseQuery('-#blurry -path:tmp')).toEqual<Predicate[]>([
      { type: 'tag', value: 'blurry', exclude: true },
      { type: 'path', value: 'tmp', exclude: true },
    ]);
  });

  it('keeps quoted phrases intact', () => {
    expect(parseQuery('description:"studio session"')).toEqual<Predicate[]>([
      { type: 'description', value: 'studio session', exclude: false },
    ]);
  });

  it('treats a bare token as a description predicate', () => {
    expect(parseQuery('sunset')).toEqual<Predicate[]>([
      { type: 'description', value: 'sunset', exclude: false },
    ]);
  });

  it('round-trips through serializeQuery', () => {
    const preds = parseQuery('#portrait -#blurry in:Studio path:"a b" hash:xyz');
    expect(parseQuery(serializeQuery({ predicates: preds }))).toEqual(preds);
  });

  it('ignores empty tokens', () => {
    expect(parseQuery('   ')).toEqual([]);
  });
});
