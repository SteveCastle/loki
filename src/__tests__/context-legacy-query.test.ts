import { buildLegacyQuery } from '../renderer/components/controls/context-query';
import type { Predicate } from '../renderer/query/types';

const p = (
  type: Predicate['type'],
  value: string,
  extra: Partial<Predicate> = {}
): Predicate => ({ type, value, exclude: false, ...extra });

test('single include predicate is unparenthesized', () => {
  expect(buildLegacyQuery([p('tag', 'cat')], 'AND')).toBe('tag:"cat"');
});

test('exclude predicate becomes NOT and is left-assoc joined', () => {
  const preds = [p('tag', 'cat'), p('description', 'blurry', { exclude: true, join: 'AND' })];
  expect(buildLegacyQuery(preds, 'AND')).toBe('(tag:"cat" AND NOT description:"blurry")');
});

test('per-predicate OR join overrides the default mode', () => {
  const preds = [p('tag', 'a'), p('path', 'photos', { join: 'OR' })];
  expect(buildLegacyQuery(preds, 'AND')).toBe('(tag:"a" OR path:"photos")');
});

test('left-associative composition is parenthesized for 3+ predicates', () => {
  const preds = [p('tag', 'a'), p('tag', 'b', { join: 'OR' }), p('path', 'c', { join: 'AND' })];
  expect(buildLegacyQuery(preds, 'AND')).toBe('((tag:"a" OR tag:"b") AND path:"c")');
});

test('visual/similar and empty-value predicates are dropped', () => {
  const preds = [
    p('visual', 'red car'),
    p('similar', '/a.jpg'),
    p('tag', ''),
    p('tag', 'cat'),
  ];
  expect(buildLegacyQuery(preds, 'AND')).toBe('tag:"cat"');
});

test('EXCLUSIVE mode uses AND as the default join', () => {
  const preds = [p('tag', 'a'), p('tag', 'b')];
  expect(buildLegacyQuery(preds, 'EXCLUSIVE')).toBe('(tag:"a" AND tag:"b")');
});

test('no representable predicates yields empty string', () => {
  expect(buildLegacyQuery([p('visual', 'x')], 'AND')).toBe('');
  expect(buildLegacyQuery([], 'AND')).toBe('');
});
