// src/__tests__/query-reducer.test.ts
import { addPredicate, removePredicate, toggleExclude, applyTagClick, setPredicateJoin, updatePredicateBlend, tagsFromQuery, addPredicateWithMode, addOrMergeSimilarityPredicate, addBlendNode, removeBlendNode, updateBlendNode, effectiveBlendNodes } from '../renderer/query/reducer';
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

  it('addPredicateWithMode appends in AND/OR', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(
      addPredicateWithMode(start, { type: 'path', value: 'p', exclude: false }, 'AND').predicates
    ).toEqual([
      { type: 'tag', value: 'a', exclude: false },
      { type: 'path', value: 'p', exclude: false },
    ]);
  });

  it('addPredicateWithMode replaces the whole query in EXCLUSIVE for any predicate type', () => {
    const start = q([
      { type: 'tag', value: 'a', exclude: false },
      { type: 'category', value: 'Studio', exclude: false },
    ]);
    expect(
      addPredicateWithMode(start, { type: 'path', value: 'p', exclude: false }, 'EXCLUSIVE').predicates
    ).toEqual([{ type: 'path', value: 'p', exclude: false }]);
  });

  it('removes a predicate by key', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }, { type: 'path', value: 'p', exclude: false }]);
    expect(removePredicate(start, '-tag:a'.replace('-', ''))).toBeDefined();
    expect(removePredicate(start, 'tag:a').predicates).toEqual([{ type: 'path', value: 'p', exclude: false }]);
  });

  it('toggles exclude when another include predicate remains', () => {
    const start = q([
      { type: 'tag', value: 'a', exclude: false },
      { type: 'tag', value: 'b', exclude: false },
    ]);
    const next = toggleExclude(start, 'tag:a');
    expect(next.predicates[0].exclude).toBe(true);
  });

  it('does NOT toggle the only include predicate to exclude (all-exclude scans the whole DB)', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(toggleExclude(start, 'tag:a')).toBe(start); // unchanged no-op
  });

  it('does NOT toggle the last include to exclude when others are already excluded', () => {
    const start = q([
      { type: 'tag', value: 'a', exclude: false },
      { type: 'tag', value: 'b', exclude: true },
    ]);
    expect(toggleExclude(start, 'tag:a')).toBe(start); // would leave zero includes
  });

  it('always allows toggling an exclude back to include', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: true }]);
    expect(toggleExclude(start, '-tag:a').predicates[0].exclude).toBe(false);
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

  it('updatePredicateBlend sets text and weight on the matching predicate only', () => {
    const start = q([
      { type: 'similar', value: 'C:/a.png', exclude: false },
      { type: 'tag', value: 'a', exclude: false },
    ]);
    const next = updatePredicateBlend(start, 'similar:C:/a.png', {
      text: 'at night',
      textWeight: 0.3,
    });
    expect(next.predicates[0]).toEqual({
      type: 'similar',
      value: 'C:/a.png',
      exclude: false,
      text: 'at night',
      textWeight: 0.3,
    });
    expect(next.predicates[1]).toEqual({ type: 'tag', value: 'a', exclude: false });
  });

  it('updatePredicateBlend keeps the predicate key stable (blend fields are not identity)', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = updatePredicateBlend(start, 'similar:C:/a.png', {
      text: 'x',
      textWeight: 0.5,
    });
    // Same key → a second update targets the same chip.
    const again = updatePredicateBlend(next, 'similar:C:/a.png', {
      text: 'y',
      textWeight: 0.8,
    });
    expect(again.predicates[0].text).toBe('y');
    expect(again.predicates[0].textWeight).toBe(0.8);
  });

  it('updatePredicateBlend clears the whole blend when text is emptied', () => {
    const start = q([
      { type: 'similar', value: 'C:/a.png', exclude: false, text: 'x', textWeight: 0.5 },
    ]);
    const next = updatePredicateBlend(start, 'similar:C:/a.png', { text: '' });
    expect(next.predicates[0]).toEqual({
      type: 'similar',
      value: 'C:/a.png',
      exclude: false,
    });
    expect('text' in next.predicates[0]).toBe(false);
    expect('textWeight' in next.predicates[0]).toBe(false);
  });

  it('updatePredicateBlend is a no-op for a missing key', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }]);
    expect(
      updatePredicateBlend(start, 'similar:nope', { text: 'x' }).predicates
    ).toEqual(start.predicates);
  });

  it('tagsFromQuery returns only included tag values', () => {
    const q2 = q([
      { type: 'tag', value: 'a', exclude: false },
      { type: 'tag', value: 'b', exclude: true },
      { type: 'path', value: 'p', exclude: false },
      { type: 'tag', value: 'c', exclude: false },
    ]);
    expect(tagsFromQuery(q2)).toEqual(['a', 'c']);
  });
});

describe('composite blend nodes', () => {
  it('EXCLUSIVE still replaces the whole query with the new similarity predicate', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'similar', value: 'C:/b.png', exclude: false },
      'EXCLUSIVE'
    );
    expect(next.predicates).toEqual([
      { type: 'similar', value: 'C:/b.png', exclude: false },
    ]);
  });

  it('AND merges a new similar image into the existing similarity chip as a node', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'similar', value: 'C:/b.png', exclude: false },
      'AND'
    );
    expect(next.predicates).toHaveLength(1);
    expect(next.predicates[0].nodes).toEqual([
      { kind: 'image', value: 'C:/b.png' },
    ]);
  });

  it('AND merges a clip capture into an existing similar chip', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'clip', value: 'data:image/png;base64,xyz', exclude: false },
      'OR'
    );
    expect(next.predicates).toHaveLength(1);
    expect(next.predicates[0].nodes).toEqual([
      { kind: 'clip', value: 'data:image/png;base64,xyz' },
    ]);
  });

  it('AND with no existing similarity chip appends a normal predicate', () => {
    const start = q([{ type: 'tag', value: 'a', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'similar', value: 'C:/b.png', exclude: false },
      'AND'
    );
    expect(next.predicates).toHaveLength(2);
  });

  it('non-similarity predicates keep plain addPredicateWithMode behavior', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'tag', value: 't', exclude: false },
      'AND'
    );
    expect(next.predicates).toHaveLength(2);
  });

  it('merge dedupes against the base value and existing nodes', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const same = addOrMergeSimilarityPredicate(
      start,
      { type: 'similar', value: 'C:/a.png', exclude: false },
      'AND'
    );
    expect(same.predicates[0].nodes).toBeUndefined();
    const once = addBlendNode(start, 'similar:C:/a.png', {
      kind: 'image',
      value: 'C:/b.png',
    });
    const twice = addBlendNode(once, 'similar:C:/a.png', {
      kind: 'image',
      value: 'C:/b.png',
    });
    expect(twice.predicates[0].nodes).toHaveLength(1);
  });

  it('addBlendNode migrates the legacy text blend into nodes[0]', () => {
    const start = q([
      { type: 'similar', value: 'C:/a.png', exclude: false, text: 'night', textWeight: 0.7 },
    ]);
    const next = addBlendNode(start, 'similar:C:/a.png', {
      kind: 'text',
      value: 'rain',
      weight: 0.5,
    });
    const p = next.predicates[0];
    expect(p.text).toBeUndefined();
    expect(p.textWeight).toBeUndefined();
    expect(p.nodes).toEqual([
      { kind: 'text', value: 'night', weight: 0.7 },
      { kind: 'text', value: 'rain', weight: 0.5 },
    ]);
  });

  it('effectiveBlendNodes shows the legacy text without mutating the predicate', () => {
    const p = { type: 'similar' as const, value: 'C:/a.png', exclude: false, text: 'night' };
    expect(effectiveBlendNodes(p)).toEqual([
      { kind: 'text', value: 'night', weight: 0.5 },
    ]);
    expect(p.text).toBe('night');
  });

  it('updateBlendNode toggles negative and adjusts weight', () => {
    const start = addBlendNode(
      q([{ type: 'similar', value: 'C:/a.png', exclude: false }]),
      'similar:C:/a.png',
      { kind: 'text', value: 'blurry', weight: 0.5 }
    );
    const neg = updateBlendNode(start, 'similar:C:/a.png', 0, { negative: true });
    expect(neg.predicates[0].nodes![0].negative).toBe(true);
    const rew = updateBlendNode(neg, 'similar:C:/a.png', 0, { weight: 0.2 });
    expect(rew.predicates[0].nodes![0]).toEqual({
      kind: 'text',
      value: 'blurry',
      weight: 0.2,
      negative: true,
    });
  });

  it('removeBlendNode drops the node and clears nodes when empty', () => {
    const start = addBlendNode(
      q([{ type: 'similar', value: 'C:/a.png', exclude: false }]),
      'similar:C:/a.png',
      { kind: 'text', value: 'x' }
    );
    const next = removeBlendNode(start, 'similar:C:/a.png', 0);
    expect(next.predicates[0].nodes).toBeUndefined();
  });
});

describe('composing from a visual (text) base', () => {
  it('AND merges a similar image into an existing visual chip as a node', () => {
    const start = q([{ type: 'visual', value: 'a red car', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'similar', value: 'C:/b.png', exclude: false },
      'AND'
    );
    expect(next.predicates).toHaveLength(1);
    expect(next.predicates[0].type).toBe('visual');
    expect(next.predicates[0].nodes).toEqual([
      { kind: 'image', value: 'C:/b.png' },
    ]);
  });

  it('AND merges a new visual text into an existing similarity chip at half strength', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'visual', value: 'at night', exclude: false },
      'OR'
    );
    expect(next.predicates).toHaveLength(1);
    expect(next.predicates[0].nodes).toEqual([
      { kind: 'text', value: 'at night', weight: 0.5 },
    ]);
  });

  it('EXCLUSIVE keeps a visual add as a full replace', () => {
    const start = q([{ type: 'similar', value: 'C:/a.png', exclude: false }]);
    const next = addOrMergeSimilarityPredicate(
      start,
      { type: 'visual', value: 'at night', exclude: false },
      'EXCLUSIVE'
    );
    expect(next.predicates).toEqual([
      { type: 'visual', value: 'at night', exclude: false },
    ]);
  });

  it('text stacking dedupes against an identical text node', () => {
    const start = q([{ type: 'visual', value: 'a red car', exclude: false }]);
    const once = addOrMergeSimilarityPredicate(
      start,
      { type: 'visual', value: 'at night', exclude: false },
      'AND'
    );
    const twice = addOrMergeSimilarityPredicate(
      once,
      { type: 'visual', value: 'at night', exclude: false },
      'AND'
    );
    expect(twice.predicates[0].nodes).toHaveLength(1);
  });
});
