import { render, fireEvent } from '@testing-library/react';
import QueryInput, { formatLimit } from '../renderer/components/query-input/QueryInput';
import { widenPredicate } from '../renderer/query/reducer';
import { queryStateKey } from '../renderer/query/history';
import { predicateKey, similarityLimit, SIMILARITY_LIMIT_MAX, type Query } from '../renderer/query/types';

const q = (preds: Query['predicates']): Query => ({ predicates: preds });

describe('widening a similarity chip', () => {
  it('adds 1000 per press, starting from the 1000 default', () => {
    const start = q([{ type: 'visual', value: 'red car', exclude: false }]);
    const key = predicateKey(start.predicates[0]);
    const once = widenPredicate(start, key);
    expect(once.predicates[0].limit).toBe(2000);
    expect(widenPredicate(once, key).predicates[0].limit).toBe(3000);
    // the chip keeps its identity (same key) so it stays the same chip
    expect(predicateKey(once.predicates[0])).toBe(key);
  });

  it('only touches similarity chips, and stops at the server ceiling', () => {
    const tag = q([{ type: 'tag', value: 'cats', exclude: false }]);
    expect(widenPredicate(tag, predicateKey(tag.predicates[0]))).toEqual(tag);
    const top = q([{ type: 'similar', value: 'a.jpg', exclude: false, limit: SIMILARITY_LIMIT_MAX }]);
    expect(widenPredicate(top, predicateKey(top.predicates[0])).predicates[0].limit).toBe(SIMILARITY_LIMIT_MAX);
  });

  it('a widened chip is a different history state; an unwidened one keeps its old key', () => {
    const base = q([{ type: 'visual', value: 'red car', exclude: false }]);
    expect(queryStateKey(base)).toBe(JSON.stringify([['visual', 'red car', false, '', '', null, [], 'blend']]));
    expect(queryStateKey(widenPredicate(base, predicateKey(base.predicates[0])))).not.toBe(queryStateKey(base));
  });

  it('formats the cap compactly', () => {
    expect(formatLimit(2000)).toBe('2k');
    expect(formatLimit(12500)).toBe('12.5k');
    expect(similarityLimit({ type: 'clip', value: 'x', exclude: false })).toBe(1000);
  });
});

describe('QueryInput widen button', () => {
  const renderInput = (query: Query, onWidenPredicate?: (key: string) => void) =>
    render(
      <QueryInput
        query={query}
        textValue=""
        onTextChange={() => {}}
        onSubmitText={() => {}}
        onRemovePredicate={() => {}}
        onToggleExclude={() => {}}
        onSetPredicateJoin={() => {}}
        onWidenPredicate={onWidenPredicate}
        onClearAll={() => {}}
        onClearText={() => {}}
      />
    );

  it('shows on similarity chips only and widens without toggling exclude', () => {
    const onWiden = jest.fn();
    const query = q([
      { type: 'tag', value: 'cats', exclude: false },
      { type: 'visual', value: 'red car', exclude: false, join: 'AND' },
    ]);
    const { container } = renderInput(query, onWiden);
    const buttons = container.querySelectorAll('.query-chip-widen');
    expect(buttons).toHaveLength(1);
    expect(buttons[0].getAttribute('title')).toContain('1,000 most similar');
    fireEvent.click(buttons[0]);
    expect(onWiden).toHaveBeenCalledWith(predicateKey(query.predicates[1]));
  });

  it('shows the widened cap on the chip', () => {
    const { container } = renderInput(q([{ type: 'similar', value: 'a.jpg', exclude: false, limit: 3000 }]), jest.fn());
    expect(container.querySelector('.query-chip-widen-count')?.textContent).toBe('3k');
  });

  it('is absent when the host does not support widening', () => {
    const { container } = renderInput(q([{ type: 'visual', value: 'x', exclude: false }]));
    expect(container.querySelector('.query-chip-widen')).toBeNull();
  });
});
