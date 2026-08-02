import { render, fireEvent } from '@testing-library/react';
import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';
import {
  queryStateKey,
  type QueryHistoryEntry,
} from '../renderer/query/history';

const EMPTY_QUERY: Query = { predicates: [] };

// A couple of recorded filter states so the dropdown has history rows in
// addition to the always-present base row. Keys use the real queryStateKey so
// the "current" marking (which compares against the live query's key) works.
const CATS_QUERY: Query = {
  predicates: [{ type: 'tag', value: 'cats', exclude: false }],
};
const DOGS_QUERY: Query = {
  predicates: [{ type: 'tag', value: 'dogs', exclude: false }],
};
const HISTORY: QueryHistoryEntry[] = [
  { key: queryStateKey(CATS_QUERY), query: CATS_QUERY, count: 12, at: Date.now() },
  { key: queryStateKey(DOGS_QUERY), query: DOGS_QUERY, count: 7, at: Date.now() },
];

function renderInput(extraProps: Partial<React.ComponentProps<typeof QueryInput>> = {}) {
  const utils = render(
    <QueryInput
      query={EMPTY_QUERY}
      textValue=""
      onTextChange={() => {}}
      onSubmitText={() => {}}
      onRemovePredicate={() => {}}
      onToggleExclude={() => {}}
      onSetPredicateJoin={() => {}}
      onClearAll={() => {}}
      onClearText={() => {}}
      history={HISTORY}
      onApplyHistory={() => {}}
      baseLabel="pictures"
      baseCount={100}
      onApplyBase={() => {}}
      {...extraProps}
    />
  );
  const dropdown = () => utils.container.querySelector('.query-input-dropdown');
  const input = utils.container.querySelector('input') as HTMLInputElement;
  return { ...utils, dropdown, input };
}

describe('QueryInput history dropdown', () => {
  it('does NOT open on mount even with autoFocus (programmatic focus is not user intent)', () => {
    const { dropdown } = renderInput({ autoFocus: true });
    expect(dropdown()).toBeNull();
  });

  it('opens when the user starts typing', () => {
    const { dropdown, input } = renderInput({ autoFocus: true });
    fireEvent.change(input, { target: { value: 'c' } });
    expect(dropdown()).not.toBeNull();
  });

  it('opens when the user intentionally mouses down on the input', () => {
    const { dropdown, input } = renderInput();
    fireEvent.mouseDown(input);
    expect(dropdown()).not.toBeNull();
  });
});

describe('QueryInput filter-state history dropdown content', () => {
  it('renders one row per history entry plus the pinned base row', () => {
    const { input, container } = renderInput();
    fireEvent.mouseDown(input);
    expect(container.querySelectorAll('.query-history-row')).toHaveLength(2);
    expect(container.querySelector('.query-history-base')).not.toBeNull();
    expect(
      container.querySelector('.query-history-base-label')?.textContent
    ).toBe('pictures');
  });

  it('clicking a history row applies that filter state', () => {
    const onApplyHistory = jest.fn();
    const { input, container } = renderInput({ onApplyHistory });
    fireEvent.mouseDown(input);
    const rows = container.querySelectorAll('.query-history-row');
    fireEvent.click(rows[1]);
    expect(onApplyHistory).toHaveBeenCalledWith(HISTORY[1]);
  });

  it('clicking the base row restores the starting library', () => {
    const onApplyBase = jest.fn();
    const { input, container } = renderInput({ onApplyBase });
    fireEvent.mouseDown(input);
    fireEvent.click(container.querySelector('.query-history-base')!);
    expect(onApplyBase).toHaveBeenCalledTimes(1);
  });

  it('marks the entry matching the ACTIVE query as current — and the base row when the query is empty', () => {
    // Active query = the cats entry.
    const { input, container } = renderInput({ query: HISTORY[0].query });
    fireEvent.mouseDown(input);
    const rows = container.querySelectorAll('.query-history-row');
    expect(rows[0].className).toContain('current');
    expect(rows[1].className).not.toContain('current');
    expect(
      container.querySelector('.query-history-base')!.className
    ).not.toContain('current');
  });

  it('with an empty query the base row is current', () => {
    const { input, container } = renderInput();
    fireEvent.mouseDown(input);
    expect(container.querySelector('.query-history-base')!.className).toContain(
      'current'
    );
    expect(container.querySelector('.query-history-row.current')).toBeNull();
  });

  it('typed text filters the history rows but never hides the base row', () => {
    const { input, container } = renderInput({ textValue: 'dog' });
    fireEvent.mouseDown(input);
    const rows = container.querySelectorAll('.query-history-row');
    expect(rows).toHaveLength(1);
    expect(rows[0].textContent).toContain('dogs');
    expect(container.querySelector('.query-history-base')).not.toBeNull();
  });

  it('keyboard: ArrowDown past the last entry reaches the base row, Enter applies it', () => {
    const onApplyBase = jest.fn();
    const { input } = renderInput({ onApplyBase });
    fireEvent.keyDown(input, { key: 'ArrowDown' }); // open, highlight row 0
    fireEvent.keyDown(input, { key: 'ArrowDown' }); // row 1
    fireEvent.keyDown(input, { key: 'ArrowDown' }); // base row
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onApplyBase).toHaveBeenCalledTimes(1);
  });
});

describe('QueryInput result navigation (resultNavCount > 0)', () => {
  it('routes arrow keys and Enter to the parent result nav', () => {
    const onResultNavMove = jest.fn();
    const onResultNavSubmit = jest.fn();
    const { input } = renderInput({
      resultNavCount: 3,
      onResultNavMove,
      onResultNavSubmit,
    });
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onResultNavMove.mock.calls).toEqual([[1], [-1]]);
    expect(onResultNavSubmit).toHaveBeenCalledTimes(1);
  });

  it('suppresses the history dropdown so it does not overlap the results', () => {
    const { dropdown, input } = renderInput({ resultNavCount: 3 });
    // Typing would normally open the dropdown; with result nav active it stays
    // closed because the parent's results surface is the navigation target.
    fireEvent.change(input, { target: { value: 'c' } });
    expect(dropdown()).toBeNull();
  });
});
