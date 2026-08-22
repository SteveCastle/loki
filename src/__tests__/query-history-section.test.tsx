import { render, fireEvent } from '@testing-library/react';
import QueryHistorySection, {
  HISTORY_BASE_KEY,
  historyRowKey,
} from '../renderer/components/query-input/query-history-section';
import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';
import {
  queryStateKey,
  type QueryHistoryEntry,
} from '../renderer/query/history';

const EMPTY_QUERY: Query = { predicates: [] };

// A couple of recorded filter states so the section has history rows in
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

function renderSection(
  extraProps: Partial<React.ComponentProps<typeof QueryHistorySection>> = {}
) {
  return render(
    <QueryHistorySection
      query={EMPTY_QUERY}
      text=""
      history={HISTORY}
      onApplyHistory={() => {}}
      baseLabel="pictures"
      baseCount={100}
      onApplyBase={() => {}}
      {...extraProps}
    />
  );
}

// The history now renders IN-FLOW as one more type-ahead section (it used to
// be a dropdown overlaying the live results under the input).
describe('QueryHistorySection content', () => {
  it('renders one row per history entry plus the pinned base row', () => {
    const { container } = renderSection();
    expect(container.querySelectorAll('.query-history-row')).toHaveLength(2);
    expect(container.querySelector('.query-history-base')).not.toBeNull();
    expect(
      container.querySelector('.query-history-base-label')?.textContent
    ).toBe('pictures');
  });

  it('clicking a history row applies that filter state', () => {
    const onApplyHistory = jest.fn();
    const { container } = renderSection({ onApplyHistory });
    const rows = container.querySelectorAll('.query-history-row');
    fireEvent.click(rows[1]);
    expect(onApplyHistory).toHaveBeenCalledWith(HISTORY[1]);
  });

  it('clicking the base row restores the starting library', () => {
    const onApplyBase = jest.fn();
    const { container } = renderSection({ onApplyBase });
    fireEvent.click(container.querySelector('.query-history-base')!);
    expect(onApplyBase).toHaveBeenCalledTimes(1);
  });

  it('marks the entry matching the ACTIVE query as current — and the base row when the query is empty', () => {
    // Active query = the cats entry.
    const { container } = renderSection({ query: HISTORY[0].query });
    const rows = container.querySelectorAll('.query-history-row');
    expect(rows[0].className).toContain('current');
    expect(rows[1].className).not.toContain('current');
    expect(
      container.querySelector('.query-history-base')!.className
    ).not.toContain('current');
  });

  it('with an empty query the base row is current', () => {
    const { container } = renderSection();
    expect(container.querySelector('.query-history-base')!.className).toContain(
      'current'
    );
    expect(container.querySelector('.query-history-row.current')).toBeNull();
  });

  it('typed text filters the history rows but never hides the base row', () => {
    const { container } = renderSection({ text: 'dog' });
    const rows = container.querySelectorAll('.query-history-row');
    expect(rows).toHaveLength(1);
    expect(rows[0].textContent).toContain('dogs');
    expect(container.querySelector('.query-history-base')).not.toBeNull();
  });

  it('shows the empty hint only when there is no history and nothing typed', () => {
    const { container } = renderSection({ history: [] });
    expect(container.querySelector('.query-input-empty')?.textContent).toBe(
      'Filters you apply will appear here'
    );
    const typed = renderSection({ history: [], text: 'x' });
    expect(typed.container.querySelector('.query-input-empty')).toBeNull();
  });
});

describe('QueryHistorySection keyboard-navigation integration', () => {
  it('reports its rows (history entries then base) as navigable items with actions', () => {
    const onItemsChange = jest.fn();
    const onApplyHistory = jest.fn();
    const onApplyBase = jest.fn();
    renderSection({ onItemsChange, onApplyHistory, onApplyBase });
    const items = onItemsChange.mock.calls.at(-1)![0];
    expect(items.map((i: { key: string }) => i.key)).toEqual([
      historyRowKey(HISTORY[0]),
      historyRowKey(HISTORY[1]),
      HISTORY_BASE_KEY,
    ]);
    items[1].action();
    expect(onApplyHistory).toHaveBeenCalledWith(HISTORY[1]);
    items[2].action();
    expect(onApplyBase).toHaveBeenCalledTimes(1);
  });

  it('renders the highlight on the row matching highlightedKey', () => {
    const { container } = renderSection({
      highlightedKey: HISTORY_BASE_KEY,
    });
    expect(
      container.querySelector('.query-history-base')!.className
    ).toContain('highlighted');
    expect(container.querySelector('.query-history-row.highlighted')).toBeNull();
  });
});

describe('QueryHistorySection collapsible mode (command palette)', () => {
  it('starts collapsed: header toggle only, no rows, no base row', () => {
    const { container } = renderSection({ collapsible: true });
    const toggle = container.querySelector('.query-history-toggle');
    expect(toggle).not.toBeNull();
    expect(toggle!.getAttribute('aria-expanded')).toBe('false');
    expect(toggle!.textContent).toContain('Session filters');
    expect(
      container.querySelector('.query-history-toggle-count')?.textContent
    ).toBe('2');
    expect(container.querySelectorAll('.query-history-row')).toHaveLength(0);
    expect(container.querySelector('.query-history-base')).toBeNull();
  });

  it('reports NO navigable items while collapsed, and the rows once expanded', () => {
    const onItemsChange = jest.fn();
    const { container } = renderSection({ collapsible: true, onItemsChange });
    expect(onItemsChange.mock.calls.at(-1)![0]).toEqual([]);

    fireEvent.click(container.querySelector('.query-history-toggle')!);
    expect(container.querySelectorAll('.query-history-row')).toHaveLength(2);
    expect(container.querySelector('.query-history-base')).not.toBeNull();
    expect(
      onItemsChange.mock.calls.at(-1)![0].map((i: { key: string }) => i.key)
    ).toEqual([
      historyRowKey(HISTORY[0]),
      historyRowKey(HISTORY[1]),
      HISTORY_BASE_KEY,
    ]);
  });

  it('collapses again on a second click and clears the reported items', () => {
    const onItemsChange = jest.fn();
    const { container } = renderSection({ collapsible: true, onItemsChange });
    const toggle = () => container.querySelector('.query-history-toggle')!;
    fireEvent.click(toggle());
    fireEvent.click(toggle());
    expect(toggle().getAttribute('aria-expanded')).toBe('false');
    expect(container.querySelectorAll('.query-history-row')).toHaveLength(0);
    expect(onItemsChange.mock.calls.at(-1)![0]).toEqual([]);
  });

  it('without the flag (sidebar) there is no toggle and rows always render', () => {
    const { container } = renderSection();
    expect(container.querySelector('.query-history-toggle')).toBeNull();
    expect(container.querySelectorAll('.query-history-row')).toHaveLength(2);
  });
});

describe('QueryInput result navigation (resultNavCount > 0)', () => {
  it('routes arrow keys and Enter to the parent result nav', () => {
    const onResultNavMove = jest.fn();
    const onResultNavSubmit = jest.fn();
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
        resultNavCount={3}
        onResultNavMove={onResultNavMove}
        onResultNavSubmit={onResultNavSubmit}
      />
    );
    const input = utils.container.querySelector('input') as HTMLInputElement;
    fireEvent.keyDown(input, { key: 'ArrowDown' });
    fireEvent.keyDown(input, { key: 'ArrowUp' });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onResultNavMove.mock.calls).toEqual([[1], [-1]]);
    expect(onResultNavSubmit).toHaveBeenCalledTimes(1);
  });
});
