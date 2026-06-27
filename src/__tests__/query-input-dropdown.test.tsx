import { render, fireEvent } from '@testing-library/react';
import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';

// Provide a non-empty history so the dropdown *would* render if it were open —
// that's the whole point of these tests: the dropdown's open/closed state, not
// whether it has items to show.
jest.mock('../renderer/hooks/useSearchHistory', () => ({
  useSearchHistory: () => ({
    history: ['cats', 'dogs'],
    addSearch: () => {},
    removeSearch: () => {},
    clearAll: () => {},
  }),
}));

const EMPTY_QUERY: Query = { predicates: [] };

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
