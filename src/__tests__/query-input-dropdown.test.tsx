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
