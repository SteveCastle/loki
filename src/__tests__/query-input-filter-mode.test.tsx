import { render, fireEvent } from '@testing-library/react';
import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';

// Mock the search-history hook so the test doesn't transitively pull in
// platform.ts (which carries unrelated, pre-existing type errors that would
// fail ts-jest). The filter-mode toggle doesn't depend on history behaviour.
jest.mock('../renderer/hooks/useSearchHistory', () => ({
  useSearchHistory: () => ({
    history: [],
    addSearch: () => {},
    removeSearch: () => {},
    clearAll: () => {},
  }),
}));

const baseProps = {
  query: { predicates: [] } as Query,
  textValue: '',
  onTextChange: () => {},
  onSubmitText: () => {},
  onRemovePredicate: () => {},
  onToggleExclude: () => {},
  onSetPredicateJoin: () => {},
  onClearAll: () => {},
  onClearText: () => {},
};

describe('QueryInput filtering-mode toggle', () => {
  it('does not render the toggle when no filteringMode is supplied', () => {
    const utils = render(<QueryInput {...baseProps} />);
    expect(
      utils.container.querySelector('.query-input-filter-mode')
    ).toBeNull();
  });

  it('renders the toggle and cycles the mode when supplied', () => {
    const onCycleFilterMode = jest.fn();
    const utils = render(
      <QueryInput
        {...baseProps}
        filteringMode="AND"
        onCycleFilterMode={onCycleFilterMode}
      />
    );
    const button = utils.container.querySelector(
      '.query-input-filter-mode'
    ) as HTMLButtonElement;
    expect(button).not.toBeNull();
    fireEvent.click(button);
    expect(onCycleFilterMode).toHaveBeenCalledTimes(1);
  });

  it('keeps the submit and clear buttons alongside the toggle (additive)', () => {
    // The old query-syntax help button was removed with the cheat sheet;
    // the toggle must coexist with the remaining input buttons.
    const utils = render(
      <QueryInput
        {...baseProps}
        filteringMode="OR"
        onCycleFilterMode={() => {}}
      />
    );
    expect(
      utils.container.querySelector('.query-input-submit')
    ).not.toBeNull();
    expect(
      utils.container.querySelector('.query-input-clear')
    ).not.toBeNull();
    expect(
      utils.container.querySelector('.query-input-filter-mode')
    ).not.toBeNull();
  });
});
