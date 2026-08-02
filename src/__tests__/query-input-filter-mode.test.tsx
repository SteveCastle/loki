import { render, fireEvent } from '@testing-library/react';
import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';

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
  history: [],
  onApplyHistory: () => {},
  baseLabel: 'pictures',
  onApplyBase: () => {},
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
