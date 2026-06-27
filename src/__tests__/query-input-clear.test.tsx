import { render, fireEvent } from '@testing-library/react';
import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';

// Mock the search-history hook so the test doesn't transitively pull in
// platform.ts (which carries unrelated, pre-existing type errors that would
// fail ts-jest). The clear button doesn't depend on history behaviour.
jest.mock('../renderer/hooks/useSearchHistory', () => ({
  useSearchHistory: () => ({
    history: [],
    addSearch: () => {},
    removeSearch: () => {},
    clearAll: () => {},
  }),
}));

// The clear button must distinguish two cases:
//  - filters present  -> full clear (chips + text), which resets the library.
//  - only text typed  -> clear the text only; a no-op on the library.
function renderInput(query: Query, textValue: string) {
  const onClearAll = jest.fn();
  const onClearText = jest.fn();
  const utils = render(
    <QueryInput
      query={query}
      textValue={textValue}
      onTextChange={() => {}}
      onSubmitText={() => {}}
      onRemovePredicate={() => {}}
      onToggleExclude={() => {}}
      onSetPredicateJoin={() => {}}
      onClearAll={onClearAll}
      onClearText={onClearText}
    />
  );
  const clearButton = utils.container.querySelector(
    '.query-input-clear'
  ) as HTMLButtonElement;
  return { onClearAll, onClearText, clearButton };
}

describe('QueryInput clear button', () => {
  it('clears everything (library reset) when predicates are present', () => {
    const query: Query = {
      predicates: [{ type: 'tag', value: 'cats', exclude: false }],
    };
    const { onClearAll, onClearText, clearButton } = renderInput(query, 'dog');
    fireEvent.click(clearButton);
    expect(onClearAll).toHaveBeenCalledTimes(1);
    expect(onClearText).not.toHaveBeenCalled();
  });

  it('clears only the text (no library reset) when no predicates are present', () => {
    const { onClearAll, onClearText, clearButton } = renderInput(
      { predicates: [] },
      'dog'
    );
    fireEvent.click(clearButton);
    expect(onClearText).toHaveBeenCalledTimes(1);
    expect(onClearAll).not.toHaveBeenCalled();
  });
});
