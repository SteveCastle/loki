import { render, fireEvent, act } from '@testing-library/react';

// Regression tests for the blend popover on similarity chips getting STUCK
// open. The old close logic latched hover/focus into refs cleared only by
// mouseleave/blur events; two ways those events never arrive:
//
//  1. The weight slider is deliberately remounted on every committed weight
//     (`key={'w'+pct}`) — commit while it has focus and React destroys the
//     focused element WITHOUT firing onBlur, so the focus latch stayed true
//     and every scheduled close bailed out (without rescheduling) forever.
//  2. There was no outside-click handler at all, so once wedged, nothing
//     short of removing the chip could close the popover.
//
// The fix makes the close timer consult the live DOM (hover/focus-within) and
// re-arm while the popover is genuinely in use, and adds a document-level
// pointerdown that hard-closes on any press outside the chip+popover.

jest.mock('../renderer/hooks/useSearchHistory', () => ({
  __esModule: true,
  useSearchHistory: () => ({
    history: [],
    addSearch: jest.fn(),
    removeSearch: jest.fn(),
    clearAll: jest.fn(),
  }),
}));

jest.mock('../renderer/hooks/useMeaningMode', () => ({
  __esModule: true,
  useMeaningMode: () => ({ meaningMode: false, setMeaningMode: jest.fn() }),
}));

jest.mock('../renderer/platform', () => ({
  __esModule: true,
  mediaUrl: (p: string) => `http://server/media/${encodeURIComponent(p)}`,
}));

import QueryInput from '../renderer/components/query-input/QueryInput';
import type { Query } from '../renderer/query/types';

const noop = () => undefined;

function makeQuery(weight: number): Query {
  return {
    predicates: [
      {
        type: 'similar',
        value: 'C:/media/a.jpg',
        exclude: false,
        nodes: [{ kind: 'text', value: 'at night', weight }],
      },
    ],
  } as Query;
}

function renderInput(query: Query) {
  const props = {
    query,
    textValue: '',
    onTextChange: noop,
    onSubmitText: noop,
    onRemovePredicate: noop,
    onToggleExclude: noop,
    onSetPredicateJoin: noop,
    // Blend handlers present → chips are hoverable/blendable.
    onAddBlendNode: noop,
    onRemoveBlendNode: noop,
    onUpdateBlendNode: noop,
    onClearAll: noop,
    onClearText: noop,
  };
  const utils = render(<QueryInput {...props} />);
  return {
    ...utils,
    rerenderQuery: (q: Query) => utils.rerender(<QueryInput {...props} query={q} />),
  };
}

const popover = (c: HTMLElement) => c.querySelector('.query-chip-blend-pop');

describe('similarity chip blend popover', () => {
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it('closes after mouseleave, but stays open while a control inside has focus', () => {
    const { container } = renderInput(makeQuery(0.5));
    const wrap = container.querySelector('.query-chip-wrap') as HTMLElement;

    fireEvent.mouseEnter(wrap);
    expect(popover(container)).not.toBeNull();

    // Focus the add-concept input, then mouse away: the popover must survive
    // (the timer re-arms while focus is inside).
    const addInput = container.querySelector(
      '.query-chip-blend-input'
    ) as HTMLInputElement;
    act(() => addInput.focus());
    fireEvent.mouseLeave(wrap);
    act(() => jest.advanceTimersByTime(600));
    expect(popover(container)).not.toBeNull();

    // Blur (focus moves elsewhere) → the next re-check closes it.
    act(() => {
      addInput.blur();
      jest.advanceTimersByTime(600);
    });
    expect(popover(container)).toBeNull();
  });

  it('recovers when the focused slider is remounted by a weight commit (the stuck-open wedge)', () => {
    const { container, rerenderQuery } = renderInput(makeQuery(0.5));
    const wrap = container.querySelector('.query-chip-wrap') as HTMLElement;

    fireEvent.mouseEnter(wrap);
    const slider = container.querySelector(
      '.query-blend-node input[type="range"]'
    ) as HTMLInputElement;
    act(() => slider.focus());

    // A committed weight changes the slider's key → React replaces the
    // focused element without ever firing onBlur.
    rerenderQuery(makeQuery(0.55));
    expect(document.activeElement).toBe(document.body); // blur was swallowed

    fireEvent.mouseLeave(wrap);
    act(() => jest.advanceTimersByTime(600));
    expect(popover(container)).toBeNull();
  });

  it('always closes on a pointerdown outside the chip and popover', () => {
    const { container } = renderInput(makeQuery(0.5));
    const wrap = container.querySelector('.query-chip-wrap') as HTMLElement;

    fireEvent.mouseEnter(wrap);
    expect(popover(container)).not.toBeNull();

    // Press inside the popover: stays open.
    fireEvent.pointerDown(popover(container) as Element);
    expect(popover(container)).not.toBeNull();

    // Press anywhere else: closes immediately, no timer involved.
    fireEvent.pointerDown(document.body);
    expect(popover(container)).toBeNull();
  });
});
