import React, { useRef } from 'react';
import { render, act } from '@testing-library/react';
import { useTransientControls } from '../renderer/components/detail/use-transient-controls';

// A touch tap on the player shows the hover-only video controls for a few
// seconds; touching the bar keeps it up; drags and mouse pointers don't count.

class FakePointerEvent extends MouseEvent {
  pointerId: number;
  pointerType: string;
  isPrimary: boolean;
  constructor(type: string, init: any = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
    this.pointerType = init.pointerType ?? 'touch';
    this.isPrimary = init.isPrimary ?? true;
  }
}

function Host() {
  const ref = useRef<HTMLDivElement>(null);
  const { visible, keepAlive } = useTransientControls(ref, 'item', 3000);
  return (
    <div>
      <div data-testid="player" ref={ref} />
      <div
        data-testid="bar"
        className={visible ? 'visible' : ''}
        onPointerDown={keepAlive}
      />
    </div>
  );
}

function pointer(
  target: Element,
  type: string,
  opts: { x?: number; y?: number; pointerType?: string } = {}
) {
  target.dispatchEvent(
    new FakePointerEvent(type, {
      bubbles: true,
      clientX: opts.x ?? 10,
      clientY: opts.y ?? 10,
      pointerType: opts.pointerType ?? 'touch',
    })
  );
}

describe('useTransientControls', () => {
  beforeAll(() => {
    (window as any).PointerEvent = FakePointerEvent;
  });
  beforeEach(() => jest.useFakeTimers());
  afterEach(() => jest.useRealTimers());

  it('shows on a touch tap and hides after the delay', () => {
    const { getByTestId } = render(<Host />);
    const player = getByTestId('player');
    const bar = getByTestId('bar');
    expect(bar.className).toBe('');
    act(() => {
      pointer(player, 'pointerdown');
      pointer(player, 'pointerup');
    });
    expect(bar.className).toBe('visible');
    act(() => jest.advanceTimersByTime(2999));
    expect(bar.className).toBe('visible');
    act(() => jest.advanceTimersByTime(1));
    expect(bar.className).toBe('');
  });

  it('a drag is a pan, not a tap', () => {
    const { getByTestId } = render(<Host />);
    const player = getByTestId('player');
    act(() => {
      pointer(player, 'pointerdown', { x: 10, y: 10 });
      pointer(player, 'pointerup', { x: 80, y: 10 });
    });
    expect(getByTestId('bar').className).toBe('');
  });

  it('ignores the mouse (hover handles it)', () => {
    const { getByTestId } = render(<Host />);
    const player = getByTestId('player');
    act(() => {
      pointer(player, 'pointerdown', { pointerType: 'mouse' });
      pointer(player, 'pointerup', { pointerType: 'mouse' });
    });
    expect(getByTestId('bar').className).toBe('');
  });

  it('touching the bar restarts the fade', () => {
    const { getByTestId } = render(<Host />);
    const player = getByTestId('player');
    const bar = getByTestId('bar');
    act(() => {
      pointer(player, 'pointerdown');
      pointer(player, 'pointerup');
    });
    act(() => jest.advanceTimersByTime(2500));
    act(() => pointer(bar, 'pointerdown'));
    act(() => jest.advanceTimersByTime(2500));
    expect(bar.className).toBe('visible');
    act(() => jest.advanceTimersByTime(600));
    expect(bar.className).toBe('');
  });
});
