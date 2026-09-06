import React from 'react';
import { render, act } from '@testing-library/react';
import { useLongPressContextMenu } from '../renderer/hooks/useLongPressContextMenu';

// A touch long press is the right-click: it is translated into a synthetic
// contextmenu event at the pressed element so the app's existing
// onContextMenu handlers (plain = command palette, shift = context palette)
// open the palettes on iPad without any per-surface touch code. One finger
// maps to plain, two fingers to shift.

// jsdom 20 has no PointerEvent; the hook only reads pointerId/pointerType.
class FakePointerEvent extends MouseEvent {
  pointerId: number;
  pointerType: string;
  isPrimary: boolean;
  constructor(type: string, init: any = {}) {
    super(type, init);
    this.pointerId = init.pointerId ?? 1;
    this.pointerType = init.pointerType ?? 'touch';
    // Like a browser: the first finger down is primary, later ones aren't.
    this.isPrimary = init.isPrimary ?? (this.pointerId === 1);
  }
}

function Host({ onContextMenu }: { onContextMenu: (e: React.MouseEvent) => void }) {
  useLongPressContextMenu();
  return (
    <div data-testid="surface" onContextMenu={onContextMenu}>
      <span data-testid="child">child</span>
    </div>
  );
}

function pointer(
  target: Element,
  type: string,
  opts: { id?: number; x?: number; y?: number; pointerType?: string } = {}
) {
  target.dispatchEvent(
    new FakePointerEvent(type, {
      bubbles: true,
      cancelable: true,
      clientX: opts.x ?? 10,
      clientY: opts.y ?? 20,
      pointerId: opts.id ?? 1,
      pointerType: opts.pointerType ?? 'touch',
    })
  );
}

describe('useLongPressContextMenu', () => {
  beforeAll(() => {
    (window as any).PointerEvent = FakePointerEvent;
  });
  beforeEach(() => {
    jest.useFakeTimers();
  });
  afterEach(() => {
    jest.useRealTimers();
  });

  function setup() {
    const received: React.MouseEvent[] = [];
    const utils = render(
      <Host onContextMenu={(e) => received.push({ ...e, shiftKey: e.shiftKey, clientX: e.clientX, clientY: e.clientY } as any)} />
    );
    return { received, child: utils.getByTestId('child') };
  }

  it('one-finger hold fires a plain contextmenu at the pressed element', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown', { x: 33, y: 44 }));
    act(() => jest.advanceTimersByTime(449));
    expect(received).toHaveLength(0);
    act(() => jest.advanceTimersByTime(1));
    expect(received).toHaveLength(1);
    expect(received[0].shiftKey).toBe(false);
    expect(received[0].clientX).toBe(33);
    expect(received[0].clientY).toBe(44);
  });

  it('two-finger hold fires a shift contextmenu (context palette)', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown', { id: 1, x: 10, y: 10 }));
    act(() => jest.advanceTimersByTime(200));
    act(() => pointer(child, 'pointerdown', { id: 2, x: 60, y: 10 }));
    // The hold is measured from the second finger.
    act(() => jest.advanceTimersByTime(300));
    expect(received).toHaveLength(0);
    act(() => jest.advanceTimersByTime(150));
    expect(received).toHaveLength(1);
    expect(received[0].shiftKey).toBe(true);
    // Anchored where the first finger landed.
    expect(received[0].clientX).toBe(10);
  });

  it('movement beyond the slop cancels the press', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown', { x: 10, y: 10 }));
    act(() => pointer(child, 'pointermove', { x: 40, y: 10 }));
    act(() => jest.advanceTimersByTime(1000));
    expect(received).toHaveLength(0);
  });

  it('small drift within the slop still fires', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown', { x: 10, y: 10 }));
    act(() => pointer(child, 'pointermove', { x: 14, y: 13 }));
    act(() => jest.advanceTimersByTime(1000));
    expect(received).toHaveLength(1);
  });

  it('releasing or losing the pointer before the timer cancels the press', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown'));
    act(() => jest.advanceTimersByTime(300));
    act(() => pointer(child, 'pointerup'));
    act(() => jest.advanceTimersByTime(1000));
    expect(received).toHaveLength(0);

    act(() => pointer(child, 'pointerdown'));
    act(() => pointer(child, 'pointercancel'));
    act(() => jest.advanceTimersByTime(1000));
    expect(received).toHaveLength(0);
  });

  it('a third finger cancels the press', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown', { id: 1 }));
    act(() => pointer(child, 'pointerdown', { id: 2 }));
    act(() => pointer(child, 'pointerdown', { id: 3 }));
    act(() => jest.advanceTimersByTime(1000));
    expect(received).toHaveLength(0);
  });

  it('ignores mouse pointers (the mouse has a real right button)', () => {
    const { received, child } = setup();
    act(() => pointer(child, 'pointerdown', { pointerType: 'mouse' }));
    act(() => jest.advanceTimersByTime(1000));
    expect(received).toHaveLength(0);
  });

  it('swallows the release click that trails the press, but not a later tap', () => {
    const { child } = setup();
    const clicks = jest.fn();
    child.addEventListener('click', clicks);
    act(() => pointer(child, 'pointerdown'));
    act(() => jest.advanceTimersByTime(450));

    // The press finger's touchend is cancelled (no synthesized click), and a
    // click arriving right after the release is absorbed anyway.
    const touchEnd = new Event('touchend', { bubbles: true, cancelable: true });
    act(() => {
      pointer(child, 'pointerup');
      child.dispatchEvent(touchEnd);
    });
    expect(touchEnd.defaultPrevented).toBe(true);
    act(() => {
      child.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(clicks).not.toHaveBeenCalled();

    // A fresh tap well after the release is a real click.
    act(() => jest.advanceTimersByTime(500));
    const laterTouchEnd = new Event('touchend', { bubbles: true, cancelable: true });
    act(() => {
      child.dispatchEvent(laterTouchEnd);
      child.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    });
    expect(laterTouchEnd.defaultPrevented).toBe(false);
    expect(clicks).toHaveBeenCalledTimes(1);
  });
});
