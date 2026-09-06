import { useEffect } from 'react';
import { absorbNextClick } from '../absorb-next-click';

// Touch equivalent of right-click.
//
// Every palette trigger in the app is an onContextMenu handler (panels
// background, list items, detail, tags, categories): plain = command palette,
// shift = context palette. Mice and trackpads fire contextmenu natively, but a
// touch long press does not on iPad — Safari never dispatches it for touch —
// and on Android/Windows it fires at the browser's own timing without any way
// to express "shift".
//
// So a long press is translated into a synthetic contextmenu event dispatched
// at the pressed element, which bubbles into the exact handler a right-click
// would have hit. Hold one finger for the command palette; hold two for the
// context palette (reported to the handlers as shiftKey). Any movement, an
// early release, or a third finger cancels the press, so pans, pinches, and
// taps are untouched.
//
// Mounted once at the app root; listens at the document so it covers every
// surface, present and future.

const LONG_PRESS_MS = 450;
// Beyond this the gesture is a pan/pinch, not a press. Generous, because a
// still finger on glass wanders a few pixels.
const MOVE_SLOP_PX = 12;
// How long after firing to keep swallowing the browser's own native
// contextmenu for the same press (Android/Windows fire it on release).
const SUPPRESS_MS = 1000;

type Point = { x: number; y: number };

export function useLongPressContextMenu(): void {
  useEffect(() => {
    if (typeof window === 'undefined' || !('PointerEvent' in window)) {
      return undefined;
    }

    const active = new Map<number, Point>();
    let timer: number | null = null;
    let anchor: { target: Element; x: number; y: number } | null = null;
    let suppressUntil = 0;
    // The fingers that made the press, until they lift. Their release must
    // not turn into a click — but a NEW tap (on the palette that just opened,
    // say) must, so suppression is keyed to these pointers, not to a clock.
    const pressPointers = new Set<number>();
    let lastPressReleaseAt = -Infinity;

    const clearTimer = () => {
      if (timer !== null) {
        window.clearTimeout(timer);
        timer = null;
      }
    };
    const cancel = () => {
      clearTimer();
      anchor = null;
    };
    const armed = () => timer !== null && anchor !== null;

    const swallowTouchEnd = (e: Event) => {
      // preventDefault on the touchend that ends the press stops the browser
      // synthesising mouse events and a click from it — otherwise the release
      // lands as a click on whatever is now under the finger (often the
      // freshly opened palette, or the detail view in touchpad mode). Pointer
      // events are dispatched before the compat touch events, so the matching
      // pointerup has usually already removed the finger from pressPointers;
      // the short grace window covers that ordering.
      if (
        pressPointers.size > 0 ||
        performance.now() - lastPressReleaseAt < 100
      ) {
        e.preventDefault();
      }
    };

    const fire = () => {
      timer = null;
      if (anchor === null) return;
      const twoFinger = active.size >= 2;
      const { x, y } = anchor;
      // The element pressed may have re-rendered away (list rows recycle);
      // fall back to whatever is under the finger now so the event still
      // reaches a live handler.
      const target = anchor.target.isConnected
        ? anchor.target
        : document.elementFromPoint(x, y);
      anchor = null;
      if (!target) return;

      suppressUntil = performance.now() + SUPPRESS_MS;
      pressPointers.clear();
      active.forEach((_, id) => pressPointers.add(id));

      const ev = new MouseEvent('contextmenu', {
        bubbles: true,
        cancelable: true,
        composed: true,
        view: window,
        clientX: x,
        clientY: y,
        screenX: x + window.screenX,
        screenY: y + window.screenY,
        button: 2,
        buttons: 2,
        shiftKey: twoFinger,
      });
      target.dispatchEvent(ev);
    };

    const isTouchLike = (e: PointerEvent) =>
      e.pointerType === 'touch' || e.pointerType === 'pen';

    const onPointerDown = (e: PointerEvent) => {
      if (!isTouchLike(e)) return;
      if (e.isPrimary) {
        // First finger of a new sequence: nothing else can be down, so any
        // pointer still tracked is one whose pointerup never reached us —
        // browsers deliver pointerup to the pointerdown target, and if that
        // element was re-rendered away mid-touch the event dies with it.
        // Left in place, a ghost finger would turn every later press into a
        // "two-finger" one.
        active.clear();
        pressPointers.clear();
        cancel();
      }
      active.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (active.size === 1) {
        anchor = {
          target: e.target as Element,
          x: e.clientX,
          y: e.clientY,
        };
        clearTimer();
        timer = window.setTimeout(fire, LONG_PRESS_MS);
      } else if (active.size === 2 && anchor !== null) {
        // Second finger: measure the hold from here so two-finger presses
        // need the same deliberate pause as one-finger ones. The press stays
        // anchored where the first finger landed.
        clearTimer();
        timer = window.setTimeout(fire, LONG_PRESS_MS);
      } else {
        cancel();
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!armed()) return;
      const start = active.get(e.pointerId);
      if (!start) return;
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      if (dx * dx + dy * dy > MOVE_SLOP_PX * MOVE_SLOP_PX) {
        cancel();
      }
    };

    const onPointerEnd = (e: PointerEvent) => {
      if (!active.has(e.pointerId)) return;
      active.delete(e.pointerId);
      // Any finger lifting (or the browser taking the pointer for a native
      // pan) before the timer fires ends the press.
      cancel();
      if (pressPointers.delete(e.pointerId)) {
        lastPressReleaseAt = performance.now();
        // Belt to the touchend braces: a click can only trail this release
        // immediately, so a tight capture-phase absorb can't catch a later,
        // legitimate tap.
        absorbNextClick(150);
      }
    };

    const onContextMenu = (e: MouseEvent) => {
      if (!e.isTrusted) return;
      if (armed()) {
        // Platforms that DO fire contextmenu for touch (Android, Windows
        // touchscreens) beat the timer. Take over so the finger-count rule
        // applies there too, instead of both events opening a palette.
        e.preventDefault();
        e.stopImmediatePropagation();
        clearTimer();
        fire();
        return;
      }
      if (performance.now() < suppressUntil) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    };

    document.addEventListener('pointerdown', onPointerDown, true);
    document.addEventListener('pointermove', onPointerMove, true);
    document.addEventListener('pointerup', onPointerEnd, true);
    document.addEventListener('pointercancel', onPointerEnd, true);
    document.addEventListener('touchend', swallowTouchEnd, {
      capture: true,
      passive: false,
    });
    window.addEventListener('contextmenu', onContextMenu, true);

    return () => {
      cancel();
      document.removeEventListener('pointerdown', onPointerDown, true);
      document.removeEventListener('pointermove', onPointerMove, true);
      document.removeEventListener('pointerup', onPointerEnd, true);
      document.removeEventListener('pointercancel', onPointerEnd, true);
      document.removeEventListener('touchend', swallowTouchEnd, {
        capture: true,
      } as EventListenerOptions);
      window.removeEventListener('contextmenu', onContextMenu, true);
    };
  }, []);
}

// Safari's page pinch-zoom fires proprietary gesture events before any
// pointer event; cancelling gesturestart is the belt to touch-action's braces
// on iOS versions that only honour one of them.
export function useBlockBrowserPinchZoom(): void {
  useEffect(() => {
    const block = (e: Event) => e.preventDefault();
    // Safari ignores user-scalable=no for pinch; cancelling a multi-finger
    // touchmove is what actually stops the page zooming. Pointer events
    // (which the detail view's pinch runs on) are unaffected by this.
    const blockMultiTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 1) e.preventDefault();
    };
    document.addEventListener('gesturestart', block, { passive: false });
    document.addEventListener('touchmove', blockMultiTouchMove, {
      passive: false,
    });
    return () => {
      document.removeEventListener('gesturestart', block);
      document.removeEventListener('touchmove', blockMultiTouchMove);
    };
  }, []);
}
