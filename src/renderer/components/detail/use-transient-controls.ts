import { useCallback, useEffect, useRef, useState } from 'react';

// The detail view's video/audio control bar is hover-revealed, which touch
// can't do. This makes a tap on the player show the bar for a moment; it
// fades once the finger has left it alone for `hideAfterMs`. Touching the bar
// itself (scrubbing, say) keeps it up. Mouse hover is untouched — the CSS
// :hover rule still applies alongside the `visible` class this drives.
//
// The tap is detected from pointer events, not click, so it works in every
// control mode: in touchpad mode the same tap also advances the cursor, and
// the bar then shows for the item that arrives.

const TAP_SLOP_PX = 12;

export function useTransientControls(
  containerRef: React.RefObject<HTMLElement>,
  // Effect key: the container mounts with the item, and ref assignment
  // doesn't re-render (see the wheel-listener effect in detail.tsx).
  mountKey: string | undefined,
  hideAfterMs = 3000
): { visible: boolean; keepAlive: () => void } {
  const [visible, setVisible] = useState(false);
  const timer = useRef<number | null>(null);

  const keepAlive = useCallback(() => {
    setVisible(true);
    if (timer.current !== null) {
      window.clearTimeout(timer.current);
    }
    timer.current = window.setTimeout(() => {
      timer.current = null;
      setVisible(false);
    }, hideAfterMs);
  }, [hideAfterMs]);

  useEffect(
    () => () => {
      if (timer.current !== null) {
        window.clearTimeout(timer.current);
      }
    },
    []
  );

  useEffect(() => {
    const container = containerRef.current;
    if (container === null) {
      return undefined;
    }
    let start: { id: number; x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => {
      if (e.pointerType !== 'touch' && e.pointerType !== 'pen') return;
      start = e.isPrimary ? { id: e.pointerId, x: e.clientX, y: e.clientY } : null;
    };
    const onUp = (e: PointerEvent) => {
      if (start === null || e.pointerId !== start.id) return;
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      start = null;
      // A drag is a pan, not a tap.
      if (dx * dx + dy * dy <= TAP_SLOP_PX * TAP_SLOP_PX) {
        keepAlive();
      }
    };
    const onCancel = () => {
      start = null;
    };
    container.addEventListener('pointerdown', onDown);
    container.addEventListener('pointerup', onUp);
    container.addEventListener('pointercancel', onCancel);
    return () => {
      container.removeEventListener('pointerdown', onDown);
      container.removeEventListener('pointerup', onUp);
      container.removeEventListener('pointercancel', onCancel);
    };
  }, [mountKey, keepAlive]);

  return { visible, keepAlive };
}
