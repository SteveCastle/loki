// Auto-scrolls a container while a react-dnd drag hovers near its top or
// bottom edge, so long lists can be scrolled mid-drag (e.g. dragging a person
// card onto another person far above/below the current scroll position).
// Unlike the copy in list.tsx this measures the container's bounding rect, so
// it works for containers that don't start at the top of the viewport.
//
// Speed has two factors, multiplied:
//   proximity — quadratic ramp across the edge zone, so speed stays gentle
//     until the pointer is right at the edge (precision for short hops);
//   dwell — after HOLD_DELAY of continuous scrolling in one direction, speed
//     accelerates up to MAX_BOOST× over RAMP_MS, so long lists don't take
//     forever. Any pause, direction flip, or drag end resets the boost.
//
// Fully imperative: the previous version collected monitor.getClientOffset()
// through useDragLayer, which re-rendered the calling component on every
// dragover event of EVERY drag type (tag drags included). Now the monitor is
// read inside a rAF loop that only runs while an accepted drag is in flight,
// and React never re-renders for it.
import { RefObject, useEffect, useRef } from 'react';
import { useDragDropManager } from 'react-dnd';

// Distance from the container edge (px) where scrolling kicks in.
const EDGE_ZONE = 72;
// Scroll speed (px per frame) at the very edge, before the dwell boost.
const BASE_SPEED = 14;
// Continuous scrolling in one direction for this long (ms) starts the boost…
const HOLD_DELAY = 400;
// …which ramps linearly to MAX_BOOST× over this long (ms).
const RAMP_MS = 1800;
const MAX_BOOST = 7;

export default function useDragAutoScroll(
  ref: RefObject<HTMLElement | null>,
  acceptTypes: string[]
) {
  // Callers pass a fresh array literal each render; keep the latest without
  // re-subscribing.
  const acceptRef = useRef(acceptTypes);
  acceptRef.current = acceptTypes;

  const manager = useDragDropManager();
  useEffect(() => {
    const monitor = manager.getMonitor();
    let frameId = 0;
    const dwell: { dir: -1 | 0 | 1; since: number } = { dir: 0, since: 0 };

    const active = () => {
      const type = monitor.getItemType();
      return (
        monitor.isDragging() &&
        typeof type === 'string' &&
        acceptRef.current.includes(type)
      );
    };

    // The loop self-perpetuates while the drag is active, so scrolling keeps
    // going with the last known pointer position even when HTML5 drag stops
    // emitting dragover events (pointer held still at an edge).
    const step = (now: number) => {
      frameId = 0;
      const el = ref.current;
      const offset = monitor.getClientOffset();
      if (!el || !offset || !active()) {
        // Drag ended (or left our types): drop any accumulated boost.
        dwell.dir = 0;
        dwell.since = 0;
        return;
      }
      const { x, y } = offset;
      const rect = el.getBoundingClientRect();
      // Only react while the pointer is over this container (with a little
      // vertical slack so overshooting an edge keeps scrolling at full speed)
      // — a drag over some other panel must not scroll this one.
      const inside =
        x >= rect.left &&
        x <= rect.right &&
        y >= rect.top - EDGE_ZONE &&
        y <= rect.bottom + EDGE_ZONE;
      let dir: -1 | 0 | 1 = 0;
      let proximity = 0;
      if (inside) {
        const fromTop = y - rect.top;
        const fromBottom = rect.bottom - y;
        if (fromTop < EDGE_ZONE) {
          dir = -1;
          proximity = Math.min(1, (EDGE_ZONE - fromTop) / EDGE_ZONE);
        } else if (fromBottom < EDGE_ZONE) {
          dir = 1;
          proximity = Math.min(1, (EDGE_ZONE - fromBottom) / EDGE_ZONE);
        }
      }

      if (dir === 0) {
        dwell.dir = 0;
        dwell.since = 0;
      } else {
        if (dwell.dir !== dir) {
          dwell.dir = dir;
          dwell.since = now;
        }
        const held = now - dwell.since - HOLD_DELAY;
        const boost =
          1 + Math.min(1, Math.max(0, held / RAMP_MS)) * (MAX_BOOST - 1);
        // Quadratic proximity: slow, controllable entry into the zone.
        el.scrollBy(0, dir * BASE_SPEED * proximity * proximity * boost);
      }
      frameId = requestAnimationFrame(step);
    };

    const sync = () => {
      if (active() && !frameId) frameId = requestAnimationFrame(step);
    };
    const unsubState = monitor.subscribeToStateChange(sync);
    const unsubOffset = monitor.subscribeToOffsetChange(sync);
    return () => {
      unsubState();
      unsubOffset();
      if (frameId) cancelAnimationFrame(frameId);
    };
  }, [manager, ref]);
}
