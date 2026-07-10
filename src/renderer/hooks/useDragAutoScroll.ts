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
import { RefObject, useEffect, useRef } from 'react';
import { useDragLayer } from 'react-dnd';

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
  const { isDragging, offset, type } = useDragLayer((monitor) => ({
    isDragging: monitor.isDragging(),
    offset: monitor.getClientOffset(),
    type: monitor.getItemType(),
  }));

  // Dwell state lives in a ref because the effect below restarts on every
  // dragover (offset change) — the boost must survive those restarts and
  // only reset when scrolling actually stops or reverses.
  const dwell = useRef<{ dir: -1 | 0 | 1; since: number }>({
    dir: 0,
    since: 0,
  });

  const active =
    isDragging &&
    offset != null &&
    typeof type === 'string' &&
    acceptTypes.includes(type);
  const x = active ? offset.x : null;
  const y = active ? offset.y : null;

  useEffect(() => {
    if (x == null || y == null) {
      // Drag ended (or left our types): drop any accumulated boost.
      dwell.current = { dir: 0, since: 0 };
      return undefined;
    }
    const el = ref.current;
    if (!el) return undefined;

    let frameId: number;
    const step = (now: number) => {
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
        dwell.current = { dir: 0, since: 0 };
      } else {
        if (dwell.current.dir !== dir) dwell.current = { dir, since: now };
        const held = now - dwell.current.since - HOLD_DELAY;
        const boost =
          1 + Math.min(1, Math.max(0, held / RAMP_MS)) * (MAX_BOOST - 1);
        // Quadratic proximity: slow, controllable entry into the zone.
        el.scrollBy(0, dir * BASE_SPEED * proximity * proximity * boost);
      }
      frameId = requestAnimationFrame(step);
    };
    frameId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameId);
    // The rAF loop keeps scrolling with the last known pointer position even
    // when HTML5 drag stops emitting dragover events (pointer held still at
    // an edge) — each new offset just restarts the loop.
  }, [x, y, ref]);
}
