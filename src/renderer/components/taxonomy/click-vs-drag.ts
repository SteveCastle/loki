/**
 * Default movement threshold (pixels) below which a mousedown→mouseup pair
 * should be treated as a click rather than a drag.
 */
export const CLICK_DRAG_THRESHOLD_PX = 5;

export type Point = { x: number; y: number };

/**
 * Returns true when the mouse moved less than `threshold` pixels on both
 * axes between mousedown and mouseup. Used by taxonomy tag rows to fire
 * selection on intentional clicks while ignoring drag gestures, since
 * HTML5 drag-and-drop suppresses native `click` events on draggable
 * elements as soon as any movement occurs.
 */
export function isClickWithinThreshold(
  start: Point | null,
  end: Point,
  threshold: number = CLICK_DRAG_THRESHOLD_PX
): boolean {
  if (!start) return false;
  return (
    Math.abs(end.x - start.x) < threshold &&
    Math.abs(end.y - start.y) < threshold
  );
}
