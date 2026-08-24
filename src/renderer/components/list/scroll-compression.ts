/*
 * Scroll-space compression for very large virtualized lists.
 *
 * Browsers clamp element dimensions at their layout engine's coordinate
 * limit — Chromium at ~33,554,400px (LayoutUnit: 1/64px fixed-point in a
 * 32-bit int), Firefox at ~17,895,700px. A virtualized list's spacer div
 * is as tall as the WHOLE list, so a few hundred thousand ~800px rows
 * blows past the cap and the scrollbar silently stops short: style.height
 * says 263M px, scrollHeight reports ~33.5M, and everything past roughly
 * item 90k is unreachable by scrolling (the cursor and detail views keep
 * working — only the scroll geometry is clamped).
 *
 * The fix (the same scheme spreadsheet grids use): cap the element at
 * MAX_SCROLL_HEIGHT and map element scroll space to virtual content space
 * with a scale factor k:
 *
 *   virtualOffset = scrollTop × k
 *   k = (totalVirtual − viewport) / (spacerHeight − viewport)
 *
 * so the element's full scroll range covers the full virtual range. Rows
 * keep their exact on-screen spacing because every visible row is shifted
 * by the same delta = virtualOffset − scrollTop; the scrollbar simply
 * becomes "denser". Below the cap k is exactly 1 and every formula is the
 * identity — behavior is byte-identical to the uncompressed list.
 *
 * Wheel input is applied 1:1 in VIRTUAL pixels (a 100px wheel tick moves
 * the content 100px, not 100×k): the wheel handler divides deltas by k
 * before applying them to scrollTop. Scrollbar dragging stays compressed,
 * which is what you want — dragging across 263M px was never precise.
 */

/** Cap for the spacer element, safely below Firefox's ~17.9M px limit
 * (Chromium allows ~33.5M). Also leaves headroom for the browser to
 * position overscan rows near the bottom edge. */
export const MAX_SCROLL_HEIGHT = 15_000_000;

/** Height the spacer element actually gets. */
export function spacerHeight(
  totalVirtual: number,
  max: number = MAX_SCROLL_HEIGHT
): number {
  return Math.min(totalVirtual, max);
}

/**
 * virtual-px per element-px. Exactly 1 while the list fits under the cap.
 * Maps the element's scrollable range [0, spacer − viewport] onto the
 * virtual range [0, total − viewport], so the scrollbar's extremes land on
 * the list's extremes.
 */
export function compressionScale(
  totalVirtual: number,
  viewport: number,
  max: number = MAX_SCROLL_HEIGHT
): number {
  if (totalVirtual <= max) return 1;
  const elementRange = Math.max(1, spacerHeight(totalVirtual, max) - viewport);
  const virtualRange = Math.max(1, totalVirtual - viewport);
  return virtualRange / elementRange;
}

/** Element scrollTop → virtual content offset. */
export function elementToVirtual(scrollTop: number, scale: number): number {
  return scrollTop * scale;
}

/** Virtual content offset → element scrollTop. */
export function virtualToElement(virtualOffset: number, scale: number): number {
  return virtualOffset / scale;
}

/**
 * How far every visible row must be pulled UP so that content at
 * virtualOffset = scrollTop×scale appears where the element has only
 * scrolled scrollTop px:
 *
 *   shift = virtualOffset − scrollTop = scrollTop × (scale − 1)
 *
 * Zero when uncompressed. All rows in the viewport share the same delta,
 * so their exact spacing is preserved, and the resulting row transforms
 * stay small (≤ the element cap) — huge translateY values would hit the
 * compositor's float32 precision.
 */
export function rowShift(scrollTop: number, scale: number): number {
  return scrollTop * (scale - 1);
}

/**
 * Virtual-space scroll target that brings a row into view with 'auto'
 * alignment: rows above the viewport align to its top, rows below align
 * to its bottom, rows already fully visible need no scroll (null).
 *
 * This exists because TanStack's scrollToIndex clamps its VIRTUAL target
 * against the element's real scrollHeight (getMaxScrollOffset reads the
 * DOM) — beyond the compressed cap every programmatic jump would land at
 * the cap instead of the row.
 */
export function cursorScrollTarget(
  rowStart: number,
  rowSize: number,
  viewport: number,
  virtualOffset: number,
  totalVirtual: number
): number | null {
  let target: number | null = null;
  if (rowStart < virtualOffset) {
    target = rowStart;
  } else if (rowStart + rowSize > virtualOffset + viewport) {
    target = rowStart + rowSize - viewport;
  }
  if (target == null) return null;
  return Math.max(0, Math.min(totalVirtual - viewport, target));
}

/** Normalize a wheel event's deltaY to pixels. */
export function wheelDeltaPx(
  deltaY: number,
  deltaMode: number,
  viewport: number
): number {
  if (deltaMode === 1) return deltaY * 16; // lines (Firefox)
  if (deltaMode === 2) return deltaY * viewport; // pages
  return deltaY;
}
