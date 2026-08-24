import {
  MAX_SCROLL_HEIGHT,
  compressionScale,
  spacerHeight,
  elementToVirtual,
  virtualToElement,
  rowShift,
  wheelDeltaPx,
  cursorScrollTarget,
} from '../renderer/components/list/scroll-compression';

const VIEWPORT = 1080;

describe('scroll compression', () => {
  it('is the identity below the element cap', () => {
    const total = 9_000 * 800; // 7.2M px — a normal big library
    expect(spacerHeight(total)).toBe(total);
    expect(compressionScale(total, VIEWPORT)).toBe(1);
    expect(elementToVirtual(123_456, 1)).toBe(123_456);
    expect(virtualToElement(123_456, 1)).toBe(123_456);
    expect(rowShift(123_456, 1)).toBe(0);
  });

  it('clamps the spacer and maps scroll extremes exactly', () => {
    // The report that motivated this: 329k items at ~800px = 263.2M px,
    // far past Chromium's ~33.5M layout limit.
    const total = 329_000 * 800;
    const k = compressionScale(total, VIEWPORT);

    expect(spacerHeight(total)).toBe(MAX_SCROLL_HEIGHT);
    expect(k).toBeGreaterThan(1);

    // Top of the scrollbar is the top of the list…
    expect(elementToVirtual(0, k)).toBe(0);
    // …and the bottom of the scrollbar is the bottom of the list: the
    // element's max scrollTop maps to the virtual max offset.
    const elementMax = MAX_SCROLL_HEIGHT - VIEWPORT;
    const virtualMax = total - VIEWPORT;
    expect(elementToVirtual(elementMax, k)).toBeCloseTo(virtualMax, 3);
  });

  it('keeps the spacer under every browser cap', () => {
    // Firefox clamps at ~17.9M; Chromium at ~33.5M.
    expect(MAX_SCROLL_HEIGHT).toBeLessThan(17_895_000);
    for (const total of [1e6, 5e7, 3e8, 1e9]) {
      expect(spacerHeight(total)).toBeLessThanOrEqual(MAX_SCROLL_HEIGHT);
    }
  });

  it('round-trips offsets through the mapping', () => {
    const total = 329_000 * 800;
    const k = compressionScale(total, VIEWPORT);
    for (const scrollTop of [0, 1, 733.5, 4_000_000, 14_998_920]) {
      const virtual = elementToVirtual(scrollTop, k);
      expect(virtualToElement(virtual, k)).toBeCloseTo(scrollTop, 6);
    }
  });

  it('positions on-screen rows exactly despite compression', () => {
    // A row whose virtual start sits d px below the virtual offset must
    // render exactly d px below the element's scroll position — the shift
    // shared by all rows preserves true spacing on screen.
    const total = 329_000 * 800;
    const k = compressionScale(total, VIEWPORT);
    const scrollTop = 7_123_456.25;
    const virtualOffset = elementToVirtual(scrollTop, k);
    const shift = rowShift(scrollTop, k);

    for (const d of [0, 1, 799, 800, VIEWPORT - 1]) {
      const rowStart = virtualOffset + d; // virtual-space row position
      const screenPos = rowStart - shift - scrollTop; // px below viewport top
      expect(screenPos).toBeCloseTo(d, 3);
    }
    // And the transform values the rows get stay small enough for the
    // compositor (start − shift ≈ element space, bounded by the cap).
    const worstTransform = total - VIEWPORT - rowShift(MAX_SCROLL_HEIGHT - VIEWPORT, k);
    expect(worstTransform).toBeLessThanOrEqual(MAX_SCROLL_HEIGHT);
  });

  it('scrollbar-drag mapping stays monotonic and dense', () => {
    const total = 329_000 * 800;
    const k = compressionScale(total, VIEWPORT);
    let prev = -1;
    for (let s = 0; s <= MAX_SCROLL_HEIGHT - VIEWPORT; s += 1_499_892) {
      const v = elementToVirtual(s, k);
      expect(v).toBeGreaterThan(prev);
      prev = v;
    }
  });

  it('cursor jumps reach any row, including past the element cap', () => {
    // TanStack's scrollToIndex clamps against the element's scrollHeight,
    // which is why cursor jumps kept cutting off at the cap — this is the
    // replacement math.
    const rowHeight = 800;
    const rows = 329_000;
    const total = rows * rowHeight;
    const k = compressionScale(total, VIEWPORT);

    // Row already fully visible → no scroll.
    expect(cursorScrollTarget(1600, rowHeight, VIEWPORT, 1600, total)).toBe(
      null
    );
    // Row above the viewport → align to its start.
    expect(cursorScrollTarget(1600, rowHeight, VIEWPORT, 40_000, total)).toBe(
      1600
    );
    // Row below the viewport → align to its end.
    expect(cursorScrollTarget(80_000, rowHeight, VIEWPORT, 0, total)).toBe(
      80_000 + rowHeight - VIEWPORT
    );

    // The row that used to be unreachable (deep past the old ~33.5M cap):
    const row = 250_000;
    const target = cursorScrollTarget(
      row * rowHeight,
      rowHeight,
      VIEWPORT,
      0,
      total
    );
    expect(target).toBe(row * rowHeight + rowHeight - VIEWPORT);
    // …and its element-space scrollTop is reachable (≤ spacer − viewport).
    const scrollTop = virtualToElement(target as number, k);
    expect(scrollTop).toBeLessThanOrEqual(MAX_SCROLL_HEIGHT - VIEWPORT);
    // Round-trip: the element lands the viewport on that exact row.
    const backToVirtual = elementToVirtual(scrollTop, k);
    expect(backToVirtual + VIEWPORT).toBeCloseTo((row + 1) * rowHeight, 3);

    // The LAST row maps exactly to the element's maximum scrollTop.
    const lastTarget = cursorScrollTarget(
      (rows - 1) * rowHeight,
      rowHeight,
      VIEWPORT,
      0,
      total
    );
    expect(lastTarget).toBe(total - VIEWPORT);
    expect(virtualToElement(lastTarget as number, k)).toBeCloseTo(
      MAX_SCROLL_HEIGHT - VIEWPORT,
      6
    );
  });

  it('normalizes wheel deltas per deltaMode', () => {
    expect(wheelDeltaPx(120, 0, VIEWPORT)).toBe(120); // pixels
    expect(wheelDeltaPx(3, 1, VIEWPORT)).toBe(48); // lines
    expect(wheelDeltaPx(1, 2, VIEWPORT)).toBe(VIEWPORT); // pages
  });
});
