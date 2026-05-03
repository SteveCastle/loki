/**
 * Tests the click-vs-drag threshold helper used by taxonomy tag rows.
 *
 * Why this exists: HTML5 drag-and-drop suppresses synthetic `click` events
 * on draggable elements when the cursor moves between mousedown and mouseup,
 * which made first-click selection of a tag flaky. We replace `onClick` with
 * a manual mousedown/mouseup pair gated by this threshold so a real click
 * (no movement) registers reliably while a drag attempt does not.
 */

import { isClickWithinThreshold } from '../renderer/components/taxonomy/click-vs-drag';

describe('isClickWithinThreshold', () => {
  it('returns false when there is no recorded start position', () => {
    expect(isClickWithinThreshold(null, { x: 0, y: 0 })).toBe(false);
  });

  it('returns true for zero movement', () => {
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 100, y: 100 })
    ).toBe(true);
  });

  it('returns true for sub-threshold movement on either axis', () => {
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 103, y: 102 })
    ).toBe(true);
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 96, y: 99 })
    ).toBe(true);
  });

  it('returns false when movement reaches the threshold on either axis', () => {
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 105, y: 100 })
    ).toBe(false);
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 100, y: 110 })
    ).toBe(false);
  });

  it('treats negative movement the same as positive movement', () => {
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 95, y: 100 })
    ).toBe(false);
    expect(
      isClickWithinThreshold({ x: 100, y: 100 }, { x: 100, y: 90 })
    ).toBe(false);
  });

  it('accepts a custom threshold', () => {
    expect(
      isClickWithinThreshold({ x: 0, y: 0 }, { x: 8, y: 0 }, 10)
    ).toBe(true);
    expect(
      isClickWithinThreshold({ x: 0, y: 0 }, { x: 10, y: 0 }, 10)
    ).toBe(false);
  });
});
