import {
  estimateFrameRate,
  snapFrameRate,
  frameStep,
  pixelToTime,
  selectDisplayTime,
  coalescedSeekTarget,
  DEFAULT_FPS,
} from '../renderer/video-frame';

describe('snapFrameRate', () => {
  it('snaps near-common estimates to the canonical rate', () => {
    expect(snapFrameRate(29.95)).toBe(29.97);
    expect(snapFrameRate(30.2)).toBe(30);
    expect(snapFrameRate(23.9)).toBe(23.976);
    expect(snapFrameRate(59.8)).toBe(59.94);
  });

  it('keeps a raw (rounded) estimate when far from any common rate', () => {
    // 47fps is >2% away from 48?/50 — no common rate near it, keep raw.
    expect(snapFrameRate(47)).toBe(47);
  });

  it('returns 0 for invalid input', () => {
    expect(snapFrameRate(0)).toBe(0);
    expect(snapFrameRate(-5)).toBe(0);
    expect(snapFrameRate(Infinity)).toBe(0);
  });
});

describe('estimateFrameRate', () => {
  it('estimates fps from evenly spaced mediaTime samples', () => {
    const t = [0, 1 / 30, 2 / 30, 3 / 30, 4 / 30, 5 / 30];
    expect(estimateFrameRate(t)).toBe(30);
  });

  it('discards loop-reset (backward) deltas from short looping videos', () => {
    // Plays to ~1s then loops back to 0 — the negative delta must be ignored.
    const t = [0.9, 0.9 + 1 / 30, 0.9 + 2 / 30, 0, 1 / 30, 2 / 30, 3 / 30];
    expect(estimateFrameRate(t)).toBe(30);
  });

  it('returns 0 when there are not enough usable samples', () => {
    expect(estimateFrameRate([])).toBe(0);
    expect(estimateFrameRate([0])).toBe(0);
    expect(estimateFrameRate([0, 0.033])).toBe(0);
  });

  it('detects 60fps', () => {
    const t = Array.from({ length: 8 }, (_, i) => i / 60);
    expect(estimateFrameRate(t)).toBe(60);
  });
});

describe('frameStep', () => {
  it('steps forward one frame at the given fps', () => {
    expect(frameStep(1.0, 30, 1, 10)).toBeCloseTo(1 + 1 / 30, 6);
  });

  it('steps backward one frame', () => {
    expect(frameStep(1.0, 30, -1, 10)).toBeCloseTo(1 - 1 / 30, 6);
  });

  it('falls back to DEFAULT_FPS when frameRate is unknown', () => {
    expect(frameStep(1.0, 0, 1, 10)).toBeCloseTo(1 + 1 / DEFAULT_FPS, 6);
    expect(frameStep(1.0, NaN, 1, 10)).toBeCloseTo(1 + 1 / DEFAULT_FPS, 6);
  });

  it('clamps at the lower bound', () => {
    expect(frameStep(0, 30, -1, 10)).toBe(0);
  });

  it('never seeks to or past the duration', () => {
    const result = frameStep(9.99, 30, 1, 10);
    expect(result).toBeLessThan(10);
  });

  it('returns 0 when duration is unknown', () => {
    expect(frameStep(5, 30, 1, 0)).toBe(0);
  });
});

describe('pixelToTime', () => {
  it('maps pixels to fractional time without rounding to whole seconds', () => {
    // The regression that motivated this work: a quarter-way drag on a 1s
    // video must yield 0.25s, not 0s.
    expect(pixelToTime(25, 100, 1)).toBeCloseTo(0.25, 6);
    expect(pixelToTime(50, 100, 2)).toBeCloseTo(1.0, 6);
  });

  it('clamps offsets outside the bar', () => {
    expect(pixelToTime(-10, 100, 2)).toBe(0);
    expect(pixelToTime(150, 100, 2)).toBe(2);
  });

  it('returns 0 for invalid bar width or duration', () => {
    expect(pixelToTime(50, 0, 2)).toBe(0);
    expect(pixelToTime(50, 100, 0)).toBe(0);
  });
});

describe('selectDisplayTime', () => {
  it('uses the live drag position while dragging', () => {
    expect(selectDisplayTime(true, 3.5, 1.0)).toBe(3.5);
  });

  it('falls back to actual time when dragging but no drag value yet', () => {
    expect(selectDisplayTime(true, null, 1.0)).toBe(1.0);
  });

  it('uses actual time when not dragging', () => {
    expect(selectDisplayTime(false, 3.5, 1.0)).toBe(1.0);
  });

  it('treats a 0 drag time as a real value, not absent', () => {
    expect(selectDisplayTime(true, 0, 5.0)).toBe(0);
  });
});

describe('coalescedSeekTarget', () => {
  it('seeks to the pending target when idle and far enough away', () => {
    expect(coalescedSeekTarget(2.0, false, 1.0)).toBe(2.0);
  });

  it('waits (null) while a seek is already in flight', () => {
    expect(coalescedSeekTarget(2.0, true, 1.0)).toBeNull();
  });

  it('does nothing when there is no pending target', () => {
    expect(coalescedSeekTarget(null, false, 1.0)).toBeNull();
  });

  it('does nothing when already at the target (within tolerance)', () => {
    expect(coalescedSeekTarget(1.0005, false, 1.0)).toBeNull();
  });
});
