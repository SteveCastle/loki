// Pure helpers for frame-accurate video scrubbing.
//
// These are deliberately free of any DOM / React dependency so they can be
// unit-tested in isolation (see src/__tests__/video-frame.test.ts). The
// `Video` component feeds raw `requestVideoFrameCallback` mediaTime samples
// into `estimateFrameRate`; `VideoControls` uses `frameStep` and `pixelToTime`
// to compute fractional seek targets.

// Used whenever the real frame rate is unknown (rVFC unavailable, variable
// frame rate with no stable estimate, or detection hasn't completed yet).
export const DEFAULT_FPS = 30;

// Common broadcast/film frame rates a noisy estimate is snapped toward.
const COMMON_RATES = [23.976, 24, 25, 29.97, 30, 50, 59.94, 60];

// Snap a raw fps estimate to the nearest common rate when it is within
// `tolerance` (relative, default 2%); otherwise return the raw estimate
// rounded to 3 decimals. Invalid input yields 0 ("unknown").
export function snapFrameRate(raw: number, tolerance = 0.02): number {
  if (!isFinite(raw) || raw <= 0) return 0;
  let best = raw;
  let bestErr = Infinity;
  for (const r of COMMON_RATES) {
    const err = Math.abs(raw - r) / r;
    if (err < bestErr) {
      bestErr = err;
      best = r;
    }
  }
  if (bestErr <= tolerance) return best;
  return Math.round(raw * 1000) / 1000;
}

// Estimate fps from a sequence of mediaTime samples (seconds) reported by
// requestVideoFrameCallback. Backward deltas (a loop reset on a short looping
// video) and absurd gaps (a pause/seek of >= 1s) are discarded. Returns 0 when
// there aren't enough usable deltas to make a stable estimate.
export function estimateFrameRate(mediaTimes: number[]): number {
  if (!Array.isArray(mediaTimes) || mediaTimes.length < 3) return 0;
  const deltas: number[] = [];
  for (let i = 1; i < mediaTimes.length; i++) {
    const d = mediaTimes[i] - mediaTimes[i - 1];
    if (d > 0 && d < 1) deltas.push(d);
  }
  if (deltas.length < 2) return 0;
  deltas.sort((a, b) => a - b);
  const mid = Math.floor(deltas.length / 2);
  const median =
    deltas.length % 2 === 0 ? (deltas[mid - 1] + deltas[mid]) / 2 : deltas[mid];
  if (median <= 0) return 0;
  return snapFrameRate(1 / median);
}

// Compute the seek target one frame away from `currentTime` in `direction`
// (+1 next, -1 previous). Falls back to DEFAULT_FPS when `frameRate` is unknown
// or invalid. Clamped to [0, duration - halfFrame] so we never land exactly on
// or past the end. Returns 0 when duration is unknown (<= 0).
export function frameStep(
  currentTime: number,
  frameRate: number,
  direction: 1 | -1,
  duration: number
): number {
  const fps = frameRate > 0 && isFinite(frameRate) ? frameRate : DEFAULT_FPS;
  const step = 1 / fps;
  if (!(duration > 0)) return 0;
  const next = currentTime + direction * step;
  const maxTime = Math.max(0, duration - step / 2);
  return Math.min(Math.max(next, 0), maxTime);
}

// Which time the scrubber should display: the live drag position while
// dragging (instant, cursor-locked), otherwise the actual decoded time. This
// decouples the thumb/label from the seek round-trip, which is the main cause
// of laggy-feeling scrubbing.
export function selectDisplayTime(
  isDragging: boolean,
  dragTime: number | null,
  actualVideoTime: number
): number {
  if (isDragging && dragTime != null) return dragTime;
  return actualVideoTime;
}

// Coalesced seeking decision: returns the time to seek to NOW, or null to wait.
// While a seek is in flight we issue no new seek (prevents a backlog that makes
// the picture lag behind the cursor); the `seeked` handler re-checks and seeks
// to the latest pending target once the element is free. This keeps exactly one
// seek in flight and always converges to `pending`.
export function coalescedSeekTarget(
  pending: number | null,
  isSeeking: boolean,
  currentTime: number,
  tolerance = 0.001
): number | null {
  if (pending == null) return null;
  if (isSeeking) return null;
  if (Math.abs(pending - currentTime) <= tolerance) return null;
  return pending;
}

// Map a horizontal pixel offset on the progress bar to a fractional time in
// seconds. The clamp keeps the result within [0, duration]. Crucially this
// does NOT round to whole seconds — that rounding was the original cause of
// "second-by-second only" scrubbing.
export function pixelToTime(
  offsetX: number,
  width: number,
  duration: number
): number {
  if (width <= 0 || duration <= 0) return 0;
  const clamped = Math.max(0, Math.min(offsetX, width));
  return (clamped / width) * duration;
}
