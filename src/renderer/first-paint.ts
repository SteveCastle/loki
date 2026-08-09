// first-paint — the "the user can see their media now" signal.
//
// Opening a file from the filesystem is the app's critical path: the window
// should show that image/video essentially immediately, and everything else
// (tag indexes, taxonomy, metadata panels) is allowed to arrive afterwards.
// This module is the one place that says when "afterwards" has begun, so
// background work can subscribe instead of each hook inventing its own timer.
//
// Two ways in: the detail viewer reports the first media element that finished
// loading, and a fallback timer fires if no media ever loads (an empty
// library, a directory picker, a broken file). Either way subscribers run
// exactly once.

import { markStartup, summariseStartup } from './startup-marks';

// Nothing is going to paint if the app opened on the directory picker, so the
// deferred work still needs to start eventually. Long enough that a normal
// media load always wins the race, short enough not to feel broken.
const FALLBACK_MS = 4000;

// Idle-callback budget after the paint. requestIdleCallback alone can starve
// indefinitely on a busy main thread, so the timeout is the real guarantee.
const IDLE_TIMEOUT_MS = 2000;

let painted = false;
let fallbackTimer: ReturnType<typeof setTimeout> | null = null;
const waiters: Array<() => void> = [];

function flush() {
  if (fallbackTimer !== null) {
    clearTimeout(fallbackTimer);
    fallbackTimer = null;
  }
  const pending = waiters.splice(0, waiters.length);
  for (const cb of pending) {
    try {
      cb();
    } catch {
      // A misbehaving subscriber must not block the others.
    }
  }
}

/** Called by the detail viewer when a media element has finished loading. */
export function notifyFirstMediaPainted(reason = 'first-media'): void {
  if (painted) return;
  painted = true;
  markStartup('first-media', { reason });
  // Close the launch out on the next tick so the summary includes the resource
  // timing for the media that just finished, not a half-written entry.
  setTimeout(() => summariseStartup(reason), 0);
  flush();
}

export function hasFirstMediaPainted(): boolean {
  return painted;
}

/**
 * Run `cb` once the first media has painted (or the fallback elapses), then
 * once the main thread is next idle. Returns an unsubscribe function.
 *
 * The idle hop matters as much as the paint: `onLoad` fires while the browser
 * is still laying out and decoding, and kicking off a multi-hundred-millisecond
 * IPC + structured clone right then stalls the frame the user is waiting for.
 */
export function onIdleAfterFirstPaint(cb: () => void): () => void {
  let cancelled = false;

  const runWhenIdle = () => {
    if (cancelled) return;
    const ric = (window as unknown as {
      requestIdleCallback?: (
        callback: () => void,
        opts?: { timeout: number }
      ) => number;
    }).requestIdleCallback;
    if (ric) {
      ric(() => {
        if (!cancelled) cb();
      }, { timeout: IDLE_TIMEOUT_MS });
    } else {
      setTimeout(() => {
        if (!cancelled) cb();
      }, 0);
    }
  };

  if (painted) {
    runWhenIdle();
    return () => {
      cancelled = true;
    };
  }

  waiters.push(runWhenIdle);
  if (fallbackTimer === null) {
    fallbackTimer = setTimeout(() => {
      fallbackTimer = null;
      // Distinguished in the log from a real paint: a launch that ends this way
      // never showed the user anything, and its timings mean something else.
      notifyFirstMediaPainted('fallback-no-media');
    }, FALLBACK_MS);
  }

  return () => {
    cancelled = true;
    const idx = waiters.indexOf(runWhenIdle);
    if (idx !== -1) waiters.splice(idx, 1);
  };
}

/** Test seam: reset module state between cases. */
export function resetFirstPaintForTests(): void {
  painted = false;
  if (fallbackTimer !== null) {
    clearTimeout(fallbackTimer);
    fallbackTimer = null;
  }
  waiters.length = 0;
}
