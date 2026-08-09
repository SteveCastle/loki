// startup-trace — main-process instrumentation for launch → first media.
//
// Opening a file from the filesystem is this app's critical path, and the only
// honest way to tune it is to measure a real launch, on real hardware, against
// the user's real library and storage. Everything here writes to
// <userData>/app-log.jsonl (see errorLog.ts) alongside the renderer's own marks
// from src/renderer/startup-marks.ts, and `scripts/startup-report.js` folds one
// launch back into a single readable timeline.
//
// Three things are recorded that nothing else can see:
//
//   1. Milestones measured from PROCESS START (`sinceLaunchMs`), which covers
//      Electron's own bootstrap — invisible to any renderer-side clock.
//   2. Event-loop stalls in the main process. This matters more than it looks:
//      the gsm:// protocol handler that streams media bytes to the viewer runs
//      on this loop, so anything that blocks it (a big SQLite result being
//      marshalled into JS, a synchronous module load) directly delays the image
//      the user is waiting for.
//   3. The gsm:// read of the opened file itself, split into stat / first byte
//      / last byte. On network storage — this library lives on Z:\ and Y:\ —
//      that stat alone can dominate everything else.
//
// Overhead is a handful of log lines and one 20ms timer that stops itself.

import { app } from 'electron';
import { logEvent } from './errorLog';

// Correlates every line of one launch — main, renderer, and gsm reads — so
// overlapping runs (or a renderer reload) can't be mistaken for one timeline.
const bootId = `${Date.now().toString(36)}-${Math.random()
  .toString(36)
  .slice(2, 8)}`;

export function getBootId(): string {
  return bootId;
}

const sinceLaunchMs = () => Math.round(process.uptime() * 1000);

/** A one-shot launch milestone, timed from process start. */
export function mark(name: string, data?: Record<string, unknown>): void {
  logEvent({
    level: 'info',
    scope: 'startup',
    message: name,
    data: { bootId, sinceLaunchMs: sinceLaunchMs(), ...(data ?? {}) },
  });
}

// ---- Event-loop stall sampling -------------------------------------------

// Sample often enough to catch the stalls that matter without being one.
const SAMPLE_INTERVAL_MS = 20;
// Below this, a late timer is just scheduling noise, not a stall.
const STALL_THRESHOLD_MS = 30;
// Enough to cover a slow cold start; the sampler stops itself afterwards so
// nothing lingers for the life of the process.
const SAMPLE_WINDOW_MS = 25_000;
// Keep the report readable — the worst offenders are what matter.
const MAX_REPORTED_STALLS = 12;

let sampler: ReturnType<typeof setInterval> | null = null;
let stopTimer: ReturnType<typeof setTimeout> | null = null;
let stalls: { atMs: number; stallMs: number }[] = [];

/**
 * Watch for periods where the main process stopped turning its event loop.
 * A timer set for 20ms that fires 400ms late means something ran for 380ms
 * without yielding — and during startup that something was almost certainly
 * sitting between the viewer and its first image.
 */
export function startLoopLagSampler(): void {
  if (sampler) return;
  let expected = Date.now() + SAMPLE_INTERVAL_MS;
  sampler = setInterval(() => {
    const now = Date.now();
    const late = now - expected;
    expected = now + SAMPLE_INTERVAL_MS;
    if (late >= STALL_THRESHOLD_MS) {
      stalls.push({ atMs: sinceLaunchMs(), stallMs: Math.round(late) });
    }
  }, SAMPLE_INTERVAL_MS);
  // Never hold the process open just to measure it.
  sampler.unref?.();
  stopTimer = setTimeout(() => stopLoopLagSampler(), SAMPLE_WINDOW_MS);
  stopTimer.unref?.();
}

/** Stop sampling and emit what was seen. Safe to call more than once. */
export function stopLoopLagSampler(): void {
  if (!sampler) return;
  clearInterval(sampler);
  sampler = null;
  if (stopTimer) {
    clearTimeout(stopTimer);
    stopTimer = null;
  }
  const worst = [...stalls]
    .sort((a, b) => b.stallMs - a.stallMs)
    .slice(0, MAX_REPORTED_STALLS)
    .sort((a, b) => a.atMs - b.atMs);
  logEvent({
    level: 'info',
    scope: 'startup',
    message: 'main-loop-stalls',
    data: {
      bootId,
      count: stalls.length,
      totalStalledMs: stalls.reduce((sum, s) => sum + s.stallMs, 0),
      worst,
    },
  });
  stalls = [];
}

// ---- gsm:// read tracing --------------------------------------------------

// Only the opening reads matter here; steady-state browsing would flood the
// log for no benefit.
const MAX_TRACED_READS = 4;
let tracedReads = 0;

export interface MediaReadTrace {
  /** Call after fs.stat resolves (or rejects). */
  stat(ok: boolean, size?: number): void;
  /** Call when the Response object is handed back to Chromium. */
  responding(status: number, extra?: Record<string, unknown>): void;
  /** Call when the body stream ends (or errors / is cancelled). */
  done(outcome: 'end' | 'error' | 'cancel', bytes?: number): void;
}

const noopTrace: MediaReadTrace = {
  stat: () => undefined,
  responding: () => undefined,
  done: () => undefined,
};

/**
 * Time one gsm:// read end to end. Returns a no-op tracer once the opening
 * burst is over, so callers never need to branch.
 *
 * The split matters: a slow `stat` is storage latency (an SMB round trip to
 * Z:\), a slow gap between `responding` and `done` is throughput, and a slow
 * gap before `stat` even resolves is a blocked event loop.
 */
export function traceMediaRead(filePath: string, range: boolean): MediaReadTrace {
  if (tracedReads >= MAX_TRACED_READS) return noopTrace;
  tracedReads += 1;
  const seq = tracedReads;
  const started = process.hrtime.bigint();
  const sinceStart = () =>
    Number(process.hrtime.bigint() - started) / 1e6;

  let statMs = -1;
  let respondMs = -1;
  let size = -1;

  return {
    stat(ok, statSize) {
      statMs = Math.round(sinceStart() * 10) / 10;
      if (typeof statSize === 'number') size = statSize;
      if (!ok) {
        logEvent({
          level: 'info',
          scope: 'startup',
          message: 'media-read',
          data: { bootId, seq, filePath, outcome: 'not-found', statMs },
        });
      }
    },
    responding(status, extra) {
      respondMs = Math.round(sinceStart() * 10) / 10;
      logEvent({
        level: 'info',
        scope: 'startup',
        message: 'media-read-respond',
        data: {
          bootId,
          seq,
          filePath,
          atMs: sinceLaunchMs(),
          range,
          status,
          size,
          statMs,
          respondMs,
          ...(extra ?? {}),
        },
      });
    },
    done(outcome, bytes) {
      logEvent({
        level: 'info',
        scope: 'startup',
        message: 'media-read',
        data: {
          bootId,
          seq,
          filePath,
          outcome,
          range,
          size,
          bytes: bytes ?? -1,
          statMs,
          respondMs,
          totalMs: Math.round(sinceStart() * 10) / 10,
        },
      });
    },
  };
}

// ---- Launch shape ---------------------------------------------------------

/**
 * What this launch was actually asked to do. Without it the timings are
 * uninterpretable: "opened a 40MB PNG from a network share" and "restored a
 * saved session" are different problems that produce the same marks.
 */
export function describeLaunch(initialFile: string): void {
  const started = process.hrtime.bigint();
  const finish = (data: Record<string, unknown>) =>
    mark('launch-target', { initialFile, ...data });

  if (!initialFile) {
    finish({ kind: 'none' });
    return;
  }
  // Deliberately async and fire-and-forget: this is diagnostics, and it must
  // never be the thing that delays the window.
  import('fs')
    .then(({ promises: fsp }) => fsp.stat(initialFile))
    .then((st) => {
      finish({
        kind: st.isDirectory() ? 'directory' : 'file',
        sizeBytes: st.isDirectory() ? -1 : st.size,
        // A stat this slow is the storage layer talking, not the app.
        statMs: Math.round((Number(process.hrtime.bigint() - started) / 1e6) * 10) / 10,
        root: initialFile.slice(0, 3),
      });
    })
    .catch(() => finish({ kind: 'unstattable' }));
}

/**
 * Called once the renderer reports first media (or gives up waiting). Closes
 * out the sampled measurements so the launch has a definite end.
 */
export function finishLaunchTrace(reason: string): void {
  mark('launch-trace-end', { reason });
  stopLoopLagSampler();
}

/** The app's own version, so a report can tell builds apart. */
export function appVersion(): string {
  try {
    return app.getVersion();
  } catch {
    return 'unknown';
  }
}
