// palette-trace — measures right-click → command palette fully ready.
//
// The palette is the app's most frequently opened surface, and its FIRST open
// of a session is reliably slower than the rest. One-time costs hide in there
// that a steady-state measurement never sees: constructing the tag-search Web
// Worker (and loading its chunk), building the first Fuse index, the first
// categories fetch, first layout of a large subtree, first style resolution.
//
// So every open is traced, and the summary records whether it was the first —
// comparing open #1 against open #2 in the same session is what isolates those
// costs. Each open emits ONE line to <userData>/app-log.jsonl under the
// `palette` scope; `scripts/palette-report.js` prints them.
//
// Cheap by construction: a handful of timestamps, one PerformanceObserver that
// only lives for the duration of an open, and one log line per open.

import { isElectron, logEvent } from './platform';

// An open that hasn't settled by now is either blocked on something pathological
// or the user closed the palette; either way, stop measuring and report.
const OPEN_TIMEOUT_MS = 6000;

interface OpenTrace {
  seq: number;
  startedAt: number;
  marks: { name: string; at: number }[];
  longTasks: { atMs: number; durationMs: number }[];
  observer: PerformanceObserver | null;
  timer: ReturnType<typeof setTimeout> | null;
  done: boolean;
}

let opens = 0;
let current: OpenTrace | null = null;

const now = (): number =>
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : 0;

/**
 * Called the instant the user asks for the palette — from the contextmenu
 * handler, BEFORE the state machine event is sent, so the measurement includes
 * the machine transition and React's re-render, not just the render itself.
 */
export function beginPaletteOpen(): void {
  if (!isElectron) return;
  // A re-open while one is in flight (rapid right-clicking) replaces it; the
  // abandoned trace is reported so it can't be mistaken for a fast open.
  if (current && !current.done) endPaletteOpen('superseded');

  opens += 1;
  const trace: OpenTrace = {
    seq: opens,
    startedAt: now(),
    marks: [],
    longTasks: [],
    observer: null,
    timer: null,
    done: false,
  };
  current = trace;

  if (typeof PerformanceObserver !== 'undefined') {
    try {
      trace.observer = new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          trace.longTasks.push({
            atMs: Math.round(entry.startTime - trace.startedAt),
            durationMs: Math.round(entry.duration),
          });
        }
      });
      trace.observer.observe({ entryTypes: ['longtask'] });
    } catch {
      trace.observer = null;
    }
  }

  trace.timer = setTimeout(() => endPaletteOpen('timeout'), OPEN_TIMEOUT_MS);
}

/** Record a milestone on the open currently being traced. Repeats are ignored. */
export function markPalette(name: string): void {
  const trace = current;
  if (!trace || trace.done) return;
  if (trace.marks.some((m) => m.name === name)) return;
  trace.marks.push({ name, at: Math.round(now() - trace.startedAt) });
}

/** True while an open is being traced — lets callers skip building mark data. */
export function isTracingPaletteOpen(): boolean {
  return !!current && !current.done;
}

/**
 * Close out the open and emit its timeline.
 *
 * `firstOpen` is the field to read first: the gap between open #1 and open #2
 * is the one-time cost, and the marks say which phase owns it.
 */
export function endPaletteOpen(reason: string): void {
  const trace = current;
  if (!trace || trace.done) return;
  trace.done = true;
  if (trace.timer) clearTimeout(trace.timer);
  if (trace.observer) {
    try {
      trace.observer.disconnect();
    } catch {
      // already gone
    }
  }

  const at = (name: string) =>
    trace.marks.find((m) => m.name === name)?.at ?? -1;
  const span = (from: string, to: string) => {
    const a = at(from);
    const b = at(to);
    return a < 0 || b < 0 ? -1 : b - a;
  };

  logEvent({
    level: 'info',
    scope: 'palette',
    message: 'palette-open',
    data: {
      seq: trace.seq,
      firstOpen: trace.seq === 1,
      reason,
      totalMs: Math.round(now() - trace.startedAt),
      marks: trace.marks,
      phases: {
        // Right-click → the palette subtree rendered.
        toRender: at('render'),
        // → measured and positioned (it renders hidden until then).
        toPositioned: span('render', 'position-ready'),
        // → the browser actually painted it.
        toPainted: span('position-ready', 'painted'),
        // → the search engine subtree mounted (deliberately deferred a task).
        toEngine: span('painted', 'engine-mounted'),
        // → its data is loaded and the fuzzy index is built.
        toDataReady: span('engine-mounted', 'data-ready'),
      },
      longTasks: {
        count: trace.longTasks.length,
        totalMs: trace.longTasks.reduce((s, t) => s + t.durationMs, 0),
        worst: [...trace.longTasks]
          .sort((a, b) => b.durationMs - a.durationMs)
          .slice(0, 10)
          .sort((a, b) => a.atMs - b.atMs),
      },
    },
  });

  current = null;
}

/** Test seam. */
export function resetPaletteTraceForTests(): void {
  if (current?.timer) clearTimeout(current.timer);
  if (current?.observer) {
    try {
      current.observer.disconnect();
    } catch {
      // ignore
    }
  }
  current = null;
  opens = 0;
}
