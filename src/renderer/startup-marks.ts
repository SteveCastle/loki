// startup-marks — renderer instrumentation for launch → first media.
//
// Opening a file from the filesystem is the app's critical path, and the only
// way to keep it fast is to see where a REAL launch spends its time, on real
// hardware, against a real library. Everything here lands in
// <userData>/app-log.jsonl next to the main process's own marks (see
// src/main/startup-trace.ts); `scripts/startup-report.js` folds one launch back
// into a single readable timeline.
//
// The milestones, in order:
//
//   bundle-start        the first line of our code to execute (boot-clock.ts)
//   imports-evaluated   the app's whole module graph has been evaluated
//   react-render        root.render() called
//   providers-ready     session store resolved, state machine interpreted
//   first-contentful-paint  the browser painted something
//   media-mounted       the <img>/<video> exists with its src set
//   first-media         the media finished loading — the user can SEE it
//
// Plus two things a plain timeline can't show:
//   * long tasks — main-thread blocks over 50ms, with when and how long
//   * the media resource's own timing, including whether it was served from
//     cache (which is how we know the preload warm-up actually paid off)
//
// `at` on every mark is milliseconds since this document started loading, so
// marks are directly comparable to each other and, via bootId, joinable to the
// main process's process-relative timings.

import { isElectron, logEvent, appArgs } from './platform';
import { bundleStartAt } from './boot-clock';

const seen = new Set<string>();
const timeline: { name: string; at: number }[] = [];

const now = (): number =>
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : 0;

const bootId = (): string =>
  (appArgs as { bootId?: string } | undefined)?.bootId ?? 'unknown';

/**
 * Record a one-shot startup milestone. Repeat calls for the same name are
 * ignored, so callers don't need their own "did I already fire this" flag.
 */
export function markStartup(
  name: string,
  data?: Record<string, unknown>,
  /** Override the timestamp for a milestone that happened before this call. */
  atOverride?: number
): void {
  // app-log.jsonl is an Electron-only file; the web build has no equivalent
  // sink, and logEvent's fallback would just spam the console.
  if (!isElectron) return;
  if (seen.has(name)) return;
  seen.add(name);
  const at = Math.round(atOverride ?? now());
  timeline.push({ name, at });
  logEvent({
    level: 'info',
    scope: 'startup',
    message: name,
    data: { bootId: bootId(), at, ...(data ?? {}) },
  });
}

// ---- Long tasks -----------------------------------------------------------

// The browser reports tasks over 50ms as "long" — that threshold is its own,
// not ours, so there is nothing to configure here. We just keep the worst.
const MAX_REPORTED_TASKS = 15;

let longTasks: { atMs: number; durationMs: number; name: string }[] = [];
let taskObserver: PerformanceObserver | null = null;

function startLongTaskObserver(): void {
  if (taskObserver || typeof PerformanceObserver === 'undefined') return;
  try {
    taskObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        longTasks.push({
          atMs: Math.round(entry.startTime),
          durationMs: Math.round(entry.duration),
          name: entry.name,
        });
      }
    });
    taskObserver.observe({ entryTypes: ['longtask'] });
  } catch {
    // longtask isn't observable everywhere; the rest of the trace still works.
    taskObserver = null;
  }
}

function stopLongTaskObserver(): void {
  if (!taskObserver) return;
  try {
    taskObserver.disconnect();
  } catch {
    // already gone
  }
  taskObserver = null;
}

// ---- Renderer event-loop stalls -------------------------------------------

// Mirrors the main process's sampler (src/main/startup-trace.ts). Long tasks
// alone can't tell apart "the renderer computed something for 5 seconds" from
// "the OS descheduled the renderer for 5 seconds" — both surface as one long
// task. This sampler pairs with them: a stall that shows up here AND lands
// between two marks with no real work between them is the machine being
// starved (heavy GPU/RAM load elsewhere), not the app doing something slow.
const RENDER_SAMPLE_MS = 25;
const RENDER_STALL_MS = 40;

let loopSampler: ReturnType<typeof setInterval> | null = null;
let loopStalls: { atMs: number; stallMs: number }[] = [];

function startLoopSampler(): void {
  if (loopSampler) return;
  let expected = Date.now() + RENDER_SAMPLE_MS;
  loopSampler = setInterval(() => {
    const t = Date.now();
    const late = t - expected;
    expected = t + RENDER_SAMPLE_MS;
    if (late >= RENDER_STALL_MS) {
      loopStalls.push({ atMs: Math.round(now()), stallMs: Math.round(late) });
    }
  }, RENDER_SAMPLE_MS);
}

function stopLoopSampler(): void {
  if (!loopSampler) return;
  clearInterval(loopSampler);
  loopSampler = null;
}

// ---- Resource timing ------------------------------------------------------

interface ResourceFacts {
  name: string;
  startMs: number;
  durationMs: number;
}

/**
 * HTTP calls the renderer made during startup — in practice the local media
 * server (/health, /config, /api/*).
 *
 * Only http(s) is captured because that is all Chromium records: `file://`
 * (the bundle) and `gsm://` (the media) never produce Resource Timing entries,
 * which is why bundle and media delivery are measured elsewhere — the bundle
 * via boot-clock, the media via the main process's own read trace in
 * src/main/startup-trace.ts. Don't add a `gsm:` filter here expecting results.
 */
function serverCallFacts(): ResourceFacts[] {
  if (typeof performance === 'undefined' || !performance.getEntriesByType) {
    return [];
  }
  try {
    return (
      performance.getEntriesByType('resource') as PerformanceResourceTiming[]
    )
      .filter((e) => e.name.startsWith('http'))
      .map((e) => ({
        name: e.name.length > 160 ? `${e.name.slice(0, 160)}…` : e.name,
        startMs: Math.round(e.startTime),
        durationMs: Math.round(e.duration),
      }))
      .sort((a, b) => a.startMs - b.startMs);
  } catch {
    return [];
  }
}

/** Document-level timing: how the HTML, CSS, and bundle request actually went. */
function navigationFacts(): Record<string, number> {
  if (typeof performance === 'undefined' || !performance.getEntriesByType) {
    return {};
  }
  try {
    const nav = performance.getEntriesByType(
      'navigation'
    )[0] as PerformanceNavigationTiming | undefined;
    if (!nav) return {};
    return {
      responseEnd: Math.round(nav.responseEnd),
      domInteractive: Math.round(nav.domInteractive),
      domContentLoaded: Math.round(nav.domContentLoadedEventEnd),
      loadEvent: Math.round(nav.loadEventEnd),
    };
  } catch {
    return {};
  }
}

function paintFacts(): Record<string, number> {
  if (typeof performance === 'undefined' || !performance.getEntriesByType) {
    return {};
  }
  try {
    const out: Record<string, number> = {};
    for (const e of performance.getEntriesByType('paint')) {
      out[e.name] = Math.round(e.startTime);
    }
    return out;
  } catch {
    return {};
  }
}

// ---- Launch summary -------------------------------------------------------

let summarised = false;

/**
 * Emit ONE line describing the whole launch, then stop measuring.
 *
 * The individual marks are the detail; this is the line to read first. It
 * carries the ordered timeline, the phase costs that actually matter, the long
 * tasks that got in the way, and how the media itself was delivered.
 */
export function summariseStartup(reason: string): void {
  if (!isElectron || summarised) return;
  summarised = true;
  stopLongTaskObserver();
  stopLoopSampler();

  const at = (name: string): number =>
    timeline.find((m) => m.name === name)?.at ?? -1;
  const span = (from: string, to: string): number => {
    const a = at(from);
    const b = at(to);
    return a < 0 || b < 0 ? -1 : b - a;
  };

  logEvent({
    level: 'info',
    scope: 'startup',
    message: 'startup-summary',
    data: {
      bootId: bootId(),
      reason,
      appVersion: (appArgs as { appVersion?: string } | undefined)?.appVersion,
      initialFile: (appArgs as { filePath?: string } | undefined)?.filePath || '',
      timeline,
      phases: {
        // Everything before our first line of JS: window creation, HTML, CSS,
        // the preload, and fetching + compiling the bundle.
        toBundleStart: Math.round(bundleStartAt),
        // Cost of evaluating the app's module graph.
        moduleGraphEval: Math.round(at('imports-evaluated') - bundleStartAt),
        toReactRender: span('imports-evaluated', 'react-render'),
        toProvidersReady: span('react-render', 'providers-ready'),
        toMediaMounted: span('providers-ready', 'media-mounted'),
        // The media element existed; this is fetch + decode.
        mediaLoad: span('media-mounted', 'first-media'),
        total: at('first-media'),
      },
      paint: paintFacts(),
      navigation: navigationFacts(),
      longTasks: {
        count: longTasks.length,
        totalMs: longTasks.reduce((sum, t) => sum + t.durationMs, 0),
        worst: [...longTasks]
          .sort((a, b) => b.durationMs - a.durationMs)
          .slice(0, MAX_REPORTED_TASKS)
          .sort((a, b) => a.atMs - b.atMs),
      },
      serverCalls: serverCallFacts(),
      loopStalls: {
        count: loopStalls.length,
        totalMs: loopStalls.reduce((s, x) => s + x.stallMs, 0),
        worst: [...loopStalls]
          .sort((a, b) => b.stallMs - a.stallMs)
          .slice(0, 10)
          .sort((a, b) => a.atMs - b.atMs),
      },
    },
  });
  loopStalls = [];
  longTasks = [];

  // Let the main process close out its own sampling against the same launch.
  try {
    (window as unknown as {
      electron?: { ipcRenderer?: { sendMessage(c: string, a: unknown[]): void } };
    }).electron?.ipcRenderer?.sendMessage('startup-first-media', [{ reason }]);
  } catch {
    // Diagnostics must never break startup.
  }
}

/**
 * Begin renderer-side tracing. Called once, as early as the bundle can manage.
 */
export function beginStartupTrace(): void {
  if (!isElectron) return;
  startLongTaskObserver();
  startLoopSampler();
  markStartup('bundle-start', undefined, bundleStartAt);
}
