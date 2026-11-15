// Run timing for media-server jobs (Jobs tab).
//
// The server stamps every job with `started_at` (first time it began
// running; kept through pause/resume and restarts), `completed_at` (when it
// reached ANY terminal state: completed, error or cancelled) and a derived
// `duration_ms` (null until both exist). Unset Go times serialize as
// "0001-01-01T00:00:00Z", so anything at or before the epoch means "unset".
// A running job's elapsed time is computed here from started_at.

export interface TimedJob {
  state: string;
  created_at?: string;
  started_at?: string;
  /** Older servers (no started_at): the current run's claim time. */
  claimed_at?: string;
  completed_at?: string;
  duration_ms?: number | null;
}

/** Epoch ms of a server timestamp, or null when unset/invalid. */
export function jobTimeMs(iso: string | undefined | null): number | null {
  if (!iso) return null;
  const t = Date.parse(iso);
  return Number.isFinite(t) && t > 0 ? t : null;
}

/**
 * Compact duration: "850ms", "42s", "3m 02s", "1h 04m", "2d 3h".
 * Mirrors renderer.formatDuration in the Go server.
 */
export function formatDuration(ms: number): string {
  const v = Number.isFinite(ms) && ms > 0 ? ms : 0;
  if (v < 1000) return `${Math.round(v)}ms`;
  const s = Math.floor(v / 1000);
  const pad = (n: number) => String(n).padStart(2, '0');
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${pad(s % 60)}s`;
  if (s < 86400) {
    return `${Math.floor(s / 3600)}h ${pad(Math.floor((s % 3600) / 60))}m`;
  }
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

const MONTHS = [
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

/**
 * Clock time for a timestamp in local time: "14:02:11" when it is on the
 * same day as `nowMs`, otherwise "Jan 3 14:02".
 */
export function formatClock(ms: number, nowMs: number = Date.now()): string {
  const d = new Date(ms);
  const n = new Date(nowMs);
  const pad = (x: number) => String(x).padStart(2, '0');
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  const sameDay =
    d.getFullYear() === n.getFullYear() &&
    d.getMonth() === n.getMonth() &&
    d.getDate() === n.getDate();
  if (sameDay) return `${hm}:${pad(d.getSeconds())}`;
  return `${MONTHS[d.getMonth()]} ${d.getDate()} ${hm}`;
}

/** Full local timestamp for tooltips: "Jan 3, 2026 14:02:11". */
export function formatFullTime(ms: number): string {
  const d = new Date(ms);
  const pad = (x: number) => String(x).padStart(2, '0');
  return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()} ${pad(
    d.getHours()
  )}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

export interface JobRunTiming {
  /** First start (epoch ms), null if the job never ran. */
  startedMs: number | null;
  /** End of the run (epoch ms), null while not finished. */
  endedMs: number | null;
  /** Final duration (ms) once ended; null otherwise. */
  durationMs: number | null;
  /** True while the job is running — elapsed should tick live. */
  live: boolean;
}

export function jobRunTiming(job: TimedJob): JobRunTiming {
  const running = job.state === 'in_progress';
  let startedMs = jobTimeMs(job.started_at);
  // Older servers had no started_at; claimed_at is the current run's start.
  if (startedMs === null && running) startedMs = jobTimeMs(job.claimed_at);
  const terminal =
    job.state === 'completed' ||
    job.state === 'error' ||
    job.state === 'cancelled';
  const endedMs = terminal ? jobTimeMs(job.completed_at) : null;
  let durationMs: number | null = null;
  if (endedMs !== null && startedMs !== null) {
    durationMs =
      typeof job.duration_ms === 'number'
        ? job.duration_ms
        : Math.max(0, endedMs - startedMs);
  }
  return {
    startedMs,
    endedMs,
    durationMs,
    live: running && startedMs !== null,
  };
}

/** Tooltip with every known timestamp, one per line. */
export function jobTimesTooltip(job: TimedJob): string {
  const t = jobRunTiming(job);
  const lines: string[] = [];
  const created = jobTimeMs(job.created_at);
  if (created !== null) lines.push(`Queued: ${formatFullTime(created)}`);
  if (t.startedMs !== null)
    lines.push(`Started: ${formatFullTime(t.startedMs)}`);
  if (t.endedMs !== null) lines.push(`Ended: ${formatFullTime(t.endedMs)}`);
  if (t.durationMs !== null)
    lines.push(`Duration: ${formatDuration(t.durationMs)}`);
  return lines.join('\n');
}
