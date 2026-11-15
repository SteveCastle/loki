import {
  formatClock,
  formatDuration,
  formatFullTime,
  jobRunTiming,
  jobTimeMs,
  jobTimesTooltip,
} from '../renderer/job-timing';

const ZERO = '0001-01-01T00:00:00Z';

describe('jobTimeMs', () => {
  it('treats Go zero times, empty and invalid strings as unset', () => {
    expect(jobTimeMs(ZERO)).toBeNull();
    expect(jobTimeMs('')).toBeNull();
    expect(jobTimeMs(undefined)).toBeNull();
    expect(jobTimeMs('nope')).toBeNull();
  });
  it('parses real timestamps', () => {
    expect(jobTimeMs('2026-01-02T03:04:05Z')).toBe(
      Date.UTC(2026, 0, 2, 3, 4, 5)
    );
  });
});

describe('formatDuration', () => {
  it.each([
    [-5, '0ms'],
    [850, '850ms'],
    [42_000, '42s'],
    [182_000, '3m 02s'],
    [3_899_000, '1h 04m'],
    [50 * 3600_000, '2d 2h'],
  ])('%d ms -> %s', (ms, want) => {
    expect(formatDuration(ms)).toBe(want);
  });
});

describe('formatClock', () => {
  it('shows only the time on the same day, the date otherwise', () => {
    const now = new Date(2026, 0, 2, 18, 0, 0).getTime();
    expect(formatClock(new Date(2026, 0, 2, 9, 5, 7).getTime(), now)).toBe(
      '09:05:07'
    );
    expect(formatClock(new Date(2025, 11, 31, 23, 1, 0).getTime(), now)).toBe(
      'Dec 31 23:01'
    );
  });
  it('formats full timestamps for tooltips', () => {
    expect(formatFullTime(new Date(2026, 0, 2, 3, 4, 5).getTime())).toBe(
      'Jan 2, 2026 03:04:05'
    );
  });
});

describe('jobRunTiming', () => {
  const start = '2026-01-02T03:04:05Z';
  const end = '2026-01-02T03:05:20Z';

  it('a queued job has no run times', () => {
    expect(
      jobRunTiming({ state: 'pending', started_at: ZERO, completed_at: ZERO })
    ).toEqual({ startedMs: null, endedMs: null, durationMs: null, live: false });
  });

  it('a running job ticks live from started_at', () => {
    const t = jobRunTiming({
      state: 'in_progress',
      started_at: start,
      completed_at: ZERO,
    });
    expect(t.live).toBe(true);
    expect(t.startedMs).toBe(Date.parse(start));
    expect(t.endedMs).toBeNull();
    expect(t.durationMs).toBeNull();
  });

  it('falls back to claimed_at for a running job from an older server', () => {
    const t = jobRunTiming({ state: 'in_progress', claimed_at: start });
    expect(t.startedMs).toBe(Date.parse(start));
    expect(t.live).toBe(true);
  });

  it('a paused job keeps its start but does not tick', () => {
    const t = jobRunTiming({ state: 'paused', started_at: start });
    expect(t.startedMs).toBe(Date.parse(start));
    expect(t.live).toBe(false);
  });

  it.each(['completed', 'error', 'cancelled'])(
    'a %s job uses the server duration_ms',
    (state) => {
      const t = jobRunTiming({
        state,
        started_at: start,
        completed_at: end,
        duration_ms: 75_000,
      });
      expect(t).toEqual({
        startedMs: Date.parse(start),
        endedMs: Date.parse(end),
        durationMs: 75_000,
        live: false,
      });
    }
  );

  it('computes the duration when the server omits duration_ms', () => {
    expect(
      jobRunTiming({ state: 'completed', started_at: start, completed_at: end })
        .durationMs
    ).toBe(75_000);
  });

  it('a job cancelled before it ran has an end but no duration', () => {
    const t = jobRunTiming({
      state: 'cancelled',
      started_at: ZERO,
      completed_at: end,
      duration_ms: null,
    });
    expect(t.startedMs).toBeNull();
    expect(t.endedMs).toBe(Date.parse(end));
    expect(t.durationMs).toBeNull();
  });

  it('builds a tooltip with every known time', () => {
    const tip = jobTimesTooltip({
      state: 'completed',
      created_at: start,
      started_at: start,
      completed_at: end,
      duration_ms: 75_000,
    });
    expect(tip.split('\n').map((l) => l.split(':')[0])).toEqual([
      'Queued',
      'Started',
      'Ended',
      'Duration',
    ]);
    expect(tip).toContain('Duration: 1m 15s');
  });
});
