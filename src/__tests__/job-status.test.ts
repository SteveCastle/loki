import {
  PathJob,
  isActiveJobState,
  jobStatusLabel,
  mergeJobProgress,
  mergeJobUpdate,
  pickActiveJob,
} from '../renderer/job-status';

const job = (over: Partial<PathJob> & { id: string }): PathJob => ({
  command: 'transcribe',
  state: 'pending',
  ...over,
});

describe('isActiveJobState', () => {
  it('treats pending, in_progress, and paused as active', () => {
    expect(isActiveJobState('pending')).toBe(true);
    expect(isActiveJobState('in_progress')).toBe(true);
    expect(isActiveJobState('paused')).toBe(true);
  });

  it('treats terminal states as inactive', () => {
    expect(isActiveJobState('completed')).toBe(false);
    expect(isActiveJobState('cancelled')).toBe(false);
    expect(isActiveJobState('error')).toBe(false);
  });
});

describe('pickActiveJob', () => {
  it('returns null when nothing matches', () => {
    expect(pickActiveJob([], ['transcribe'])).toBeNull();
    expect(
      pickActiveJob([job({ id: 'a', command: 'hash' })], ['transcribe'])
    ).toBeNull();
    expect(
      pickActiveJob([job({ id: 'a', state: 'completed' })], ['transcribe'])
    ).toBeNull();
  });

  it('returns the newest (last) matching active job', () => {
    const jobs = [
      job({ id: 'old', state: 'in_progress' }),
      job({ id: 'other-command', command: 'embed', state: 'in_progress' }),
      job({ id: 'new' }),
    ];
    expect(pickActiveJob(jobs, ['transcribe'])?.id).toBe('new');
  });

  it('skips terminal jobs even when they are newest', () => {
    const jobs = [
      job({ id: 'running', state: 'in_progress' }),
      job({ id: 'done', state: 'completed' }),
    ];
    expect(pickActiveJob(jobs, ['transcribe'])?.id).toBe('running');
  });
});

describe('jobStatusLabel', () => {
  it('labels queue and pause states', () => {
    expect(jobStatusLabel(job({ id: 'a' }), 'Transcribing')).toBe('Queued…');
    expect(jobStatusLabel(job({ id: 'a', state: 'paused' }))).toBe('Paused');
  });

  it('shows progress only for multi-item jobs', () => {
    expect(
      jobStatusLabel(
        job({
          id: 'a',
          state: 'in_progress',
          progress_done: 2,
          progress_total: 5,
        }),
        'Transcribing'
      )
    ).toBe('Transcribing… 2/5');
    // A single-file job's 0/1 → 1/1 counter is noise, not signal.
    expect(
      jobStatusLabel(
        job({
          id: 'a',
          state: 'in_progress',
          progress_done: 0,
          progress_total: 1,
        }),
        'Transcribing'
      )
    ).toBe('Transcribing…');
  });
});

describe('mergeJobUpdate / mergeJobProgress', () => {
  const tracked = [job({ id: 'a' }), job({ id: 'b', command: 'embed' })];

  it('updates tracked jobs in place', () => {
    const next = mergeJobUpdate(tracked, job({ id: 'a', state: 'completed' }));
    expect(next.find((j) => j.id === 'a')?.state).toBe('completed');
    expect(next.find((j) => j.id === 'b')?.state).toBe('pending');
  });

  it('ignores unknown job ids (membership changes come from refetch)', () => {
    expect(mergeJobUpdate(tracked, job({ id: 'zzz' }))).toBe(tracked);
    expect(mergeJobProgress(tracked, 'zzz', 1, 2)).toBe(tracked);
  });

  it('merges progress counters', () => {
    const next = mergeJobProgress(tracked, 'b', 3, 9);
    const b = next.find((j) => j.id === 'b');
    expect(b?.progress_done).toBe(3);
    expect(b?.progress_total).toBe(9);
  });
});
