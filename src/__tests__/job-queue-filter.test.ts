import { jobMatches, NO_FILTER, typesLabel, type QueueJob } from '../renderer/components/metadata/job-queue';

const job = (over: Partial<QueueJob>): QueueJob => ({
  id: 'abc-123',
  command: 'reshoot',
  arguments: ['--duration', '15', '--prompt', 'she waves at the camera'],
  input: 'Z:/photos/beach.png',
  state: 'pending',
  created_at: '2026-11-23T16:00:00Z',
  ...over,
});

describe('job queue filter', () => {
  it('passes everything with no filter', () => {
    expect(jobMatches(job({}), NO_FILTER)).toBe(true);
  });

  it('needs every search word in the command, arguments, input, id or title', () => {
    expect(jobMatches(job({}), { ...NO_FILTER, q: 'WAVES beach' })).toBe(true);
    expect(jobMatches(job({}), { ...NO_FILTER, q: 'abc-123' })).toBe(true);
    expect(jobMatches(job({}), { ...NO_FILTER, q: 'Animate' }, 'Animate · beach.png')).toBe(true);
    expect(jobMatches(job({}), { ...NO_FILTER, q: 'waves dog' })).toBe(false);
  });

  it('filters by type and state, with active = queued, running or paused', () => {
    expect(jobMatches(job({}), { ...NO_FILTER, types: ['dream'] })).toBe(false);
    expect(jobMatches(job({}), { ...NO_FILTER, types: ['dream', 'reshoot'], state: 'pending' })).toBe(true);
    expect(jobMatches(job({ state: 'paused' }), { ...NO_FILTER, state: 'active' })).toBe(true);
    expect(jobMatches(job({ state: 'completed' }), { ...NO_FILTER, state: 'active' })).toBe(false);
    expect(jobMatches(job({ state: 'error' }), { ...NO_FILTER, state: 'error' })).toBe(true);
  });

  it('hides the checked types in Hide mode', () => {
    const hide = { ...NO_FILTER, types: ['reshoot', 'enhance-prompt'], typeMode: 'hide' as const };
    expect(jobMatches(job({}), hide)).toBe(false);
    expect(jobMatches(job({ command: 'dream' }), hide)).toBe(true);
    expect(jobMatches(job({ command: 'dream' }), { ...hide, types: [] })).toBe(true);
  });

  it('labels the type button', () => {
    expect(typesLabel(NO_FILTER)).toBe('All types');
    expect(typesLabel({ types: ['dream', 'reshoot'], typeMode: 'show' })).toBe('dream, reshoot');
    expect(typesLabel({ types: ['a', 'b', 'c'], typeMode: 'hide' })).toBe('All but 3 types');
  });
});
