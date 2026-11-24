import { buildJobs, settingsFor } from '../renderer/components/transform/intents';
import { withAutoEnhance } from '../renderer/components/transform/enhance-prompt';
import { submitJobs } from '../renderer/components/transform/store';

jest.mock('../renderer/components/transform/engine-setup', () => ({
  ensureEngines: async () => 'ready',
}));

// jsdom has no AbortSignal.timeout, which submitJobs passes as its signal.
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

const IMG = 'C:/media/photo.jpg';

describe('Enhance first (withAutoEnhance)', () => {
  const direct = (over: Record<string, unknown> = {}) =>
    settingsFor('direct', { prompt: '<Picture 1> waves at the camera', variations: 2, ...over });

  it('leaves jobs alone when the box is off', () => {
    const { jobs } = buildJobs('direct', direct(), [IMG]);
    expect(withAutoEnhance(jobs, 'direct', direct())).toBe(jobs);
  });

  it('gives each run its own rewrite, seeded with the run seed', () => {
    const s = direct({ autoEnhance: true });
    const { jobs } = buildJobs('direct', s, [IMG], { random: () => 0.25 });
    const out = withAutoEnhance(jobs, 'direct', s);
    expect(out).toHaveLength(2);
    out.forEach((job, i) => {
      expect(job.enhance).toMatchObject({
        prompt: '<Picture 1> waves at the camera',
        model: 'h3-video',
        task: 'video',
        seed: Number(jobs[i].fields.seed),
        references: [{ token: '<Picture 1>', kind: 'image', path: IMG }],
      });
    });
    expect(out[0].enhance?.seed).not.toBe(out[1].enhance?.seed);
  });

  it('repeats the rewrite seed for a fixed seed', () => {
    const s = direct({ autoEnhance: true, seedMode: 'fixed', seed: 1234, variations: 1 });
    const a = withAutoEnhance(buildJobs('direct', s, [IMG]).jobs, 'direct', s);
    const b = withAutoEnhance(buildJobs('direct', s, [IMG]).jobs, 'direct', s);
    expect(a[0].enhance?.seed).toBe(1234);
    expect(b[0].enhance?.seed).toBe(1234);
  });

  it('skips intents whose prompt has no guide', () => {
    const s = settingsFor('upscale', { autoEnhance: true });
    const { jobs } = buildJobs('upscale', s, [IMG]);
    expect(withAutoEnhance(jobs, 'upscale', s).every((j) => !j.enhance)).toBe(true);
  });

  it('posts an enhanced run to the rewrite endpoint with the run as `then`', async () => {
    const fetchMock = jest.fn().mockResolvedValue({ ok: true, json: async () => ({ jobId: 'enh-1' }) });
    (global as any).fetch = fetchMock;
    const s = direct({ autoEnhance: true, variations: 1 });
    const jobs = withAutoEnhance(buildJobs('direct', s, [IMG]).jobs, 'direct', s);
    const ids = await submitJobs(jobs, { mediaServerBase: 'http://srv', authToken: null });
    expect(ids).toEqual(['enh-1']);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('http://srv/api/transform/enhance-prompt');
    const body = JSON.parse(init.body);
    expect(body.then).toEqual({ input: jobs[0].input, fields: jobs[0].fields });
    expect(body.seed).toBe(Number(jobs[0].fields.seed));
  });
});
