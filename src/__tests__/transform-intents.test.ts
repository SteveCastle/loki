import {
  availability,
  buildJobs,
  cameraViewLabel,
  cameraViewPreset,
  runSummary,
  snapAzimuth,
  classify,
  estimateSeconds,
  formatDuration,
  mediaKind,
  nativeCanvas,
  parseSize,
  planFor,
  referenceTokens,
  remapTokens,
  reshootFrames,
  retouchPreset,
  settingsFor,
  stepsFor,
  stepsForSettings,
  FAST_STEPS,
  tokenIssues,
  usesReferences,
} from '../renderer/components/transform/intents';

const IMG = 'C:/media/photo.jpg';
const IMG2 = 'C:/media/other.png';
const VID = 'C:/media/clip.mp4';
const AUD = 'C:/media/song.mp3';

describe('classification', () => {
  it('sorts paths by media kind', () => {
    expect(mediaKind('a.JPG')).toBe('image');
    expect(mediaKind('a.mkv')).toBe('video');
    expect(mediaKind('a.flac')).toBe('audio');
    expect(mediaKind('a.txt')).toBe('other');
    const c = classify([IMG, VID, AUD, 'x.txt', IMG2, '']);
    expect(c.images).toEqual([IMG, IMG2]);
    expect(c.videos).toEqual([VID]);
    expect(c.audios).toEqual([AUD]);
    expect(c.other).toEqual(['x.txt']);
  });
});

describe('availability', () => {
  it('gates intents on the selection', () => {
    const one = classify([IMG]);
    expect(availability('restore', one).ok).toBe(true);
    expect(usesReferences('edit', one)).toBe(false);
    expect(usesReferences('edit', classify([IMG, IMG2]))).toBe(true);
    expect(usesReferences('restore', classify([IMG, IMG2]))).toBe(false);
    expect(availability('alive', classify([VID])).ok).toBe(false);
    expect(availability('direct', classify([AUD])).ok).toBe(true);
    // retouch takes a single video (a frame is sampled), not several
    expect(availability('restore', classify([VID])).ok).toBe(true);
    expect(availability('restore', classify([VID, 'C:/media/b.mp4'])).ok).toBe(false);
  });
});

describe('buildJobs: retouch', () => {
  const fixed = { random: () => 0.5 };

  it('restore uses the preset and no prompt', () => {
    const r = buildJobs('restore', settingsFor('restore'), [IMG], fixed);
    expect(r.error).toBeUndefined();
    expect(r.jobs).toHaveLength(1);
    expect(r.jobs[0].input).toBe('retouch "C:/media/photo.jpg"');
    expect(r.jobs[0].fields.preset).toBe('restore');
    expect(r.jobs[0].fields.prompt).toBeUndefined();
    expect(r.jobs[0].fields.steps).toBe('25');
    expect(r.jobs[0].fields.scale).toBeUndefined();
  });

  it('upscale passes the factor; 1x passes nothing', () => {
    const s = settingsFor('upscale', { scale: 3 });
    expect(buildJobs('upscale', s, [IMG], fixed).jobs[0].fields.scale).toBe('3');
    expect(buildJobs('upscale', { ...s, scale: 1 }, [IMG], fixed).jobs[0].fields.scale).toBeUndefined();
  });

  it('wallpaper picks the preset from the target', () => {
    expect(retouchPreset('wallpaper', settingsFor('wallpaper'))).toBe('4kify');
    const phone = settingsFor('wallpaper', { wallpaperTarget: 'phone' });
    expect(buildJobs('wallpaper', phone, [IMG], fixed).jobs[0].fields.preset).toBe('4kify-phone');
  });

  it('edit needs a prompt and sends it verbatim', () => {
    expect(buildJobs('edit', settingsFor('edit'), [IMG]).error).toMatch(/Describe/);
    const s = { ...settingsFor('edit'), prompt: 'make it "night"\nwith rain' };
    const j = buildJobs('edit', s, [IMG], fixed).jobs[0];
    expect(j.fields.prompt).toBe('make it "night"\nwith rain');
    expect(j.fields.preset).toBeUndefined();
  });

  it('edit can resize in the same pass (scale and custom size)', () => {
    const base = { ...settingsFor('edit'), prompt: 'x' };
    expect(buildJobs('edit', { ...base, sizeMode: 'scale', scale: 2 }, [IMG], fixed).jobs[0].fields.scale).toBe('2');
    const c = buildJobs('edit', { ...base, sizeMode: 'custom', customSize: '1920 × 1080' }, [IMG], fixed).jobs[0].fields;
    expect(c.size).toBe('1920x1080');
    expect(c.scale).toBeUndefined();
  });

  it('edit with several images makes ONE run that shows the model all of them as references', () => {
    const s = { ...settingsFor('edit'), prompt: 'put <image2> into <image1>' };
    const r = buildJobs('edit', s, [IMG, IMG2], fixed);
    expect(r.jobs).toHaveLength(1);
    expect(r.jobs[0].input).toBe('retouch "C:/media/photo.jpg\nC:/media/other.png"');
    expect(r.jobs[0].fields.combine).toBe('1');
    expect(r.jobs[0].fields.prompt).toBe('put <image2> into <image1>');
    expect(r.jobs[0].label).toMatch(/2 images as references/);
    // one image: a plain edit
    expect(buildJobs('edit', s, [IMG], fixed).jobs[0].fields.combine).toBeUndefined();
    // presets stay per-image batches
    expect(buildJobs('restore', settingsFor('restore'), [IMG, IMG2], fixed).jobs[0].fields.combine).toBeUndefined();
  });

  it('a manual prompt replaces a preset\'s built-in text but keeps the preset (size, references)', () => {
    const s = { ...settingsFor('wallpaper'), append: 'keep grain', manualPrompt: 'Outpaint <image1> to 16:9' };
    const f = buildJobs('wallpaper', s, [IMG], fixed).jobs[0].fields;
    expect(f.preset).toBe('4kify');
    expect(f.prompt).toBe('Outpaint <image1> to 16:9');
    expect(f.append).toBeUndefined();
    expect(buildJobs('wallpaper', { ...s, manualPrompt: '  ' }, [IMG], fixed).error).toMatch(/Write the prompt/);
  });

  it('a single video carries the sampled frame time', () => {
    const r = buildJobs('restore', settingsFor('restore'), [VID], { ...fixed, videoTime: 12.3456 });
    expect(r.jobs[0].fields.time).toBe('12.346');
    expect(buildJobs('restore', settingsFor('restore'), [IMG], { ...fixed, videoTime: 4 }).jobs[0].fields.time).toBeUndefined();
  });

  it('variations queue distinct seeds; fixed seeds are reproducible', () => {
    const s = { ...settingsFor('restore'), variations: 3, seedMode: 'fixed' as const, seed: 100 };
    const seeds = buildJobs('restore', s, [IMG]).jobs.map((j) => j.fields.seed);
    expect(new Set(seeds).size).toBe(3);
    expect(seeds[0]).toBe('100');
    expect(buildJobs('restore', s, [IMG]).jobs.map((j) => j.fields.seed)).toEqual(seeds);
    expect(buildJobs('restore', { ...s, variations: 99 }, [IMG]).jobs).toHaveLength(4);
  });

  it('quality maps to steps per engine', () => {
    expect(stepsFor('retouch', 'draft')).toBe(12);
    expect(stepsFor('reshoot', 'fine')).toBe(32);
    expect(buildJobs('restore', { ...settingsFor('restore'), quality: 'fine' }, [IMG], fixed).jobs[0].fields.steps).toBe('40');
  });
});

describe('buildJobs: reshoot', () => {
  const fixed = { random: () => 0.5 };

  it('alive makes one job per image with animate + describe + shake', () => {
    const s = { ...settingsFor('alive'), describe: 'a woman in a sunlit room', shake: 'handheld' as const };
    const r = buildJobs('alive', s, [IMG, IMG2], fixed);
    expect(r.jobs).toHaveLength(2);
    expect(r.jobs[0].input).toBe('reshoot "C:/media/photo.jpg"');
    expect(r.jobs[0].fields).toMatchObject({ animate: '1', describe: 'a woman in a sunlit room', shake: 'handheld', duration: '5', steps: '20' });
    expect(r.jobs[0].fields.noaudio).toBeUndefined();
  });

  it('direct needs a prompt and sends all references in one job', () => {
    expect(buildJobs('direct', settingsFor('direct'), [IMG]).error).toMatch(/Describe/);
    const s = { ...settingsFor('direct'), prompt: 'She dances to <Audio 1>', refSize: 'max' as const, keepSoundtrack: false };
    const r = buildJobs('direct', s, [IMG, VID, AUD], fixed);
    expect(r.jobs).toHaveLength(1);
    expect(r.jobs[0].input).toBe('reshoot "C:/media/photo.jpg\nC:/media/clip.mp4\nC:/media/song.mp3"');
    expect(r.jobs[0].fields).toMatchObject({ prompt: 'She dances to <Audio 1>', refsize: 'max', novideoaudio: '1' });
  });

  it('clamps the duration', () => {
    const s = { ...settingsFor('alive'), describe: 'x', duration: 99 };
    expect(buildJobs('alive', s, [IMG], fixed).jobs[0].fields.duration).toBe('15');
  });
});

describe('the fast video model (FastH3 Omni)', () => {
  const fixed = { random: () => 0.5 };

  it('H3 is the default and sends no model field (old engines keep working)', () => {
    expect(settingsFor('alive').reshootModel).toBe('h3');
    const alive = buildJobs('alive', { ...settingsFor('alive'), describe: 'x' }, [IMG], fixed).jobs[0].fields;
    expect(alive.model).toBeUndefined();
    expect(alive.steps).toBe('20');
    const direct = buildJobs('direct', { ...settingsFor('direct'), prompt: 'x', quality: 'fine' }, [IMG], fixed).jobs[0].fields;
    expect(direct.model).toBeUndefined();
    expect(direct.steps).toBe('32');
  });

  it('fast sends model=fast and always 8 steps, whatever the quality', () => {
    for (const quality of ['draft', 'balanced', 'fine'] as const) {
      const s = { ...settingsFor('direct'), prompt: 'x', quality, reshootModel: 'fast' as const };
      expect(buildJobs('direct', s, [IMG, AUD], fixed).jobs[0].fields).toMatchObject({ model: 'fast', steps: String(FAST_STEPS) });
    }
    const alive = buildJobs('alive', { ...settingsFor('alive'), describe: 'x', reshootModel: 'fast' }, [IMG, IMG2], fixed);
    expect(alive.jobs.map((j) => [j.fields.model, j.fields.steps])).toEqual([['fast', '8'], ['fast', '8']]);
    expect(FAST_STEPS).toBe(8);
  });

  it('the model never touches image runs', () => {
    expect(stepsForSettings('retouch', { quality: 'balanced', reshootModel: 'fast' })).toBe(25);
    const f = buildJobs('restore', { ...settingsFor('restore'), reshootModel: 'fast' }, [IMG], fixed).jobs[0].fields;
    expect(f.model).toBeUndefined();
    expect(f.steps).toBe('25');
  });

  it('the plan and the time estimate follow the fast step count', () => {
    const src = { width: 973, height: 1646 };
    const h3 = planFor('alive', settingsFor('alive', { duration: 5 }), [IMG], src);
    const fast = planFor('alive', settingsFor('alive', { duration: 5, reshootModel: 'fast', quality: 'fine' }), [IMG], src);
    expect(h3.steps).toBe(20);
    expect(fast.steps).toBe(8);
    expect(estimateSeconds(fast)).toBeLessThan(estimateSeconds(h3) * 0.6);
  });

  it('the last-run summary names the model instead of a quality', () => {
    expect(runSummary('alive', { ...settingsFor('alive'), describe: 'a dog', reshootModel: 'fast' })).toContain('fast model');
    expect(runSummary('alive', { ...settingsFor('alive'), describe: 'a dog' })).toContain('balanced');
  });
});

describe('geometry helpers', () => {
  it('parses sizes', () => {
    expect(parseSize('1920x1080')).toEqual([1920, 1080]);
    expect(parseSize(' 640 × 480 ')).toEqual([640, 480]);
    expect(parseSize('wide')).toBeNull();
    expect(parseSize('0x0')).toBeNull();
  });

  it('snaps to the native canvases like loki-reshoot', () => {
    expect(nativeCanvas(973, 1646)).toMatchObject({ label: '9:16', width: 768, height: 1344 });
    expect(nativeCanvas(1920, 1080)).toMatchObject({ label: '16:9', width: 1344, height: 768 });
    expect(nativeCanvas(1000, 1000)).toMatchObject({ label: '1:1', width: 768, height: 768 });
    expect(nativeCanvas(1080, 1350)).toMatchObject({ label: '3:4', width: 768, height: 1024 });
  });

  it('snaps frame counts up to 17k+5', () => {
    expect(reshootFrames(5)).toBe(124);
    expect(reshootFrames(1)).toBe(39);
    expect(reshootFrames(15)).toBe(362);
  });
});

describe('plan and estimate', () => {
  it('names the output like the server does', () => {
    expect(planFor('restore', settingsFor('restore'), [IMG]).outputName).toBe('photo_restored.png');
    expect(planFor('upscale', settingsFor('upscale'), [IMG], { width: 1000, height: 500 })).toMatchObject({ outputName: 'photo_up.png', outWidth: 2000, outHeight: 1000 });
    expect(planFor('wallpaper', settingsFor('wallpaper'), [IMG])).toMatchObject({ outputName: 'photo_4k.png', outWidth: 3840, outHeight: 2160 });
    expect(planFor('edit', { ...settingsFor('edit'), prompt: 'x' }, [IMG]).outputName).toBe('photo_edit.png');
    expect(planFor('alive', settingsFor('alive'), [IMG], { width: 973, height: 1646 })).toMatchObject({ outputName: 'photo_reshoot.mp4', outWidth: 768, outHeight: 1344, frames: 124 });
  });

  it('counts the results, not just the jobs', () => {
    const three = [IMG, IMG2, 'C:/media/c.png'];
    expect(planFor('restore', settingsFor('restore'), three)).toMatchObject({ jobCount: 1, outputs: 3 });
    expect(planFor('restore', settingsFor('restore', { variations: 2 }), three)).toMatchObject({ jobCount: 2, outputs: 6 });
    expect(planFor('edit', { ...settingsFor('edit'), prompt: 'x' }, three)).toMatchObject({ jobCount: 1, outputs: 1 });
    expect(planFor('edit', { ...settingsFor('edit'), prompt: 'x', variations: 2 }, three)).toMatchObject({ jobCount: 2, outputs: 2 });
    expect(planFor('alive', settingsFor('alive'), three)).toMatchObject({ jobCount: 3, outputs: 3 });
  });

  it('estimates grow with size and length', () => {
    const small = estimateSeconds(planFor('upscale', settingsFor('upscale', { scale: 1.5 }), [IMG], { width: 500, height: 500 }));
    const big = estimateSeconds(planFor('upscale', settingsFor('upscale', { scale: 4 }), [IMG], { width: 2000, height: 2000 }));
    expect(big).toBeGreaterThan(small * 5);
    const short = estimateSeconds(planFor('alive', settingsFor('alive', { duration: 2 }), [IMG], { width: 973, height: 1646 }));
    const long = estimateSeconds(planFor('alive', settingsFor('alive', { duration: 12 }), [IMG], { width: 973, height: 1646 }));
    expect(long).toBeGreaterThan(short * 2);
    // measured: 5 s at 768x1344, 20 steps ≈ 4.5 min end to end
    const five = estimateSeconds(planFor('alive', settingsFor('alive', { duration: 5 }), [IMG], { width: 973, height: 1646 }));
    expect(five).toBeGreaterThan(200);
    expect(five).toBeLessThan(400);
  });

  it('formats durations', () => {
    expect(formatDuration(12)).toBe('10 s');
    expect(formatDuration(100)).toBe('2 min');
    expect(formatDuration(4000)).toBe('1 h 7 min');
  });
});

describe('reference tokens', () => {
  const C = 'C:/media/c.png';

  it('names every image of a multi-image edit, and only <image1> for a per-image run', () => {
    expect(referenceTokens('edit', [IMG, IMG2, C])).toEqual({ [IMG]: '<image1>', [IMG2]: '<image2>', [C]: '<image3>' });
    expect(referenceTokens('edit', [IMG])).toEqual({ [IMG]: '<image1>' });
    expect(referenceTokens('restore', [IMG, IMG2])).toEqual({ [IMG]: '', [IMG2]: '' });
    expect(referenceTokens('restore', [IMG, IMG2], true)).toEqual({ [IMG]: '<image1>', [IMG2]: '<image1>' });
    expect(referenceTokens('direct', [IMG, VID, AUD])).toEqual({ [IMG]: '<Picture 1>', [VID]: '<Video 1>', [AUD]: '<Audio 1>' });
    expect(referenceTokens('alive', [IMG], true)).toEqual({ [IMG]: '<Picture 1>' });
  });

  it('tokens follow their files when the inputs are reordered or one is removed', () => {
    const prompt = 'put the jacket of <image2> on <image1>, lit like <image3>';
    expect(remapTokens(prompt, 'edit', [IMG, IMG2, C], [IMG2, IMG, C])).toBe('put the jacket of <image1> on <image2>, lit like <image3>');
    // IMG2 removed: <image3> becomes <image2>; the removed file's token is left for validation to flag
    expect(remapTokens(prompt, 'edit', [IMG, IMG2, C], [IMG, C])).toBe('put the jacket of <image2> on <image1>, lit like <image2>');
  });

  it('reports unknown and unused tokens', () => {
    expect(tokenIssues('use <image3> with <image1>', ['<image1>', '<image2>'])).toEqual({ unknown: ['<image3>'], unused: ['<image2>'] });
    expect(tokenIssues('no tokens', ['', 'photo'])).toEqual({ unknown: [], unused: [] });
  });
});

describe('buildJobs: manual prompt on reshoot', () => {
  it('bring to life with a hand-written prompt drops the living-photo expansion', () => {
    const s = { ...settingsFor('alive'), describe: 'a dog', manualPrompt: '<Picture 1> wags its tail' };
    const f = buildJobs('alive', s, [IMG], { random: () => 0.5 }).jobs[0].fields;
    expect(f.animate).toBeUndefined();
    expect(f.describe).toBeUndefined();
    expect(f.prompt).toBe('<Picture 1> wags its tail');
  });

  it('direct uses the manual prompt over the shape prompt', () => {
    const s = { ...settingsFor('direct'), prompt: 'a', manualPrompt: 'b' };
    expect(buildJobs('direct', s, [IMG], { random: () => 0.5 }).jobs[0].fields.prompt).toBe('b');
  });
});

describe('New angle (camera-angle presets)', () => {
  const fixed = { random: () => 0.5 };
  it('maps orbit + height to the server preset ids', () => {
    expect(cameraViewPreset(180, 'eye')).toBe('reverse');
    expect(cameraViewPreset(90, 'eye')).toBe('right-90');
    expect(cameraViewPreset(315, 'high')).toBe('left-45-high');
    expect(cameraViewPreset(225, 'low')).toBe('left-135-low');
    expect(cameraViewPreset(0, 'high')).toBe('front-high');
    expect(cameraViewPreset(0, 'eye')).toBe('');
    expect(snapAzimuth(-40)).toBe(315);
    expect(snapAzimuth(200)).toBe(180);
    expect(cameraViewLabel(180, 'high')).toBe('Reverse, high angle');
  });
  it('queues a retouch job with the view preset and names the output after it', () => {
    const s = settingsFor('angle', { cameraAzimuth: 90, cameraElevation: 'low', append: 'keep the scarf' });
    const f = buildJobs('angle', s, [IMG], fixed).jobs[0].fields;
    expect(f.preset).toBe('right-90-low');
    expect(f.append).toBe('keep the scarf');
    expect(f.prompt).toBeUndefined();
    expect(planFor('angle', s, [IMG]).outputName).toBe('photo_right-90-low.png');
    expect(runSummary('angle', s)).toMatch(/^90° right, low angle/);
  });
  it('defaults to reverse and refuses the original view', () => {
    expect(retouchPreset('angle', settingsFor('angle'))).toBe('reverse');
    const same = settingsFor('angle', { cameraAzimuth: 0, cameraElevation: 'eye' });
    expect(buildJobs('angle', same, [IMG], fixed).error).toMatch(/Move the camera/);
  });
  it('runs once per image, each as its own <image1>', () => {
    expect(buildJobs('angle', settingsFor('angle'), [IMG, IMG2], fixed).jobs[0].fields.combine).toBeUndefined();
    expect(usesReferences('angle', classify([IMG, IMG2]))).toBe(false);
  });
});
