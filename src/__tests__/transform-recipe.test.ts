import { BASE_SETTINGS, buildJobs, settingsFor, type TransformSettings } from '../renderer/components/transform/intents';
import { carriedPins, nearestQuality, suggestTaskName, taskFromRecord } from '../renderer/components/transform/recipe';
import { loadTemplates, resetTemplateCache, saveTemplate, templateSettings } from '../renderer/components/transform/store';

/* A Transform result back into a saved task: the flow stamps its recipe into
 * every job (`meta`, through the engines' generation record), and the
 * Generation panel turns a record into an intent + settings for saveTemplate. */

const IMG = 'C:/media/photo.jpg';
const REF = 'C:/media/jacket.png';
const REF2 = 'C:/media/style.png';
const AUD = 'C:/media/song.mp3';

const recipeOf = (fields: Record<string, string>) => JSON.parse(fields.meta);

/** The record an engine stamps around a recipe, as generation-metadata.ts hands it over. */
const recordWith = (meta: unknown, tool = 'loki-retouch') => ({ loki: 1, tool, version: '0.1.0', args: ['--meta=x'], prompt: 'final', meta });

describe('the recipe stamped into each job', () => {
  test('only when asked: previews and older callers send what they always sent', () => {
    const s = { ...settingsFor('restore') };
    expect(buildJobs('restore', s, [IMG]).jobs[0].fields.meta).toBeUndefined();
    const meta = recipeOf(buildJobs('restore', s, [IMG], { recipe: true }).jobs[0].fields);
    expect(meta).toEqual({ kind: 'loki-transform', v: 1, intent: 'restore', settings: s, paths: [IMG] });
  });

  test('with pinned references each run carries the settings as set (pins included) and its own inputs', () => {
    const s: TransformSettings = { ...settingsFor('edit'), prompt: 'dress <image1> in <image2>', pinned: [REF] };
    const { jobs } = buildJobs('edit', s, [IMG, 'C:/media/b.jpg'], { recipe: true });
    expect(jobs).toHaveLength(2);
    expect(recipeOf(jobs[0].fields).settings.pinned).toEqual([REF]);
    expect(recipeOf(jobs[0].fields).paths).toEqual([IMG, REF]);
    expect(recipeOf(jobs[1].fields).paths).toEqual(['C:/media/b.jpg', REF]);
  });

  test('every variation and every Animate photo is stamped', () => {
    const alive = { ...settingsFor('alive'), describe: 'a dog', variations: 2 };
    const { jobs } = buildJobs('alive', alive, [IMG, REF], { recipe: true });
    expect(jobs).toHaveLength(4);
    expect(jobs.map((j) => recipeOf(j.fields).paths)).toEqual([[IMG], [IMG], [REF], [REF]]);
  });
});

describe('rebuilding a task from a stamped recipe', () => {
  test('a multi-image Edit: the other images become pins, so <image2> still finds the jacket', () => {
    const s: TransformSettings = { ...settingsFor('edit'), prompt: 'put the jacket of <image2> on <image1>', quality: 'fine', seedMode: 'fixed', seed: 42 };
    const meta = recipeOf(buildJobs('edit', s, [IMG, REF], { recipe: true }).jobs[0].fields);
    const task = taskFromRecord(recordWith(meta));
    expect(task).toMatchObject({ intent: 'edit', approximate: false });
    expect(task!.settings).toEqual({ ...s, pinned: [REF] });
    // On a new image it runs as one job, new image first, the jacket after it.
    const rerun = buildJobs('edit', task!.settings, ['C:/media/new.jpg']);
    expect(rerun.jobs[0].input).toBe(`retouch "C:/media/new.jpg\n${REF}"`);
    expect(rerun.jobs[0].fields.prompt).toBe('put the jacket of <image2> on <image1>');
  });

  test('a custom prompt rewritten whole in Review is saved as the prompt', () => {
    const meta = { kind: 'loki-transform', v: 1, intent: 'direct', settings: { ...settingsFor('direct'), prompt: 'draft', manualPrompt: 'She dances to <Audio 1>' }, paths: [IMG, AUD] };
    const task = taskFromRecord(recordWith(meta, 'loki-reshoot'))!;
    expect(task.settings.prompt).toBe('She dances to <Audio 1>');
    expect(task.settings.manualPrompt).toBeNull();
    expect(task.settings.pinned).toEqual([AUD]);
  });

  test("a preset keeps its hand-written prompt and settings; it takes no pins", () => {
    const s = { ...settingsFor('upscale'), scale: 3, manualPrompt: 'upscale, keep the grain', pinned: [REF] };
    const task = taskFromRecord(recordWith({ kind: 'loki-transform', v: 1, intent: 'upscale', settings: s, paths: [IMG, REF2] }))!;
    expect(task.intent).toBe('upscale');
    expect(task.settings.scale).toBe(3);
    expect(task.settings.manualPrompt).toBe('upscale, keep the grain');
    expect(task.settings.pinned).toEqual([]);
  });

  test('settings from the file are checked: unknown keys and wrong types are dropped', () => {
    const task = taskFromRecord(recordWith({ kind: 'loki-transform', v: 1, intent: 'restore', settings: { quality: 'fine', scale: 'big', evil: 1, pinned: [3, REF] } }))!;
    expect(task.settings.quality).toBe('fine');
    expect(task.settings.scale).toBe(BASE_SETTINGS.scale);
    expect(task.settings).not.toHaveProperty('evil');
  });

  test('the video model is restored from a recipe; an unknown one falls back to H3', () => {
    const rec = (reshootModel: unknown) => taskFromRecord(recordWith({ kind: 'loki-transform', v: 1, intent: 'alive', settings: { describe: 'a dog', reshootModel } }, 'loki-reshoot'))!;
    expect(rec('fast').settings.reshootModel).toBe('fast');
    expect(rec('turbo').settings.reshootModel).toBe('h3');
    expect(rec(undefined).settings.reshootModel).toBe('h3');
  });

  test('a recipe for an intent this version does not know falls back to the engine fields', () => {
    const rec = { ...recordWith({ kind: 'loki-transform', v: 9, intent: 'teleport', settings: {} }), args: ['--preset', 'restore', 'in.png'], preset: 'restore', steps: 25 };
    expect(taskFromRecord(rec)).toMatchObject({ intent: 'restore', approximate: true });
  });
});

describe('carried pins', () => {
  test('everything after the first input; a Direct run of nothing but pins keeps them all', () => {
    expect(carriedPins('edit', [IMG, REF, REF2])).toEqual([REF, REF2]);
    expect(carriedPins('edit', [IMG, AUD])).toEqual([]); // Edit pins images only
    expect(carriedPins('direct', [REF, AUD], [REF, AUD])).toEqual([REF, AUD]);
    expect(carriedPins('direct', ['-', REF])).toEqual([]);
  });
});

describe('rebuilding from an older record (no recipe)', () => {
  const retouch = (over: Record<string, unknown>) => ({ loki: 1, tool: 'loki-retouch', version: '0.1.0', steps: 25, seed: 7, width: 2048, height: 1536, inputs: [{ tag: '<image1>', path: IMG, width: 1024, height: 768 }], ...over });

  test('an edit with a reference', () => {
    const task = taskFromRecord(retouch({ args: ['--prompt=dress <image1> in <image2>', '--steps', '40', '--ref', REF, '-o', 'o.png', IMG], prompt: 'dress <image1> in <image2>', steps: 40, inputs: [{ tag: '<image1>', path: IMG }, { tag: '<image2>', path: REF }] }))!;
    expect(task).toMatchObject({ intent: 'edit', approximate: true });
    expect(task.settings).toMatchObject({ prompt: 'dress <image1> in <image2>', quality: 'fine', seedMode: 'random', seed: 7, pinned: [REF] });
  });

  test('presets: upscale factor from the sizes, a replaced prompt, the phone wallpaper, an appended direction', () => {
    expect(taskFromRecord(retouch({ args: ['--preset', 'upscale', IMG], preset: 'upscale' }))!.settings).toMatchObject({ sizeMode: 'scale', scale: 2 });
    const up = taskFromRecord(retouch({ args: ['--preset', 'upscale', '--scale', '3', IMG], preset: 'upscale' }))!;
    expect(up.settings.scale).toBe(3);
    const manual = taskFromRecord(retouch({ args: ['--preset', 'restore', '--prompt=restore gently', IMG], preset: 'restore', prompt: 'restore gently' }))!;
    expect(manual.settings.manualPrompt).toBe('restore gently');
    expect(taskFromRecord(retouch({ args: ['--preset', '4kify-phone', IMG], preset: '4kify-phone' }))!.settings.wallpaperTarget).toBe('phone');
    const appended = taskFromRecord(retouch({ args: ['--preset', 'restore', '--append=keep the film grain', IMG], preset: 'restore' }))!;
    expect(appended.settings).toMatchObject({ append: 'keep the film grain', manualPrompt: null });
  });

  test('reshoot: Animate and Direct', () => {
    const base = { loki: 1, tool: 'loki-reshoot', steps: 32, seed: 3, seconds: 5.208, audio: true, ref_image_size: 'max' };
    const alive = taskFromRecord({ ...base, animate: true, describe: 'a dog on a porch', shake: 'handheld', prompt: '<long expanded prompt>', args: ['--animate', IMG, '--describe=a dog on a porch', '--prompt=it yawns', '-d', '5'], inputs: [{ tag: '<Picture 1>', path: IMG }] })!;
    expect(alive).toMatchObject({ intent: 'alive', approximate: true });
    expect(alive.settings).toMatchObject({ describe: 'a dog on a porch', shake: 'handheld', prompt: 'it yawns', duration: 5, quality: 'fine', refSize: 'max', pinned: [] });
    const direct = taskFromRecord({ ...base, audio: false, prompt: 'She dances to <Audio 1>', args: ['--prompt=She dances to <Audio 1>', '--no-video-audio', '-i', IMG, '-a', AUD], inputs: [{ tag: '<Picture 1>', path: IMG }, { tag: '<Audio 1>', path: AUD }] })!;
    expect(direct.settings).toMatchObject({ prompt: 'She dances to <Audio 1>', duration: 5.25, noAudio: true, keepSoundtrack: false, pinned: [AUD] });
  });

  test('not a Transform-shaped run: nothing to save', () => {
    expect(taskFromRecord(null)).toBeNull();
    expect(taskFromRecord({ loki: 1, tool: 'loki-something' })).toBeNull();
    expect(taskFromRecord(retouch({ args: [IMG], prompt: '' }))).toBeNull(); // an edit needs its prompt
  });

  test('reshoot on the fast model: from model_id or --model, without a quality guessed from its 8 steps', () => {
    const base = { loki: 1, tool: 'loki-reshoot', steps: 8, seed: 3, seconds: 5.208, audio: true, prompt: 'She dances', inputs: [{ tag: '<Picture 1>', path: IMG }] };
    const byId = taskFromRecord({ ...base, model: 'FastH3 Omni', model_id: 'fast', args: ['--model', 'fast', '--prompt=She dances', '-i', IMG] })!;
    expect(byId.settings).toMatchObject({ reshootModel: 'fast', quality: 'balanced' });
    expect(buildJobs('direct', byId.settings, [IMG]).jobs[0].fields).toMatchObject({ model: 'fast', steps: '8' });
    expect(taskFromRecord({ ...base, args: ['--model=fast', '--prompt=She dances', '-i', IMG] })!.settings.reshootModel).toBe('fast');
    // An H3 record (or one from before the model field): H3, quality from the steps as before.
    const h3 = taskFromRecord({ ...base, steps: 12, model: 'MiniMax H3', model_id: 'h3', args: ['--prompt=She dances', '-i', IMG] })!;
    expect(h3.settings).toMatchObject({ reshootModel: 'h3', quality: 'draft' });
  });

  test('quality from steps', () => {
    expect(nearestQuality('retouch', 12)).toBe('draft');
    expect(nearestQuality('reshoot', 20)).toBe('balanced');
    expect(nearestQuality('retouch', undefined)).toBe('balanced');
  });
});

describe('saving the rebuilt task', () => {
  beforeEach(() => {
    window.localStorage.clear();
    resetTemplateCache();
  });

  test('a preset saves without a prompt and comes back with its hand-written prompt', () => {
    const s = { ...settingsFor('restore'), manualPrompt: 'restore gently', quality: 'fine' as const };
    const task = { intent: 'restore' as const, settings: s, approximate: false };
    expect(suggestTaskName(task)).toBe('Restore · restore gently');
    const t = saveTemplate('restore', suggestTaskName(task), '', s);
    expect(t).not.toBeNull();
    expect(loadTemplates('restore')).toHaveLength(1);
    expect(templateSettings('restore', t!)).toMatchObject({ manualPrompt: 'restore gently', quality: 'fine', prompt: '' });
  });

  test('a custom intent still needs its prompt, and never keeps a one-off review rewrite', () => {
    expect(saveTemplate('edit', 'Empty', '', settingsFor('edit'))).toBeNull();
    const t = saveTemplate('edit', 'Night', 'make it night', { ...settingsFor('edit'), manualPrompt: 'something else' })!;
    expect(t.settings).not.toHaveProperty('manualPrompt');
    expect(templateSettings('edit', t).manualPrompt).toBeNull();
  });

  test('names: a custom prompt by its first words, presets by what they do', () => {
    expect(suggestTaskName({ intent: 'edit', settings: { ...settingsFor('edit'), prompt: 'Turn it into a snowy winter day, keep the people' }, approximate: false })).toBe('Turn it into a snowy winter…');
    expect(suggestTaskName({ intent: 'upscale', settings: { ...settingsFor('upscale'), scale: 4 }, approximate: false })).toBe('Upscale 4×');
    expect(suggestTaskName({ intent: 'angle', settings: { ...settingsFor('angle'), cameraAzimuth: 180, cameraElevation: 'high' }, approximate: false })).toBe('Reverse, high angle');
  });
});

describe('loki-dream results back into tasks', () => {
  test('a stamped Generate recipe comes back exactly, with no inputs to carry', () => {
    const s: TransformSettings = { ...settingsFor('generate'), prompt: 'a red fox', aspect: 'portrait', quality: 'draft', variations: 3 };
    const meta = recipeOf(buildJobs('generate', s, [IMG], { recipe: true }).jobs[0].fields);
    expect(meta.paths).toEqual([]);
    const task = taskFromRecord(recordWith(meta, 'loki-dream'));
    expect(task).toMatchObject({ intent: 'generate', approximate: false });
    expect(task?.settings).toMatchObject({ prompt: 'a red fox', aspect: 'portrait', quality: 'draft', variations: 3, pinned: [] });
    expect(suggestTaskName(task!)).toBe('a red fox');
  });

  test('an unknown aspect in a recipe falls back to the default', () => {
    const meta = { kind: 'loki-transform', v: 1, intent: 'generate', settings: { prompt: 'x', aspect: 'triangle' }, paths: [] };
    expect(taskFromRecord(recordWith(meta, 'loki-dream'))?.settings.aspect).toBe('square');
  });

  test('a hand-run loki-dream is rebuilt from its command line', () => {
    const gen = taskFromRecord({ loki: 1, tool: 'loki-dream', prompt: 'a castle', seed: 9, args: ['--prompt', 'a castle', '--size', '1344x768', '--steps', '48', '--negative', 'blurry', '-o', 'out'] });
    expect(gen).toMatchObject({ intent: 'generate', approximate: true });
    expect(gen?.settings).toMatchObject({ prompt: 'a castle', aspect: 'landscape', quality: 'fine', negative: 'blurry', seed: 9 });
    const odd = taskFromRecord({ loki: 1, tool: 'loki-dream', prompt: 'a castle', args: ['--size', '1000x600', '--turbo'] });
    expect(odd?.settings).toMatchObject({ aspect: 'custom', customSize: '1000x600', quality: 'draft' });
    const re = taskFromRecord({ loki: 1, tool: 'loki-dream', prompt: 'oil painting', args: ['--init', IMG, '--strength', '0.4', '--steps', '30'] });
    expect(re?.intent).toBe('reimagine');
    expect(re?.settings).toMatchObject({ strength: 0.4, quality: 'balanced' });
    // the Scene editor's inpainting is not a Transform task
    expect(taskFromRecord({ loki: 1, tool: 'loki-dream', prompt: 'x', args: ['--init', IMG, '--mask', 'm.png'] })).toBeNull();
  });

  test('nearestQuality knows the dream step table', () => {
    expect(nearestQuality('dream', 8)).toBe('draft');
    expect(nearestQuality('dream', 28)).toBe('balanced');
    expect(nearestQuality('dream', 50)).toBe('fine');
  });
});
