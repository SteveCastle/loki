import { buildJobs, dreamStepsFor, dreamUnguided, planFor, runSummary, settingsFor } from '../renderer/components/transform/intents';
import { applyPromptGuide, dreamPromptGuide, DREAM_PROMPT_GUIDES, downloadGb, dreamRecipeOf, effectiveDreamModel, parseDreamCatalog, sameModelFile } from '../renderer/components/dream/dream-models';
import { matchesQuery } from '../renderer/components/dream/model-picker';

// The shape GET /api/dream/models answers with (loki-dream --list-models --json).
const BODY = {
  default: 'kromoss_int8_convrot.safetensors',
  where: 'this machine',
  fetched: '2026-10-31T12:00:00Z',
  models: [
    {
      file: 'kromoss_int8_convrot.safetensors',
      label: 'Kromoss',
      architecture: 'moss',
      architecture_name: 'MOSS-Image',
      distilled: false,
      steps: 50,
      cfg: 3,
      turbo: true,
      loras: true,
      local: true,
      format: 'int8',
      size_gb: 6.4,
    },
    {
      file: 'krea2_turbo_fp8_scaled.safetensors',
      label: 'Krea 2 Turbo',
      architecture: 'krea2',
      architecture_name: 'Krea 2',
      distilled: true,
      steps: 8,
      cfg: 1,
      turbo: false,
      loras: false,
      local: true,
      format: 'fp8',
      text_encoder: { file: 'qwen3vl_4b_bf16.safetensors', present: false, download_gb: 8.9 },
      vae: { file: 'qwen_image_vae.safetensors', present: true },
    },
    {
      file: 'anima-base.safetensors',
      label: 'Anima',
      architecture: 'anima',
      architecture_name: 'Anima',
      distilled: false,
      steps: 30,
      cfg: 4,
      turbo: false,
      loras: false,
      local: false,
      download_gb: 4.2,
    },
    { file: '', label: 'no file: dropped' },
    'junk',
  ],
};

const fixed = { random: () => 0.5 };
const gen = (over: Partial<ReturnType<typeof settingsFor>> = {}) => ({ ...settingsFor('generate'), prompt: 'a fox', ...over });

describe('dream model catalog', () => {
  const c = parseDreamCatalog(BODY);

  it('parses the server list, dropping malformed entries', () => {
    expect(c.models.map((m) => m.file)).toEqual(['kromoss_int8_convrot.safetensors', 'krea2_turbo_fp8_scaled.safetensors', 'anima-base.safetensors']);
    expect(c.defaultFile).toBe('kromoss_int8_convrot.safetensors');
    expect(c.models[1]).toMatchObject({ arch: 'krea2', archName: 'Krea 2', distilled: true, steps: 8, turbo: false });
    expect(parseDreamCatalog(null).models).toEqual([]);
  });

  it('resolves the default and matches file names loosely', () => {
    expect(effectiveDreamModel(c, '')?.label).toBe('Kromoss');
    expect(effectiveDreamModel(c, 'ANIMA-BASE')?.arch).toBe('anima');
    expect(sameModelFile('a.safetensors', 'A')).toBe(true);
    expect(sameModelFile('', '')).toBe(false);
  });

  it('adds up what a first run downloads', () => {
    expect(downloadGb(c.models[0])).toBe(0);
    expect(downloadGb(c.models[1])).toBe(8.9);
    expect(downloadGb(c.models[2])).toBe(4.2);
  });

  it('searches every term across name, file and architecture', () => {
    expect(matchesQuery(c.models[1], 'krea fp8')).toBe(true);
    expect(matchesQuery(c.models[1], 'krea nvfp4')).toBe(false);
    expect(matchesQuery(c.models[1], 'krea turbo')).toBe(true);
    expect(matchesQuery(c.models[0], 'moss')).toBe(true);
    expect(matchesQuery(c.models[2], 'flux')).toBe(false);
  });

  it('MOSS-Image keeps Kromoss quality table and Turbo', () => {
    const s = gen({ dreamModel: '', ...dreamRecipeOf(c.models[0]), quality: 'draft' });
    expect(buildJobs('generate', s, [], fixed).jobs[0].fields).toMatchObject({ turbo: '1' });
    expect(dreamStepsFor({ ...s, quality: 'fine' })).toBe(50);
  });

  it('a distilled model sends its file and neither turbo, steps nor negative', () => {
    const m = c.models[1];
    const s = gen({ dreamModel: m.file, dreamLabel: m.label, ...dreamRecipeOf(m), quality: 'fine', negative: 'blurry' });
    const f = buildJobs('generate', s, [], fixed).jobs[0].fields;
    expect(f.model).toBe('krea2_turbo_fp8_scaled.safetensors');
    expect(f.turbo).toBeUndefined();
    expect(f.steps).toBeUndefined();
    expect(f.negative).toBeUndefined();
    expect(dreamUnguided(s)).toBe(true);
    expect(planFor('generate', s, [])).toMatchObject({ steps: 8, turbo: true });
    expect(runSummary('generate', s)).toContain('Krea 2 Turbo');
  });

  it('a base model scales its own steps and keeps the negative prompt', () => {
    const m = c.models[2];
    const s = gen({ dreamModel: m.file, ...dreamRecipeOf(m), negative: 'blurry' });
    expect(dreamStepsFor({ ...s, quality: 'draft' })).toBe(15);
    expect(dreamStepsFor({ ...s, quality: 'balanced' })).toBe(30);
    expect(dreamStepsFor({ ...s, quality: 'fine' })).toBe(48);
    const f = buildJobs('generate', { ...s, quality: 'draft' }, [], fixed).jobs[0].fields;
    expect(f).toMatchObject({ model: 'anima-base.safetensors', steps: '15', negative: 'blurry' });
    expect(f.turbo).toBeUndefined();
  });

  it('the default model sends no model field', () => {
    expect(buildJobs('generate', gen(), [], fixed).jobs[0].fields.model).toBeUndefined();
  });

  it('fills the sample prompt and negative only over empty boxes or another model sample', () => {
    const anima = dreamPromptGuide('anima');
    expect(anima.prompt.startsWith('masterpiece, best quality, score_7, safe, ')).toBe(true);
    expect(anima.negative).toBe('worst quality, low quality, score_1, score_2, score_3, artist name');
    expect(dreamPromptGuide('unknown-arch')).toBe(DREAM_PROMPT_GUIDES.moss);
    // empty: both filled
    expect(applyPromptGuide({ prompt: '', negative: '' }, anima, true)).toEqual({ prompt: anima.prompt, negative: anima.negative });
    // another model's untouched samples: swapped
    const moss = DREAM_PROMPT_GUIDES.moss;
    expect(applyPromptGuide({ prompt: moss.prompt, negative: moss.negative }, anima, true)).toEqual({ prompt: anima.prompt, negative: anima.negative });
    // the user's own text: kept
    expect(applyPromptGuide({ prompt: 'my fox', negative: 'no hats' }, anima, true)).toEqual({});
    // Reimagine: the negative only
    expect(applyPromptGuide({ prompt: '', negative: '' }, anima, false)).toEqual({ negative: anima.negative });
    // already this model's: nothing to change
    expect(applyPromptGuide({ prompt: anima.prompt, negative: anima.negative }, anima, true)).toEqual({});
  });
});
