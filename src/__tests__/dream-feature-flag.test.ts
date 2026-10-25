// Builds without LOKI_DREAM=1 offer no loki-dream (Kromoss) features: no
// Generate / Reimagine intents, no runs of them, no tasks rebuilt from
// loki-dream results.

type Intents = typeof import('../renderer/components/transform/intents');
type Recipe = typeof import('../renderer/components/transform/recipe');

function load(flag: string | undefined): { intents: Intents; recipe: Recipe } {
  const saved = process.env.LOKI_DREAM;
  if (flag === undefined) delete process.env.LOKI_DREAM;
  else process.env.LOKI_DREAM = flag;
  let out!: { intents: Intents; recipe: Recipe };
  jest.isolateModules(() => {
    out = {
      // eslint-disable-next-line global-require
      intents: require('../renderer/components/transform/intents'),
      // eslint-disable-next-line global-require
      recipe: require('../renderer/components/transform/recipe'),
    };
  });
  process.env.LOKI_DREAM = saved;
  return out;
}

const IMG = 'C:/media/photo.jpg';

describe('without LOKI_DREAM', () => {
  const { intents, recipe } = load(undefined);

  test('Generate and Reimagine are not offered', () => {
    const ids = intents.INTENTS.map((i) => i.id);
    expect(ids).not.toContain('generate');
    expect(ids).not.toContain('reimagine');
    expect(ids).toContain('edit');
    expect(ids).toContain('alive');
    expect(intents.INTENTS.some((i) => i.engine === 'dream')).toBe(false);
  });

  test('a remembered Generate run resolves but cannot run', () => {
    expect(intents.intentById('generate').engine).toBe('dream');
    expect(intents.availability('generate', intents.classify([])).ok).toBe(false);
    expect(intents.availability('reimagine', intents.classify([IMG])).ok).toBe(false);
    expect(intents.availability('edit', intents.classify([IMG])).ok).toBe(true);
  });

  test('loki-dream results are not rebuilt into tasks', () => {
    expect(recipe.taskFromRecord({ loki: 1, tool: 'loki-dream', prompt: 'a castle', args: ['--prompt', 'a castle'] })).toBeNull();
  });
});

describe('with LOKI_DREAM=1', () => {
  const { intents } = load('1');

  test('Generate and Reimagine are offered', () => {
    const ids = intents.INTENTS.map((i) => i.id);
    expect(ids).toContain('generate');
    expect(ids).toContain('reimagine');
    expect(intents.availability('generate', intents.classify([])).ok).toBe(true);
  });
});
