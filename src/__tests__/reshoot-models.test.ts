import { parseReshootModels } from '../renderer/components/transform/reshoot-models';

/* GET /api/reshoot/models: the server's extra video models, checked as data. */
test('models from the server are checked; anything else means none', () => {
  expect(parseReshootModels({ models: [{ file: 'turbo.safetensors', label: 'Turbo', recipe: 'fast' }, { file: 'base.safetensors', recipe: 'weird' }, { label: 'no file' }, null] })).toEqual([
    { file: 'turbo.safetensors', label: 'Turbo', recipe: 'fast' },
    { file: 'base.safetensors', label: 'base.safetensors', recipe: 'h3' },
  ]);
  expect(parseReshootModels(null)).toEqual([]);
  expect(parseReshootModels({ models: 'nope' })).toEqual([]);
});
