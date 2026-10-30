import { pickEngine } from '../renderer/components/scene/panel';
import type { ReshootModelChoice } from '../renderer/components/transform/reshoot-models';

// A motion layer's Engine control: H3 / Fast clear the extra model, an extra model sets its file and its recipe.
describe('scene motion engine choice', () => {
  const extra: ReshootModelChoice[] = [{ file: 'extra-model.safetensors', label: 'Extra', recipe: 'fast' }];

  it('an extra model sends its file with its recipe', () => {
    expect(pickEngine('file:extra-model.safetensors', extra)).toEqual({ model: 'fast', dit: 'extra-model.safetensors' });
  });

  it('H3 and Fast clear a chosen extra model', () => {
    expect(pickEngine('h3', extra)).toEqual({ model: 'h3', dit: '' });
    expect(pickEngine('fast', extra)).toEqual({ model: 'fast', dit: '' });
  });

  it('a file the server no longer lists is kept (its generation says so)', () => {
    expect(pickEngine('file:gone.safetensors', extra)).toEqual({ dit: 'gone.safetensors' });
  });
});
