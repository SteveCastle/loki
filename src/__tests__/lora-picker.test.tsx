import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import LoraPicker from '../renderer/components/dream/lora-picker';
import { loraFits, lorasField, parseLoraList, type LoraList } from '../renderer/components/dream/loras';
import { buildJobs, runSummary, settingsFor } from '../renderer/components/transform/intents';
import { taskFromRecord } from '../renderer/components/transform/recipe';

const BODY = {
  loras: [
    { path: 'C:\\data\\loras\\zkx\\zkx.safetensors', file: 'zkx.safetensors', name: 'zkx', arch: 'moss', trigger: 'zkx', base_model: 'kromoss_bf16.safetensors', rank: 16, step: 1000, steps: 1000, run: 'zkx', size_mb: 94, modified: '2026-11-24T18:00:00Z' },
    { path: 'C:\\data\\loras\\zkx\\checkpoints\\step-000500\\zkx-step000500.safetensors', file: 'zkx-step000500.safetensors', name: 'zkx', arch: 'moss', trigger: 'zkx', step: 500, steps: 1000, checkpoint: true, run: 'zkx', size_mb: 94, modified: '2026-11-24T17:40:00Z' },
    { path: 'C:\\models\\loras\\inkwash.safetensors', file: 'inkwash.safetensors', name: 'inkwash', arch: 'moss', size_mb: 70, modified: '2026-11-20T10:00:00Z' },
    { path: 'C:\\models\\loras\\sdxl_detail.safetensors', file: 'sdxl_detail.safetensors', name: 'sdxl_detail', arch: '', size_mb: 50, modified: '2026-11-01T10:00:00Z' },
    { file: 'no path' },
  ],
  dirs: ['C:\\data\\loras', 'C:\\models\\loras'],
};
const LIST: LoraList = { ...parseLoraList(BODY), loaded: true, loading: false };
const KROMOSS = { file: 'kromoss_bf16.safetensors', label: 'Kromoss', arch: 'moss', archName: 'MOSS-Image', distilled: false, steps: 50, cfg: 3, turbo: true, local: true };
const FINETUNE = { ...KROMOSS, file: 'my_moss_finetune.safetensors', label: 'My fine-tune' };
const KREA = { ...KROMOSS, file: 'krea2_turbo.safetensors', label: 'Krea 2', arch: 'krea2', archName: 'Krea 2', turbo: false };

describe('LoRAs: list, fit and job field', () => {
  it('parses the server list, skipping entries without a path', () => {
    expect(LIST.loras.map((l) => l.name)).toEqual(['zkx', 'zkx', 'inkwash', 'sdxl_detail']);
    expect(LIST.loras[1].checkpoint).toBe(true);
    expect(LIST.loras[0].baseModel).toBe('kromoss_bf16.safetensors');
  });

  it('fits MOSS-Image LoRAs to MOSS-Image models only, noting another base model', () => {
    expect(loraFits(LIST.loras[0], KROMOSS).ok).toBe(true);
    expect(loraFits(LIST.loras[0], undefined).ok).toBe(true);
    expect(loraFits(LIST.loras[0], FINETUNE)).toEqual({ ok: true, note: 'trained on kromoss_bf16' });
    expect(loraFits(LIST.loras[0], KREA).ok).toBe(false);
    expect(loraFits(LIST.loras[3], KROMOSS)).toEqual({ ok: false, why: 'not a LoRA type the engine can apply' });
  });

  it('writes one path:scale per line, the bare path at full strength, dropping zero', () => {
    expect(
      lorasField([
        { path: 'C:\\a.safetensors', scale: 1 },
        { path: 'C:\\b.safetensors', scale: 0.75 },
        { path: 'C:\\c.safetensors', scale: 0 },
      ])
    ).toBe('C:\\a.safetensors\nC:\\b.safetensors:0.75');
  });

  it('Generate sends the LoRAs to a MOSS-Image model, and none to another architecture', () => {
    const fixed = { random: () => 0.5 };
    const s = { ...settingsFor('generate'), prompt: 'zkx, a man', dreamLoras: [{ path: 'C:\\zkx.safetensors', scale: 0.8, name: 'zkx' }] };
    expect(buildJobs('generate', s, [], fixed).jobs[0].fields.loras).toBe('C:\\zkx.safetensors:0.8');
    expect(runSummary('generate', s)).toContain('LoRA zkx');
    const other = { ...s, dreamModel: 'krea2_turbo.safetensors', dreamTurbo: false, dreamDistilled: true };
    expect(buildJobs('generate', other, [], fixed).jobs[0].fields.loras).toBeUndefined();
    expect(runSummary('generate', other)).not.toContain('LoRA');
  });

  it('a result made with LoRAs rebuilds with them (not the turbo LoRA)', () => {
    const task = taskFromRecord({
      loki: 1,
      tool: 'loki-dream',
      prompt: 'zkx, a man',
      steps: 8,
      args: ['--turbo', '-p', 'zkx, a man', '--lora', 'C:\\l\\zkx.safetensors:0.8'],
      loras: [
        { path: 'C:\\models\\8-steps-dmd2-lora.safetensors', strength: 1 },
        { path: 'C:\\l\\zkx.safetensors', strength: 0.8 },
      ],
    });
    expect(task?.settings.dreamLoras).toEqual([{ path: 'C:\\l\\zkx.safetensors', scale: 0.8, name: 'zkx' }]);
  });
});

describe('LoRAs: the picker', () => {
  const setup = (value = [] as { path: string; scale: number; name?: string; trigger?: string }[], model: any = KROMOSS, prompt = '') => {
    const onChange = jest.fn();
    const onTrigger = jest.fn();
    render(<LoraPicker list={LIST} model={model} value={value} onChange={onChange} onRefresh={jest.fn()} prompt={prompt} onTrigger={onTrigger} />);
    return { onChange, onTrigger };
  };

  it('offers the fitting LoRAs (checkpoints on request), several at a time', () => {
    const { onChange } = setup();
    expect(screen.getByText('3 fit this model')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Add a LoRA' }));
    const options = () => screen.getAllByRole('option');
    expect(options()).toHaveLength(3); // the checkpoint is hidden
    expect((options()[2] as HTMLButtonElement).disabled).toBe(true); // the SDXL LoRA
    fireEvent.click(screen.getByLabelText('Checkpoints'));
    expect(options()).toHaveLength(4);
    fireEvent.click(options()[0]);
    expect(onChange).toHaveBeenLastCalledWith([{ path: LIST.loras[0].path, scale: 1, name: 'zkx', trigger: 'zkx' }]);
    fireEvent.change(screen.getByLabelText('Search LoRAs'), { target: { value: 'ink' } });
    expect(options()).toHaveLength(1);
  });

  it('chosen LoRAs: strength, the trigger word for the prompt, removal', () => {
    const chosen = [
      { path: LIST.loras[0].path, scale: 1, name: 'zkx', trigger: 'zkx' },
      { path: LIST.loras[2].path, scale: 0.6, name: 'inkwash' },
    ];
    const { onChange, onTrigger } = setup(chosen, KROMOSS, 'a man on a bench');
    fireEvent.change(screen.getByLabelText('zkx strength'), { target: { value: '0.7' } });
    expect(onChange).toHaveBeenLastCalledWith([{ ...chosen[0], scale: 0.7 }, chosen[1]]);
    fireEvent.click(screen.getByRole('button', { name: '+ zkx' }));
    expect(onTrigger).toHaveBeenCalledWith('zkx');
    fireEvent.click(screen.getByRole('button', { name: 'Remove inkwash' }));
    expect(onChange).toHaveBeenLastCalledWith([chosen[0]]);
  });

  it('keeps chosen LoRAs greyed when the model takes none, and hides a trigger already in the prompt', () => {
    setup([{ path: LIST.loras[0].path, scale: 1, name: 'zkx', trigger: 'zkx' }], KREA, 'zkx, a man');
    expect(screen.getByText(/not used: Krea 2 models take no LoRAs here/)).toBeTruthy();
    expect(screen.queryByRole('button', { name: '+ zkx' })).toBeNull();
    expect((screen.getByLabelText('zkx strength') as HTMLInputElement).disabled).toBe(true);
  });
});
