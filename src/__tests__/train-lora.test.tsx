import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// jsdom has no AbortSignal.timeout (the job POST passes one as its signal).
if (typeof (AbortSignal as any).timeout !== 'function') {
  (AbortSignal as any).timeout = () => new AbortController().signal;
}

import { TrainLoraDialog } from '../renderer/components/train-lora/train-lora-dialog';
import { buildTrainLoraJob, defaultSteps, defaultTrainModel, DEFAULT_SETTINGS, estimateMinutes, isTrainImage, runName, trainableModels } from '../renderer/components/train-lora/plan';
import { resetDreamModelsForTests } from '../renderer/components/dream/dream-models';
import { branchesFrom, buildContinueJob, continueDefaults, parseLoraRuns } from '../renderer/components/train-lora/runs';

const RUNS = {
  dir: 'C:\data\loras',
  runs: [
    {
      name: 'inkwash', dir: 'C:\data\loras\inkwash', kind: 'style', trigger: 'inkwash', model: 'kromoss_bf16.safetensors',
      rank: 16, resolution: 1024, lr: 5e-5, steps: 1000, images: 35, step: 1000, loss: 0.405, finished: true, modified: '2026-11-24T13:00:00Z',
      checkpoints: [
        { step: 500, dir: 'C:\data\loras\inkwash\checkpoints\step-000500', samples: ['C:\s\step-000500_0.png', 'C:\s\step-000500_1.png'] },
        { step: 1000, dir: 'C:\data\loras\inkwash\checkpoints\step-001000', samples: ['C:\s\step-001000_0.png', 'C:\s\step-001000_1.png'] },
      ],
    },
    { name: 'empty', dir: 'C:\data\loras\empty', steps: 1000, images: 3, step: 0, finished: false, checkpoints: [], modified: '' },
  ],
};

const CATALOG = {
  default: 'krea2_turbo_int8_convrot.safetensors',
  models: [
    { file: 'kromoss_int8_convrot.safetensors', label: 'Kromoss int8', architecture: 'moss', architecture_name: 'MOSS-Image', local: true },
    { file: 'kromoss_bf16.safetensors', label: 'Kromoss bf16', architecture: 'moss', architecture_name: 'MOSS-Image', local: true },
    { file: 'my_moss_finetune.safetensors', label: 'My MOSS fine-tune', architecture: 'moss', architecture_name: 'MOSS-Image', local: true },
    { file: 'krea2_turbo_int8_convrot.safetensors', label: 'Krea 2 Turbo', architecture: 'krea2', architecture_name: 'Krea 2', local: true },
    { file: 'moss_missing.safetensors', label: 'Not here', architecture: 'moss', architecture_name: 'MOSS-Image', local: false },
  ],
};

const IMGS = ['C:/set/a.jpg', 'C:/set/b.png', 'C:/set/c.webp'];

describe('Train LoRA: the job', () => {
  it('sends the images as the target and the settings as the fields the server pins (TestUIContractTrainLora)', () => {
    const job = buildTrainLoraJob([...IMGS, 'C:/set/clip.mp4'], {
      ...DEFAULT_SETTINGS,
      name: 'Old Man',
      trigger: 'zkx',
      kind: 'character',
      className: 'man',
      steps: 1500,
      rank: 32,
      resolution: 768,
    });
    expect(job.input).toBe(`train-lora "${IMGS.join('\n')}"`);
    expect(job.fields).toEqual({ name: 'Old Man', trigger: 'zkx', kind: 'character', class: 'man', captions: 'auto', steps: '1500', rank: '32', resolution: '768' });
  });

  it('leaves out empty settings, the class of kinds without one, and auto steps', () => {
    const job = buildTrainLoraJob(IMGS, { ...DEFAULT_SETTINGS, kind: 'style', className: 'ignored', captions: 'library' });
    expect(job.fields).toEqual({ kind: 'style', captions: 'library', rank: '16', resolution: '1024' });
  });

  it('trains on a library query (the last token) when the palette is widened to the library', () => {
    const job = buildTrainLoraJob(IMGS, { ...DEFAULT_SETTINGS, trigger: 'zkx' }, 'cXVlcnk=');
    expect(job.input).toBe('train-lora --query64=cXVlcnk=');
    expect(job.fields.trigger).toBe('zkx');
  });

  it('offers the MOSS-Image checkpoints on the server, defaulting to Kromoss bf16, and sends the pick', () => {
    const ms = trainableModels(CATALOG.models.map((m) => ({ file: m.file, arch: m.architecture, local: m.local })));
    expect(ms.map((m) => m.file)).toEqual(['kromoss_int8_convrot.safetensors', 'kromoss_bf16.safetensors', 'my_moss_finetune.safetensors']);
    expect(defaultTrainModel(ms)).toBe('kromoss_bf16.safetensors');
    expect(defaultTrainModel(ms.filter((m) => !m.file.includes('bf16')))).toBe('kromoss_int8_convrot.safetensors');
    expect(buildTrainLoraJob(IMGS, { ...DEFAULT_SETTINGS, model: 'my_moss_finetune.safetensors' }).fields.model).toBe('my_moss_finetune.safetensors');
    expect(buildTrainLoraJob(IMGS, DEFAULT_SETTINGS).fields.model).toBeUndefined();
  });

  it('continues a run: the style at 1e-4 to 2500 steps, branching from an earlier checkpoint', () => {
    const [run] = parseLoraRuns(RUNS);
    const d = continueDefaults(run);
    expect(d).toEqual({ checkpoint: run.checkpoints[1].dir, steps: 2500, lr: 1e-4, saveEvery: 0 });
    expect(branchesFrom(run, run.checkpoints[1])).toBeNull();
    expect(branchesFrom(run, run.checkpoints[0])).toBe('inkwash-from500');
    expect(buildContinueJob(d)).toEqual({ input: 'train-lora', fields: { resume: run.checkpoints[1].dir, steps: '2500', lr: '0.0001' } });
    expect(buildContinueJob({ ...d, lr: 0 }).fields.lr).toBeUndefined();
  });

  it('plans like loki-tune', () => {
    expect(defaultSteps(10)).toBe(1000);
    expect(defaultSteps(40)).toBe(2000);
    expect(defaultSteps(400)).toBe(3000);
    expect(defaultSteps(35, true)).toBe(2450);
    expect(defaultSteps(10, true)).toBe(2000);
    expect(runName({ name: '', trigger: 'My Style!' })).toBe('My_Style');
    expect(runName({ name: '', trigger: '' })).toBe('lora');
    expect(isTrainImage('x.JPEG')).toBe(true);
    expect(isTrainImage('x.mp4')).toBe(false);
    expect(estimateMinutes(20, { steps: 0, resolution: 512 })).toBeLessThan(estimateMinutes(20, { steps: 0, resolution: 1024 }));
  });
});

describe('Train LoRA: the dialog', () => {
  let fetchMock: jest.Mock;
  let createOk = true;
  const createCall = () => fetchMock.mock.calls.find(([url]) => String(url).endsWith('/create'));
  beforeEach(() => {
    window.localStorage.clear();
    resetDreamModelsForTests();
    createOk = true;
    fetchMock = jest.fn(async (url: string) =>
      String(url).includes('/api/dream/models')
        ? { ok: true, status: 200, json: async () => CATALOG }
        : String(url).includes('/api/lora-runs')
        ? { ok: true, status: 200, json: async () => RUNS }
        : createOk
        ? { ok: true, status: 200, json: async () => ({ id: 'job-1' }) }
        : { ok: false, status: 500, json: async () => ({}) }
    );
    (global as any).fetch = fetchMock;
  });

  it('needs a trigger for a character, then submits and closes', async () => {
    const onClose = jest.fn();
    render(<TrainLoraDialog req={{ paths: [...IMGS, 'C:/set/notes.txt'], authToken: 'tok' }} onClose={onClose} />);
    expect(screen.getByText(/3 images \(1 other file skipped\)/)).toBeTruthy();
    const start = screen.getByRole('button', { name: 'Start training' }) as HTMLButtonElement;
    expect(start.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText('Trigger word'), { target: { value: 'zkx' } });
    fireEvent.change(screen.getByLabelText('What the trigger is'), { target: { value: 'man' } });
    expect(start.disabled).toBe(false);
    fireEvent.click(start);
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    const [url, init] = createCall()!;
    expect(url).toMatch(/\/create$/);
    expect(init.headers.Authorization).toBe('Bearer tok');
    const body = JSON.parse(init.body);
    expect(body.input).toBe(`train-lora "${IMGS.join('\n')}"`);
    expect(body.fields).toMatchObject({ trigger: 'zkx', class: 'man', kind: 'character', captions: 'auto' });
    expect(body.fields.steps).toBeUndefined();
  });

  it('a style needs no trigger, hides the class, and remembers the kind', async () => {
    const onClose = jest.fn();
    const { unmount } = render(<TrainLoraDialog req={{ paths: IMGS }} onClose={onClose} />);
    fireEvent.click(screen.getByRole('radio', { name: 'Style' }));
    expect(screen.queryByLabelText('What the trigger is')).toBeNull();
    fireEvent.click(screen.getByRole('radio', { name: 'Steps' }));
    fireEvent.change(screen.getByLabelText('Steps'), { target: { value: '1750' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start training' }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(JSON.parse(createCall()![1].body).fields).toMatchObject({ kind: 'style', steps: '1750' });
    unmount();
    render(<TrainLoraDialog req={{ paths: IMGS }} onClose={jest.fn()} />);
    expect(screen.getByRole('radio', { name: 'Style' }).getAttribute('aria-checked')).toBe('true');
  });

  it('a library query shows its size and submits the query', async () => {
    const onClose = jest.fn();
    render(<TrainLoraDialog req={{ paths: IMGS, query64: 'cXVlcnk=', count: 240, label: 'Entire library' }} onClose={onClose} />);
    expect(screen.getByText(/240 images in Entire library/)).toBeTruthy();
    expect(screen.getByText(/Auto \(3000 steps\)/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Trigger word'), { target: { value: 'zkx' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start training' }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(JSON.parse(createCall()![1].body).input).toBe('train-lora --query64=cXVlcnk=');
  });

  it('a query of unknown size (a tag) still starts', async () => {
    render(<TrainLoraDialog req={{ paths: [], query64: 'dGFn', count: null, label: 'tag: portrait' }} onClose={jest.fn()} />);
    expect(screen.getByText(/All images in tag: portrait/)).toBeTruthy();
    expect(screen.getByText(/Auto \(~50 passes\)/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Trigger word'), { target: { value: 'zkx' } });
    expect((screen.getByRole('button', { name: 'Start training' }) as HTMLButtonElement).disabled).toBe(false);
  });

  it('continues a run from a checkpoint (opened without images)', async () => {
    const onClose = jest.fn();
    render(<TrainLoraDialog req={{ paths: [], authToken: 'tok', mode: 'continue' }} onClose={onClose} />);
    await waitFor(() => expect(screen.getByRole('option', { name: /inkwash/ })).toBeTruthy());
    // the run without checkpoints is not offered
    expect(screen.queryByRole('option', { name: /empty/ })).toBeNull();
    expect((screen.getByLabelText('Train until step') as HTMLInputElement).value).toBe('2500');
    expect(screen.getAllByRole('img')).toHaveLength(2);
    fireEvent.click(screen.getByRole('radio', { name: '500' }));
    expect(screen.getByText(/continues in a new run, inkwash-from500/)).toBeTruthy();
    fireEvent.click(screen.getByRole('radio', { name: '1000' }));
    fireEvent.change(screen.getByLabelText('Train until step'), { target: { value: '3000' } });
    fireEvent.click(screen.getByRole('button', { name: 'Continue training' }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(JSON.parse(createCall()![1].body)).toEqual({
      input: 'train-lora',
      fields: { resume: RUNS.runs[0].checkpoints[1].dir, steps: '3000', lr: '0.0001' },
    });
  });

  it('warns about a small set and shows a failed submit', async () => {
    createOk = false;
    const onClose = jest.fn();
    render(<TrainLoraDialog req={{ paths: IMGS }} onClose={onClose} />);
    expect(screen.getByText(/Few images/)).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Trigger word'), { target: { value: 'zkx' } });
    fireEvent.click(screen.getByRole('button', { name: 'Start training' }));
    await waitFor(() => expect(screen.getByText(/Could not start the training/)).toBeTruthy());
    expect(onClose).not.toHaveBeenCalled();
  });
});
