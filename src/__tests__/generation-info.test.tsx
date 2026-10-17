import React from 'react';
import { render, fireEvent, act } from '@testing-library/react';
import GenerationInfo from '../renderer/components/metadata/generation-info';

const comfyPrompt = {
  '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'flux1-dev.safetensors' } },
  '6': { class_type: 'CLIPTextEncode', inputs: { text: 'a lighthouse at dusk', clip: ['4', 1] } },
  '7': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry', clip: ['4', 1] } },
  '3': { class_type: 'KSampler', inputs: { seed: 9, steps: 20, cfg: 4, sampler_name: 'euler', scheduler: 'simple', denoise: 1, model: ['4', 0], positive: ['6', 0], negative: ['7', 0] } },
};
const workflow = { nodes: [{ id: 3, type: 'KSampler' }], links: [], extra: { note: 'big' } };

let written: string[];
beforeEach(() => {
  written = [];
  Object.assign(navigator, { clipboard: { writeText: (t: string) => { written.push(t); return Promise.resolve(); } } });
});

test('renders the generator, prompts and settings, and copies the whole workflow JSON', async () => {
  const chunks = { prompt: JSON.stringify(comfyPrompt), workflow: JSON.stringify(workflow) };
  const { getByText, queryByText } = render(<GenerationInfo chunks={chunks} />);
  expect(getByText('Generation · ComfyUI')).toBeTruthy();
  expect(getByText('a lighthouse at dusk')).toBeTruthy();
  expect(getByText('blurry')).toBeTruthy();
  expect(getByText('flux1-dev.safetensors')).toBeTruthy();
  expect(getByText('euler')).toBeTruthy();

  await act(async () => { fireEvent.click(getByText('Copy workflow JSON')); });
  expect(written).toHaveLength(1);
  expect(JSON.parse(written[0])).toEqual(workflow);
  expect(written[0]).toContain('\n');   // pretty-printed, as ComfyUI's own save does
  expect(getByText('Copied')).toBeTruthy();

  await act(async () => { fireEvent.click(getByText('Copy API prompt JSON')); });
  expect(JSON.parse(written[1])).toEqual(comfyPrompt);

  // The raw records are there on demand, not in the way.
  expect(queryByText(/embedded text \(2\)/)).toBeTruthy();
  expect(queryByText('Copy parameters')).toBeNull();
});

test('Lowkey files name the tool, show the command and copy it', async () => {
  const rec = { loki: 1, tool: 'loki-retouch', version: '0.1.0', model: 'Qwen Image 2.1', args: ['-p', 'add snow', 'a.jpg'], prompt: 'add snow', seed: 5, steps: 25, width: 640, height: 480 };
  const { getByText, queryByText } = render(<GenerationInfo chunks={{ loki: JSON.stringify(rec) }} />);
  expect(getByText('Generation · Lowkey · loki-retouch 0.1.0')).toBeTruthy();
  expect(getByText('add snow')).toBeTruthy();
  expect(getByText('loki-retouch -p "add snow" a.jpg')).toBeTruthy();
  expect(getByText('Qwen Image 2.1')).toBeTruthy();
  await act(async () => { fireEvent.click(getByText('Copy command')); });
  expect(written).toEqual(['loki-retouch -p "add snow" a.jpg']);
  await act(async () => { fireEvent.click(getByText('Copy generation JSON')); });
  expect(JSON.parse(written[1])).toEqual(rec);
  expect(queryByText('Copy workflow JSON')).toBeNull();
});

test('WebUI files get the parameters block copy, and files without records render nothing', () => {
  const parameters = 'a cat\nNegative prompt: dog\nSteps: 20, Sampler: Euler a, CFG scale: 7, Seed: 1, Model: v1-5';
  const { getByText, queryByText } = render(<GenerationInfo chunks={{ parameters }} />);
  expect(getByText('Generation · Stable Diffusion WebUI')).toBeTruthy();
  expect(getByText('a cat')).toBeTruthy();
  expect(getByText('Copy parameters')).toBeTruthy();
  expect(queryByText('Copy workflow JSON')).toBeNull();

  const { container } = render(<GenerationInfo chunks={undefined} />);
  expect(container.innerHTML).toBe('');
});
