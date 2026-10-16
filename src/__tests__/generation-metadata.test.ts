import { parseGenerationMetadata, parseWebUiParameters } from '../renderer/generation-metadata';

/* A small but realistic ComfyUI API prompt: checkpoint → two text encoders
 * → KSampler, with the positive text routed through a primitive node and a
 * LoRA on the model path. */
const comfyPrompt = {
  '4': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'sd_xl_base_1.0.safetensors' } },
  '10': { class_type: 'LoraLoader', inputs: { lora_name: 'detail.safetensors', strength_model: 0.8, strength_clip: 1, model: ['4', 0], clip: ['4', 1] } },
  '5': { class_type: 'EmptyLatentImage', inputs: { width: 1024, height: 768, batch_size: 1 } },
  '20': { class_type: 'PrimitiveNode', inputs: { value: 'a lighthouse at dusk, oil painting' } },
  '6': { class_type: 'CLIPTextEncode', inputs: { text: ['20', 0], clip: ['10', 1] }, _meta: { title: 'CLIP Text Encode (Prompt)' } },
  '7': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry, watermark', clip: ['10', 1] }, _meta: { title: 'Negative' } },
  '3': {
    class_type: 'KSampler',
    inputs: {
      seed: 123456789, steps: 28, cfg: 6.5, sampler_name: 'dpmpp_2m', scheduler: 'karras', denoise: 1,
      model: ['10', 0], positive: ['6', 0], negative: ['7', 0], latent_image: ['5', 0],
    },
  },
  '8': { class_type: 'VAEDecode', inputs: { samples: ['3', 0], vae: ['4', 2] } },
  '9': { class_type: 'SaveImage', inputs: { filename_prefix: 'ComfyUI', images: ['8', 0] } },
};

const comfyWorkflow = { last_node_id: 20, nodes: [{ id: 3, type: 'KSampler', widgets_values: [1, 'fixed', 28, 6.5, 'dpmpp_2m', 'karras', 1] }], links: [] };

test('ComfyUI PNG text: prompt walks links, settings sweep upstream, both JSONs copyable', () => {
  const info = parseGenerationMetadata({
    prompt: JSON.stringify(comfyPrompt),
    workflow: JSON.stringify(comfyWorkflow),
  });
  expect(info?.generator).toBe('ComfyUI');
  expect(info?.prompt).toBe('a lighthouse at dusk, oil painting');
  expect(info?.negativePrompt).toBe('blurry, watermark');
  const params = Object.fromEntries(info!.params);
  expect(params).toMatchObject({
    seed: '123456789', steps: '28', cfg: '6.5', sampler: 'dpmpp_2m', scheduler: 'karras', denoise: '1',
    model: 'sd_xl_base_1.0.safetensors', size: '1024×768', loras: 'detail.safetensors (0.8)',
  });
  expect(JSON.parse(info!.workflowJson!)).toEqual(comfyWorkflow);
  expect(JSON.parse(info!.promptJson!)).toEqual(comfyPrompt);
  expect(info!.workflowJson).toContain('\n  ');   // pretty-printed for pasting
});

test('ComfyUI SamplerCustomAdvanced: conditioning comes through the guider, seed through the noise node', () => {
  const prompt = {
    '1': { class_type: 'UNETLoader', inputs: { unet_name: 'flux1-dev.safetensors', weight_dtype: 'default' } },
    '2': { class_type: 'CLIPTextEncode', inputs: { text: 'a red fox in snow', clip: ['9', 0] } },
    '3': { class_type: 'FluxGuidance', inputs: { conditioning: ['2', 0], guidance: 3.5 } },
    '4': { class_type: 'BasicGuider', inputs: { model: ['1', 0], conditioning: ['3', 0] } },
    '5': { class_type: 'RandomNoise', inputs: { noise_seed: 42 } },
    '6': { class_type: 'KSamplerSelect', inputs: { sampler_name: 'euler' } },
    '7': { class_type: 'BasicScheduler', inputs: { scheduler: 'simple', steps: 20, denoise: 1, model: ['1', 0] } },
    '8': { class_type: 'SamplerCustomAdvanced', inputs: { noise: ['5', 0], guider: ['4', 0], sampler: ['6', 0], sigmas: ['7', 0], latent_image: ['10', 0] } },
    '9': { class_type: 'DualCLIPLoader', inputs: { clip_name1: 't5xxl.safetensors', clip_name2: 'clip_l.safetensors' } },
    '10': { class_type: 'EmptySD3LatentImage', inputs: { width: 1024, height: 1024, batch_size: 1 } },
  };
  const info = parseGenerationMetadata({ prompt: JSON.stringify(prompt) });
  expect(info?.prompt).toBe('a red fox in snow');
  expect(info?.negativePrompt).toBeUndefined();
  expect(Object.fromEntries(info!.params)).toMatchObject({
    seed: '42', steps: '20', guidance: '3.5', sampler: 'euler', scheduler: 'simple',
    model: 'flux1-dev.safetensors', size: '1024×1024',
  });
});

test('ComfyUI webp/jpeg Exif fields carry prompt:/workflow: prefixes; video comments carry both as one object', () => {
  const exif = parseGenerationMetadata({ Model: `prompt:${JSON.stringify(comfyPrompt)}`, Make: `workflow:${JSON.stringify(comfyWorkflow)}` });
  expect(exif?.generator).toBe('ComfyUI');
  expect(exif?.prompt).toBe('a lighthouse at dusk, oil painting');
  expect(JSON.parse(exif!.workflowJson!)).toEqual(comfyWorkflow);

  const video = parseGenerationMetadata({ comment: JSON.stringify({ prompt: comfyPrompt, workflow: comfyWorkflow }), encoder: 'Lavf60' });
  expect(video?.generator).toBe('ComfyUI');
  expect(video?.prompt).toBe('a lighthouse at dusk, oil painting');
  expect(JSON.parse(video!.promptJson!)).toEqual(comfyPrompt);
});

test('ComfyUI editor graph alone: stock node widget layouts are read positionally', () => {
  const workflow = {
    nodes: [
      { id: 1, type: 'CLIPTextEncode', title: 'Positive', widgets_values: ['a castle'] },
      { id: 2, type: 'CLIPTextEncode', title: 'Negative Prompt', widgets_values: ['ugly'] },
      { id: 3, type: 'KSampler', widgets_values: [7, 'randomize', 30, 7, 'euler_ancestral', 'normal', 0.6] },
      { id: 4, type: 'CheckpointLoaderSimple', widgets_values: ['dreamshaper.safetensors'] },
      { id: 5, type: 'EmptyLatentImage', widgets_values: [512, 768, 1] },
    ],
  };
  const info = parseGenerationMetadata({ workflow: JSON.stringify(workflow) });
  expect(info?.prompt).toBe('a castle');
  expect(info?.negativePrompt).toBe('ugly');
  expect(Object.fromEntries(info!.params)).toMatchObject({ seed: '7', steps: '30', cfg: '7', sampler: 'euler_ancestral', denoise: '0.6', model: 'dreamshaper.safetensors', size: '512×768' });
  expect(info?.promptJson).toBeUndefined();
});

const webui = `masterpiece, a cat wearing a hat
on a chair
Negative prompt: lowres, bad anatomy
Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 3141592653, Size: 512x768, Model hash: 6ce0161689, Model: v1-5-pruned-emaonly, Lora hashes: "detail: 1a2b3c", Version: v1.9.4`;

test('WebUI parameters block: multi-line prompt, negative, and the settings line', () => {
  const info = parseGenerationMetadata({ parameters: webui });
  expect(info?.generator).toBe('Stable Diffusion WebUI');
  expect(info?.prompt).toBe('masterpiece, a cat wearing a hat\non a chair');
  expect(info?.negativePrompt).toBe('lowres, bad anatomy');
  expect(Object.fromEntries(info!.params)).toMatchObject({
    steps: '30', sampler: 'DPM++ 2M Karras', 'cfg scale': '7', seed: '3141592653', size: '512x768',
    model: 'v1-5-pruned-emaonly', 'lora hashes': 'detail: 1a2b3c', version: 'v1.9.4',
  });
  expect(info?.parametersText).toBe(webui);
  expect(info?.workflowJson).toBeUndefined();
});

test('WebUI parameters arrive via Exif UserComment on jpeg/webp, and a block without a negative prompt still parses', () => {
  expect(parseGenerationMetadata({ UserComment: webui })?.generator).toBe('Stable Diffusion WebUI');
  const noNeg = parseWebUiParameters('just a prompt\nSteps: 20, Sampler: Euler a, Seed: 1');
  expect(noNeg.prompt).toBe('just a prompt');
  expect(noNeg.negativePrompt).toBeUndefined();
  expect(noNeg.params).toEqual([['steps', '20'], ['sampler', 'Euler a'], ['seed', '1']]);
});

test('files with no records yield nothing; unrecognised text is still listed for copying', () => {
  expect(parseGenerationMetadata(undefined)).toBeNull();
  expect(parseGenerationMetadata({})).toBeNull();
  const other = parseGenerationMetadata({ Software: 'Photoshop', Comment: 'hello' });
  expect(other?.generator).toBe('Unknown');
  expect(other?.raw).toEqual({ Software: 'Photoshop', Comment: 'hello' });
});

test('ComfyUI pair-carrying nodes answer by output slot, and a zeroed conditioning is an empty negative', () => {
  const prompt = {
    '1': { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'ltx.safetensors' } },
    '3': { class_type: 'CLIPTextEncode', inputs: { text: 'a dancer on a rooftop', clip: ['1', 1] } },
    '4': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry', clip: ['1', 1] } },
    '22': { class_type: 'LTXVConditioning', inputs: { positive: ['3', 0], negative: ['4', 0], frame_rate: 25 } },
    '81': { class_type: 'LTXVCropGuides', inputs: { positive: ['22', 0], negative: ['22', 1], latent: ['43', 0] } },
    '82': { class_type: 'CFGGuider', inputs: { model: ['1', 0], positive: ['81', 0], negative: ['81', 1], cfg: 3 } },
    '41': { class_type: 'SamplerCustomAdvanced', inputs: { guider: ['82', 0], noise: ['11', 0], sampler: ['8', 0], sigmas: ['9', 0], latent_image: ['43', 0] } },
    '11': { class_type: 'RandomNoise', inputs: { noise_seed: 77 } },
    '8': { class_type: 'KSamplerSelect', inputs: { sampler_name: 'euler' } },
    '9': { class_type: 'LTXVScheduler', inputs: { steps: 30, max_shift: 2, base_shift: 0.95 } },
    '43': { class_type: 'EmptyLTXVLatentVideo', inputs: { width: 768, height: 512, length: 97, batch_size: 1 } },
  };
  const info = parseGenerationMetadata({ prompt: JSON.stringify(prompt) });
  expect(info?.prompt).toBe('a dancer on a rooftop');
  expect(info?.negativePrompt).toBe('blurry');
  expect(Object.fromEntries(info!.params)).toMatchObject({ seed: '77', cfg: '3', sampler: 'euler', steps: '30', size: '768×512', model: 'ltx.safetensors' });

  const zeroed = { ...prompt, '4': { class_type: 'ConditioningZeroOut', inputs: { conditioning: ['3', 0] } } };
  const z = parseGenerationMetadata({ prompt: JSON.stringify(zeroed) });
  expect(z?.prompt).toBe('a dancer on a rooftop');
  expect(z?.negativePrompt).toBe('');
});
