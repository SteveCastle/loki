import { parseGenerationMetadata, parseWebUiParameters, shellQuote } from '../renderer/generation-metadata';

/* The record loki-retouch stamps into its PNGs (iTXt chunk `loki`), as the
 * Rust side writes it: markers first, then the run. */
const retouchRecord = {
  loki: 1, tool: 'loki-retouch', version: '0.1.0', model: 'Qwen Image 2.1', created: '2026-10-16T21:04:05Z',
  args: ['--preset', '4kify', '-p', 'make it night, with rain', 'C:\\shots\\photo 1.jpg'],
  prompt: 'make it night, with rain', preset: '4kify', seed: 3, steps: 25, shift: 0.69,
  width: 3840, height: 2160, ref_resolution: 0,
  inputs: [{ tag: '<image1>', path: 'C:\\shots\\photo 1.jpg', width: 1024, height: 768 }, { tag: '<image2>', path: 'jacket.png', width: 512, height: 512 }],
};

test('loki-retouch PNG record: Lowkey generator, prompt, settings, inputs, and a re-runnable command', () => {
  const info = parseGenerationMetadata({ loki: JSON.stringify(retouchRecord) });
  expect(info?.generator).toBe('Lowkey');
  expect(info?.tool).toBe('loki-retouch 0.1.0');
  expect(info?.prompt).toBe('make it night, with rain');
  expect(info?.negativePrompt).toBeUndefined();
  expect(Object.fromEntries(info!.params)).toMatchObject({
    model: 'Qwen Image 2.1', preset: '4kify', seed: '3', steps: '25', shift: '0.69', size: '3840×2160', 'reference size': 'auto',
    '<image1>': 'C:\\shots\\photo 1.jpg (1024×768)', '<image2>': 'jacket.png (512×512)', created: '2026-10-16 21:04:05 UTC',
  });
  expect(info?.commandLine).toBe('loki-retouch --preset 4kify -p "make it night, with rain" "C:\\shots\\photo 1.jpg"');
  expect(JSON.parse(info!.recordJson!)).toEqual(retouchRecord);
  expect(info?.workflowJson).toBeUndefined();
  expect(info?.promptJson).toBeUndefined();
});

test('loki-reshoot mp4 record arrives as the container comment and is not mistaken for a ComfyUI comment', () => {
  const rec = {
    loki: 1, tool: 'loki-reshoot', version: '0.1.0', model: 'MiniMax H3', created: '2026-10-16T21:10:00Z',
    args: ['--animate', 'face.png', '-d', '5'],
    prompt: 'A living photo…', seed: 1234, steps: 20, width: 768, height: 1024, frames: 124, fps: 24, seconds: 5.167,
    audio: true, crf: 18, ref_image_size: 'match', native: true, fit: 'crop', animate: true, shake: 'subtle',
    inputs: [
      { tag: '<Picture 1>', path: 'face.png', width: 1200, height: 1600 },
      { tag: '<Video 1>', path: 'dance.mp4', start: 2, duration: 4, audio_tag: '<Audio 1>' },
      { tag: '<Audio 2>', path: 'theme.mp3', start: 0 },
    ],
  };
  const info = parseGenerationMetadata({ comment: JSON.stringify(rec), encoder: 'Lavf61', major_brand: 'isom' });
  expect(info?.generator).toBe('Lowkey');
  expect(info?.tool).toBe('loki-reshoot 0.1.0');
  expect(info?.prompt).toBe('A living photo…');
  expect(Object.fromEntries(info!.params)).toMatchObject({
    model: 'MiniMax H3', seed: '1234', steps: '20', size: '768×1024', length: '5.167 s · 124 frames @ 24 fps', audio: 'yes', crf: '18',
    'reference images': 'match', fit: 'crop', 'living photo': 'yes', shake: 'subtle',
    '<Picture 1>': 'face.png (1200×1600)', '<Video 1>': 'dance.mp4 (from 2 s, 4 s, soundtrack <Audio 1>)', '<Audio 2>': 'theme.mp3',
  });
  expect(info?.commandLine).toBe('loki-reshoot --animate face.png -d 5');
  expect(info?.raw).toHaveProperty('encoder', 'Lavf61');

  // A newer record (more fields, higher version) still reads; a generated-size note appears when it differs.
  const later = parseGenerationMetadata({ loki: JSON.stringify({ ...retouchRecord, loki: 2, width: 1000, height: 700, generated_width: 1008, generated_height: 704, future_field: { a: 1 } }) });
  expect(later?.generator).toBe('Lowkey');
  expect(Object.fromEntries(later!.params).size).toBe('1000×700 (generated at 1008×704)');

  // Not ours: a `loki` key without the tool marker, or no record at all.
  expect(parseGenerationMetadata({ loki: '{"loki":1}' })?.generator).toBe('Unknown');
  expect(parseGenerationMetadata({ comment: 'loki was here' })?.generator).toBe('Unknown');
});

test('shellQuote leaves plain words bare and quotes the rest', () => {
  expect(shellQuote('--preset')).toBe('--preset');
  expect(shellQuote('photos/a-b_c.jpg')).toBe('photos/a-b_c.jpg');
  expect(shellQuote('C:\\x\\y.png')).toBe('C:\\x\\y.png');
  expect(shellQuote('two words')).toBe('"two words"');
  expect(shellQuote('say "hi"')).toBe('"say \\"hi\\""');
  expect(shellQuote('C:\\my shots\\a.jpg')).toBe('"C:\\my shots\\a.jpg"');
  expect(shellQuote('')).toBe('""');
  expect(shellQuote('<image1> and <image2>')).toBe('"<image1> and <image2>"');
});

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
