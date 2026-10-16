/*
 * Generation metadata — what an AI image tool wrote into the file, made
 * readable. The backend (Electron main or the Go server) hands over the raw
 * embedded text records (`textChunks`, keyword → text); this turns them
 * into one shape for the panel: which tool, the prompts, the settings that
 * matter, and the JSON documents worth copying out whole.
 *
 * Pure functions, no DOM — the same code runs for both environments and
 * under Jest.
 */

export interface GenerationInfo {
  /** Which tool wrote the file, as the panel should name it. */
  generator: 'ComfyUI' | 'Stable Diffusion WebUI' | 'NovelAI' | 'InvokeAI' | 'Unknown';
  prompt?: string;
  negativePrompt?: string;
  /** Ordered label/value pairs: steps, sampler, model… */
  params: Array<[string, string]>;
  /** ComfyUI's editor graph (what "Load" in ComfyUI wants), pretty-printed. */
  workflowJson?: string;
  /** ComfyUI's API-format prompt (what /prompt accepts), pretty-printed. */
  promptJson?: string;
  /** WebUI's one-block parameters text, verbatim. */
  parametersText?: string;
  /** Every raw record, for the fallback listing. */
  raw: Record<string, string>;
}

type Json = any;

/** Parse JSON leniently: ComfyUI writes plain JSON, but some tools wrap
 * it in whitespace or a "prompt:" prefix. */
function parseJson(text: string | undefined): Json | null {
  if (!text) return null;
  const s = text.trim();
  if (!s.startsWith('{') && !s.startsWith('[')) return null;
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

/** Return the first chunk whose keyword matches (case-insensitive). */
function chunk(chunks: Record<string, string>, ...names: string[]): string | undefined {
  for (const name of names) {
    const hit = Object.keys(chunks).find((k) => k.toLowerCase() === name.toLowerCase());
    if (hit && chunks[hit]) return chunks[hit];
  }
  return undefined;
}

/* ---- ComfyUI ---------------------------------------------------------- */

const isLink = (v: Json): v is [string, number] =>
  Array.isArray(v) && v.length === 2 && (typeof v[0] === 'string' || typeof v[0] === 'number');

const fmt = (v: Json): string =>
  typeof v === 'number' ? String(+v.toFixed(4)) : String(v);

/**
 * Pull prompts and settings out of a ComfyUI API-format prompt: a map of
 * node id → {class_type, inputs}. Links are `[nodeId, slot]` inputs. The
 * strategy is graph-shaped rather than node-name-shaped so custom nodes
 * mostly work: find the sampler, follow its positive/negative wires back
 * to text, and sweep everything upstream of it for the settings.
 */
function parseComfyPrompt(graph: Json): Partial<GenerationInfo> {
  const nodes: Record<string, { class_type: string; inputs: Record<string, Json>; _meta?: { title?: string } }> = {};
  for (const [id, node] of Object.entries(graph ?? {})) {
    if (node && typeof node === 'object' && typeof (node as Json).class_type === 'string') {
      nodes[id] = { inputs: {}, ...(node as Json) };
      nodes[id].inputs ??= {};
    }
  }
  const ids = Object.keys(nodes);
  if (!ids.length) return {};

  /** The text a conditioning wire ultimately carries. */
  const resolveText = (id: string, slot = 0, depth = 0, seen = new Set<string>()): string | undefined => {
    const node = nodes[id];
    if (!node || depth > 12 || seen.has(id)) return undefined;
    seen.add(id);
    const inputs = node.inputs;
    // A zeroed conditioning is an explicitly empty prompt, not whatever
    // text happens to sit behind it.
    if (/zeroout|zero_out/i.test(node.class_type)) return '';
    // Nodes that carry a positive/negative PAIR through (LTXVConditioning,
    // crop guides…) answer by output slot: 0 is the positive side, 1 the
    // negative. Without this, following "negative" lands on the positive
    // text because the pair's first input wins.
    if (isLink(inputs.positive) && isLink(inputs.negative)) {
      const side = slot === 1 ? inputs.negative : inputs.positive;
      const t = resolveText(String(side[0]), side[1], depth + 1, seen);
      if (t !== undefined) return t;
    }
    for (const key of ['text', 'prompt', 'string', 'value', 'text_g', 'positive_prompt', 'clip_l', 't5xxl']) {
      const v = inputs[key];
      if (typeof v === 'string') {
        if (key === 'text_g' && typeof inputs.text_l === 'string' && inputs.text_l !== v) return `${v}\n${inputs.text_l}`;
        return v;
      }
      if (isLink(v)) {
        const t = resolveText(String(v[0]), v[1], depth + 1, seen);
        if (t !== undefined) return t;
      }
    }
    // Pass-through nodes (combine, controlnet apply, guidance…): follow the
    // conditioning-ish wires first, then anything else that is a link.
    const order = Object.entries(inputs).sort(([a], [b]) => rank(a) - rank(b));
    for (const [, v] of order) {
      if (!isLink(v)) continue;
      const t = resolveText(String(v[0]), v[1], depth + 1, seen);
      if (t !== undefined) return t;
    }
    return undefined;
  };
  const rank = (key: string) =>
    /^(conditioning|positive|negative|cond)/i.test(key) ? 0 : /^(clip|model|vae|latent|image|mask|samples|pixels)/i.test(key) ? 2 : 1;

  // The sampler: prefer one that owns positive/negative wires (KSampler,
  // KSamplerAdvanced), else any sampler-named node (SamplerCustom… wire
  // their conditioning through a guider, which the upstream sweep finds).
  const samplers = ids.filter((id) => /sampler/i.test(nodes[id].class_type) && !/select|sampler_name/i.test(nodes[id].class_type));
  const sampler = samplers.find((id) => nodes[id].inputs.positive !== undefined) ?? samplers[samplers.length - 1];

  const params: Array<[string, string]> = [];
  const seenParam = new Set<string>();
  const put = (label: string, value: Json) => {
    if (value === undefined || value === null || value === '' || isLink(value)) return;
    if (seenParam.has(label)) return;
    seenParam.add(label);
    params.push([label, fmt(value)]);
  };
  const loras: string[] = [];
  let width: Json;
  let height: Json;

  // Sweep upstream of the sampler (or the whole graph without one), nearest
  // nodes first so the sampler's own steps/cfg win over a scheduler's.
  const sweep = (start: string[]) => {
    const seen = new Set<string>();
    const queue = [...start];
    while (queue.length) {
      const id = queue.shift()!;
      if (seen.has(id) || !nodes[id]) continue;
      seen.add(id);
      const { inputs, class_type } = nodes[id];
      put('seed', inputs.seed ?? inputs.noise_seed);
      put('steps', inputs.steps);
      put('cfg', inputs.cfg ?? inputs.cfg_scale);
      put('guidance', inputs.guidance);
      put('sampler', inputs.sampler_name);
      put('scheduler', inputs.scheduler);
      put('denoise', inputs.denoise);
      put('shift', inputs.shift);
      put('model', inputs.ckpt_name ?? inputs.unet_name ?? inputs.model_name);
      put('clip', inputs.clip_name ?? inputs.clip_name1);
      put('vae', inputs.vae_name);
      if (typeof inputs.lora_name === 'string' && !loras.includes(inputs.lora_name)) {
        const strength = inputs.strength_model ?? inputs.strength;
        loras.push(typeof strength === 'number' ? `${inputs.lora_name} (${fmt(strength)})` : inputs.lora_name);
      }
      if (/latent/i.test(class_type) && typeof inputs.width === 'number' && typeof inputs.height === 'number') {
        width ??= inputs.width;
        height ??= inputs.height;
      }
      for (const v of Object.values(inputs)) if (isLink(v)) queue.push(String(v[0]));
    }
  };
  sweep(sampler ? [sampler] : ids);
  if (width !== undefined) put('size', `${width}×${height}`);
  if (loras.length) put('loras', loras.join(', '));

  let prompt: string | undefined;
  let negativePrompt: string | undefined;
  if (sampler) {
    const { inputs } = nodes[sampler];
    const guider = isLink(inputs.guider) ? nodes[String(inputs.guider[0])]?.inputs : null;
    const pos = inputs.positive ?? guider?.positive ?? guider?.conditioning;
    const neg = inputs.negative ?? guider?.negative;
    if (isLink(pos)) prompt = resolveText(String(pos[0]), pos[1]);
    if (isLink(neg)) negativePrompt = resolveText(String(neg[0]), neg[1]);
  }
  if (prompt === undefined) {
    // No sampler we recognise: the text encoders themselves, titled or
    // not, still say what was asked for.
    const encoders = ids.filter((id) => /textencode|text_encode|prompt/i.test(nodes[id].class_type) && resolveText(id));
    const negIdx = encoders.findIndex((id) => /neg/i.test(nodes[id]._meta?.title ?? ''));
    const posIdx = encoders.findIndex((id, i) => i !== negIdx);
    if (posIdx >= 0) prompt = resolveText(encoders[posIdx]);
    if (negIdx >= 0) negativePrompt = resolveText(encoders[negIdx]);
    else if (encoders.length > 1 && posIdx === 0) negativePrompt = resolveText(encoders[1]);
  }
  return { prompt, negativePrompt, params };
}

/**
 * Fallback for files that carry only the editor graph: widget values are
 * positional, so this reads the stock nodes by their known layouts.
 */
function parseComfyWorkflow(workflow: Json): Partial<GenerationInfo> {
  const nodes: Json[] = Array.isArray(workflow?.nodes) ? workflow.nodes : [];
  const params: Array<[string, string]> = [];
  const texts: Array<{ text: string; title: string }> = [];
  for (const n of nodes) {
    const w = Array.isArray(n.widgets_values) ? n.widgets_values : [];
    const type = String(n.type ?? '');
    if (type === 'CLIPTextEncode' && typeof w[0] === 'string') texts.push({ text: w[0], title: String(n.title ?? '') });
    if (type === 'KSampler' && w.length >= 7 && !params.length) {
      params.push(['seed', fmt(w[0])], ['steps', fmt(w[2])], ['cfg', fmt(w[3])],
        ['sampler', fmt(w[4])], ['scheduler', fmt(w[5])], ['denoise', fmt(w[6])]);
    }
    if (type === 'CheckpointLoaderSimple' && typeof w[0] === 'string') params.push(['model', w[0]]);
    if (type === 'EmptyLatentImage' && w.length >= 2) params.push(['size', `${w[0]}×${w[1]}`]);
  }
  const neg = texts.find((t) => /neg/i.test(t.title));
  const pos = texts.find((t) => t !== neg);
  const prompt = pos?.text;
  const negativePrompt = neg?.text ?? (texts.length > 1 && pos === texts[0] ? texts[1].text : undefined);
  return { prompt, negativePrompt, params };
}

/* ---- Stable Diffusion WebUI ------------------------------------------- */

/**
 * The WebUI block: prompt, an optional "Negative prompt:" line, then one
 * line of "Key: value, Key: value" where values can be quoted strings or
 * JSON objects (Hashes, TI hashes).
 */
export function parseWebUiParameters(text: string): Partial<GenerationInfo> {
  const lines = text.replace(/\r\n/g, '\n').split('\n');
  let settingsIdx = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (/^Steps: \d+/.test(lines[i]) || /^[A-Z][\w ]*: [^,]*,\s*[A-Z][\w ]*: /.test(lines[i])) { settingsIdx = i; break; }
  }
  const body = (settingsIdx >= 0 ? lines.slice(0, settingsIdx) : lines).join('\n');
  const negAt = body.indexOf('\nNegative prompt: ');
  const startsNeg = body.startsWith('Negative prompt: ');
  const prompt = startsNeg ? '' : negAt >= 0 ? body.slice(0, negAt) : body;
  const negativePrompt = startsNeg ? body.slice('Negative prompt: '.length)
    : negAt >= 0 ? body.slice(negAt + '\nNegative prompt: '.length) : undefined;
  const params: Array<[string, string]> = [];
  if (settingsIdx >= 0) {
    const re = /([A-Za-z][\w .\-/]*?): ("(?:[^"\\]|\\.)*"|\{[^}]*\}|[^,]*)(?:,\s*|$)/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(lines[settingsIdx])) !== null) {
      let value = m[2].trim();
      if (value.startsWith('"') && value.endsWith('"')) {
        try { value = JSON.parse(value); } catch { value = value.slice(1, -1); }
      }
      if (value) params.push([m[1].trim().toLowerCase(), value]);
    }
  }
  return {
    prompt: prompt.trim() || undefined,
    negativePrompt: negativePrompt?.trim() || undefined,
    params,
  };
}

/* ---- entry point -------------------------------------------------------- */

const pretty = (v: Json) => JSON.stringify(v, null, 2);

/** Interpret a file's embedded text records. Null when there are none. */
export function parseGenerationMetadata(
  chunks: Record<string, string> | undefined | null
): GenerationInfo | null {
  if (!chunks || !Object.keys(chunks).length) return null;
  const raw = chunks;

  // ComfyUI: PNG keywords, or the webp/jpeg Exif "prompt:"/"workflow:"
  // prefixed fields, or a video comment holding both.
  let promptText = chunk(chunks, 'prompt');
  let workflowText = chunk(chunks, 'workflow');
  for (const text of Object.values(chunks)) {
    if (!promptText && text.startsWith('prompt:')) promptText = text.slice(7);
    if (!workflowText && text.startsWith('workflow:')) workflowText = text.slice(9);
  }
  const comment = chunk(chunks, 'comment', 'description');
  const commentJson = parseJson(comment);
  if (commentJson && typeof commentJson === 'object' && (commentJson.prompt || commentJson.workflow)) {
    promptText ??= commentJson.prompt ? JSON.stringify(commentJson.prompt) : undefined;
    workflowText ??= commentJson.workflow ? JSON.stringify(commentJson.workflow) : undefined;
  }
  const promptJson = parseJson(promptText);
  const workflowJson = parseJson(workflowText);
  if (promptJson || workflowJson) {
    const parsed = promptJson ? parseComfyPrompt(promptJson) : parseComfyWorkflow(workflowJson);
    const fromGraph = !parsed.prompt && workflowJson ? parseComfyWorkflow(workflowJson) : null;
    return {
      generator: 'ComfyUI',
      prompt: parsed.prompt ?? fromGraph?.prompt,
      negativePrompt: parsed.negativePrompt ?? fromGraph?.negativePrompt,
      params: parsed.params?.length ? parsed.params : fromGraph?.params ?? [],
      workflowJson: workflowJson ? pretty(workflowJson) : undefined,
      promptJson: promptJson ? pretty(promptJson) : undefined,
      raw,
    };
  }

  // Stable Diffusion WebUI (and Forge, SD.Next…): the parameters block.
  const parameters = chunk(chunks, 'parameters', 'UserComment')
    ?? Object.values(chunks).find((t) => /\nSteps: \d+/.test(t) || /Negative prompt: /.test(t));
  if (parameters && (/Steps: \d+/.test(parameters) || /Negative prompt: /.test(parameters))) {
    const parsed = parseWebUiParameters(parameters);
    return { generator: 'Stable Diffusion WebUI', ...parsed, params: parsed.params ?? [], parametersText: parameters, raw };
  }

  if (chunk(chunks, 'invokeai_metadata') || chunk(chunks, 'invokeai_graph')) {
    const meta = parseJson(chunk(chunks, 'invokeai_metadata'));
    const params: Array<[string, string]> = [];
    for (const key of ['seed', 'steps', 'cfg_scale', 'scheduler', 'width', 'height']) {
      if (meta?.[key] !== undefined) params.push([key.replace('_scale', ''), fmt(meta[key])]);
    }
    if (meta?.model?.name ?? meta?.model?.model_name) params.push(['model', meta.model.name ?? meta.model.model_name]);
    return {
      generator: 'InvokeAI',
      prompt: meta?.positive_prompt,
      negativePrompt: meta?.negative_prompt,
      params,
      workflowJson: chunk(chunks, 'invokeai_graph') ? pretty(parseJson(chunk(chunks, 'invokeai_graph')) ?? chunk(chunks, 'invokeai_graph')) : undefined,
      raw,
    };
  }

  if (/novelai/i.test(chunk(chunks, 'Software') ?? '')) {
    const meta = parseJson(chunk(chunks, 'Comment'));
    const params: Array<[string, string]> = [];
    for (const key of ['seed', 'steps', 'scale', 'sampler', 'width', 'height']) {
      if (meta?.[key] !== undefined) params.push([key === 'scale' ? 'cfg' : key, fmt(meta[key])]);
    }
    return {
      generator: 'NovelAI',
      prompt: chunk(chunks, 'Description') ?? meta?.prompt,
      negativePrompt: meta?.uc,
      params,
      raw,
    };
  }

  return { generator: 'Unknown', params: [], raw };
}
