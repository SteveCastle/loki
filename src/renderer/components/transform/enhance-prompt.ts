// Prompt enhancement: the Enhance button next to a Transform prompt asks the server's text model (loki-scribe) to
// rewrite the prompt the way the target model reads best (POST /api/transform/enhance-prompt, guides in
// media-server/tasks/prompt_guides/). Only ever on request: a run never waits for it, and the result lands in the
// prompt box where it can be edited or undone.
import { mediaServerBase } from '../../platform';
import { effectiveDreamModel, type DreamCatalog } from '../dream/dream-models';
import { mediaKind, type IntentId, type TransformSettings } from './intents';

export type EnhanceTask = 'generate' | 'reimagine' | 'edit' | 'video';

export interface EnhanceTarget {
  /** The server's guide key. */
  model: string;
  task: EnhanceTask;
  /** The model's name, for the button's tooltip. */
  label: string;
}

export interface EnhanceRequestBody {
  prompt: string;
  model: string;
  task: EnhanceTask;
  references: Array<{ token: string; kind: string; path?: string }>;
  image?: string;
  imageToken?: string;
}

export interface EnhanceResult {
  prompt: string;
  model: string;
  missingTokens?: string[];
}

// loki-dream architecture → guide key.
const DREAM_GUIDES: Record<string, string> = {
  moss: 'moss-image',
  krea2: 'krea2',
  flux2: 'flux2-klein',
  anima: 'anima',
  sdxl: 'sdxl',
};

/** Which guide a prompt is rewritten with, or null when the intent's prompt can't be enhanced. */
export function enhanceTarget(intent: IntentId, s: TransformSettings, catalog: DreamCatalog | undefined): EnhanceTarget | null {
  if (intent === 'generate' || intent === 'reimagine') {
    const m = catalog ? effectiveDreamModel(catalog, s.dreamModel) : undefined;
    const arch = m?.arch || 'moss';
    return { model: DREAM_GUIDES[arch] || 'moss-image', task: intent, label: m?.archName || 'MOSS-Image' };
  }
  if (intent === 'edit') {
    return s.retouchModel === 'h3'
      ? { model: 'h3-image', task: 'edit', label: 'MiniMax H3 Image' }
      : { model: 'qwen-image-edit', task: 'edit', label: 'Qwen Image Edit' };
  }
  // Every video model (H3, Fast, the extra files) is MiniMax H3.
  if (intent === 'direct') return { model: 'h3-video', task: 'video', label: 'MiniMax H3' };
  return null;
}

/**
 * The request for a prompt: its reference tokens, and the picture the text model looks at (the image being
 * edited or redrawn, else Direct's first picture).
 */
export function enhanceBody(
  target: EnhanceTarget,
  prompt: string,
  refs: Array<{ token: string; path: string }>,
  paths: string[]
): EnhanceRequestBody {
  const body: EnhanceRequestBody = {
    prompt,
    model: target.model,
    task: target.task,
    // The server shows the text model every image and (a frame of) every video here.
    references: refs.map((r) => ({ token: r.token, kind: mediaKind(r.path), path: r.path })),
  };
  if (target.task === 'reimagine') {
    // the picture being redrawn has no token
    const src = paths.find((p) => mediaKind(p) === 'image');
    if (src) body.image = src;
  } else if (target.task === 'edit' && refs.length === 0) {
    // a single-image edit lists no tokens, but the image is still <image1>
    const src = paths.find((p) => mediaKind(p) === 'image');
    if (src) {
      body.image = src;
      body.imageToken = '<image1>';
    }
  }
  return body;
}

export async function requestEnhance(body: EnhanceRequestBody, authToken: string | undefined, signal: AbortSignal): Promise<EnhanceResult> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (authToken) headers.Authorization = `Bearer ${authToken}`;
  const res = await fetch(`${mediaServerBase}/api/transform/enhance-prompt`, { method: 'POST', headers, body: JSON.stringify(body), signal });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data as EnhanceResult;
}
