// The diffusion models loki-dream can run (GET /api/dream/models: the engine's own --list-models, cached by the
// server), for the model pickers of Generate / Reimagine and the Scene editor's "new scene from a prompt".
//
// The list can be long (a ComfyUI models folder full of checkpoints): it is fetched once and shared by every picker,
// refetched when it is older than a minute and a picker mounts, and on demand (Refresh asks the engine to rescan).
import { useCallback, useEffect, useState } from 'react';
import { isElectron, mediaServerBase } from '../../platform';
import { serverAuthHeaders } from '../../server-auth';

export interface DreamModelNeed {
  file: string;
  present: boolean;
  downloadGb?: number;
}

export interface DreamModel {
  /** The file name a job sends as `model`. */
  file: string;
  label: string;
  /** moss | krea2 | flux2 | anima | sdxl */
  arch: string;
  archName: string;
  /** Few steps without guidance. */
  distilled: boolean;
  steps: number;
  cfg: number;
  /** Turbo = the MOSS-Image distillation LoRA. */
  turbo: boolean;
  /** The checkpoint is on the engine's machine; false = the first run downloads `downloadGb`. */
  local: boolean;
  format?: string;
  sizeGb?: number;
  downloadGb?: number;
  baseModel?: string;
  textEncoder?: DreamModelNeed;
  vae?: DreamModelNeed;
}

export interface DreamCatalog {
  models: DreamModel[];
  /** The file a job without `model` runs (the server's dreamModel setting, else the engine's default). */
  defaultFile: string;
  where?: string;
  /** Why the list is empty or partial. */
  note?: string;
  loaded: boolean;
  loading: boolean;
  error?: string;
}

const EMPTY: DreamCatalog = {
  models: [],
  defaultFile: '',
  loaded: false,
  loading: false,
};

const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
const str = (v: unknown): string => (typeof v === 'string' ? v : '');

function parseNeed(v: unknown): DreamModelNeed | undefined {
  if (!v || typeof v !== 'object') return undefined;
  const o = v as Record<string, unknown>;
  return {
    file: str(o.file),
    present: o.present === true,
    downloadGb: num(o.download_gb),
  };
}

/** The answer's models, each checked (it is data from the server). */
export function parseDreamCatalog(body: unknown): Omit<DreamCatalog, 'loaded' | 'loading'> {
  const o = body && typeof body === 'object' ? (body as Record<string, unknown>) : {};
  const list = Array.isArray(o.models) ? o.models : [];
  const models: DreamModel[] = [];
  list.forEach((m) => {
    if (!m || typeof m !== 'object') return;
    const r = m as Record<string, unknown>;
    const file = str(r.file);
    if (!file) return;
    models.push({
      file,
      label: str(r.label) || file.replace(/\.safetensors$/i, ''),
      arch: str(r.architecture) || 'moss',
      archName: str(r.architecture_name) || str(r.architecture) || 'Unknown',
      distilled: r.distilled === true,
      steps: num(r.steps) ?? 0,
      cfg: num(r.cfg) ?? 0,
      turbo: r.turbo === true,
      local: r.local !== false,
      format: str(r.format) || undefined,
      sizeGb: num(r.size_gb),
      downloadGb: num(r.download_gb),
      baseModel: str(r.base_model) || undefined,
      textEncoder: parseNeed(r.text_encoder),
      vae: parseNeed(r.vae),
    });
  });
  return {
    models,
    defaultFile: str(o.default),
    where: str(o.where) || undefined,
    note: str(o.note) || undefined,
  };
}

/** Same model file? (case-insensitive, with or without `.safetensors`) */
export function sameModelFile(a: string, b: string): boolean {
  const n = (x: string) => x.toLowerCase().replace(/\.safetensors$/, '');
  return !!a && !!b && n(a) === n(b);
}

export function findDreamModel(c: Pick<DreamCatalog, 'models'>, file: string): DreamModel | undefined {
  return c.models.find((m) => sameModelFile(m.file, file));
}

/** The model a choice runs: the file chosen, or for '' the server's default. */
export function effectiveDreamModel(c: Pick<DreamCatalog, 'models' | 'defaultFile'>, file: string): DreamModel | undefined {
  return findDreamModel(c, file || c.defaultFile);
}

/** Gigabytes a first run of the model downloads (checkpoint, text encoder, VAE); 0 = ready. */
export function downloadGb(m: DreamModel): number {
  let gb = m.local ? 0 : m.downloadGb || 0;
  if (m.textEncoder && !m.textEncoder.present) gb += m.textEncoder.downloadGb || 0;
  if (m.vae && !m.vae.present) gb += m.vae.downloadGb || 0;
  return Math.round(gb * 10) / 10;
}

/** The settings a model's recipe sets (see TransformSettings.dreamModel). */
export function dreamRecipeOf(m: DreamModel | undefined): {
  dreamTurbo: boolean;
  dreamDistilled: boolean;
  dreamSteps: number;
} {
  if (!m) return { dreamTurbo: true, dreamDistilled: false, dreamSteps: 0 };
  return {
    dreamTurbo: m.turbo,
    dreamDistilled: m.distilled,
    dreamSteps: m.turbo ? 0 : m.steps,
  };
}

/** "Turbo · 8 steps", "Base · 30 steps", for a model's line. */
export function variantLabel(m: DreamModel): string {
  if (m.turbo) return `${m.steps} steps · Turbo available`;
  return `${m.distilled ? 'Distilled' : 'Base'} · ${m.steps} steps`;
}

// --- prompt guides -------------------------------------------------------------------------------------------
//
// A starting prompt and negative per architecture, written the way each model's card says it was trained to read:
// - MOSS-Image (Kromoss): a full natural-language description; text in "quotes" is rendered. Its pipeline's own
//   negative is empty, so the sample one is short and generic.
// - Krea 2: photographic captions in plain sentences (subject, place, light, colour, camera / film stock).
// - FLUX.2 klein: subject first, then action, setting, style and light, in precise prose; it is steered by what the
//   prompt says rather than by a negative, so that one stays minimal (and its distilled checkpoints ignore it).
// - Anima: the card's own prefix and negative: lowercase Danbooru tags with spaces (score tags keep underscores),
//   optionally followed by a natural-language sentence. Illustration only; it does not do photorealism.
// - SDXL: comma-separated phrases, the subject first, then style, light and quality words; CLIP reads about 75 tokens
//   per chunk and takes (word:1.2) weights. Checkpoints differ a lot, so the sample stays general.

export interface DreamPromptGuide {
  prompt: string;
  negative: string;
  /** One line on how to write for the model. */
  tip: string;
}

export const DREAM_PROMPT_GUIDES: Record<string, DreamPromptGuide> = {
  moss: {
    prompt:
      'A photograph of a red fox curled up asleep in fresh snow at the edge of a pine forest, early morning, soft golden backlight catching the frost on its fur, shallow depth of field, 85mm lens',
    negative: 'blurry, low quality, jpeg artifacts, watermark, deformed, extra limbs',
    tip: 'Describe the whole picture in sentences: subject, setting, style, lighting and camera. Text in "quotes" is drawn as lettering.',
  },
  krea2: {
    prompt:
      'Candid photograph of an elderly fisherman mending a bright orange net on a wooden pier at dawn, his weathered hands in sharp focus, mist over the harbour behind him, muted film colours, natural light, shot on 35mm film',
    negative: 'blurry, low quality, oversaturated, plastic skin, cgi, 3d render, watermark, text',
    tip: 'Write it like a photo caption in plain sentences: who and what, where, the light and colour, and the camera or film look. Krea 2 is at its best with photographic realism.',
  },
  flux2: {
    prompt:
      'A ceramic coffee cup on a sunlit wooden café table, steam rising from it, a folded newspaper beside it, warm morning light through the window, shallow depth of field, editorial food photography',
    negative: 'blurry, low quality, distorted, watermark',
    tip: 'Lead with the main subject, then the action, setting, style and lighting. FLUX.2 follows long, precise prose: describe what you want rather than what to avoid.',
  },
  anima: {
    prompt:
      'masterpiece, best quality, score_7, safe, 1girl, solo, silver hair, long hair, blue eyes, white hoodie, holding umbrella, rain, city street, night, neon lights, reflections. She looks up at the glowing shop signs.',
    negative: 'worst quality, low quality, score_1, score_2, score_3, artist name',
    tip: 'Start with "masterpiece, best quality, score_7, safe," then lowercase tags with spaces (not underscores), optionally a sentence after. Anime and illustration only: it does not do photorealism.',
  },
  sdxl: {
    prompt:
      'a lighthouse on a rocky cliff at sunset, crashing waves, dramatic clouds, warm golden light, highly detailed, sharp focus, cinematic composition',
    negative: 'lowres, blurry, worst quality, low quality, jpeg artifacts, watermark, signature, text, deformed',
    tip: 'Short comma-separated phrases, the subject first, then style, light and quality words. (word:1.2) makes a phrase count more. Fine-tuned checkpoints often expect their own tags: check the model card.',
  },
};

/** The guide for an architecture (MOSS-Image's for an unknown one). */
export function dreamPromptGuide(arch: string | undefined): DreamPromptGuide {
  return DREAM_PROMPT_GUIDES[arch || ''] || DREAM_PROMPT_GUIDES.moss;
}

const isSample = (text: string, key: 'prompt' | 'negative') => Object.values(DREAM_PROMPT_GUIDES).some((g) => g[key] === text.trim());

/**
 * What a model's guide changes: the prompt (when `withPrompt`) and the negative, each only while it is empty or
 * still another model's sample: the user's own text is never replaced.
 */
export function applyPromptGuide(s: { prompt: string; negative: string }, g: DreamPromptGuide, withPrompt: boolean): { prompt?: string; negative?: string } {
  const out: { prompt?: string; negative?: string } = {};
  if (withPrompt && (!s.prompt.trim() || isSample(s.prompt, 'prompt')) && s.prompt !== g.prompt) out.prompt = g.prompt;
  if ((!s.negative.trim() || isSample(s.negative, 'negative')) && s.negative !== g.negative) out.negative = g.negative;
  return out;
}

let shared: DreamCatalog = EMPTY;
let fetchedAt = 0;
let inflight: Promise<void> | null = null;
const listeners = new Set<(c: DreamCatalog) => void>();

function publish(c: DreamCatalog) {
  shared = c;
  listeners.forEach((l) => l(c));
}

async function load(authToken: string | undefined, refresh: boolean): Promise<void> {
  if (inflight && !refresh) return inflight;
  publish({ ...shared, loading: true, error: undefined });
  const ctl = new AbortController();
  const giveUp = window.setTimeout(() => ctl.abort(), refresh ? 90000 : 30000);
  const run = (async () => {
    try {
      const res = await fetch(`${mediaServerBase}/api/dream/models${refresh ? '?refresh=1' : ''}`, {
        // the library's token when the caller has it, else the viewer's stored credential (the Scene window)
        headers: authToken ? { Authorization: `Bearer ${authToken}` } : serverAuthHeaders(),
        credentials: isElectron ? 'omit' : 'include',
        signal: ctl.signal,
      });
      // an older server has no such endpoint: no list, the default model only
      if (res.status === 404) {
        publish({
          ...EMPTY,
          loaded: true,
          note: 'This server cannot list its image models yet; jobs use its default model.',
        });
      } else if (!res.ok) {
        publish({
          ...shared,
          loading: false,
          error: `The model list failed (${res.status})`,
        });
      } else {
        publish({
          ...parseDreamCatalog(await res.json()),
          loaded: true,
          loading: false,
        });
        fetchedAt = Date.now();
      }
    } catch (e) {
      publish({
        ...shared,
        loading: false,
        error: ctl.signal.aborted ? 'The model list took too long' : e instanceof Error ? e.message : String(e),
      });
    } finally {
      window.clearTimeout(giveUp);
      inflight = null;
    }
  })();
  inflight = run;
  return run;
}

/**
 * The shared catalog and a refresh (asks the engine to rescan its model folders). Nothing loads unless `enabled`;
 * without an `authToken` the viewer's stored server credential is used.
 */
export function useDreamModels(enabled: boolean, authToken?: string | null): { catalog: DreamCatalog; refresh: () => void } {
  const [catalog, setCatalog] = useState<DreamCatalog>(shared);
  useEffect(() => {
    listeners.add(setCatalog);
    setCatalog(shared);
    if (enabled && (!shared.loaded || Date.now() - fetchedAt > 60000)) load(authToken || undefined, false);
    return () => {
      listeners.delete(setCatalog);
    };
  }, [enabled, authToken]);
  const refresh = useCallback(() => {
    if (enabled) load(authToken || undefined, true);
  }, [enabled, authToken]);
  return { catalog, refresh };
}

/** Tests: forget the shared list. */
export function resetDreamModelsForTests() {
  shared = EMPTY;
  fetchedAt = 0;
  inflight = null;
}
