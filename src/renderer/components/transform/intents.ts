// Transform intents: what a user can ask the local AI engines to do with the
// selected media, and the pure logic that turns a choice + settings into job
// requests for the media server's `retouch` (loki-retouch, images) and
// `reshoot` (loki-reshoot, video) tasks. No React in here: it is unit-tested
// and shared by the palette's one-click chips and the Transform flow.

export type EngineId = 'retouch' | 'reshoot';
export type IntentId =
  | 'restore'
  | 'upscale'
  | 'wallpaper'
  | 'edit'
  | 'alive'
  | 'direct';

export type MediaKind = 'image' | 'video' | 'audio' | 'other';

const IMAGE_RE = /\.(jpe?g|jfif|png|webp|avif|bmp|tiff?|gif)$/i;
const VIDEO_RE = /\.(mp4|webm|mov|mkv|avi|m4v)$/i;
const AUDIO_RE = /\.(mp3|wav|flac|ogg|m4a|aac|opus)$/i;

export function mediaKind(path: string): MediaKind {
  if (IMAGE_RE.test(path)) return 'image';
  if (VIDEO_RE.test(path)) return 'video';
  if (AUDIO_RE.test(path)) return 'audio';
  return 'other';
}

export interface Inputs {
  images: string[];
  videos: string[];
  audios: string[];
  other: string[];
}

export function classify(paths: string[]): Inputs {
  const out: Inputs = { images: [], videos: [], audios: [], other: [] };
  paths.forEach((p) => {
    if (!p) return;
    const k = mediaKind(p);
    if (k === 'image') out.images.push(p);
    else if (k === 'video') out.videos.push(p);
    else if (k === 'audio') out.audios.push(p);
    else out.other.push(p);
  });
  return out;
}

export type Quality = 'draft' | 'balanced' | 'fine';
export type SizeMode = 'same' | 'scale' | 'custom';
export type Shake = 'none' | 'subtle' | 'handheld';
export type WallpaperTarget = 'desktop' | 'phone';

export interface TransformSettings {
  /** Free-form instruction (edit / direct). */
  prompt: string;
  /**
   * A hand-written replacement for the WHOLE prompt the engine would expand (a preset's built-in text, the
   * living-photo prompt), written in the review step. null = use the generated prompt. Never remembered.
   */
  manualPrompt: string | null;
  /** One sentence naming the subject (alive). */
  describe: string;
  /** Extra direction appended to a preset's built-in prompt. */
  append: string;
  sizeMode: SizeMode;
  /** Output size factor (sizeMode = scale). */
  scale: number;
  /** `WxH` (sizeMode = custom). */
  customSize: string;
  wallpaperTarget: WallpaperTarget;
  quality: Quality;
  seedMode: 'random' | 'fixed';
  seed: number;
  /** How many alternatives to queue (different seeds). */
  variations: number;
  shake: Shake;
  /** Seconds of video. */
  duration: number;
  refSize: 'match' | 'max';
  noAudio: boolean;
  keepSoundtrack: boolean;
}

export interface IntentDef {
  id: IntentId;
  engine: EngineId;
  group: 'image' | 'video';
  title: string;
  tagline: string;
  /** Longer explanation shown in the Shape phase. */
  blurb: string;
  /** Runs straight from the palette chip with remembered settings. */
  oneClick: boolean;
  /** Needs the user to type something before it can run. */
  needsText: 'prompt' | 'describe' | null;
  defaults: Partial<TransformSettings>;
  /** Example prompts shown as hints / "surprise me". */
  examples: string[];
  /** Examples for a multi-image selection (the images become references). */
  referenceExamples?: string[];
}

export const BASE_SETTINGS: TransformSettings = {
  prompt: '',
  manualPrompt: null,
  describe: '',
  append: '',
  sizeMode: 'same',
  scale: 2,
  customSize: '',
  wallpaperTarget: 'desktop',
  quality: 'balanced',
  seedMode: 'random',
  seed: 0,
  variations: 1,
  shake: 'subtle',
  duration: 5,
  refSize: 'match',
  noAudio: false,
  keepSoundtrack: true,
};

export const INTENTS: IntentDef[] = [
  {
    id: 'restore',
    engine: 'retouch',
    group: 'image',
    title: 'Restore',
    tagline: 'Clean up noise, blur and compression',
    blurb:
      'Faithful restoration at the original size: removes grain and compression artifacts and recovers natural detail without redrawing anything.',
    oneClick: true,
    needsText: null,
    defaults: { sizeMode: 'same' },
    examples: [],
  },
  {
    id: 'upscale',
    engine: 'retouch',
    group: 'image',
    title: 'Upscale',
    tagline: 'Real super-resolution, ratio kept',
    blurb:
      'Enlarges the image and adds the fine detail the low resolution lost, keeping every edge and contour exactly where it is.',
    oneClick: true,
    needsText: null,
    defaults: { sizeMode: 'scale', scale: 2 },
    examples: [],
  },
  {
    id: 'wallpaper',
    engine: 'retouch',
    group: 'image',
    title: 'Wallpaper',
    tagline: 'Restore and outpaint to fit a screen',
    blurb:
      'Restores the photo, then extends the scene beyond its borders to fill a 4K desktop (or vertical phone) canvas.',
    oneClick: true,
    needsText: null,
    defaults: { sizeMode: 'same', wallpaperTarget: 'desktop' },
    examples: [],
  },
  {
    id: 'edit',
    engine: 'retouch',
    group: 'image',
    title: 'Edit with words',
    tagline: 'Change anything by describing it',
    blurb:
      'Write what should change and the model edits the image, keeping the rest. It can also resize in the same pass. With several images selected the model sees them all as references: <image1> is the one edited, <image2>, <image3>… the others.',
    oneClick: false,
    needsText: 'prompt',
    defaults: { sizeMode: 'same' },
    examples: [
      'Turn it into a snowy winter day, keep the people and composition',
      'Make it night, with warm light coming from the windows',
      'Remove the person on the left and fill in the background',
      'Change the jacket to red leather',
    ],
    referenceExamples: [
      'Put the person from <image2> into the scene of <image1>',
      'Dress the person in <image1> in the jacket from <image2>',
      'Use the style of <image2> for <image1>',
    ],
  },
  {
    id: 'alive',
    engine: 'reshoot',
    group: 'video',
    title: 'Bring to life',
    tagline: 'A still becomes a living photo',
    blurb:
      'Natural ambient life, subtle resting movement and an optional hand-held camera feel, keeping identity and framing. One video per image.',
    oneClick: false,
    needsText: 'describe',
    defaults: { duration: 5, shake: 'subtle' },
    examples: [
      'the young woman taking a mirror selfie in a sunlit room',
      'a golden retriever lying on a porch at sunset',
      'a street scene in the rain at night',
    ],
  },
  {
    id: 'direct',
    engine: 'reshoot',
    group: 'video',
    title: 'Direct a scene',
    tagline: 'Images, clips and music in, video out',
    blurb:
      'Full control: reference images (<Picture 1>…), clips (<Video 1>…) and audio (<Audio 1>…) plus a prompt become one video with generated sound.',
    oneClick: false,
    needsText: 'prompt',
    defaults: { duration: 5 },
    examples: [
      'Animate <Picture 1> as one continuous shot. She dances to the music of <Audio 1>. Audio: rhythmic music from <Audio 1>.',
      'The person in <Picture 1> performs the motion of <Video 1>.',
    ],
  },
];

export function intentById(id: IntentId): IntentDef {
  return INTENTS.find((i) => i.id === id) as IntentDef;
}

/** Defaults for an intent, with any remembered overrides on top. */
export function settingsFor(
  id: IntentId,
  remembered?: Partial<TransformSettings> | null
): TransformSettings {
  return { ...BASE_SETTINGS, ...intentById(id).defaults, ...(remembered || {}) };
}

export interface Availability {
  ok: boolean;
  /** Why it is unavailable (shown on the greyed-out card). */
  reason?: string;
}

/** Whether an intent can run on this selection. */
export function availability(id: IntentId, inputs: Inputs): Availability {
  const { images, videos, audios } = inputs;
  const stills = images.length;
  switch (id) {
    case 'restore':
    case 'upscale':
    case 'wallpaper':
    case 'edit':
      if (stills > 0 || videos.length === 1) return { ok: true };
      return { ok: false, reason: 'Select an image' };
    case 'alive':
      if (stills > 0) return { ok: true };
      return { ok: false, reason: 'Select an image' };
    case 'direct':
      if (stills + videos.length + audios.length > 0) return { ok: true };
      return { ok: false, reason: 'Select images, clips or audio' };
    default:
      return { ok: false };
  }
}

/**
 * Whether a retouch intent shows the model every selected image at once (ONE run: the first is <image1>, the rest
 * references) instead of running once per image. Only a free-form edit does: presets are per-image treatments.
 */
export function usesReferences(id: IntentId, inputs: Inputs): boolean {
  return id === 'edit' && inputs.images.length > 1;
}

/** Whether the intent's prompt field IS the whole prompt the engine runs (nothing is expanded around it). */
export function promptIsVerbatim(id: IntentId): boolean {
  return id === 'edit' || id === 'direct';
}

/**
 * The reference token the model knows each input by: <image1>… for retouch, <Picture 1>/<Video 1>/<Audio 1> for
 * reshoot. Inputs a prompt cannot refer to map to '' (a preset's batch) or a plain label ('photo').
 * `manual` = the prompt is hand-written, so every input the model sees gets its token.
 */
export function referenceTokens(id: IntentId | null, paths: string[], manual = false): Record<string, string> {
  const out: Record<string, string> = {};
  if (!id) return out;
  const inputs = classify(paths);
  if (intentById(id).engine === 'retouch') {
    const targets = inputs.images.length ? inputs.images : inputs.videos.slice(0, 1);
    const shared = usesReferences(id, inputs);
    targets.forEach((p, i) => {
      // A per-image run sees only its own image, always as <image1>.
      if (id === 'edit' || manual) out[p] = shared ? `<image${i + 1}>` : '<image1>';
      else out[p] = '';
    });
    return out;
  }
  inputs.images.forEach((p, i) => {
    // Bring to life runs once per photo: each is that run's <Picture 1>.
    if (id === 'alive') out[p] = manual ? '<Picture 1>' : 'photo';
    else out[p] = `<Picture ${i + 1}>`;
  });
  inputs.videos.forEach((p, i) => {
    out[p] = `<Video ${i + 1}>`;
  });
  inputs.audios.forEach((p, i) => {
    out[p] = `<Audio ${i + 1}>`;
  });
  return out;
}

const TOKEN_RE = /<image\d+>|<(?:Picture|Video|Audio) \d+>/g;

/**
 * Rewrite the reference tokens in `text` after the inputs were reordered or removed, so every token keeps pointing
 * at the same file. Tokens of removed files are left as written (validation flags them).
 */
export function remapTokens(text: string, id: IntentId, before: string[], after: string[]): string {
  if (!text) return text;
  const was = referenceTokens(id, before, true);
  const now = referenceTokens(id, after, true);
  const map: Record<string, string> = {};
  Object.keys(was).forEach((p) => {
    if (was[p] && now[p]) map[was[p]] = now[p];
  });
  return text.replace(TOKEN_RE, (t) => map[t] || t);
}

export interface TokenIssues {
  /** Tokens in the prompt that match no input. */
  unknown: string[];
  /** Tokens of inputs the prompt never mentions. */
  unused: string[];
}

/** Compare the tokens a prompt uses with the ones its inputs have. */
export function tokenIssues(prompt: string, tokens: string[]): TokenIssues {
  const used = new Set(prompt.match(TOKEN_RE) || []);
  const known = new Set(tokens.filter((t) => t.startsWith('<')));
  return {
    unknown: Array.from(used).filter((t) => !known.has(t)),
    unused: Array.from(known).filter((t) => !used.has(t)),
  };
}

export const QUALITY_STEPS: Record<EngineId, Record<Quality, number>> = {
  retouch: { draft: 12, balanced: 25, fine: 40 },
  reshoot: { draft: 12, balanced: 20, fine: 32 },
};

export const MAX_VARIATIONS = 4;
export const MIN_DURATION = 1;
export const MAX_DURATION = 15;

export function stepsFor(engine: EngineId, q: Quality): number {
  return QUALITY_STEPS[engine][q];
}

/** `WxH` → [w, h] (positive integers) or null. */
export function parseSize(s: string): [number, number] | null {
  const m = /^\s*(\d{2,5})\s*[x×X]\s*(\d{2,5})\s*$/.exec(s || '');
  if (!m) return null;
  const w = parseInt(m[1], 10);
  const h = parseInt(m[2], 10);
  return w > 0 && h > 0 ? [w, h] : null;
}

// --- native canvas (mirrors loki-reshoot's `--native`) -----------------------

const SUPPORTED_RATIOS: Array<[string, number]> = [
  ['1:1', 1],
  ['4:3', 4 / 3],
  ['3:4', 3 / 4],
  ['16:9', 16 / 9],
  ['9:16', 9 / 16],
];
const CANVAS_MAX_PIXELS = 768 * 1344;
const r32 = (x: number) => Math.max(32, Math.round(x / 32) * 32);

export function nativeCanvas(
  w: number,
  h: number
): { label: string; width: number; height: number } {
  const a = w / h;
  let best = SUPPORTED_RATIOS[0];
  SUPPORTED_RATIOS.forEach((r) => {
    if (Math.abs(Math.log(a / r[1])) < Math.abs(Math.log(a / best[1])))
      best = r;
  });
  const ratio = best[1];
  let [nw, nh] = ratio >= 1 ? [768 * ratio, 768] : [768, 768 / ratio];
  if (nw * nh > CANVAS_MAX_PIXELS) {
    const s = Math.sqrt(CANVAS_MAX_PIXELS / (nw * nh));
    nw *= s;
    nh *= s;
  }
  return { label: best[0], width: r32(nw), height: r32(nh) };
}

/** Frame count loki-reshoot generates for a duration (snaps up to 17k+5 @ 24 fps). */
export function reshootFrames(seconds: number): number {
  let n = Math.max(5, Math.round(seconds * 24));
  while (n % 17 !== 5) n += 1;
  return n;
}

// --- job building -----------------------------------------------------------

export interface JobRequest {
  /** First token = task id, last (quoted) token = newline-joined input paths. */
  input: string;
  /** Verbatim `--key value` options (empty values are dropped by the server). */
  fields: Record<string, string>;
  /** Human label for toasts / review. */
  label: string;
}

export interface BuildContext {
  /** Seconds into a single video input to sample (retouch on video). */
  videoTime?: number;
  /** Seed source for 'random' mode (injectable for tests). */
  random?: () => number;
}

const q = (paths: string[]) => `"${paths.join('\n')}"`;

function seeds(s: TransformSettings, ctx: BuildContext): number[] {
  const n = Math.min(MAX_VARIATIONS, Math.max(1, Math.round(s.variations)));
  const rnd = ctx.random || Math.random;
  const base =
    s.seedMode === 'fixed' ? Math.max(0, Math.floor(s.seed)) : Math.floor(rnd() * 2147483647);
  return Array.from({ length: n }, (_, i) => (base + i * 7919) % 2147483647);
}

function put(
  f: Record<string, string>,
  key: string,
  value: string | number | boolean | undefined
) {
  if (value === undefined || value === false || value === '') return;
  f[key] = value === true ? '1' : String(value);
}

/** The retouch preset for an intent + settings ('' = free-form prompt). */
export function retouchPreset(id: IntentId, s: TransformSettings): string {
  if (id === 'restore') return 'restore';
  if (id === 'upscale') return 'upscale';
  if (id === 'wallpaper') return s.wallpaperTarget === 'phone' ? '4kify-phone' : '4kify';
  return '';
}

export interface BuildResult {
  jobs: JobRequest[];
  error?: string;
}

/** Turn an intent + settings + selected paths into the jobs to queue. */
export function buildJobs(
  id: IntentId,
  s: TransformSettings,
  paths: string[],
  ctx: BuildContext = {}
): BuildResult {
  const def = intentById(id);
  const inputs = classify(paths);
  const avail = availability(id, inputs);
  if (!avail.ok) return { jobs: [], error: avail.reason || 'Not available' };
  const steps = stepsFor(def.engine, s.quality);
  const seedList = seeds(s, ctx);
  const jobs: JobRequest[] = [];

  if (def.engine === 'retouch') {
    const manual = s.manualPrompt !== null ? s.manualPrompt.trim() : null;
    const prompt = manual ?? s.prompt.trim();
    if (manual === '') return { jobs: [], error: 'Write the prompt, or reset it to the generated one' };
    if (id === 'edit' && !prompt) return { jobs: [], error: 'Describe what should change' };
    // Stills, or the single video (a frame is sampled from it).
    const targets = inputs.images.length > 0 ? inputs.images : inputs.videos.slice(0, 1);
    const shared = usesReferences(id, inputs);
    seedList.forEach((seed) => {
      const f: Record<string, string> = {};
      // A manual prompt replaces the preset's text; the preset still sets the size and reference handling.
      put(f, 'preset', retouchPreset(id, s));
      if (id === 'edit' || manual !== null) put(f, 'prompt', prompt);
      else put(f, 'append', s.append.trim());
      if (id !== 'wallpaper') {
        if (s.sizeMode === 'scale' && s.scale > 0 && s.scale !== 1) put(f, 'scale', s.scale);
        if (s.sizeMode === 'custom' && parseSize(s.customSize)) put(f, 'size', s.customSize.replace(/\s+/g, '').replace(/[×X]/, 'x'));
      }
      put(f, 'steps', steps);
      put(f, 'seed', seed);
      // One run: targets[0] is <image1>, the others ride along as <image2>… (loki-retouch --ref).
      if (shared) put(f, 'combine', true);
      if (ctx.videoTime && inputs.images.length === 0) put(f, 'time', ctx.videoTime.toFixed(3));
      jobs.push({
        input: `retouch ${q(targets)}`,
        fields: f,
        label: shared
          ? `${def.title} · ${targets.length} images as references`
          : `${def.title} · ${targets.length} ${targets.length === 1 ? 'item' : 'items'}`,
      });
    });
    return { jobs };
  }

  // reshoot
  const manual = s.manualPrompt !== null ? s.manualPrompt.trim() : null;
  if (manual === '') return { jobs: [], error: 'Write the prompt, or reset it to the generated one' };
  if (id === 'direct' && !(manual ?? s.prompt.trim()))
    return { jobs: [], error: 'Describe the scene' };
  const common = (f: Record<string, string>, seed: number) => {
    put(f, 'duration', Math.min(MAX_DURATION, Math.max(MIN_DURATION, s.duration)));
    put(f, 'steps', steps);
    put(f, 'seed', seed);
    if (s.refSize === 'max') put(f, 'refsize', 'max');
    put(f, 'noaudio', s.noAudio);
    put(f, 'novideoaudio', !s.keepSoundtrack);
  };
  if (id === 'alive') {
    inputs.images.slice(0, 12).forEach((img) => {
      seedList.forEach((seed) => {
        const f: Record<string, string> = {};
        if (manual !== null) {
          // The whole prompt is hand-written: no living-photo expansion, the photo is a plain <Picture 1>.
          put(f, 'prompt', manual);
        } else {
          put(f, 'animate', true);
          put(f, 'describe', s.describe.trim());
          put(f, 'shake', s.shake);
          put(f, 'prompt', s.prompt.trim());
        }
        common(f, seed);
        jobs.push({ input: `reshoot ${q([img])}`, fields: f, label: `Bring to life · ${img.split(/[\\/]/).pop()}` });
      });
    });
    return { jobs };
  }
  const all = [...inputs.images, ...inputs.videos, ...inputs.audios];
  seedList.forEach((seed) => {
    const f: Record<string, string> = {};
    put(f, 'prompt', manual ?? s.prompt.trim());
    common(f, seed);
    jobs.push({ input: `reshoot ${q(all)}`, fields: f, label: `Direct · ${all.length} references` });
  });
  return { jobs };
}

// --- plan (what the review phase shows) -------------------------------------

export interface Plan {
  jobCount: number;
  /** Files that will be produced (images x variations; one video per job). */
  outputs: number;
  /** e.g. "photo_up.png" */
  outputName: string;
  /** Output dimensions when they can be derived. */
  outWidth?: number;
  outHeight?: number;
  frames?: number;
  engine: EngineId;
  steps: number;
}

const SUFFIX: Record<string, string> = {
  '': '_edit',
  restore: '_restored',
  upscale: '_up',
  '4kify': '_4k',
  '4kify-phone': '_phone',
};

function stem(p: string): string {
  const base = p.split(/[\\/]/).pop() || p;
  const dot = base.lastIndexOf('.');
  return dot > 0 ? base.slice(0, dot) : base;
}

export function planFor(
  id: IntentId,
  s: TransformSettings,
  paths: string[],
  source?: { width: number; height: number } | null
): Plan {
  const def = intentById(id);
  const inputs = classify(paths);
  const steps = stepsFor(def.engine, s.quality);
  const first = inputs.images[0] || inputs.videos[0] || inputs.audios[0] || '';
  const built = buildJobs(id, s, paths, { random: () => 0.5 });
  const jobCount = built.jobs.length;
  const variations = Math.min(MAX_VARIATIONS, Math.max(1, Math.round(s.variations)));
  const outputs =
    def.engine === 'retouch'
      ? (usesReferences(id, inputs) ? 1 : Math.max(1, inputs.images.length || 1)) * variations
      : jobCount;
  if (def.engine === 'reshoot') {
    const canvas = source ? nativeCanvas(source.width, source.height) : undefined;
    return {
      jobCount,
      outputs,
      outputName: `${stem(first)}_reshoot.mp4`,
      outWidth: canvas?.width,
      outHeight: canvas?.height,
      frames: reshootFrames(s.duration),
      engine: 'reshoot',
      steps,
    };
  }
  const preset = retouchPreset(id, s);
  const out: Plan = { jobCount, outputs, outputName: `${stem(first)}${SUFFIX[preset] || '_edit'}.png`, engine: 'retouch', steps };
  if (id === 'wallpaper') {
    const [w, h] = s.wallpaperTarget === 'phone' ? [1296, 2800] : [3840, 2160];
    out.outWidth = w;
    out.outHeight = h;
  } else if (source) {
    if (s.sizeMode === 'scale') {
      out.outWidth = Math.round(source.width * s.scale);
      out.outHeight = Math.round(source.height * s.scale);
    } else if (s.sizeMode === 'custom' && parseSize(s.customSize)) {
      const [w, h] = parseSize(s.customSize) as [number, number];
      out.outWidth = w;
      out.outHeight = h;
    } else {
      out.outWidth = source.width;
      out.outHeight = source.height;
    }
  }
  return out;
}

// --- time estimate -----------------------------------------------------------

/** Rough wall-clock seconds for ONE job on an RTX 4090-class GPU (fitted to measured runs). */
export function estimateSeconds(plan: Plan): number {
  if (plan.engine === 'retouch') {
    const mp = plan.outWidth && plan.outHeight ? (plan.outWidth * plan.outHeight) / 1e6 : 1;
    return 12 + plan.steps * 0.4 * mp;
  }
  const w = plan.outWidth || 768;
  const h = plan.outHeight || 1344;
  const latentT = Math.floor(((plan.frames || 124) - 5) / 17) * 5 + 2;
  const tokens = latentT * (h / 32) * (w / 32);
  const step = 0.655 + 1.419e-4 * tokens + 3.846e-9 * tokens * tokens;
  return 35 + plan.steps * step + 8 + tokens * 0.0003;
}

export function formatDuration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 90) return `${Math.max(5, Math.round(s / 5) * 5)} s`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}
