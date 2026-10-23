// Transform intents: what a user can ask the local AI engines to do with the
// selected media (or, for Generate, with nothing at all), and the pure logic
// that turns a choice + settings into job requests for the media server's
// `retouch` (loki-retouch, images), `reshoot` (loki-reshoot, video) and
// `dream` (loki-dream, text -> image) tasks. No React in here: it is
// unit-tested and shared by the palette's one-click chips and the Transform flow.

export type EngineId = 'retouch' | 'reshoot' | 'dream';
export type IntentId =
  | 'restore'
  | 'upscale'
  | 'wallpaper'
  | 'angle'
  | 'edit'
  | 'alive'
  | 'direct'
  | 'generate'
  | 'reimagine';

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
/**
 * Which loki-reshoot model runs a video: `h3` (MiniMax H3, the engine's default; Quality picks its steps) or `fast`
 * (FastH3 Omni, distilled for exactly FAST_STEPS steps: ~4-5x faster sampling with sparse attention, maybe lower quality).
 */
export type ReshootModel = 'h3' | 'fast';
export type SizeMode = 'same' | 'scale' | 'custom';
export type Shake = 'none' | 'subtle' | 'handheld';
export type WallpaperTarget = 'desktop' | 'phone';
export type CameraElevation = 'eye' | 'high' | 'low';
/** Generate: the picture's shape (DREAM_ASPECTS), or a custom `WxH` (customSize). */
export type DreamAspect = 'square' | 'landscape' | 'portrait' | 'custom';

/**
 * The camera-angle presets (media-server/tasks/retouch_views.go): an orbit in 45° steps, measured toward the right
 * of the original frame (0 = where the photo was taken from, 180 = directly behind), at eye level, high or low.
 * The ids must match the server's.
 */
export const CAMERA_AZIMUTHS: Array<{ az: number; id: string; label: string }> = [
  { az: 0, id: 'front', label: 'Front' },
  { az: 45, id: 'right-45', label: '45° right' },
  { az: 90, id: 'right-90', label: '90° right' },
  { az: 135, id: 'right-135', label: '135° right' },
  { az: 180, id: 'reverse', label: 'Reverse' },
  { az: 225, id: 'left-135', label: '135° left' },
  { az: 270, id: 'left-90', label: '90° left' },
  { az: 315, id: 'left-45', label: '45° left' },
];

/** Snap any angle to the nearest orbit position (0…315). */
export function snapAzimuth(deg: number): number {
  return ((Math.round(deg / 45) * 45) % 360 + 360) % 360;
}

function azimuthEntry(az: number) {
  return CAMERA_AZIMUTHS.find((a) => a.az === snapAzimuth(az)) || CAMERA_AZIMUTHS[0];
}

/** The retouch preset for a camera position; '' for the original view (front, eye level). */
export function cameraViewPreset(az: number, elev: CameraElevation): string {
  const a = azimuthEntry(az);
  if (elev === 'eye') return a.az === 0 ? '' : a.id;
  return `${a.id}-${elev}`;
}

/** e.g. "Reverse, high angle"; "Original view" for front at eye level. */
export function cameraViewLabel(az: number, elev: CameraElevation): string {
  const a = azimuthEntry(az);
  if (elev === 'eye') return a.az === 0 ? 'Original view' : a.label;
  return `${a.label}, ${elev} angle`;
}

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
  /** New angle: camera orbit in degrees toward the right of the original frame (multiples of 45). */
  cameraAzimuth: number;
  /** New angle: camera height. */
  cameraElevation: CameraElevation;
  quality: Quality;
  seedMode: 'random' | 'fixed';
  seed: number;
  /**
   * Pinned references (Edit, Direct): files kept from one flow to the next that ride along with whatever is
   * selected, after it, so a saved prompt can name them by the same tokens every time. See runGroups.
   */
  pinned: string[];
  /** How many alternatives to queue (different seeds). */
  variations: number;
  shake: Shake;
  /** Seconds of video. */
  duration: number;
  refSize: 'match' | 'max';
  /** Video model (reshoot only). */
  reshootModel: ReshootModel;
  noAudio: boolean;
  keepSoundtrack: boolean;
  /** What the picture should not have (Generate, Reimagine; the turbo quality ignores it). */
  negative: string;
  /** Generate: the picture's shape. */
  aspect: DreamAspect;
  /** Reimagine: how much of the image is redrawn, 0-1. */
  strength: number;
}

export interface IntentDef {
  id: IntentId;
  engine: EngineId;
  /** The Choose step's section: edit images, make video, or make new pictures. */
  group: 'image' | 'video' | 'create';
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
  cameraAzimuth: 180,
  cameraElevation: 'eye',
  quality: 'balanced',
  seedMode: 'random',
  seed: 0,
  variations: 1,
  shake: 'subtle',
  duration: 5,
  refSize: 'match',
  reshootModel: 'h3',
  noAudio: false,
  keepSoundtrack: true,
  pinned: [],
  negative: '',
  aspect: 'square',
  strength: 0.8,
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
    id: 'angle',
    engine: 'retouch',
    group: 'image',
    title: 'New angle',
    tagline: 'Re-shoot the subject from another camera position',
    blurb:
      'Moves the camera around the subject — a full turn in 45° steps, at eye level, from above or from below — while the subject keeps exactly the same spot and pose. The model paints the sides and background the new viewpoint reveals.',
    oneClick: false,
    needsText: null,
    defaults: { sizeMode: 'same', cameraAzimuth: 180, cameraElevation: 'eye' },
    examples: [],
  },
  {
    id: 'edit',
    engine: 'retouch',
    group: 'image',
    title: 'Edit',
    tagline: 'Describe a change and the model paints it in',
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
    title: 'Animate',
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
    title: 'Direct',
    tagline: 'Direct a scene: images, clips and music in, video out',
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
  {
    id: 'generate',
    engine: 'dream',
    group: 'create',
    title: 'Generate',
    tagline: 'A new picture from a description',
    blurb:
      'Describe a picture and the model paints it from scratch: no image needed. The results are added to your library, in a Generated folder; anything selected is left alone.',
    oneClick: false,
    needsText: 'prompt',
    defaults: { aspect: 'square' },
    examples: [
      'A lighthouse on a rocky coast at dusk, waves breaking on the rocks, dramatic clouds, photograph',
      'A cozy reading nook with a cat asleep on a stack of books, warm afternoon light through the window',
      'Portrait of an old fisherman with a weathered face and a knit cap, 85mm lens, shallow depth of field',
      'A neon-lit street market in the rain at night, cinematic',
    ],
  },
  {
    id: 'reimagine',
    engine: 'dream',
    group: 'create',
    title: 'Reimagine',
    tagline: 'Redraw an image from a description',
    blurb:
      'The model repaints the image toward your description, keeping as much of the original as the strength allows: subtle keeps the colors and detail, strong little more than the layout. Describe the whole result, not just the change. One new image per image, saved beside it.',
    oneClick: false,
    needsText: 'prompt',
    defaults: { strength: 0.8 },
    examples: [
      'The same scene as an oil painting with visible brush strokes',
      'The same scene in deep winter, everything covered in fresh snow',
      'A watercolor illustration of this scene in soft pastel colors',
    ],
  },
];

export function intentById(id: IntentId): IntentDef {
  return INTENTS.find((i) => i.id === id) as IntentDef;
}

/**
 * The two kinds of intent, named the same everywhere in the UI: a preset runs a built-in prompt (you can add to it,
 * or replace it in Review); a custom one runs the prompt you write. Prompts you save are a third, "saved", kind of
 * starting point for custom intents (store.ts).
 */
export type IntentKind = 'preset' | 'custom';

export function intentKind(id: IntentId): IntentKind {
  return intentById(id).needsText === 'prompt' ? 'custom' : 'preset';
}

/** A name for a new saved prompt: its first few words. */
export function suggestTemplateName(prompt: string): string {
  const flat = prompt.replace(/\s+/g, ' ').trim();
  if (flat.length <= 32) return flat;
  const cut = flat.slice(0, 32);
  const sp = cut.lastIndexOf(' ');
  return `${(sp > 12 ? cut.slice(0, sp) : cut).replace(/[.,;:]+$/, '')}…`;
}

function quoteSnippet(text: string, max = 48): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return `“${flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat}”`;
}

/** One line describing a run's configuration, for "Last run" (what it did and how, not which files). */
export function runSummary(id: IntentId, s: TransformSettings): string {
  const parts: string[] = [];
  const manual = s.manualPrompt !== null && s.manualPrompt.trim() ? s.manualPrompt : null;
  if (manual) parts.push(`your prompt ${quoteSnippet(manual)}`);
  else if (promptIsVerbatim(id)) parts.push(quoteSnippet(s.prompt));
  else if (id === 'alive' && s.describe.trim()) parts.push(quoteSnippet(s.describe));
  else if (s.append.trim()) parts.push(`+ ${quoteSnippet(s.append, 32)}`);
  if (id === 'angle') parts.unshift(cameraViewLabel(s.cameraAzimuth, s.cameraElevation));
  const engine = intentById(id).engine;
  if (id === 'wallpaper') parts.push(s.wallpaperTarget === 'phone' ? 'phone' : 'desktop');
  else if (engine === 'retouch') {
    if (s.sizeMode === 'scale') parts.push(`${s.scale}×`);
    else if (s.sizeMode === 'custom' && parseSize(s.customSize)) parts.push(s.customSize);
  } else if (id === 'generate') {
    const size = dreamSize(s);
    if (size) parts.push(`${size[0]}×${size[1]}`);
  } else if (id === 'reimagine') parts.push(`strength ${strengthOf(s)}`);
  else parts.push(`${s.duration.toFixed(1).replace(/\.0$/, '')} s`);
  parts.push(engine === 'reshoot' && s.reshootModel === 'fast' ? 'fast model' : qualityLabel(engine, s.quality));
  if (s.variations > 1) parts.push(`${s.variations} variations`);
  if (s.seedMode === 'fixed') parts.push(`seed ${s.seed}`);
  const refs = pinnable(id) ? (s.pinned || []).length : 0;
  if (refs) parts.push(`${refs} ${refs === 1 ? 'reference' : 'references'}`);
  return parts.join(' · ');
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
    case 'angle':
    case 'edit':
      if (stills > 0 || videos.length === 1) return { ok: true };
      return { ok: false, reason: 'Select an image' };
    case 'alive':
      if (stills > 0) return { ok: true };
      return { ok: false, reason: 'Select an image' };
    case 'direct':
      if (stills + videos.length + audios.length > 0) return { ok: true };
      return { ok: false, reason: 'Select images, clips or audio' };
    case 'generate':
      // Needs nothing: whatever is selected is left alone.
      return { ok: true };
    case 'reimagine':
      if (stills > 0) return { ok: true };
      return { ok: false, reason: 'Select an image' };
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

/** Whether the intent can keep pinned references: the custom ones that take references (Edit images, Direct a video). */
export function pinnable(id: IntentId | null): boolean {
  return id === 'edit' || id === 'direct';
}

/** The pins that count for an intent: Edit references are images; Direct takes any image, clip or audio. */
export function activePins(id: IntentId | null, pinned: string[] | undefined): string[] {
  if (!pinnable(id) || !pinned || !pinned.length) return [];
  return id === 'edit' ? classify(pinned).images : pinned.filter((p) => mediaKind(p) !== 'other');
}

/**
 * With pinned references in play, the runs to make: one per selected item, each `[target, ...pins]`, so the
 * selected item is always first (<image1>, or the first of its kind for Direct) and the pins follow it in pinned
 * order. A selected file that is also pinned counts as a pin. Direct with nothing but pins selected is one run of
 * the pins. null = no pins (the selection alone decides, as before).
 */
export function runGroups(id: IntentId, paths: string[], pinned: string[] | undefined): string[][] | null {
  const pins = activePins(id, pinned);
  if (!pins.length) return null;
  const rest = paths.filter((p) => !pins.includes(p));
  if (id === 'edit') return classify(rest).images.map((t) => [t, ...pins]);
  const targets = rest.filter((p) => mediaKind(p) !== 'other');
  return targets.length ? targets.map((t) => [t, ...pins]) : [pins];
}

/**
 * Every reference a run used besides the item it was for: the pins, or (unpinned) the inputs that rode along with
 * the first one — Edit's other images (one combined run, <image2>…), Direct's other media. As pins, in this order,
 * they keep their tokens on any new selection, so a remembered run brings all of them back.
 */
export function runReferences(id: IntentId, paths: string[], pinned: string[] | undefined): string[] {
  if (!pinnable(id)) return [];
  const pins = activePins(id, pinned);
  if (pins.length) return pins;
  if (id === 'edit') return classify(paths).images.slice(1);
  return paths.filter((p) => mediaKind(p) !== 'other').slice(1);
}

/** Whether the intent's prompt field IS the whole prompt the engine runs (nothing is expanded around it). */
export function promptIsVerbatim(id: IntentId): boolean {
  return id === 'edit' || id === 'direct' || id === 'generate' || id === 'reimagine';
}

/** Whether the intent takes the selection at all (Generate makes something new and leaves it alone). */
export function usesSelection(id: IntentId): boolean {
  return id !== 'generate';
}

/**
 * The reference token the model knows each input by: <image1>… for retouch, <Picture 1>/<Video 1>/<Audio 1> for
 * reshoot. Inputs a prompt cannot refer to map to '' (a preset's batch) or a plain label ('photo').
 * `manual` = the prompt is hand-written, so every input the model sees gets its token.
 */
export function referenceTokens(id: IntentId | null, paths: string[], manual = false, pinned: string[] = []): Record<string, string> {
  const out: Record<string, string> = {};
  if (!id) return out;
  const groups = runGroups(id, paths, pinned);
  if (groups) {
    // Every run's target has the same token (it is first in its own run); the pins' come from the first run.
    groups.forEach((g, i) => {
      const t = referenceTokens(id, g, manual);
      if (i === 0) Object.assign(out, t);
      else out[g[0]] = t[g[0]];
    });
    // Edit with no image to edit yet: the pins keep the tokens they will have once one is selected.
    if (!groups.length) activePins(id, pinned).forEach((p, i) => (out[p] = `<image${i + 2}>`));
    return out;
  }
  const inputs = classify(paths);
  if (id === 'generate') return out;
  if (intentById(id).engine === 'dream') {
    // Reimagine: each image is the starting point of its own run; the prompt has no way to name it.
    inputs.images.forEach((p) => {
      out[p] = '';
    });
    return out;
  }
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
    // Animate runs once per photo: each is that run's <Picture 1>.
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
export function remapTokens(
  text: string,
  id: IntentId,
  before: string[],
  after: string[],
  pinsBefore: string[] = [],
  pinsAfter: string[] = pinsBefore
): string {
  if (!text) return text;
  const was = referenceTokens(id, before, true, pinsBefore);
  const now = referenceTokens(id, after, true, pinsAfter);
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
  // Kromoss: 50 steps at CFG 3 is the model's own setting (Fine); draft = the 8-step turbo LoRA at CFG 1.
  dream: { draft: 8, balanced: 30, fine: 50 },
};

/** What a quality is called: loki-dream's draft is its turbo mode. */
export function qualityLabel(engine: EngineId, q: Quality): string {
  return engine === 'dream' && q === 'draft' ? 'turbo' : q;
}

export const MAX_VARIATIONS = 4;
/** loki-dream makes up to this many images in one run (consecutive seeds, one model load). */
export const MAX_DREAM_COUNT = 8;

/** How many variations an intent can queue: Generate's are one run of the engine, so it allows more. */
export function maxVariations(id: IntentId): number {
  return id === 'generate' ? MAX_DREAM_COUNT : MAX_VARIATIONS;
}
export const MIN_DURATION = 1;
export const MAX_DURATION = 15;

export function stepsFor(engine: EngineId, q: Quality): number {
  return QUALITY_STEPS[engine][q];
}

/** The fast video model (FastH3 Omni) is distilled for exactly this many steps, whatever the quality. */
export const FAST_STEPS = 8;

/** The step count a run of these settings uses: the quality's, except the fast video model's fixed count. */
export function stepsForSettings(engine: EngineId, s: Pick<TransformSettings, 'quality' | 'reshootModel'>): number {
  return engine === 'reshoot' && s.reshootModel === 'fast' ? FAST_STEPS : stepsFor(engine, s.quality);
}

/** `WxH` → [w, h] (positive integers) or null. */
export function parseSize(s: string): [number, number] | null {
  const m = /^\s*(\d{2,5})\s*[x×X]\s*(\d{2,5})\s*$/.exec(s || '');
  if (!m) return null;
  const w = parseInt(m[1], 10);
  const h = parseInt(m[2], 10);
  return w > 0 && h > 0 ? [w, h] : null;
}

// --- loki-dream canvas ----------------------------------------------------------

/** Generate's shapes: Kromoss's ~1 MP training sizes. */
export const DREAM_ASPECTS: Array<{ id: Exclude<DreamAspect, 'custom'>; label: string; width: number; height: number }> = [
  { id: 'square', label: 'Square', width: 1024, height: 1024 },
  { id: 'landscape', label: 'Landscape', width: 1344, height: 768 },
  { id: 'portrait', label: 'Portrait', width: 768, height: 1344 },
];

/** The model works in multiples of DREAM_GRID px (its 16x VAE), around a megapixel. */
export const DREAM_GRID = 16;
export const DREAM_MAX_PIXELS = 1024 * 1024;
const DREAM_MIN_SIDE = 256;

/** Snap a custom size to what the model makes: multiples of DREAM_GRID, at most ~DREAM_MAX_PIXELS, ratio kept. */
export function snapDreamSize(w: number, h: number): [number, number] {
  let fw = Math.max(1, w);
  let fh = Math.max(1, h);
  if (fw * fh > DREAM_MAX_PIXELS) {
    const k = Math.sqrt(DREAM_MAX_PIXELS / (fw * fh));
    fw *= k;
    fh *= k;
  }
  const snap = (x: number) => Math.max(DREAM_MIN_SIDE, Math.round(x / DREAM_GRID) * DREAM_GRID);
  let sw = snap(fw);
  let sh = snap(fh);
  // Rounding both sides up can push the area just over the cap: take the larger side down a notch.
  while (sw * sh > DREAM_MAX_PIXELS && Math.max(sw, sh) > DREAM_MIN_SIDE) {
    if (sw >= sh) sw -= DREAM_GRID;
    else sh -= DREAM_GRID;
  }
  return [sw, sh];
}

/** The size Generate makes for these settings, or null for a custom size that is not a size yet. */
export function dreamSize(s: Pick<TransformSettings, 'aspect' | 'customSize'>): [number, number] | null {
  if (s.aspect === 'custom') {
    const p = parseSize(s.customSize);
    return p ? snapDreamSize(p[0], p[1]) : null;
  }
  const a = DREAM_ASPECTS.find((x) => x.id === s.aspect) || DREAM_ASPECTS[0];
  return [a.width, a.height];
}

/**
 * Reimagine's strength presets. Kromoss holds on to the picture far longer than the number suggests: at 0.6 little
 * more than a detail changes, 0.8 restyles colors and light, and from about 0.88 only the layout is left (turbo's
 * 8 steps make 0.88-0.97 the same run).
 */
export const STRENGTH_PRESETS: Array<{ value: number; label: string; sub: string }> = [
  { value: 0.6, label: 'Subtle', sub: 'keeps colors and detail' },
  { value: 0.8, label: 'Medium', sub: 'keeps the composition' },
  { value: 0.9, label: 'Strong', sub: 'keeps the layout' },
];

/** Reimagine's strength, in range (0.05-1) and rounded to the slider's hundredths. */
export function strengthOf(s: Pick<TransformSettings, 'strength'>): number {
  const v = Number.isFinite(s.strength) ? s.strength : BASE_SETTINGS.strength;
  return Math.round(Math.min(1, Math.max(0.05, v)) * 100) / 100;
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
  /**
   * Stamp the run's recipe into each job (`meta`, see recipeJson) so its results can be saved as a task again.
   * true = these settings; the pinned-references pass hands down the settings as the user set them (pins included).
   */
  recipe?: true | TransformSettings;
}

/** `kind` of the recipe the Transform flow stamps into its results (the generation record's `meta`). */
export const RECIPE_KIND = 'loki-transform';
export const RECIPE_VERSION = 1;

/**
 * The recipe one job stamps into its results: the intent, every setting as the user set them, and the job's inputs in
 * order (the first is the item the run was for; recipe.ts turns the rest into pinned references). See recipe.ts.
 */
export function recipeJson(id: IntentId, s: TransformSettings, paths: string[]): string {
  return JSON.stringify({ kind: RECIPE_KIND, v: RECIPE_VERSION, intent: id, settings: s, paths });
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
  if (id === 'angle') return cameraViewPreset(s.cameraAzimuth, s.cameraElevation);
  return '';
}

export interface BuildResult {
  jobs: JobRequest[];
  error?: string;
}

/** New angle with the camera still where the photo was taken from. */
export const SAME_VIEW = 'Move the camera: this is the view the photo already has';

/** Edit with pins but no image of its own selected. */
export const NO_TARGET = 'Select the image to edit: the pinned references ride along with it';

/** Turn an intent + settings + selected paths into the jobs to queue. */
export function buildJobs(
  id: IntentId,
  s: TransformSettings,
  paths: string[],
  ctx: BuildContext = {}
): BuildResult {
  const def = intentById(id);
  const groups = runGroups(id, paths, s.pinned);
  if (groups) {
    // Pinned references: one run per selected item, each with the pins after it.
    if (!groups.length) return { jobs: [], error: NO_TARGET };
    const pins = activePins(id, s.pinned).length;
    const out: JobRequest[] = [];
    for (const g of groups) {
      const b = buildJobs(id, { ...s, pinned: [] }, g, ctx.recipe === true ? { ...ctx, recipe: s } : ctx);
      if (b.error) return b;
      b.jobs.forEach((j) => out.push({ ...j, label: `${def.title} · ${g[0].split(/[\\/]/).pop()} + ${pins} pinned` }));
    }
    return { jobs: out };
  }
  const inputs = classify(paths);
  const avail = availability(id, inputs);
  if (!avail.ok) return { jobs: [], error: avail.reason || 'Not available' };
  const dreamProblem = def.engine === 'dream' ? dreamError(id, s) : null;
  if (dreamProblem) return { jobs: [], error: dreamProblem };
  const steps = stepsForSettings(def.engine, s);
  const seedList = seeds(s, ctx);
  const jobs: JobRequest[] = [];
  const recipe = ctx.recipe === true ? s : ctx.recipe;
  const stamp = (f: Record<string, string>, runPaths: string[]) => {
    if (recipe) f.meta = recipeJson(id, recipe, runPaths);
  };

  if (def.engine === 'dream') return { jobs: buildDreamJobs(id, s, inputs, ctx, stamp) };

  if (def.engine === 'retouch') {
    const manual = s.manualPrompt !== null ? s.manualPrompt.trim() : null;
    const prompt = manual ?? s.prompt.trim();
    if (manual === '') return { jobs: [], error: 'Write the prompt, or reset it to the generated one' };
    if (id === 'edit' && !prompt) return { jobs: [], error: 'Describe what should change' };
    if (id === 'angle' && !retouchPreset(id, s) && manual === null) return { jobs: [], error: SAME_VIEW };
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
      stamp(f, targets);
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
    // Only the fast model is named: H3 is the engine's default, so its jobs run on engines from before --model.
    if (s.reshootModel === 'fast') put(f, 'model', 'fast');
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
        stamp(f, [img]);
        jobs.push({ input: `reshoot ${q([img])}`, fields: f, label: `Animate · ${img.split(/[\\/]/).pop()}` });
      });
    });
    return { jobs };
  }
  const all = [...inputs.images, ...inputs.videos, ...inputs.audios];
  seedList.forEach((seed) => {
    const f: Record<string, string> = {};
    put(f, 'prompt', manual ?? s.prompt.trim());
    common(f, seed);
    stamp(f, all);
    jobs.push({ input: `reshoot ${q(all)}`, fields: f, label: `Direct · ${all.length} references` });
  });
  return { jobs };
}

/** Why a Generate / Reimagine run can't be built yet, or null. */
function dreamError(id: IntentId, s: TransformSettings): string | null {
  const manual = s.manualPrompt !== null ? s.manualPrompt.trim() : null;
  if (manual === '') return 'Write the prompt, or reset it to the generated one';
  if (!(manual ?? s.prompt.trim())) return id === 'generate' ? 'Describe the picture' : 'Describe the result';
  if (id === 'generate' && !dreamSize(s)) return 'Enter a width and height';
  return null;
}

/**
 * loki-dream jobs. Generate: ONE job with no input making every variation (`count`: consecutive seeds, one model
 * load), written to the library's Generated folder; the selection is ignored. Reimagine: one job per image and
 * variation, the image as the starting point.
 */
function buildDreamJobs(
  id: IntentId,
  s: TransformSettings,
  inputs: Inputs,
  ctx: BuildContext,
  stamp: (f: Record<string, string>, runPaths: string[]) => void
): JobRequest[] {
  const prompt = (s.manualPrompt !== null ? s.manualPrompt : s.prompt).trim();
  const common = (f: Record<string, string>, seed: number) => {
    put(f, 'prompt', prompt);
    // Turbo is its own step count at CFG 1, which ignores a negative prompt: send neither.
    if (s.quality === 'draft') put(f, 'turbo', true);
    else {
      put(f, 'steps', stepsFor('dream', s.quality));
      put(f, 'negative', s.negative.trim());
    }
    put(f, 'seed', seed);
  };
  if (id === 'generate') {
    const [w, h] = dreamSize(s) as [number, number];
    const n = Math.min(MAX_DREAM_COUNT, Math.max(1, Math.round(s.variations)));
    const f: Record<string, string> = {};
    common(f, seeds({ ...s, variations: 1 }, ctx)[0]);
    put(f, 'size', `${w}x${h}`);
    if (n > 1) put(f, 'count', n);
    stamp(f, []);
    return [{ input: 'dream', fields: f, label: `Generate · ${n} ${n === 1 ? 'image' : 'images'}` }];
  }
  const jobs: JobRequest[] = [];
  inputs.images.forEach((img) => {
    seeds(s, ctx).forEach((seed) => {
      const f: Record<string, string> = {};
      common(f, seed);
      put(f, 'strength', strengthOf(s));
      stamp(f, [img]);
      jobs.push({ input: `dream ${q([img])}`, fields: f, label: `Reimagine · ${img.split(/[\\/]/).pop()}` });
    });
  });
  return jobs;
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
  /** loki-dream's turbo mode (no classifier-free guidance: half the work per step). */
  turbo?: boolean;
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
  const steps = stepsForSettings(def.engine, s);
  const first = inputs.images[0] || inputs.videos[0] || inputs.audios[0] || '';
  const built = buildJobs(id, s, paths, { random: () => 0.5 });
  const jobCount = built.jobs.length;
  const variations = Math.min(maxVariations(id), Math.max(1, Math.round(s.variations)));
  if (def.engine === 'dream') {
    const turbo = s.quality === 'draft';
    if (id === 'generate') {
      const size = dreamSize(s);
      return { jobCount, outputs: variations, outputName: 'dream_….png', outWidth: size?.[0], outHeight: size?.[1], engine: 'dream', steps, turbo };
    }
    return {
      jobCount,
      outputs: jobCount,
      outputName: `${stem(first)}_dream.png`,
      // The engine works at the input's size (on its 16 px grid).
      outWidth: source ? Math.round(source.width / DREAM_GRID) * DREAM_GRID : undefined,
      outHeight: source ? Math.round(source.height / DREAM_GRID) * DREAM_GRID : undefined,
      engine: 'dream',
      steps,
      turbo,
    };
  }
  const outputs =
    def.engine === 'retouch' && !runGroups(id, paths, s.pinned)
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
  const suffix = id === 'angle' && preset ? `_${preset}` : SUFFIX[preset] || '_edit';
  const out: Plan = { jobCount, outputs, outputName: `${stem(first)}${suffix}.png`, engine: 'retouch', steps };
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
  if (plan.engine === 'dream') {
    // Not yet measured: a guess for Kromoss (a model load, then CFG doubles the work per step unless turbo).
    const mp = plan.outWidth && plan.outHeight ? (plan.outWidth * plan.outHeight) / 1e6 : 1;
    return 15 + plan.steps * (plan.turbo ? 0.35 : 0.7) * mp;
  }
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
