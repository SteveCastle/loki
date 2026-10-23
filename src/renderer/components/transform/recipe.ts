// A Transform result back into a saved task: the generation record our engines stamp into what they write
// (tools/CONVENTIONS.md, "Provenance"; parsed by generation-metadata.ts) becomes an intent + settings for
// saveTemplate, ready to run on a different image.
//
// Results made by the Transform flow carry its recipe in the record's `meta` (intents.ts: recipeJson): the intent
// and every setting exactly as the user set them, so the task is rebuilt as it was. Older results (and runs of the
// CLIs by hand) only have the engine's own view: the final prompt, steps, seed, sizes and the command line. Those are
// rebuilt as closely as they allow and marked `approximate`.
//
// Either way the run's other inputs become pinned references, so a prompt naming <image2> or <Audio 1> still finds
// them when the saved task runs on a new image (which takes the first input's place). No React in here.
import {
  BASE_SETTINGS,
  DREAM_ASPECTS,
  INTENTS,
  QUALITY_STEPS,
  RECIPE_KIND,
  activePins,
  cameraViewLabel,
  intentById,
  intentKind,
  parseSize,
  pinnable,
  settingsFor,
  suggestTemplateName,
  MAX_DURATION,
  MIN_DURATION,
  type EngineId,
  type IntentId,
  type Quality,
  type TransformSettings,
} from './intents';

export interface RebuiltTask {
  intent: IntentId;
  /** Full settings; `prompt` is the saved task's prompt ('' for a preset). */
  settings: TransformSettings;
  /** Rebuilt from the engine's own fields (no recipe was stamped): close to the run, not exact. */
  approximate: boolean;
}

type Json = any;

const isIntent = (v: unknown): v is IntentId => typeof v === 'string' && INTENTS.some((i) => i.id === v);

/** Only the settings this version knows, each of the type it expects: a recipe is data from a file. */
function knownSettings(raw: Json): Partial<TransformSettings> {
  const out: Record<string, unknown> = {};
  if (!raw || typeof raw !== 'object') return out;
  (Object.keys(BASE_SETTINGS) as Array<keyof TransformSettings>).forEach((k) => {
    const v = raw[k];
    const want = BASE_SETTINGS[k];
    if (k === 'manualPrompt') {
      if (v === null || typeof v === 'string') out[k] = v;
    } else if (k === 'reshootModel') {
      if (v === 'h3' || v === 'fast') out[k] = v;
    } else if (k === 'aspect') {
      if (v === 'square' || v === 'landscape' || v === 'portrait' || v === 'custom') out[k] = v;
    } else if (k === 'pinned') {
      if (Array.isArray(v)) out[k] = v.filter((p) => typeof p === 'string' && p);
    } else if (typeof v === typeof want && !(typeof v === 'number' && !Number.isFinite(v))) {
      out[k] = v;
    }
  });
  return out as Partial<TransformSettings>;
}

/**
 * The pinned references a rebuilt task keeps: every input of the run but the first (the item it ran on, whose place
 * the new selection takes), in run order. A run of nothing but pins (Direct) keeps them all.
 */
export function carriedPins(id: IntentId, paths: string[], pinned: string[] = []): string[] {
  const real = paths.filter((p) => p && p !== '-');
  const onlyPins = real.length > 0 && real.every((p) => pinned.includes(p));
  const rest = onlyPins ? real : real.slice(1);
  return activePins(id, Array.from(new Set(rest)));
}

function fromRecipe(meta: Json): RebuiltTask | null {
  if (!meta || meta.kind !== RECIPE_KIND || !isIntent(meta.intent)) return null;
  const id: IntentId = meta.intent;
  const s: TransformSettings = { ...settingsFor(id), ...knownSettings(meta.settings) };
  // A custom intent's hand-written review prompt is what ran: it becomes the saved prompt.
  if (intentKind(id) === 'custom' && s.manualPrompt !== null && s.manualPrompt.trim()) s.prompt = s.manualPrompt;
  if (intentKind(id) === 'custom') s.manualPrompt = null;
  const paths: string[] = Array.isArray(meta.paths) ? meta.paths.filter((p: unknown) => typeof p === 'string') : [];
  s.pinned = pinnable(id) ? carriedPins(id, paths, s.pinned) : [];
  if (intentKind(id) === 'custom' && !s.prompt.trim()) return null;
  return { intent: id, settings: s, approximate: false };
}

// --- older results: the engine's own fields -------------------------------------------------------------------

/** The value of a flag in an argv (`--flag value` or `--flag=value`), any of its spellings. */
function argValue(args: string[], ...names: string[]): string | undefined {
  for (let i = 0; i < args.length; i += 1) {
    const a = args[i];
    for (const n of names) {
      if (a === n && i + 1 < args.length) return args[i + 1];
      if (a.startsWith(`${n}=`)) return a.slice(n.length + 1);
    }
  }
  return undefined;
}

const hasArg = (args: string[], ...names: string[]) => args.some((a) => names.includes(a) || names.some((n) => a.startsWith(`${n}=`)));

/** The quality whose step count is closest to what ran. */
export function nearestQuality(engine: EngineId, steps: unknown): Quality {
  if (typeof steps !== 'number') return 'balanced';
  const table = QUALITY_STEPS[engine];
  return (Object.keys(table) as Quality[]).reduce((best, q) => (Math.abs(table[q] - steps) < Math.abs(table[best] - steps) ? q : best), 'balanced');
}

const SCALES = [1.5, 2, 3, 4];

function inputPaths(rec: Json): string[] {
  return Array.isArray(rec.inputs) ? rec.inputs.map((i: Json) => (i && typeof i.path === 'string' ? i.path : '')).filter(Boolean) : [];
}

function fromRetouch(rec: Json, args: string[]): RebuiltTask | null {
  const preset: string = typeof rec.preset === 'string' ? rec.preset : argValue(args, '--preset') || (hasArg(args, '--upscale') ? 'upscale' : '');
  const byPreset: Record<string, IntentId> = { restore: 'restore', upscale: 'upscale', '4kify': 'wallpaper', '4kify-phone': 'wallpaper' };
  // No engine preset: a free-form edit (a camera angle also lands here: the server expanded it into the prompt).
  const id: IntentId = byPreset[preset] || 'edit';
  const s: TransformSettings = { ...settingsFor(id) };
  const prompt = typeof rec.prompt === 'string' ? rec.prompt.trim() : '';
  const explicit = argValue(args, '--prompt', '-p');
  if (id === 'edit') {
    s.prompt = prompt;
  } else {
    if (explicit !== undefined || hasArg(args, '--prompt-file', '-P')) s.manualPrompt = (explicit ?? prompt).trim();
    s.append = (argValue(args, '--append') || '').trim();
  }
  if (id === 'wallpaper') s.wallpaperTarget = preset === '4kify-phone' ? 'phone' : 'desktop';
  const scale = parseFloat(argValue(args, '--scale') ?? argValue(args, '--upscale') ?? '');
  const size = argValue(args, '--size');
  if (Number.isFinite(scale) && scale > 0) {
    s.sizeMode = 'scale';
    s.scale = scale;
  } else if (size && parseSize(size)) {
    s.sizeMode = 'custom';
    s.customSize = size;
  } else if (id === 'upscale') {
    // Upscale without a factor on the command line: from the delivered size against the input's.
    const src = Array.isArray(rec.inputs) && rec.inputs[0] ? rec.inputs[0] : null;
    const ratio = src && typeof src.width === 'number' && typeof rec.width === 'number' ? rec.width / src.width : 2;
    s.scale = SCALES.reduce((b, x) => (Math.abs(x - ratio) < Math.abs(b - ratio) ? x : b), 2);
  }
  s.quality = nearestQuality('retouch', rec.steps);
  if (typeof rec.seed === 'number') s.seed = rec.seed;
  s.pinned = id === 'edit' ? carriedPins(id, inputPaths(rec)) : [];
  if (id === 'edit' && !s.prompt) return null;
  return { intent: id, settings: s, approximate: true };
}

function fromReshoot(rec: Json, args: string[]): RebuiltTask | null {
  const id: IntentId = rec.animate === true ? 'alive' : 'direct';
  const s: TransformSettings = { ...settingsFor(id) };
  const prompt = typeof rec.prompt === 'string' ? rec.prompt.trim() : '';
  if (id === 'alive') {
    s.describe = typeof rec.describe === 'string' ? rec.describe : '';
    if (rec.shake === 'none' || rec.shake === 'subtle' || rec.shake === 'handheld') s.shake = rec.shake;
    // The extra direction given with --animate (the record's prompt is the whole expanded one).
    s.prompt = (argValue(args, '--prompt', '-p') || '').trim();
  } else {
    s.prompt = prompt;
  }
  const d = parseFloat(argValue(args, '--duration', '-d') ?? '');
  const seconds = Number.isFinite(d) ? d : typeof rec.seconds === 'number' ? Math.round(rec.seconds * 4) / 4 : s.duration;
  s.duration = Math.min(MAX_DURATION, Math.max(MIN_DURATION, seconds));
  // The fast model always runs its own fixed steps: the quality (which picks H3's steps) stays at its default.
  if (rec.model_id === 'fast' || argValue(args, '--model') === 'fast') s.reshootModel = 'fast';
  else s.quality = nearestQuality('reshoot', rec.steps);
  if (rec.ref_image_size === 'max') s.refSize = 'max';
  s.noAudio = rec.audio === false;
  s.keepSoundtrack = !hasArg(args, '--no-video-audio');
  if (typeof rec.seed === 'number') s.seed = rec.seed;
  s.pinned = id === 'direct' ? carriedPins(id, inputPaths(rec)) : [];
  if (id === 'direct' && !s.prompt) return null;
  return { intent: id, settings: s, approximate: true };
}

function fromDream(rec: Json, args: string[]): RebuiltTask | null {
  // An inpainting run (--mask) is the Scene editor's, not a Transform task.
  if (hasArg(args, '--mask')) return null;
  const id: IntentId = hasArg(args, '--init') ? 'reimagine' : 'generate';
  const s: TransformSettings = { ...settingsFor(id) };
  s.prompt = (typeof rec.prompt === 'string' ? rec.prompt : argValue(args, '--prompt', '-p') || '').trim();
  if (!s.prompt) return null;
  if (hasArg(args, '--turbo')) s.quality = 'draft';
  else {
    const steps = parseFloat(argValue(args, '--steps') ?? '');
    s.quality = nearestQuality('dream', Number.isFinite(steps) ? steps : typeof rec.steps === 'number' ? rec.steps : 50);
    s.negative = (argValue(args, '--negative') || '').trim();
  }
  if (id === 'reimagine') {
    const strength = parseFloat(argValue(args, '--strength') ?? '');
    if (Number.isFinite(strength)) s.strength = strength;
  } else {
    const size = parseSize(argValue(args, '--size') || '');
    const match = size && DREAM_ASPECTS.find((a) => a.width === size[0] && a.height === size[1]);
    if (match) s.aspect = match.id;
    else if (size) {
      s.aspect = 'custom';
      s.customSize = `${size[0]}x${size[1]}`;
    }
  }
  if (typeof rec.seed === 'number') s.seed = rec.seed;
  return { intent: id, settings: s, approximate: true };
}

/** A generation record (findLokiRecord) as a task to save, or null when it is not a Transform-shaped run. */
export function taskFromRecord(rec: Json): RebuiltTask | null {
  if (!rec || typeof rec !== 'object') return null;
  const exact = fromRecipe(rec.meta);
  if (exact) return exact;
  const args: string[] = Array.isArray(rec.args) ? rec.args.filter((a: unknown) => typeof a === 'string') : [];
  if (rec.tool === 'loki-retouch') return fromRetouch(rec, args);
  if (rec.tool === 'loki-reshoot') return fromReshoot(rec, args);
  if (rec.tool === 'loki-dream') return fromDream(rec, args);
  return null;
}

/** A name for the saved task: a custom prompt's first words, else what the preset does. */
export function suggestTaskName(task: RebuiltTask): string {
  const { intent: id, settings: s } = task;
  if (intentKind(id) === 'custom') return suggestTemplateName(s.prompt);
  const extra = s.manualPrompt?.trim() || s.append.trim() || (id === 'alive' ? s.describe.trim() : '');
  let base = intentById(id).title;
  if (id === 'upscale') base = `Upscale ${s.scale}×`;
  else if (id === 'wallpaper') base = s.wallpaperTarget === 'phone' ? 'Phone wallpaper' : 'Wallpaper';
  else if (id === 'angle') base = cameraViewLabel(s.cameraAzimuth, s.cameraElevation);
  return extra ? `${base} · ${suggestTemplateName(extra)}` : base;
}
