// Train LoRA: what the dialog offers and the job it submits. The server's `train-lora` task (media-server/tasks/
// train_lora.go) turns the fields into a loki-tune command line; media-server/tasks/train_lora_test.go pins the
// field names (TestUIContractTrainLora).

export type LoraKind = 'character' | 'style' | 'object' | 'concept' | 'general';
export type LoraCaptions = 'auto' | 'library' | 'none';

export interface KindInfo {
  id: LoraKind;
  label: string;
  /** what the LoRA learns, and so what the captions leave to the trigger */
  hint: string;
  /** a class noun (woman, mug) makes sense for this kind */
  hasClass: boolean;
  classPlaceholder?: string;
}

export const KINDS: KindInfo[] = [
  {
    id: 'character',
    label: 'Character',
    hint: 'A person or character. Captions leave who they are (face, hair, build, signature items) to the trigger and describe clothing, pose, setting and lighting, so those stay promptable.',
    hasClass: true,
    classPlaceholder: 'woman, man, dog…',
  },
  {
    id: 'style',
    label: 'Style',
    hint: 'A look: medium, technique, palette, rendering. Captions describe the content in full and never name the style, so the LoRA carries it.',
    hasClass: false,
  },
  {
    id: 'object',
    label: 'Object',
    hint: "A product or object. Captions leave its design to the trigger and describe where it is, the angle, hands, surroundings and light.",
    hasClass: true,
    classPlaceholder: 'mug, sneaker, car…',
  },
  {
    id: 'concept',
    label: 'Concept',
    hint: 'A pose, action, effect or composition, named only by the trigger. Everything else is described.',
    hasClass: false,
  },
  {
    id: 'general',
    label: 'General',
    hint: 'Everything in the images is described plainly: a fine-tune toward this set as a whole.',
    hasClass: false,
  },
];

export const RESOLUTIONS = [512, 768, 1024] as const;
export const RANKS = [16, 32, 64] as const;

export interface TrainLoraSettings {
  name: string;
  trigger: string;
  kind: LoraKind;
  className: string;
  captions: LoraCaptions;
  /** 0 = the engine's default (about 50 epochs, 1000-3000) */
  steps: number;
  rank: number;
  resolution: number;
  /** MOSS-Image checkpoint file to train against; '' = loki-tune's default (Kromoss bf16, else int8) */
  model: string;
}

export const DEFAULT_SETTINGS: TrainLoraSettings = {
  name: '',
  trigger: '',
  kind: 'character',
  className: '',
  captions: 'auto',
  steps: 0,
  rank: 16,
  resolution: 1024,
  model: '',
};

const IMAGE_EXT = /\.(png|jpe?g|jfif|webp|bmp|tiff?)$/i;

export function isTrainImage(path: string): boolean {
  return IMAGE_EXT.test(path);
}

/** loki-tune's default: about 50 epochs, 1000-3000 steps; a style about 70 epochs, at least 2000. */
export function defaultSteps(images: number, style = false): number {
  if (style) return Math.min(3000, Math.max(2000, images * 70));
  return Math.min(3000, Math.max(1000, images * 50));
}

/** Seconds per step on an RTX 4090 (bf16 Kromoss), by training resolution. */
const SECS_PER_STEP: Record<number, number> = { 512: 0.55, 768: 1.0, 1024: 1.7 };

/** Rough training time in minutes (steps + captioning and previews). */
export function estimateMinutes(images: number, s: Pick<TrainLoraSettings, 'steps' | 'resolution'>): number {
  const steps = s.steps > 0 ? s.steps : defaultSteps(images);
  const secs = steps * (SECS_PER_STEP[s.resolution] ?? 1.7) + images * 6 + (steps / 250) * 12 + 60;
  return Math.round(secs / 60);
}

/** A file-name-safe run name, as the server makes it (loraRunName). */
export function runName(s: Pick<TrainLoraSettings, 'name' | 'trigger'>): string {
  const n = (s.name.trim() || s.trigger.trim()).replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^[._]+|[._]+$/g, '');
  return n || 'lora';
}

/** The trigger is what prompts call the subject with; these kinds are useless without one. */
export function needsTrigger(kind: LoraKind): boolean {
  return kind === 'character' || kind === 'object' || kind === 'concept';
}

export interface TrainLoraJob {
  input: string;
  fields: Record<string, string>;
}

/** POST /create body: the images as the job's newline-separated target (or, with `query64`, the library query, which
 * the server resolves to every matching image; it must stay the last token), the settings as fields. */
export function buildTrainLoraJob(paths: string[], s: TrainLoraSettings, query64?: string): TrainLoraJob {
  const images = paths.filter(isTrainImage);
  const kind = KINDS.find((k) => k.id === s.kind);
  const fields: Record<string, string> = {
    name: s.name.trim(),
    trigger: s.trigger.trim(),
    kind: s.kind,
    class: kind?.hasClass ? s.className.trim() : '',
    captions: s.captions,
    rank: String(s.rank),
    resolution: String(s.resolution),
    model: s.model.trim(),
  };
  if (s.steps > 0) fields.steps = String(Math.round(s.steps));
  for (const k of Object.keys(fields)) if (!fields[k]) delete fields[k];
  return { input: query64 ? `train-lora --query64=${query64}` : `train-lora "${images.join('\n')}"`, fields };
}

/** The checkpoints a LoRA can be trained against: MOSS-Image files on the server's machine (a missing fine-tune cannot
 * be downloaded, and other architectures are not trainable). */
export function trainableModels<M extends { arch: string; local: boolean }>(models: M[]): M[] {
  return models.filter((m) => m.arch === 'moss' && m.local);
}

/** loki-tune's default base among them: Kromoss bf16 when present, else the int8 file. */
export function defaultTrainModel<M extends { file: string }>(models: M[]): string {
  const f = (re: RegExp) => models.find((m) => re.test(m.file))?.file;
  return f(/^kromoss.*bf16/i) ?? f(/^kromoss.*(int8|convrot)/i) ?? '';
}
