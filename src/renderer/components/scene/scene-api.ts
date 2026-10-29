// The Scene editor's media-server API (media-server/scene_api.go). Types mirror
// tasks/scene.go's JSON. Electron authenticates with the viewer's bearer token;
// the web build rides the session cookie.
import { isElectron, mediaServerBase } from '../../platform';
import { serverAuthHeaders } from '../../server-auth';
import { maskBounds, type MaskOp, type MaskOpts, type Rect } from './mask';
import type { SceneRefItem } from './refs';

/** still = an edited area, motion = an animated one, paint = the user's own colour painting (never generated). */
export type LayerKind = 'still' | 'motion' | 'paint';

export interface SceneVersion {
  path: string;
  region: Rect;
  prompt: string;
  duration?: number;
  audio?: boolean;
  note?: string;
  jobId?: string;
  /** The stills it was generated from ("<layer id>:<version path>", bottom up; see stackBelow). */
  below?: string[];
  /** The images the engine was given: the crop, then the references in order (generations only). */
  inputs?: string[];
  /** motion: its roto (a matte tracked through the clip) */
  track?: SceneTrack | null;
  /**
   * Its own mask: the layer's when it was generated, then whatever is changed while it is shown. Showing the version
   * puts it back on the layer. Editor-owned (the server keeps it by path); absent on versions from before.
   */
  mask?: VersionMask;
  /** A still version blended into the picture under it (server-made; absent while it is being made or when off). */
  stitch?: SceneStitch | null;
  created: string;
}

/**
 * A stitched still version: `path` is the patch (the version's region, alpha = the blend) shown in place of the
 * version through its mask; `seam` draws the seam, `heal` where it still shows.
 */
export interface SceneStitch {
  path: string;
  seam?: string;
  heal?: string;
  key: string;
  report: {
    band: number;
    /** Mean displacement corrected near the edge (px), and its 95th percentile. */
    drift: number;
    driftMax: number;
    /** The overall colour shift corrected (0-255). */
    tone: number;
    /** Grain added (0-255 sd). */
    grainAdded: number;
    seamLength: number;
    residual: number;
    /** Pixels where the seam still shows (Heal redraws them). */
    healPixels: number;
    skipped?: string;
  };
}

const roundRect = (r: Rect): Rect => ({ x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.w), h: Math.round(r.h) });

/** A layer's mask as drawn (never a fit to a result: that is re-made from the version itself). */
export interface VersionMask {
  shapes?: MaskOp[];
  maskOpts?: MaskOpts;
  feather: number;
}

/** What a generation is made with: the mask without a fit to the shown result, its raster and the crop around it. */
export interface GenMask extends VersionMask {
  mask: string;
  region: Rect;
}

/** A prompt for loki-segment, in the pixels of what is segmented (clip pixels for a roto). */
export interface SegPrompt {
  obj: number;
  frame: number;
  text?: string;
  /** [x, y, 1 = this | 0 = not this] */
  points?: Array<[number, number, number]>;
  /** [x, y, w, h] */
  box?: [number, number, number, number];
  /** clicks/box: which of SAM's readings to track (0 = best; the one chosen with Tab) */
  candidate?: number;
}

/** A motion version's roto (server-owned): the job, what it was tracked from, and the matte it made. */
export interface SceneTrack {
  jobId: string;
  state: string;
  done: number;
  total: number;
  error?: string;
  prompts: SegPrompt[];
  /** The matte: <dir>/00000.png ... one per clip frame */
  dir?: string;
  frames?: number;
  /** The clip with the matte as alpha (VP9 webm), for the stage */
  video?: string;
}

/** One pick: on the picture (frame pixels), or with layer + version on a motion clip's frame (clip pixels). */
export interface SegmentRequest {
  text?: string;
  points?: Array<[number, number, number]>;
  box?: [number, number, number, number];
  everything?: boolean;
  multimask?: boolean;
  layer?: string;
  version?: number;
  frame?: number;
}

/** An object found: its mask (white, alpha = coverage) as a PNG data URL covering rect, and its box. */
export interface SegmentObject {
  id: number;
  label: string;
  score: number;
  bbox: Rect;
  mask: string;
  rect: Rect;
  /** clicks/boxes: the other readings (part / object / whole), best first */
  candidates?: SegmentObject[];
}

export interface SegmentResult {
  width: number;
  height: number;
  objects: SegmentObject[] | null;
  ms: number;
}

export interface SceneJob {
  id: string;
  state: string;
  done: number;
  total: number;
  region: Rect;
  prompt: string;
  input?: string;
  line?: string;
  /** What the version it makes will be called (e.g. "re-finished v2"); empty for a plain generation. */
  note?: string;
  /** The step running now: the engine (retouch / reshoot) or scene-lock (fixing the edges to the still). */
  step?: string;
}

/** One spoken line of a motion layer: who says it (empty = the person in the picture), the language, the words. */
export interface DialogueLine {
  speaker?: string;
  lang?: string;
  text: string;
}

export const DIALOGUE_LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Japanese', 'Chinese', 'Korean', 'Russian', 'Hindi', 'Arabic'];

const UNSPACED = new Set(['Japanese', 'Chinese']);

/**
 * Roughly how long the lines take to say (seconds): ~2.5 words a second (about 6 characters a second for
 * languages written without spaces), plus a short pause between lines.
 */
export function dialogueSeconds(lines: DialogueLine[] | undefined): number {
  const spoken = (lines || []).filter((l) => l.text.trim());
  if (!spoken.length) return 0;
  const talk = spoken.reduce((sum, l) => {
    const t = l.text.trim();
    if (UNSPACED.has(l.lang || '')) return sum + t.replace(/\s+/g, '').length / 6;
    return sum + t.split(/\s+/).length / 2.5;
  }, 0);
  return talk + 0.4 * (spoken.length - 1) + 0.6;
}

/** Per-layer generation settings (stored as the layer's `options`). */
export interface LayerOptions {
  context?: 'tight' | 'normal' | 'wide' | 'full';
  /** still: sampling steps */
  steps?: number;
  /** still: the engine: loki-retouch edits by instruction (default), loki-dream (Kromoss) redraws the area from a description */
  engine?: 'retouch' | 'dream';
  /** still, dream: how much of the area is redrawn (0-1; default 1, a sketch 0.75) */
  strength?: number;
  /** motion */
  duration?: number;
  model?: 'h3' | 'fast';
  sound?: boolean;
  megapixels?: number;
  describe?: string;
  /** motion: playback rate (0.25-4, default 1) */
  speed?: number;
  /** motion: seconds of scene time its loop is shifted by, so layers don't pulse together (default 0) */
  offset?: number;
  /** still: a clean plate, the picture under that motion layer's roto with its subject removed (not "below" it) */
  cleanFor?: string;
  /** motion: composite through the shown version's roto matte (default on when it has one) */
  matte?: boolean;
  /** motion: soundtrack gain (0-2, default 1; 0 mutes it) */
  volume?: number;
  /** motion: what its generated sound should be ("crackling fire, soft wind"); empty = the ambience of what moves */
  soundPrompt?: string;
  /** motion: allow speech and music in its sound (default never) */
  voices?: boolean;
  /** motion: lines spoken in order (dialogue implies sound) */
  dialogue?: DialogueLine[];
  /** motion: an audio or video file whose voice the first speaker speaks in */
  voiceRef?: string;
  /** extra images (and for motion videos / audio) the generation is given; see refs.ts */
  refs?: SceneRefItem[];
  /** paint: a guide only (used as a reference, never in the picture) */
  guide?: boolean;
  /** paint: the character "Detect pose" read the figure from (mask pieces, frame px), for finishShapes */
  subject?: MaskOp[];
  /**
   * still: fit the mask to the shown version, adding wherever it changed the picture near the mask, so a result
   * that corrects a rough sketch's shape (or casts a shadow) is not cut off at the sketch's edge.
   */
  /** still: blend the shown version's edges into the picture under it (stitch; default on) */
  stitch?: boolean;
  fit?: boolean;
  /** still: a finish layer ("Finish with AI"): its prompt is what the paint is, wrapped in this grounding. */
  finish?: FinishMode;
  /** still, finish: the paint layer it finishes */
  paint?: string;
  /** still, finish: the user's own grounding ({what} = the prompt, {guide} = the guide's token); empty = the default */
  grounding?: string;
  /** motion: lock the static parts to the still (default on; see media-server/tasks/scene_lock.go) */
  lock?: boolean;
  /** motion: output at the region's size with the still's fine detail (default on) */
  detail?: boolean;
  seed?: number;
}

export type FinishMode = 'sketch' | 'colour' | 'repose';

/**
 * How a finish layer's words are read: the instruction around them. Style-neutral ("this picture", never
 * "photograph"): the scene may be a photo or a painting. Mirrors sceneFinishGrounding in media-server/tasks/scene_finish.go.
 */
export const FINISH_GROUNDING: Record<FinishMode, string> = {
  sketch:
    'In <image1>, the rough painted sketch{where} stands in for {what}: replace that sketch, and only it, with {what} as it would really appear in this picture. ' +
    '{target} shows the sketch alone on black, in exactly the same place: it marks the only part of <image1> to change. ' +
    'The sketch is a crude placeholder that only marks where it goes, roughly how big it is and a hint of its colours, ' +
    'so do not copy its flat paint or hard outline: correct its shape and proportions, and give it the real look, light, texture and focus ' +
    "it would have here, in the picture's own style. Leave everything that was not sketched exactly as it is: " +
    'do not change, move or replace any other figure or object, even one that looks like {what}.',
  colour:
    'Finish the colouring of <image1>, whose new base colours are already laid in: {what}. ' +
    'The new colours are {colours}: every object in one of them keeps it exactly; only even out patchy or uneven colour within an object. ' +
    '{self} is the same picture before it was recoloured, for its light, shading and detail only, not its colours. ' +
    'Keep every line, texture and detail, and the composition, unchanged.',
  repose:
    'New pose for the person in <image1>: {pose}. ' +
    'Make the person in <image1> do the exact same pose as shown in {guide}: {what}. ' +
    '{guide} only shows the pose: where the head, shoulders, elbows, hands, hips, knees and feet go. ' +
    "Follow the direction each part of the body points in {guide}, but keep this person's own body: the same proportions, limb lengths, build and size in the frame as in <image1>, so it is clearly the same body, only moved. " +
    'Only the arms and legs that are placed differently in {guide} move; the rest of the body stays exactly where it is. ' +
    'They have exactly two arms, two hands and two legs, each joined naturally to the body; an arm or leg that moves leaves its old place, which then shows the background or their clothing instead. ' +
    '{self} is the same picture again, for their face and body. ' +
    'Keep their identity exactly: the same face, hair, skin and figure, and the same clothing, which drapes and folds naturally with the new pose. ' +
    'Keep the background, the lighting, the camera angle and the framing of <image1> unchanged.',
};

/**
 * What "Finish with AI" can do with a painting, for people. guide: the paint is read as a guide (a reference over the
 * clean picture, kept out of it) rather than as part of the picture; expand: how far past the paint the layer's mask
 * reaches, as a share of the picture's short side (a pose moves limbs well outside a stick figure). Mirrors
 * finishUsesGuide in media-server/tasks/scene_finish.go.
 */
export const FINISH_MODES: Record<
  FinishMode,
  { label: string; help: string; guide: boolean; expand: number; refNote: string; ask: string; placeholder: string; added: string }
> = {
  sketch: {
    label: 'Add',
    help: 'Turn the rough sketch into a real thing in the picture: draw a box-ish house, get a house',
    guide: false,
    expand: 0.01,
    refNote: '',
    ask: 'What is it?',
    placeholder: 'e.g. the sun · a stone cottage with ivy · a woman in a blue dress (optional)',
    added: 'Added an Edit layer that makes your sketch real: say what it is, then generate.',
  },
  colour: {
    label: 'Recolour',
    help: 'A base colour coat from rough paint: each object the paint is mostly on takes its colour, edge to edge, keeping its shading',
    guide: true,
    expand: 0.01,
    refNote: 'the rough colour guide',
    ask: 'Light or shading? (optional)',
    placeholder: 'Leave empty for the base colours alone (exact shades, a few seconds). e.g. warm late-afternoon light, soft shadows',
    added: 'Added a Recolour layer: generate for the base colours, or describe light and shading first to have the AI finish them.',
  },
  repose: {
    label: 'Repose',
    help: 'Move a character into the pose you drew over them: a rough stick figure is enough',
    guide: true,
    expand: 0.06,
    refNote: 'the pose drawing',
    ask: 'Anything about the pose?',
    placeholder: 'e.g. waving hello · sitting on the step (optional)',
    added: 'Added an Edit layer that reposes the character, masked over them and the new pose: add any notes, then generate.',
  },
};

/**
 * A finish layer's mask shapes for mode over paint, keeping the user's own (prev, less any earlier cover). Repose
 * also covers the character as they stand now: the engine redraws the whole person, and a mask of only the drawing
 * left the old body showing round the new one (ghost hips, a second arm). The cover is the character Detect pose read
 * (paint.options.subject), else the drawing's box padded out to a body's width (joints sit inside the body).
 */
export function finishShapes(paint: SceneLayer, mode: FinishMode, prev: MaskOp[]): MaskOp[] {
  const own = prev.filter((op) => !op.subject);
  // Recolour colours whole objects, past where the rough paint reaches: the cover is the paint's box padded out (the
  // base coat leaves everything it doesn't recolour exactly as it was).
  if (mode === 'colour') {
    const b = maskBounds(paint.shapes || []);
    if (!b) return own;
    const pad = 0.1 * Math.max(b.w, b.h);
    return [...own, { t: 'rect', x: b.x - pad, y: b.y - pad, w: b.w + 2 * pad, h: b.h + 2 * pad, subject: true }];
  }
  if (mode !== 'repose') return own;
  const known = paint.options?.subject;
  if (known?.length) return [...own, ...known.map((op): MaskOp => ({ ...op, erase: undefined, subject: true }))];
  const b = maskBounds(paint.shapes || []);
  if (!b) return own;
  const pad = 0.15 * Math.max(b.w, b.h);
  return [...own, { t: 'rect', x: b.x - pad, y: b.y - pad, w: b.w + 2 * pad, h: b.h + 2 * pad, subject: true }];
}

/** The Kromoss strength a finish starts at when the layer sets none. Mirrors sceneDreamArgs. */
export function finishStrength(finish: FinishMode | undefined): number {
  return finish === 'sketch' || finish === 'repose' ? 0.9 : finish === 'colour' ? 0.8 : 1;
}

/**
 * Fill a grounding's placeholders: {what} the words, {guide} the paint reference's token, {where} where the sketch is
 * (sketchWhere), {target} the token of the sketch alone on black (Retouch sketch finishes; the server adds it as the
 * last reference). With no words a "…: {what}." clause (or ", even one that looks like {what}.") folds away and any
 * other {what} reads "the real thing"; with no target its sentence goes. {pose} (Repose: the mannequin's pose in words)
 * and {self} (Repose: the token of the picture given again for identity) likewise go with their sentence when empty.
 * Mirrors composeFinish / sceneFinishPrompt.
 */
export function composeFinish(grounding: string, what: string, guide: string, where = '', target = '', pose = '', self = '', colours = ''): string {
  const w = what.trim().replace(/\.+$/, '');
  let out = fillOrDrop(fillOrDrop(fillOrDrop(grounding, '{pose}', pose), '{self}', self), '{colours}', colours);
  if (!w) out = out.split(': {what}.').join('.').split(', even one that looks like {what}.').join('.');
  out = fillOrDrop(out, '{target}', target);
  return out
    .split('{what}')
    .join(w || 'the real thing')
    .split('{where}')
    .join(where)
    .split('{target}')
    .join(target)
    .split('{guide}')
    .join(guide);
}

/** key filled with val, or the sentences naming key dropped when val is empty. Mirrors fillOrDrop in scene_finish.go. */
function fillOrDrop(s: string, key: string, val: string): string {
  if (val) return s.split(key).join(val);
  const k = key.replace(/[{}]/g, '\\$&');
  return s.replace(new RegExp('[^.]*' + k + '[^.]*\\.', 'g'), '').trim();
}

/**
 * Where paint (its bounds) sits in a crop: " at the bottom left", " on the right", " in the middle"; '' when it fills
 * most of it or misses it. Mirrors sketchWhere in media-server/tasks/scene_finish.go.
 */
export function sketchWhere(paint: Rect, region: Rect): string {
  if (paint.w <= 0 || paint.h <= 0 || region.w <= 0 || region.h <= 0) return '';
  if (paint.w * paint.h > 0.5 * region.w * region.h) return '';
  const cx = paint.x + Math.floor(paint.w / 2);
  const cy = paint.y + Math.floor(paint.h / 2);
  if (cx < region.x || cy < region.y || cx > region.x + region.w || cy > region.y + region.h) return '';
  const third = (c: number, start: number, size: number) => {
    const f = (c - start) / size;
    if (f < 1 / 3) return 0;
    return f > 2 / 3 ? 2 : 1;
  };
  const v = ['top', '', 'bottom'][third(cy, region.y, region.h)];
  const h = ['left', '', 'right'][third(cx, region.x, region.w)];
  if (!v && !h) return ' in the middle';
  if (!v) return ` on the ${h}`;
  if (!h) return ` at the ${v}`;
  return ` at the ${v} ${h}`;
}

export interface SceneLayer {
  id: string;
  name: string;
  kind: LayerKind;
  hidden?: boolean;
  region: Rect;
  mask: string;
  shapes?: MaskOp[];
  /** Whole-mask settings (invert, expand, opacity), already applied to `mask`. */
  maskOpts?: MaskOpts;
  feather: number;
  prompt: string;
  options?: LayerOptions;
  versions: SceneVersion[];
  active: number;
  job?: SceneJob | null;
  /** Generations started while `job` runs, oldest first: each takes its turn on the GPU. */
  queued?: SceneJob[];
  error?: string;
}

export interface SceneRender {
  jobId: string;
  format: 'video' | 'still';
  state: string;
  done: number;
  total: number;
  output?: string;
  error?: string;
}

export interface Scene {
  id: string;
  name: string;
  source: string;
  sourceTime?: number;
  base: string;
  width: number;
  height: number;
  layers: SceneLayer[];
  duration: number;
  /** Each motion layer loops at its own length (default false: retimed to fit the scene length, see loopSpeed). */
  freeLoops?: boolean;
  render?: SceneRender | null;
  /** Stitches are being made (polled until they are). */
  stitching?: boolean;
  created: string;
  updated: string;
}

export interface DreamStatus {
  jobId: string;
  state: string;
  done: number;
  total: number;
  output?: string;
  error?: string;
}

export interface SceneSummary {
  id: string;
  name: string;
  source: string;
  base: string;
  layers: number;
  motion: number;
  updated: string;
}

export class SceneApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

const NOT_SIGNED_IN =
  'Not signed in to the media server for this library. Sign in from the database tab of the viewer’s command palette, then try again.';

async function call<T>(method: string, path: string, body?: unknown, timeoutMs = 20000): Promise<T> {
  // The viewer's own credential, read fresh per call (server-auth.ts): a
  // sign-in in the viewer applies here without reopening the window.
  const headers: Record<string, string> = { ...serverAuthHeaders() };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  let res: Response;
  try {
    res = await fetch(`${mediaServerBase}${path}`, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: isElectron ? 'omit' : 'include',
      // Every server call is bounded: a hung request must never pin the socket pool.
      signal: AbortSignal.timeout(timeoutMs),
      // A redirect is the server sending a signed-out client to its login
      // page — an auth failure, not a dead server (see below).
      redirect: 'manual',
    });
  } catch (e) {
    const msg = e instanceof Error && e.name === 'TimeoutError' ? 'the media server did not answer in time' : 'cannot reach the media server';
    throw new SceneApiError(msg, 0);
  }
  if (res.type === 'opaqueredirect' || (res.status >= 300 && res.status < 400)) {
    throw new SceneApiError(NOT_SIGNED_IN, 401);
  }
  const text = await res.text();
  if (!res.ok) {
    let msg = text.trim();
    try {
      const parsed = JSON.parse(msg);
      if (parsed && parsed.error) msg = parsed.error;
    } catch {
      // plain-text error
    }
    if (res.status === 401 || res.status === 403) msg = NOT_SIGNED_IN;
    throw new SceneApiError(msg || `HTTP ${res.status}`, res.status);
  }
  return (text ? JSON.parse(text) : undefined) as T;
}

const enc = encodeURIComponent;

export type LoopKind = 'crossfade' | 'natural' | 'boomerang';

export const sceneApi = {
  list: (source?: string) => call<SceneSummary[]>('GET', `/api/scenes${source ? `?source=${enc(source)}` : ''}`),
  // Starting a scene normalises the base frame with ffmpeg: allow it longer.
  open: (source: string, time: number, fresh = false) => call<Scene>('POST', '/api/scenes', { source, time, fresh }, 60000),
  get: (id: string) => call<Scene>('GET', `/api/scenes/${enc(id)}`),
  save: (scene: Scene) => call<Scene>('PUT', `/api/scenes/${enc(scene.id)}`, scene),
  remove: (id: string) => call<{ ok: boolean }>('DELETE', `/api/scenes/${enc(id)}`),
  generate: (id: string, layer: string, mask?: GenMask) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/generate`, { mask }, 60000),
  /** Cancel one generation (running or queued) by its job id, or all of the layer's when job is omitted. */
  cancel: (id: string, layer: string, job?: string) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/cancel`, job ? { job } : {}),
  deleteVersion: (id: string, layer: string, n: number) => call<Scene>('DELETE', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}`),
  /** Discard every version of a layer but n. */
  keepOnly: (id: string, layer: string, n: number) => call<Scene>('DELETE', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions?keep=${n}`),
  /** Redo version n's finish (edges, detail) from the engine's output with the layer's current settings: a new version. */
  refinish: (id: string, layer: string, n: number) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/refinish`, {}),
  /** Redraw where version n's stitched seam still shows (loki-dream), as a new version. */
  heal: (id: string, layer: string, n: number) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/heal`, {}),
  loop: (id: string, layer: string, n: number, kind: LoopKind) =>
    call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/loop`, { kind }, 180000),
  /** Where version n moves: a PNG data URL (alpha = motion) covering `region`. */
  /** Where still version n changed the picture it was made from, near the layer's mask: a PNG data URL (alpha = change) covering `region`. */
  changeMap: (id: string, layer: string, n: number) =>
    call<{ dataUrl: string; region: Rect }>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/change`, {}, 60000),
  motionMap: (id: string, layer: string, n: number) =>
    call<{ dataUrl: string; region: Rect }>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/motion`, {}, 120000),
  render: (id: string, format: 'video' | 'still') => call<Scene>('POST', `/api/scenes/${enc(id)}/render`, { format }),
  // The first pick loads the model and encodes the picture: allow it longer.
  /** The body pose of the person in box (frame px) of the picture under paint layer `layer`: OpenPose's 18 joints. */
  pose: (id: string, layer: string, box: Rect) =>
    call<{ joints: Array<{ x: number; y: number; score: number }> }>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/pose`, { box: roundRect(box) }, 60000),
  segment: (id: string, req: SegmentRequest) => call<SegmentResult>('POST', `/api/scenes/${enc(id)}/segment`, req, 120000),
  track: (id: string, layer: string, n: number, prompts: SegPrompt[]) =>
    call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/track`, { prompts }),
  /** Frame f of version n's roto matte as a mask piece over rect (clip pixels), and the clip's size. */
  matteFrame: (id: string, layer: string, n: number, frame: number) =>
    call<{ mask: string; rect: Rect; width: number; height: number }>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/matte`, { frame }),
  /** A new picture from a prompt (loki-dream) to start a scene from; dreamStatus follows it. */
  dream: (req: { prompt: string; negative?: string; size?: string; turbo?: boolean }) => call<DreamStatus>('POST', '/api/scene-dream', req),
  dreamStatus: (job: string) => call<DreamStatus>('GET', `/api/scene-dream/${enc(job)}`),
  untrack: (id: string, layer: string, n: number) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/untrack`, {}),
};

/**
 * The stills a layer at index i is generated from: the visible still layers below it with a version shown, as
 * "<layer id>:<version path>", bottom up. Mirrors sceneBelow in media-server/tasks/scene.go.
 */
export function stackBelow(s: Scene, i: number): string[] {
  const out: string[] = [];
  s.layers.slice(0, Math.max(0, i)).forEach((l) => {
    // A clean plate only shows round the matte of the layer above: not what it was made from.
    if (l.options?.cleanFor) return;
    const v = l.versions[l.active];
    if (!l.hidden && v && l.kind === 'still') out.push(`${l.id}:${v.path}`);
    if (paintInPicture(l)) out.push(`${l.id}:paint`);
  });
  return out;
}

/** A paint layer that is part of the picture (visible, painted, not a guide). Mirrors paintInPicture in scene.go. */
export function paintInPicture(l: SceneLayer): boolean {
  return l.kind === 'paint' && !l.hidden && !!l.mask && !l.options?.guide;
}

/**
 * Whether the stills under a layer's shown version are no longer the ones it was made from (a layer below was
 * moved, hidden or switched to another version). Versions from before this was recorded are never stale.
 */
export function versionStale(s: Scene, l: SceneLayer): boolean {
  const v = l.versions[l.active];
  if (!v || !v.below) return false;
  const now = stackBelow(
    s,
    s.layers.findIndex((x) => x.id === l.id)
  );
  return now.length !== v.below.length || now.some((x, k) => x !== v.below?.[k]);
}

/** The length a scene exports at (seconds): its own, else the longest visible motion clip, else 5. Mirrors sceneDuration. */
export function sceneLength(s: Scene): number {
  if (s.duration > 0) return s.duration;
  let d = 0;
  s.layers.forEach((l) => {
    const v = l.versions[l.active];
    // A layer played faster loops sooner.
    if (l.kind === 'motion' && v && !l.hidden) d = Math.max(d, (v.duration || 0) / layerTiming(l.options).speed);
  });
  return d > 0 ? d : 5;
}

/** A motion layer's own timing with defaults and limits applied. Mirrors sceneLayerTiming in scene_loop.go. */
export function layerTiming(o: LayerOptions | undefined): { speed: number; offset: number; volume: number } {
  const speed = o && typeof o.speed === 'number' && o.speed > 0 ? Math.min(4, Math.max(0.25, o.speed)) : 1;
  const offset = o && typeof o.offset === 'number' ? Math.max(0, o.offset) : 0;
  const volume = o && typeof o.volume === 'number' ? Math.min(2, Math.max(0, o.volume)) : 1;
  return { speed, offset, volume };
}

/**
 * The PTS multiplier a motion layer plays at in the scene (1 = as generated, 2 = half speed): its own speed, then
 * (synced loops) fitted so whole loops fill the scene length. Mirrors the export (sceneRenderTask).
 */
export function layerRate(s: Scene, l: SceneLayer): number {
  const own = 1 / layerTiming(l.options).speed;
  const d = l.versions[l.active]?.duration || 0;
  return s.freeLoops ? own : own * loopSpeed(sceneLength(s), d * own);
}

/**
 * Synced loops: the PTS multiplier that fits a whole number of a d-second clip's loops into a scene of length L
 * (1 = as generated, > 1 = slower). Mirrors loopSpeed in media-server/tasks/scene_loop.go.
 */
export function loopSpeed(L: number, d: number): number {
  if (!(L > 0) || !(d > 0)) return 1;
  const n = Math.max(1, Math.round(L / d));
  return L / (n * d);
}

/**
 * What a layer's generation is doing, for people: Queued (waiting for the GPU), Generating (the engine), Finishing
 * (the scene-lock pass: static edges, detail; also a Re-finish) or Paused, with its progress when known.
 */
export function jobPhase(job: SceneJob): { label: 'Queued' | 'Generating' | 'Finishing' | 'Paused'; percent: number | null } {
  const percent = job.total > 0 ? Math.round((job.done / job.total) * 100) : null;
  if (job.state === 'paused') return { label: 'Paused', percent };
  // The finish step queues behind nothing heavy: once it's the current step it is finishing, even if not started.
  if (job.step === 'scene-lock') return { label: 'Finishing', percent };
  if (job.state === 'pending') return { label: 'Queued', percent: null };
  return { label: 'Generating', percent };
}

/** The most generations a layer may have in flight (running + queued). Mirrors sceneMaxGenerations in scene.go. */
export const MAX_GENERATIONS = 8;

/** A layer's generations still to come: the running one plus those queued behind it. */
export function generationsInFlight(l: SceneLayer): number {
  return (l.job ? 1 : 0) + (l.queued?.length || 0);
}

/** Whether a layer (or the scene's export) has server work in flight. */
export function sceneBusy(s: Scene): boolean {
  const live = (st: string | undefined) => !!st && ['pending', 'in_progress', 'paused'].includes(st);
  return s.layers.some((l) => !!l.job || l.versions.some((v) => live(v.track?.state))) || live(s.render?.state) || !!s.stitching;
}

/**
 * Fold the server's copy into the local one: versions, job, error and the export are the server's; everything the
 * editor owns (masks, prompts, names, order, and each version's mask) stays local, so an autosave answer never undoes
 * typing in flight. A layer whose version list grew shows the new version (the server selected it).
 */
export function mergeServer(local: Scene, server: Scene): Scene {
  const byId = new Map(server.layers.map((l) => [l.id, l]));
  return {
    ...local,
    render: server.render,
    stitching: server.stitching,
    updated: server.updated,
    layers: local.layers.map((l) => {
      const s = byId.get(l.id);
      if (!s) return l;
      const grew = s.versions.length !== l.versions.length;
      const masks = new Map(l.versions.filter((v) => v.mask).map((v) => [v.path, v.mask]));
      let versions = masks.size ? s.versions.map((v) => (masks.has(v.path) ? { ...v, mask: masks.get(v.path) } : v)) : s.versions;
      // A stitch is made for the mask the server has: while a mask edit is still on its way it would show the old one.
      if (s.mask !== l.mask) versions = versions.map((v) => (v.stitch ? { ...v, stitch: null } : v));
      versions.forEach((v) => v.stitch && stitchMasks.set(v.stitch, s.mask));
      return { ...l, versions, job: s.job || null, queued: s.queued || [], error: s.error, active: grew ? s.active : l.active };
    }),
  };
}

/** The mask each stitch was made for (as the server had it when the stitch arrived). */
const stitchMasks = new WeakMap<SceneStitch, string>();

/**
 * The stitch to show for a layer: its shown version's, when it is a still that is stitched and its mask is still the
 * one the stitch was made for (a mask being edited shows plainly until the new stitch arrives).
 */
export function shownStitch(l: SceneLayer): SceneStitch | null {
  if (l.kind !== 'still' || l.options?.stitch === false) return null;
  const st = l.versions[l.active]?.stitch;
  if (!st) return null;
  const made = stitchMasks.get(st);
  return made === undefined || made === l.mask ? st : null;
}
