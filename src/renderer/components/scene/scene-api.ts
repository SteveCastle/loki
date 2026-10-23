// The Scene editor's media-server API (media-server/scene_api.go). Types mirror
// tasks/scene.go's JSON. Electron authenticates with the viewer's bearer token;
// the web build rides the session cookie.
import { isElectron, mediaServerBase, store } from '../../platform';
import type { MaskOp, MaskOpts, Rect } from './mask';
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
  created: string;
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
  /**
   * still: fit the mask to the shown version, adding wherever it changed the picture near the mask, so a result
   * that corrects a rough sketch's shape (or casts a shadow) is not cut off at the sketch's edge.
   */
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

export type FinishMode = 'sketch' | 'colour';

/**
 * How a finish layer's words are read: the instruction around them. Style-neutral ("this picture", never
 * "photograph"): the scene may be a photo or a painting. Mirrors sceneFinishGrounding in media-server/tasks/scene_finish.go.
 */
export const FINISH_GROUNDING: Record<FinishMode, string> = {
  sketch:
    'In <image1>, a rough painted sketch stands in for {what}: replace it with {what} as it would really appear in this picture. ' +
    'The sketch is a crude placeholder that only marks where it goes, roughly how big it is and a hint of its colours, ' +
    'so do not copy its flat paint or hard outline: correct its shape and proportions, and give it the real look, light, texture and focus ' +
    "it would have here, in the picture's own style. Leave everything that was not sketched as it is.",
  colour:
    'Colour <image1> using the rough colours painted on {guide} as a guide: {what}. ' +
    'The paint is loose and only marks which colour goes where: give each whole object or area the colour painted mostly over it, ' +
    'following the shapes and edges of <image1> rather than the edges of the brush strokes. ' +
    'Keep every line and detail of <image1> as it is, and finish with clean, natural shading.',
};

/** Fill a grounding's placeholders; with no words a "…: {what}." clause folds away, any other {what} reads "the real thing". Mirrors composeFinish. */
export function composeFinish(grounding: string, what: string, guide: string): string {
  const w = what.trim().replace(/\.+$/, '');
  let out = grounding;
  if (!w) out = out.split(': {what}.').join('.');
  return out
    .split('{what}')
    .join(w || 'the real thing')
    .split('{guide}')
    .join(guide);
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

function authHeaders(): Record<string, string> {
  if (!isElectron) return {};
  const token = store.get('authToken', null) as string | null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function call<T>(method: string, path: string, body?: unknown, timeoutMs = 20000): Promise<T> {
  const headers: Record<string, string> = { ...authHeaders() };
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
      redirect: 'error',
    });
  } catch (e) {
    const msg = e instanceof Error && e.name === 'TimeoutError' ? 'the media server did not answer in time' : 'cannot reach the media server';
    throw new SceneApiError(msg, 0);
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
    if (res.status === 401 || res.status === 403) msg = 'Not signed in to the media server. Sign in from the viewer, then reopen this window.';
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
  generate: (id: string, layer: string) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/generate`, {}, 60000),
  /** Cancel one generation (running or queued) by its job id, or all of the layer's when job is omitted. */
  cancel: (id: string, layer: string, job?: string) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/cancel`, job ? { job } : {}),
  deleteVersion: (id: string, layer: string, n: number) => call<Scene>('DELETE', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}`),
  /** Discard every version of a layer but n. */
  keepOnly: (id: string, layer: string, n: number) => call<Scene>('DELETE', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions?keep=${n}`),
  /** Redo version n's finish (edges, detail) from the engine's output with the layer's current settings: a new version. */
  refinish: (id: string, layer: string, n: number) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/refinish`, {}),
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
  return s.layers.some((l) => !!l.job || l.versions.some((v) => live(v.track?.state))) || live(s.render?.state);
}

/**
 * Fold the server's copy into the local one: versions, job, error and the export are the server's; everything the
 * editor owns (masks, prompts, names, order) stays local, so an autosave answer never undoes typing in flight. A
 * layer whose version list grew shows the new version (the server selected it).
 */
export function mergeServer(local: Scene, server: Scene): Scene {
  const byId = new Map(server.layers.map((l) => [l.id, l]));
  return {
    ...local,
    render: server.render,
    updated: server.updated,
    layers: local.layers.map((l) => {
      const s = byId.get(l.id);
      if (!s) return l;
      const grew = s.versions.length !== l.versions.length;
      return { ...l, versions: s.versions, job: s.job || null, queued: s.queued || [], error: s.error, active: grew ? s.active : l.active };
    }),
  };
}
