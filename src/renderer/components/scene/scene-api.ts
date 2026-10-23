// The Scene editor's media-server API (media-server/scene_api.go). Types mirror
// tasks/scene.go's JSON. Electron authenticates with the viewer's bearer token;
// the web build rides the session cookie.
import { isElectron, mediaServerBase, store } from '../../platform';
import type { MaskOp, Rect } from './mask';
import type { SceneRefItem } from './refs';

export type LayerKind = 'still' | 'motion';

export interface SceneVersion {
  path: string;
  region: Rect;
  prompt: string;
  duration?: number;
  audio?: boolean;
  note?: string;
  jobId?: string;
  created: string;
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
  /** motion: lock the static parts to the still (default on; see media-server/tasks/scene_lock.go) */
  lock?: boolean;
  /** motion: output at the region's size with the still's fine detail (default on) */
  detail?: boolean;
  seed?: number;
}

export interface SceneLayer {
  id: string;
  name: string;
  kind: LayerKind;
  hidden?: boolean;
  region: Rect;
  mask: string;
  shapes?: MaskOp[];
  feather: number;
  prompt: string;
  options?: LayerOptions;
  versions: SceneVersion[];
  active: number;
  job?: SceneJob | null;
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
  cancel: (id: string, layer: string) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/cancel`, {}),
  deleteVersion: (id: string, layer: string, n: number) => call<Scene>('DELETE', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}`),
  /** Discard every version of a layer but n. */
  keepOnly: (id: string, layer: string, n: number) => call<Scene>('DELETE', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions?keep=${n}`),
  /** Redo version n's finish (edges, detail) from the engine's output with the layer's current settings: a new version. */
  refinish: (id: string, layer: string, n: number) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/refinish`, {}),
  loop: (id: string, layer: string, n: number, kind: LoopKind) =>
    call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/loop`, { kind }, 180000),
  /** Where version n moves: a PNG data URL (alpha = motion) covering `region`. */
  motionMap: (id: string, layer: string, n: number) =>
    call<{ dataUrl: string; region: Rect }>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/motion`, {}, 120000),
  render: (id: string, format: 'video' | 'still') => call<Scene>('POST', `/api/scenes/${enc(id)}/render`, { format }),
};

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

/** Whether a layer (or the scene's export) has server work in flight. */
export function sceneBusy(s: Scene): boolean {
  return s.layers.some((l) => !!l.job) || (!!s.render && ['pending', 'in_progress', 'paused'].includes(s.render.state));
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
      return { ...l, versions: s.versions, job: s.job || null, error: s.error, active: grew ? s.active : l.active };
    }),
  };
}
