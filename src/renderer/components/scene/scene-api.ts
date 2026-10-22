// The Scene editor's media-server API (media-server/scene_api.go). Types mirror
// tasks/scene.go's JSON. Electron authenticates with the viewer's bearer token;
// the web build rides the session cookie.
import { isElectron, mediaServerBase, store } from '../../platform';
import type { MaskOp, Rect } from './mask';

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
  /** motion: lock the static parts to the still (default on; see media-server/tasks/scene_lock.go) */
  lock?: boolean;
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
  boomerang: (id: string, layer: string, n: number) => call<Scene>('POST', `/api/scenes/${enc(id)}/layers/${enc(layer)}/versions/${n}/boomerang`, {}, 120000),
  render: (id: string, format: 'video' | 'still') => call<Scene>('POST', `/api/scenes/${enc(id)}/render`, { format }),
};

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
