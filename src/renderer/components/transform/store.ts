// Tiny external store for the Transform Studio: the palette (which unmounts its
// contents when it hides) opens the studio through here, and a root-level host
// renders it. Also persists the last-used settings per intent.
import { useSyncExternalStore } from 'react';
import type { IntentId, TransformSettings, JobRequest } from './intents';

export type StudioPhase = 'choose' | 'shape' | 'review';

export interface StudioRequest {
  /** The working set: right-clicked file plus any multi-selection, in order. */
  paths: string[];
  /** Playback position of a single video target, seconds (frame sampling). */
  videoTime?: number;
  intent?: IntentId;
  phase?: StudioPhase;
}

let current: StudioRequest | null = null;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

export function openTransformStudio(req: StudioRequest): void {
  current = req;
  emit();
}

export function closeTransformStudio(): void {
  if (current === null) return;
  current = null;
  emit();
}

export function getStudioRequest(): StudioRequest | null {
  return current;
}

export function useStudioRequest(): StudioRequest | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => current,
    () => null
  );
}

// --- remembered settings -----------------------------------------------------

const STORAGE_KEY = 'loki.transform.settings.v1';
type Remembered = Partial<Record<IntentId, Partial<TransformSettings>>>;

// Per-intent text (prompts, descriptions) is deliberately NOT remembered: a
// stale "make it night" must never ride along into the next image.
const REMEMBER_KEYS: Array<keyof TransformSettings> = [
  'sizeMode',
  'scale',
  'wallpaperTarget',
  'quality',
  'shake',
  'duration',
  'refSize',
  'noAudio',
  'keepSoundtrack',
];

export function loadRemembered(id: IntentId): Partial<TransformSettings> {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const all = JSON.parse(raw) as Remembered;
    return all[id] || {};
  } catch {
    return {};
  }
}

export function saveRemembered(id: IntentId, s: TransformSettings): void {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const all: Remembered = raw ? JSON.parse(raw) : {};
    const picked: Record<string, unknown> = {};
    REMEMBER_KEYS.forEach((k) => {
      picked[k] = s[k];
    });
    all[id] = picked as Partial<TransformSettings>;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable: settings just don't stick
  }
}

// --- submission ---------------------------------------------------------------

export interface SubmitContext {
  mediaServerBase: string;
  authToken: string | null;
}

/** POST each job to the media server's /create. Resolves with the ids; rejects on the first failure. */
export async function submitJobs(
  jobs: JobRequest[],
  ctx: SubmitContext
): Promise<string[]> {
  const ids: string[] = [];
  for (const job of jobs) {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (ctx.authToken) headers.Authorization = `Bearer ${ctx.authToken}`;
    // eslint-disable-next-line no-await-in-loop
    const res = await fetch(`${ctx.mediaServerBase}/create`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ input: job.input, fields: job.fields }),
      signal: AbortSignal.timeout(10000),
      redirect: 'error',
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    try {
      // eslint-disable-next-line no-await-in-loop
      const body = (await res.json()) as { id?: string };
      if (body && body.id) ids.push(body.id);
    } catch {
      // older servers may not echo an id
    }
  }
  return ids;
}
