// First-use setup for the local AI engines. The media server can install an
// engine (the loki-* executable plus its model weights, 17-42 GB) on demand,
// but a download that size must never start unannounced: every job submission
// goes through ensureEngines(), which asks the media server what is missing
// and, when something is, shows a confirmation (EngineSetupDialog) with the
// size, then downloads with progress before the job is queued.
import { useSyncExternalStore } from 'react';
import {
  cancelModelDownload,
  fetchStatus,
  isDownloadingState,
  startModelDownload,
  type DepStatus,
} from '../../onboarding/api';
import type { JobRequest } from './intents';

export type EngineKind = 'retouch' | 'reshoot';

/** Dependency ids per engine: [executable, model weights]. */
export const ENGINES: Record<EngineKind, { title: string; ids: [string, string] }> = {
  retouch: { title: 'AI image editing', ids: ['loki-retouch', 'qwen-image-2.1'] },
  reshoot: { title: 'AI video from references', ids: ['loki-reshoot', 'minimax-h3-ref2va'] },
};

/** Which engine a job needs, from the task id at the start of its input. */
export function engineOfJob(job: Pick<JobRequest, 'input'>): EngineKind | null {
  const task = job.input.trim().split(/\s+/)[0];
  if (task === 'retouch' || task === '4kify') return 'retouch';
  if (task === 'reshoot') return 'reshoot';
  return null;
}

export interface EngineNeed {
  kind: EngineKind;
  title: string;
  /** What would be downloaded (already-installed parts are omitted). */
  items: DepStatus[];
  bytes: number;
}

const isInstalled = (d: DepStatus) => d.state === 'installed' || d.state === 'ready';

/**
 * What an engine still needs, or null when it can run now: everything is
 * installed, the executable is already on the server's PATH (the server then
 * uses it and installs nothing), or the deps API doesn't know the engine
 * (older server: don't gate, the job reports its own error).
 */
export function planEngineSetup(status: DepStatus[], kind: EngineKind): EngineNeed | null {
  const [toolId, modelId] = ENGINES[kind].ids;
  const tool = status.find((d) => d.id === toolId);
  const model = status.find((d) => d.id === modelId);
  if (!tool || !model) return null;
  if (isInstalled(tool) && tool.detail?.source === 'path') return null;
  const items = [tool, model].filter((d) => !isInstalled(d));
  if (items.length === 0) return null;
  return {
    kind,
    title: ENGINES[kind].title,
    items,
    bytes: items.reduce((n, d) => n + (d.size_bytes ?? 0), 0),
  };
}

// --- dialog state -------------------------------------------------------------

export type SetupPhase = 'confirm' | 'downloading' | 'failed';

export interface SetupState {
  need: EngineNeed;
  phase: SetupPhase;
  bytesDone: number;
  bytesTotal: number;
  error?: string;
}

/** How a setup request ended. */
export type SetupResult = 'ready' | 'cancelled' | 'background';

/** Thrown by submitJobs when the engine was not made ready; callers treat it as "don't queue", not as a failure. */
export class EngineSetupDeferred extends Error {
  readonly result: Exclude<SetupResult, 'ready'>;

  constructor(result: Exclude<SetupResult, 'ready'>) {
    super(result === 'background' ? 'engine download continues in the background' : 'engine setup cancelled');
    this.name = 'EngineSetupDeferred';
    this.result = result;
  }
}

let state: SetupState | null = null;
let settle: ((r: SetupResult) => void) | null = null;
let pollTimer: ReturnType<typeof setTimeout> | null = null;
let apiBase = '';
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((l) => l());
}

function set(next: SetupState | null) {
  state = next;
  emit();
}

function finish(r: SetupResult) {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = null;
  const s = settle;
  settle = null;
  set(null);
  if (s) s(r);
}

export function getEngineSetup(): SetupState | null {
  return state;
}

export function useEngineSetup(): SetupState | null {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
    () => null
  );
}

function progressOf(need: EngineNeed, status: DepStatus[]): { done: number; total: number } {
  let done = 0;
  let total = 0;
  need.items.forEach((item) => {
    const cur = status.find((d) => d.id === item.id) ?? item;
    const size = cur.size_bytes ?? item.size_bytes ?? 0;
    total += size;
    if (isInstalled(cur)) done += size;
    else if (isDownloadingState(cur.state)) done += Math.min(size, cur.detail?.bytes_done ?? 0);
  });
  return { done, total };
}

async function poll() {
  if (!state || state.phase !== 'downloading') return;
  const { need } = state;
  try {
    const status = await fetchStatus(apiBase);
    if (!state || state.phase !== 'downloading') return;
    const cur = need.items.map((i) => status.find((d) => d.id === i.id));
    const failed = cur.find((d) => d && (d.state === 'failed' || d.state === 'cancelled'));
    if (failed) {
      set({ ...state, phase: 'failed', error: failed.error || failed.detail?.error || 'The download did not finish.' });
      return;
    }
    if (cur.every((d) => d && isInstalled(d))) {
      finish('ready');
      return;
    }
    const p = progressOf(need, status);
    set({ ...state, bytesDone: p.done, bytesTotal: p.total });
  } catch {
    // transient: keep polling
  }
  pollTimer = setTimeout(poll, 1500);
}

/** The user agreed: start (or resume) every missing piece and follow its progress. */
export async function beginEngineDownload(): Promise<void> {
  if (!state) return;
  const { need } = state;
  set({ ...state, phase: 'downloading', bytesDone: 0, bytesTotal: need.bytes, error: undefined });
  try {
    await Promise.all(
      need.items.filter((d) => !isDownloadingState(d.state)).map((d) => startModelDownload(d.id, apiBase))
    );
  } catch (e) {
    if (state) set({ ...state, phase: 'failed', error: `Could not start the download (${e instanceof Error ? e.message : String(e)}).` });
    return;
  }
  poll();
}

/** Close without downloading, or stop the download in progress (what was fetched is kept for next time). */
export function cancelEngineSetup(): void {
  if (!state) return;
  const { need, phase } = state;
  if (phase === 'downloading') {
    need.items.forEach((d) => {
      cancelModelDownload(d.id, apiBase).catch(() => undefined);
    });
  }
  finish('cancelled');
}

/** Leave the download running on the server and let the user get on with things. */
export function continueEngineSetupInBackground(): void {
  if (state?.phase === 'downloading') finish('background');
}

/**
 * Make sure the engines these jobs need are installed, asking first when that
 * means a download. Resolves 'ready' when the jobs may be queued.
 */
export async function ensureEngines(jobs: Array<Pick<JobRequest, 'input'>>, base: string): Promise<SetupResult> {
  const kinds = Array.from(new Set(jobs.map(engineOfJob).filter((k): k is EngineKind => k !== null)));
  if (kinds.length === 0) return 'ready';
  let status: DepStatus[];
  try {
    status = await fetchStatus(base);
    if (!Array.isArray(status)) return 'ready';
  } catch {
    return 'ready'; // deps API unreachable: don't block, the job reports its own error
  }
  for (const kind of kinds) {
    const need = planEngineSetup(status, kind);
    if (need) {
      if (state) return 'cancelled'; // another request is already on screen
      apiBase = base;
      // eslint-disable-next-line no-await-in-loop
      const result = await new Promise<SetupResult>((resolve) => {
        settle = resolve;
        // Something already downloading (started elsewhere) skips the question.
        const running = need.items.some((d) => isDownloadingState(d.state));
        set({ need, phase: 'confirm', bytesDone: 0, bytesTotal: need.bytes });
        if (running) beginEngineDownload();
      });
      if (result !== 'ready') return result;
    }
  }
  return 'ready';
}
