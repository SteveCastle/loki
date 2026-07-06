export type Category = 'bundled' | 'optional' | 'model' | 'tool';

export interface DepStatus {
  id: string;
  category: Category;
  name: string;
  /** User-facing capability this dependency unlocks (e.g. "Auto-tagging"). */
  feature?: string;
  description?: string;
  state: 'ready' | 'missing' | 'broken' | 'installed' | 'not_installed' | 'queued' | 'downloading' | 'verifying' | 'failed' | 'cancelled';
  version?: string;
  size_bytes?: number;
  path?: string;
  error?: string;
  detail?: any;
}

/** True for the states where a download can be started. */
export function isDownloadableState(s: DepStatus['state']): boolean {
  return s === 'missing' || s === 'failed' || s === 'cancelled';
}

/** True while an install is in flight. */
export function isDownloadingState(s: DepStatus['state']): boolean {
  return s === 'downloading' || s === 'queued' || s === 'verifying';
}

// `base` lets Electron (different origin from the media server) reach the
// same endpoints; the web UI passes '' and stays same-origin.
// Every fetch carries a timeout: an untimed request to the media-server
// origin can pin one of Chromium's 6 per-origin sockets indefinitely and
// starve every other call to the server.
export async function fetchStatus(base = ''): Promise<DepStatus[]> {
  const r = await fetch(`${base}/api/deps/status`, {
    signal: AbortSignal.timeout(10000),
  });
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

// Shared, deduplicated status fetch. Point-of-use surfaces (one per
// "Generate" button, remounted on every media change) used to each fire
// their own /api/deps/status request; a burst of those was enough to occupy
// the whole socket pool. All callers within the TTL share one request.
const statusCache = new Map<string, { at: number; promise: Promise<DepStatus[]> }>();
const STATUS_CACHE_TTL_MS = 15_000;

export function fetchStatusShared(
  base = '',
  force = false
): Promise<DepStatus[]> {
  const now = Date.now();
  const hit = statusCache.get(base);
  if (!force && hit && now - hit.at < STATUS_CACHE_TTL_MS) return hit.promise;
  const promise = fetchStatus(base).catch((e) => {
    // Don't cache failures — let the next caller retry.
    if (statusCache.get(base)?.promise === promise) statusCache.delete(base);
    throw e;
  });
  statusCache.set(base, { at: now, promise });
  return promise;
}

export async function startModelDownload(id: string, base = ''): Promise<void> {
  const r = await fetch(`${base}/api/deps/models/${encodeURIComponent(id)}/download`, {
    method: 'POST',
    signal: AbortSignal.timeout(10000),
  });
  if (!r.ok && r.status !== 202) throw new Error(`download ${r.status}`);
}

export async function cancelModelDownload(id: string, base = ''): Promise<void> {
  await fetch(`${base}/api/deps/models/${encodeURIComponent(id)}/cancel`, {
    method: 'POST',
    signal: AbortSignal.timeout(10000),
  });
}

export async function deleteModel(id: string, base = ''): Promise<void> {
  await fetch(`${base}/api/deps/models/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    signal: AbortSignal.timeout(10000),
  });
}
