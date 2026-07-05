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
export async function fetchStatus(base = ''): Promise<DepStatus[]> {
  const r = await fetch(`${base}/api/deps/status`);
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function startModelDownload(id: string, base = ''): Promise<void> {
  const r = await fetch(`${base}/api/deps/models/${encodeURIComponent(id)}/download`, { method: 'POST' });
  if (!r.ok && r.status !== 202) throw new Error(`download ${r.status}`);
}

export async function cancelModelDownload(id: string, base = ''): Promise<void> {
  await fetch(`${base}/api/deps/models/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export async function deleteModel(id: string, base = ''): Promise<void> {
  await fetch(`${base}/api/deps/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
