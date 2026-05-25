export type Category = 'bundled' | 'optional' | 'model';

export interface DepStatus {
  id: string;
  category: Category;
  name: string;
  state: 'ready' | 'missing' | 'broken' | 'installed' | 'not_installed' | 'queued' | 'downloading' | 'verifying' | 'failed' | 'cancelled';
  version?: string;
  size_bytes?: number;
  path?: string;
  error?: string;
  detail?: any;
}

export async function fetchStatus(): Promise<DepStatus[]> {
  const r = await fetch('/api/deps/status');
  if (!r.ok) throw new Error(`status ${r.status}`);
  return r.json();
}

export async function startModelDownload(id: string): Promise<void> {
  const r = await fetch(`/api/deps/models/${encodeURIComponent(id)}/download`, { method: 'POST' });
  if (!r.ok && r.status !== 202) throw new Error(`download ${r.status}`);
}

export async function cancelModelDownload(id: string): Promise<void> {
  await fetch(`/api/deps/models/${encodeURIComponent(id)}/cancel`, { method: 'POST' });
}

export async function deleteModel(id: string): Promise<void> {
  await fetch(`/api/deps/models/${encodeURIComponent(id)}`, { method: 'DELETE' });
}
