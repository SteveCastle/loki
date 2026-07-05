import { useCallback, useEffect, useState } from 'react';
import {
  fetchStatus,
  isDownloadableState,
  isDownloadingState,
  startModelDownload,
  type DepStatus,
} from './api';
import { depsApiBase } from './requirements';

// Tracks one dependency for a point-of-use surface (e.g. the per-file
// "Generate" buttons). Fetches once on mount and polls only while a download
// is in flight, so idle panels stay cheap. When the deps API is unreachable
// `dep` stays null and callers should not gate their action.
export function useDepRequirement(depId: string): {
  dep: DepStatus | null;
  needsDownload: boolean;
  downloading: boolean;
  pct: number;
  download: () => Promise<void>;
} {
  const [dep, setDep] = useState<DepStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      const items = await fetchStatus(depsApiBase);
      setDep(items.find((d) => d.id === depId) ?? null);
    } catch {
      setDep(null);
    }
  }, [depId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const downloading = !!dep && isDownloadingState(dep.state);
  useEffect(() => {
    if (!downloading) return undefined;
    const t = window.setInterval(refresh, 2000);
    return () => window.clearInterval(t);
  }, [downloading, refresh]);

  const needsDownload = !!dep && isDownloadableState(dep.state);
  const inst = dep?.detail || {};
  const done: number = inst.bytes_done ?? 0;
  const total: number = inst.bytes_total ?? dep?.size_bytes ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : 0;

  const download = async () => {
    await startModelDownload(depId, depsApiBase);
    await refresh();
  };

  return { dep, needsDownload, downloading, pct, download };
}
