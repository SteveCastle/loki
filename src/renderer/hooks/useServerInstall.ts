import { useCallback, useEffect, useState } from 'react';
import { invoke, on, isElectron } from '../platform';

// Mirrors ServerInstallStatus / ServerInstallProgress in
// src/main/server-install.ts (main process).
export type ServerInstallPhase =
  | 'idle'
  | 'resolving'
  | 'downloading'
  | 'installing'
  | 'starting'
  | 'done'
  | 'error';

export interface ServerInstallProgress {
  phase: ServerInstallPhase;
  received?: number;
  total?: number;
  version?: string;
  error?: string;
}

export interface ServerInstallStatus {
  supported: boolean;
  running: boolean;
  installedPath: string | null;
  prompted: boolean;
  progress: ServerInstallProgress;
}

export const isServerInstallBusy = (p: ServerInstallProgress) =>
  p.phase === 'resolving' ||
  p.phase === 'downloading' ||
  p.phase === 'installing' ||
  p.phase === 'starting';

export function describeServerInstall(p: ServerInstallProgress): string {
  switch (p.phase) {
    case 'resolving':
      return 'Finding the latest server release…';
    case 'downloading': {
      const mb = (n = 0) => (n / 1024 / 1024).toFixed(0);
      return p.total
        ? `Downloading… ${mb(p.received)} / ${mb(p.total)} MB`
        : `Downloading… ${mb(p.received)} MB`;
    }
    case 'installing':
      return 'Installing…';
    case 'starting':
      return 'Starting the server…';
    case 'done':
      return 'The server is running.';
    case 'error':
      return p.error || 'Install failed';
    default:
      return '';
  }
}

/**
 * Desktop-only: status of the local Lowkey Media Server install plus the
 * actions to install / start it. Progress streams from the main process, so
 * every mounted consumer (first-run prompt, palette widget) stays in sync.
 * `status` stays null in web mode — the SPA is served by the server itself.
 */
export function useServerInstall(enabled = true) {
  const [status, setStatus] = useState<ServerInstallStatus | null>(null);
  const [progress, setProgress] = useState<ServerInstallProgress>({
    phase: 'idle',
  });

  const refresh = useCallback(async () => {
    if (!isElectron) return null;
    try {
      const s = (await invoke('server-install-status')) as ServerInstallStatus;
      setStatus(s);
      setProgress(s.progress);
      return s;
    } catch {
      return null;
    }
  }, []);

  useEffect(() => {
    if (!isElectron || !enabled) return undefined;
    refresh();
    return on('server-install-progress', (p) => {
      setProgress(p as ServerInstallProgress);
      const phase = (p as ServerInstallProgress).phase;
      if (phase === 'done' || phase === 'error') refresh();
    });
  }, [enabled, refresh]);

  const install = useCallback(() => invoke('server-install'), []);
  const start = useCallback(() => invoke('server-start'), []);
  const dismissPrompt = useCallback(
    () => invoke('server-install-dismiss-prompt'),
    []
  );

  return { status, progress, refresh, install, start, dismissPrompt };
}
