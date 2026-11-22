// A temporary override of what the detail view shows, without touching the
// library or its cursor. The metadata Duplicates tab uses it to look at a
// group member in place; whoever sets a peek owns clearing it (the tab does
// on unmount and whenever the in-context item changes), and the detail
// view's close button clears it too.
import { useSyncExternalStore } from 'react';

let peekPath: string | null = null;
const listeners = new Set<() => void>();

export function setDetailPeek(path: string | null): void {
  if (path === peekPath) return;
  peekPath = path;
  listeners.forEach((l) => l());
}

export function getDetailPeek(): string | null {
  return peekPath;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useDetailPeek(): string | null {
  return useSyncExternalStore(subscribe, getDetailPeek);
}
