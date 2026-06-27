import { useState, useEffect, useCallback } from 'react';
import { sessionStore } from '../platform';

const STORAGE_KEY = 'searchHistory';
const MAX_HISTORY = 100;

// Module-level shared state. The history is a single logical list, but it's
// consumed by more than one hook instance at a time (e.g. the command palette
// renders a QueryInput *and* records selections from its results surface). A
// listener registry keeps every mounted instance in sync, so a save made from
// one instance is immediately reflected in another instance's dropdown.
let historyCache: string[] | null = null;
const listeners = new Set<(history: string[]) => void>();

function emit(next: string[]) {
  historyCache = next;
  listeners.forEach((listener) => listener(next));
  sessionStore.set(STORAGE_KEY, next).catch(() => {});
}

async function loadHistory(): Promise<string[]> {
  if (historyCache !== null) return historyCache;
  try {
    const data = await sessionStore.get(STORAGE_KEY);
    historyCache = Array.isArray(data) ? data : [];
  } catch {
    historyCache = [];
  }
  return historyCache;
}

export function useSearchHistory() {
  const [history, setHistory] = useState<string[]>(historyCache ?? []);

  useEffect(() => {
    const listener = (next: string[]) => setHistory(next);
    listeners.add(listener);
    // Pull the current shared value (cached or freshly loaded) into this
    // instance. Subsequent mutations arrive via the listener.
    loadHistory().then((loaded) => setHistory(loaded));
    return () => {
      listeners.delete(listener);
    };
  }, []);

  const addSearch = useCallback((query: string) => {
    const trimmed = query.trim();
    if (!trimmed) return;
    const prev = historyCache ?? [];
    const next = [trimmed, ...prev.filter((item) => item !== trimmed)].slice(
      0,
      MAX_HISTORY
    );
    emit(next);
  }, []);

  const removeSearch = useCallback((query: string) => {
    const prev = historyCache ?? [];
    emit(prev.filter((item) => item !== query));
  }, []);

  const clearAll = useCallback(() => {
    emit([]);
  }, []);

  return { history, addSearch, removeSearch, clearAll };
}
