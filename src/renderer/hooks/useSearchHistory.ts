import { useState, useEffect, useCallback } from 'react';
import { sessionStore } from '../platform';

const STORAGE_KEY = 'searchHistory';
const MAX_HISTORY = 100;

let historyCache: string[] | null = null;

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

function persistHistory(history: string[]) {
  historyCache = history;
  sessionStore.set(STORAGE_KEY, history).catch(() => {});
}

export function useSearchHistory() {
  const [history, setHistory] = useState<string[]>(historyCache ?? []);

  useEffect(() => {
    loadHistory().then(setHistory);
  }, []);

  const addSearch = useCallback((query: string) => {
    const trimmed = query.trim();
    if (!trimmed) return;
    setHistory((prev) => {
      const filtered = prev.filter((item) => item !== trimmed);
      const next = [trimmed, ...filtered].slice(0, MAX_HISTORY);
      persistHistory(next);
      return next;
    });
  }, []);

  const removeSearch = useCallback((query: string) => {
    setHistory((prev) => {
      const next = prev.filter((item) => item !== query);
      persistHistory(next);
      return next;
    });
  }, []);

  const clearAll = useCallback(() => {
    persistHistory([]);
    setHistory([]);
  }, []);

  return { history, addSearch, removeSearch, clearAll };
}
