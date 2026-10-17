// When each saved item (a saved Transform task, a server Workflow) was last
// picked, for the palette's Saved list "Recent" order. Keys are the list's
// entry keys (savedTransformKey / savedWorkflowKey). Per browser, like the
// saved transforms themselves; storage failures just mean no history.

const RECENT_KEY = 'loki.saved.recent.v1';
const SORT_KEY = 'loki.saved.sort.v1';
/** Enough history for any real list; the oldest entries are dropped. */
const MAX_ENTRIES = 500;

export type SavedSort = 'name' | 'recent';

export const savedTransformKey = (intent: string, id: string) => `t:${intent}:${id}`;
export const savedWorkflowKey = (id: string) => `w:${id}`;

export function loadRecent(): Record<string, number> {
  try {
    const raw = window.localStorage.getItem(RECENT_KEY);
    const v = raw ? JSON.parse(raw) : {};
    return v && typeof v === 'object' ? v : {};
  } catch {
    return {};
  }
}

export function markUsed(key: string, at: number = Date.now()): void {
  try {
    const all = { ...loadRecent(), [key]: at };
    const keys = Object.keys(all);
    if (keys.length > MAX_ENTRIES) {
      keys.sort((a, b) => all[a] - all[b]).slice(0, keys.length - MAX_ENTRIES).forEach((k) => delete all[k]);
    }
    window.localStorage.setItem(RECENT_KEY, JSON.stringify(all));
  } catch {
    // storage unavailable
  }
}

export function loadSavedSort(): SavedSort {
  try {
    return window.localStorage.getItem(SORT_KEY) === 'recent' ? 'recent' : 'name';
  } catch {
    return 'name';
  }
}

export function saveSavedSort(sort: SavedSort): void {
  try {
    window.localStorage.setItem(SORT_KEY, sort);
  } catch {
    // storage unavailable
  }
}
