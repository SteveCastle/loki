// Session-only memory for the per-call custom description prompt.
//
// "Session-only" means in-memory for the lifetime of the renderer window —
// cleared on reload / app restart, and never persisted to disk. A simple
// module-level singleton fits the contract; no React state, no XState, no
// localStorage.

let lastCustomPrompt = '';
let cachedDefaultPrompt: string | null = null;

export function getLastCustomPrompt(): string {
  return lastCustomPrompt;
}

export function setLastCustomPrompt(value: string): void {
  // Only remember non-empty submissions so that clearing the textarea
  // doesn't wipe the previously-used prompt.
  if (value.trim() === '') return;
  lastCustomPrompt = value;
}

export function getCachedDefaultPrompt(): string | null {
  return cachedDefaultPrompt;
}

export function setCachedDefaultPrompt(value: string): void {
  cachedDefaultPrompt = value;
}

// Test-only reset. Exported under a deliberately ugly name to discourage
// production callers.
export function __resetCustomPromptStoreForTests(): void {
  lastCustomPrompt = '';
  cachedDefaultPrompt = null;
}
