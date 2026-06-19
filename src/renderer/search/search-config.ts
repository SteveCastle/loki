// Shared debounce window for keystroke-driven type-ahead searches: the tag
// fuzzy search (useTagSearch) and the IPC-backed suggestion queries — path
// lookups and per-category counts (SuggestionSections). One knob so every
// downstream search across the taxonomy sidebar and the command palette
// coalesces bursts of keystrokes on the same cadence.
export const SEARCH_DEBOUNCE_MS = 250;
