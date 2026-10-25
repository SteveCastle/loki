// Debounce window for the keystroke-driven queries that hit the database — the
// per-category counts in SuggestionSections. Those serialize on the single DB
// connection, so a burst of keystrokes must coalesce into one query.
export const SEARCH_DEBOUNCE_MS = 250;

// The tag type-ahead itself (useTagSearch). Matching runs off-thread and
// answers in ~5–20ms even over the full ~190K-tag table (see tag-matcher.ts),
// so this only coalesces a fast typist's burst rather than protecting anything.
export const TAG_SEARCH_DEBOUNCE_MS = 40;
