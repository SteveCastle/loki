// Isolated tag-search worker bootstrap.
//
// The webpack-recognised `new Worker(new URL('...', import.meta.url))` form is
// what pins the worker as its own chunk at build time — but it also embeds the
// `import.meta` meta-property, which the CommonJS ts-jest transform refuses to
// compile (TS1343). Keeping it in this one tiny module, loaded lazily via
// require() from tag-search-service ONLY when a real Worker exists, keeps the
// rest of the search service importable (and unit-testable) under jsdom.
export function createTagSearchWorker(): Worker {
  return new Worker(
    new URL('../components/taxonomy/tag-search.worker.ts', import.meta.url)
  );
}
