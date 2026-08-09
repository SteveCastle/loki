// boot-clock — the first line of our code that runs in the renderer.
//
// Deliberately import-free, and deliberately the FIRST import in index.tsx, so
// that ES module evaluation order makes this the earliest timestamp the bundle
// can take. Everything else in the renderer (React, XState, the platform layer,
// the whole component tree) evaluates after it.
//
// The gap between `bundleStartAt` and the `imports-evaluated` mark is therefore
// exactly the cost of evaluating the app's module graph — a number that is
// otherwise invisible, and that was ~600ms of a cold launch before the hls.js
// and moment removals. Keep this file dependency-free or it stops measuring
// what it claims to.

export const bundleStartAt =
  typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : 0;
