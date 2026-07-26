// Small deterministic RNG helpers shared by the shuffle (filter.ts) and the
// battle pairing (battle-pairing.ts). Both need "same seed string → same
// order" so a re-render doesn't reshuffle the list, while a new seed must
// produce a genuinely unrelated order.

// FNV-1a, 32-bit. Turns a seed string into a number to feed the mixers below.
export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

// splitmix32 finalizer — full avalanche, so seeds that differ by one bit
// produce unrelated outputs. This matters because seeds are minted from a
// counter: without avalanche, consecutive seeds shift every rank by roughly
// the same amount and the "new" shuffle is the old one rotated.
export function mix32(x: number): number {
  let h = x | 0;
  h = Math.imul(h ^ (h >>> 16), 0x21f0aaad);
  h = Math.imul(h ^ (h >>> 15), 0x735a2d97);
  return (h ^ (h >>> 15)) >>> 0;
}

// mulberry32 — tiny deterministic PRNG. Returns floats in [0, 1).
export function mulberry32(seed: number): () => number {
  let a = mix32(seed) | 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const seededRng = (seed: string) => mulberry32(hashString(seed));

// Fisher-Yates, seeded. O(n) and uniform over all permutations (unlike
// sorting by a per-item hash, which is O(n log n) and only as uniform as the
// hash). Does not mutate the input.
export function seededShuffle<T>(items: readonly T[], seed: string): T[] {
  const out = items.slice();
  const rng = seededRng(seed);
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = out[i];
    out[i] = out[j];
    out[j] = tmp;
  }
  return out;
}
