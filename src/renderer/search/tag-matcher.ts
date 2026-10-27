// tag-matcher — the ranked type-ahead matcher behind tag search.
//
// Replaces Fuse.js. Fuse runs a Bitap fuzzy match against EVERY entry on EVERY
// query, and over a real tag table (~190K rows, 97% autotagger "Suggested")
// that measured 235ms for a 2-char query and over a second for "blonde hair" —
// off the main thread, but the results trailed the typing by that much. Its
// ranking was also poor for type-ahead: "al" ranked "Classical" above
// "Al Pacino", and any query that fuzzily hit the CATEGORY name ("sug") matched
// all 183K Suggested tags.
//
// This matcher ranks in tiers, cheapest first, and only falls through to the
// typo-tolerant pass when the exact tiers can't fill the requested limit:
//
//   0  label equals the query
//   1  label starts with the query as whole words ("al" -> "Al Pacino")
//   2  label starts with the query
//   3  the query starts a word inside the label
//   4  every query word starts some label word (any order)
//   5  the query occurs anywhere in the label
//   6  every query word occurs somewhere in the label
//   7  fuzzy: every query word is within 1–2 edits of a label word's prefix
//   8  the query starts the category name (fills leftover slots only)
//
// Within a tier, shorter labels rank first (the closest match to what was
// typed), then alphabetical for stability. Normalisation folds case, accents,
// and the `_`/`-` separators the autotagger uses, so "blonde hair" finds
// "blonde_hair" exactly.
//
// Pure and dependency-free: the worker and the synchronous fallback in
// tag-search-service.ts share this one implementation.

export interface PackedTags {
  /** Labels joined by SEP, in source order. */
  labels: string;
  /** Per-tag index into `categories`. */
  categoryIds: Int32Array;
  /** Distinct category names. */
  categories: string[];
}

export const SEP = '\u0000';

export interface TagMatchIndex {
  norm: string[];
  /** Bitmask of the a–z / 0–9 buckets each normalised label contains. */
  masks: Int32Array;
  categoryIds: Int32Array;
  categoryNorm: string[];
}

const COMBINING = /[̀-ͯ]/g;
const SEPARATORS = /[\s_\-./]+/g;

export function normalize(s: string): string {
  let out = s.toLowerCase();
  // NFD + strip combining marks folds accents ("Beyoncé" -> "beyonce"). Skip
  // the cost for the (overwhelmingly common) pure-ASCII label.
  // eslint-disable-next-line no-control-regex
  if (/[^\x00-\x7f]/.test(out)) out = out.normalize('NFD').replace(COMBINING, '');
  return out.replace(SEPARATORS, ' ').trim();
}

function charMask(s: string): number {
  let m = 0;
  for (let i = 0; i < s.length; i += 1) {
    const c = s.charCodeAt(i);
    if (c >= 97 && c <= 122) m |= 1 << (c - 97);
    else if (c >= 48 && c <= 57) m |= 1 << 26;
    else if (c !== 32) m |= 1 << 27;
  }
  return m;
}

function popcount(n: number): number {
  let v = n - ((n >>> 1) & 0x55555555);
  v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
  return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

export function packTags(tags: { label: string; category: string }[]): PackedTags {
  const categoryIds = new Int32Array(tags.length);
  const categories: string[] = [];
  const catIndex = new Map<string, number>();
  const labels = new Array<string>(tags.length);
  for (let i = 0; i < tags.length; i += 1) {
    const t = tags[i];
    labels[i] = t.label;
    const cat = t.category ?? '';
    let id = catIndex.get(cat);
    if (id === undefined) {
      id = categories.length;
      categories.push(cat);
      catIndex.set(cat, id);
    }
    categoryIds[i] = id;
  }
  return { labels: labels.join(SEP), categoryIds, categories };
}

export function buildIndex(packed: PackedTags): TagMatchIndex {
  const raw = packed.labels.length ? packed.labels.split(SEP) : [];
  const norm = new Array<string>(raw.length);
  const masks = new Int32Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) {
    const n = normalize(raw[i]);
    norm[i] = n;
    masks[i] = charMask(n);
  }
  return {
    norm,
    masks,
    categoryIds: packed.categoryIds,
    categoryNorm: packed.categories.map(normalize),
  };
}

// Edits allowed for a fuzzy word match. Short words get none: at 1–3 chars a
// single edit matches nearly everything.
function allowedEdits(len: number): number {
  if (len >= 8) return 2;
  if (len >= 4) return 1;
  return 0;
}

// Reused DP rows for the edit-distance pass (no per-call allocation).
let rowPrev = new Int32Array(64);
let rowCur = new Int32Array(64);
let rowPrev2 = new Int32Array(64);

// Smallest optimal-string-alignment distance between `a` and ANY prefix of
// `b` (so a partially typed word can match a longer label word), bounded:
// returns max+1 as soon as the distance is certain to exceed `max`.
function prefixDistance(a: string, b: string, bStart: number, bEnd: number, max: number): number {
  const m = a.length;
  const n = Math.min(bEnd - bStart, m + max);
  if (n + 1 > rowCur.length) {
    rowPrev = new Int32Array(n + 1);
    rowCur = new Int32Array(n + 1);
    rowPrev2 = new Int32Array(n + 1);
  }
  for (let j = 0; j <= n; j += 1) rowPrev[j] = j;
  for (let i = 1; i <= m; i += 1) {
    rowCur[0] = i;
    let rowMin = i;
    const ac = a.charCodeAt(i - 1);
    for (let j = 1; j <= n; j += 1) {
      const bc = b.charCodeAt(bStart + j - 1);
      const cost = ac === bc ? 0 : 1;
      let v = rowPrev[j - 1] + cost;
      const del = rowPrev[j] + 1;
      if (del < v) v = del;
      const ins = rowCur[j - 1] + 1;
      if (ins < v) v = ins;
      if (
        i > 1 &&
        j > 1 &&
        ac === b.charCodeAt(bStart + j - 2) &&
        a.charCodeAt(i - 2) === bc
      ) {
        const tr = rowPrev2[j - 2] + 1;
        if (tr < v) v = tr;
      }
      rowCur[j] = v;
      if (v < rowMin) rowMin = v;
    }
    if (rowMin > max) return max + 1;
    const t = rowPrev2;
    rowPrev2 = rowPrev;
    rowPrev = rowCur;
    rowCur = t;
  }
  // rowPrev holds the last row; distance to the best prefix is its minimum.
  let best = max + 1;
  for (let j = 0; j <= n; j += 1) if (rowPrev[j] < best) best = rowPrev[j];
  return best;
}

// Does query word `w` fuzzily match the prefix of some word in `label`?
function fuzzyWord(w: string, label: string, max: number): boolean {
  let start = 0;
  const len = label.length;
  while (start < len) {
    let end = label.indexOf(' ', start);
    if (end < 0) end = len;
    // A word shorter than w by more than `max` can't be within range.
    if (end - start + max >= w.length && prefixDistance(w, label, start, end, max) <= max) {
      return true;
    }
    start = end + 1;
  }
  return false;
}

function startsWord(label: string, word: string): boolean {
  return label.startsWith(word) || label.indexOf(' ' + word) >= 0;
}

/**
 * Rank the index against `query` and return up to `limit` source indices,
 * best first. Queries are normalised the same way labels are.
 */
export function matchTags(index: TagMatchIndex, query: string, limit: number): number[] {
  const q = normalize(query);
  if (!q || limit <= 0) return [];
  const words = q.split(' ');
  const multi = words.length > 1;
  const { norm } = index;
  const n = norm.length;

  const TIERS = 9;
  const FUZZY = 7;
  const CATEGORY = 8;
  const buckets: number[][] = Array.from({ length: TIERS }, () => []);

  for (let i = 0; i < n; i += 1) {
    const label = norm[i];
    const at = label.indexOf(q);
    if (at === 0) {
      if (label.length === q.length) buckets[0].push(i);
      else buckets[label.charCodeAt(q.length) === 32 ? 1 : 2].push(i);
    } else if (at > 0) {
      // A later occurrence may still sit on a word boundary.
      buckets[label.charCodeAt(at - 1) === 32 || label.indexOf(' ' + q) >= 0 ? 3 : 5].push(i);
    } else if (multi) {
      let allPrefix = true;
      let allSub = true;
      for (let k = 0; k < words.length && allSub; k += 1) {
        const w = words[k];
        if (label.indexOf(w) < 0) allSub = false;
        else if (allPrefix && !startsWord(label, w)) allPrefix = false;
      }
      if (allSub) buckets[allPrefix ? 4 : 6].push(i);
    }
  }

  const exactCount = buckets.reduce((s, b) => s + b.length, 0);

  // Typo-tolerant pass, only when the exact tiers left room.
  if (exactCount < limit) {
    const maxima = words.map((w) => allowedEdits(w.length));
    if (maxima.some((m) => m > 0)) {
      const qMasks = words.map(charMask);
      const totalMax = maxima.reduce((s, m) => s + m, 0);
      const qMask = charMask(q);
      const taken = new Uint8Array(n);
      for (let t = 0; t < FUZZY; t += 1) for (const i of buckets[t]) taken[i] = 1;
      for (let i = 0; i < n; i += 1) {
        if (taken[i]) continue;
        const lm = index.masks[i];
        // Every edit can account for at most one query character the label
        // lacks, so a label missing more distinct characters can't match.
        if (popcount(qMask & ~lm) > totalMax) continue;
        const label = norm[i];
        let ok = true;
        for (let k = 0; k < words.length && ok; k += 1) {
          const w = words[k];
          const max = maxima[k];
          if (max === 0 || popcount(qMasks[k] & ~lm) === 0) {
            // Exact (prefix-of-word) requirement for short words; also try it
            // first for longer ones since it is far cheaper than the DP.
            if (startsWord(label, w)) continue;
            if (max === 0) {
              ok = false;
              continue;
            }
          }
          if (!fuzzyWord(w, label, max)) ok = false;
        }
        if (ok) buckets[FUZZY].push(i);
      }
    }
  }

  // Category fallback: tags whose category name starts with the query.
  let filled = buckets.reduce((s, b) => s + b.length, 0);
  if (filled < limit) {
    const cats = new Set<number>();
    index.categoryNorm.forEach((c, id) => {
      if (c && startsWords(c, words)) cats.add(id);
    });
    if (cats.size > 0) {
      const seen = new Uint8Array(n);
      for (let t = 0; t < CATEGORY; t += 1) for (const i of buckets[t]) seen[i] = 1;
      for (let i = 0; i < n && filled < limit; i += 1) {
        if (!seen[i] && cats.has(index.categoryIds[i])) {
          buckets[CATEGORY].push(i);
          filled += 1;
        }
      }
    }
  }

  const out: number[] = [];
  const byLength = (a: number, b: number) =>
    norm[a].length - norm[b].length || (norm[a] < norm[b] ? -1 : norm[a] > norm[b] ? 1 : a - b);
  for (let t = 0; t < TIERS && out.length < limit; t += 1) {
    const b = buckets[t];
    if (b.length === 0) continue;
    const room = limit - out.length;
    if (b.length > room) {
      // Partial order is enough: only `room` of them will be returned.
      out.push(...selectSmallest(b, room, byLength));
    } else {
      b.sort(byLength);
      out.push(...b);
    }
  }
  return out;
}

function startsWords(label: string, words: string[]): boolean {
  return words.every((w) => startsWord(label, w));
}

// The `k` smallest of `items` by `cmp`, sorted — via a bounded max-heap, so a
// 50K-match bucket costs O(n log k) rather than a full sort.
function selectSmallest(items: number[], k: number, cmp: (a: number, b: number) => number): number[] {
  const heap: number[] = [];
  const up = (i0: number) => {
    let i = i0;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (cmp(heap[i], heap[p]) <= 0) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const down = (i0: number) => {
    let i = i0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && cmp(heap[l], heap[m]) > 0) m = l;
      if (r < heap.length && cmp(heap[r], heap[m]) > 0) m = r;
      if (m === i) break;
      [heap[i], heap[m]] = [heap[m], heap[i]];
      i = m;
    }
  };
  for (const it of items) {
    if (heap.length < k) {
      heap.push(it);
      up(heap.length - 1);
    } else if (cmp(it, heap[0]) < 0) {
      heap[0] = it;
      down(0);
    }
  }
  return heap.sort(cmp);
}
