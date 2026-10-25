// natural-key — a precomputed collation key that orders strings EXACTLY like
// the `natural-compare` package, but with plain `<` comparison.
//
// Sorting a 1M-item library by name through naturalCompare took ~17–20s of
// main-thread time: each of the ~20M comparisons re-walks both strings,
// re-maps every character and re-parses digit runs. Encoding each path once
// (O(n)) and then comparing the keys as ordinary strings moves all of that out
// of the comparator.
//
// How natural-compare orders (v1.4.0, String.alphabet unset):
//   - every character maps to a code (see `code` below), compared numerically;
//   - when BOTH sides are at a digit 1–9, the whole digit run (zeros included
//     after the first digit) is compared as a number and both sides skip past
//     it. A leading '0' is NOT a number start — it compares as a character.
//   - a string that is a prefix of another sorts first.
//
// The key emits one char per mapped code, so code order == key order. Codes
// 67–75 belong only to the digits 1–9, so a number start is emitted as the
// marker code 67 (it compares against any non-digit exactly as its first digit
// would), then a length char, then the digits: equal-length digit strings
// compare lexically as numbers do, and a shorter run is a smaller number.
// Two different numbers therefore differ at the length char or the digits;
// equal numbers fall through to whatever follows, as in natural-compare.

function code(c: number): number {
  if (c < 45 || c > 127) return c;
  if (c < 46) return 65; // -
  if (c < 48) return c - 1;
  if (c < 58) return c + 18; // 0-9
  if (c < 65) return c - 11;
  if (c < 91) return c + 11; // A-Z
  if (c < 97) return c - 37;
  if (c < 123) return c + 5; // a-z
  return c - 63;
}

const NUMBER_MARK = 67;

// Reused scratch buffer for building a key's char codes.
let buf = new Uint16Array(256);

function encode(s: string, tail: number): number {
  const n = s.length;
  // Worst case one number run per digit adds 2 chars each; a 1-digit run is 3.
  if (buf.length < n * 3 + tail) buf = new Uint16Array(n * 3 + tail + 64);
  let o = 0;
  let i = 0;
  while (i < n) {
    const c = s.charCodeAt(i);
    if (c >= 49 && c <= 57) {
      let j = i + 1;
      while (j < n) {
        const d = s.charCodeAt(j);
        if (d < 48 || d > 57) break;
        j += 1;
      }
      buf[o++] = NUMBER_MARK;
      buf[o++] = j - i;
      for (let k = i; k < j; k += 1) buf[o++] = s.charCodeAt(k);
      i = j;
    } else {
      buf[o++] = code(c);
      i += 1;
    }
  }
  return o;
}

function toString(len: number): string {
  // fromCharCode.apply has an argument-count ceiling; chunk long keys.
  if (len <= 4096) {
    return String.fromCharCode.apply(null, buf.subarray(0, len) as unknown as number[]);
  }
  let out = '';
  for (let i = 0; i < len; i += 4096) {
    out += String.fromCharCode.apply(
      null,
      buf.subarray(i, Math.min(len, i + 4096)) as unknown as number[]
    );
  }
  return out;
}

export function naturalKey(s: string): string {
  return toString(encode(s, 0));
}

/**
 * Stable sort of `items` by the natural order of `text(item)` — the order
 * `naturalCompare(text(a), text(b))` gives — in a fraction of the time.
 *
 * Each key gets a NUL separator plus the item's index as two fixed-width chars
 * appended, so the comparator-less native `Array.prototype.sort()` (string
 * order by UTF-16 code unit, ~3x faster than any JS comparator) both sorts and
 * breaks ties by input order. NUL sorts below every key char, so a key that is
 * a prefix of another still sorts first. Paths never contain NUL.
 */
export function sortByNaturalKey<T>(items: T[], text: (item: T) => string): T[] {
  const n = items.length;
  const keys = new Array<string>(n);
  for (let i = 0; i < n; i += 1) {
    let o = encode(text(items[i]), 3);
    buf[o++] = 0;
    buf[o++] = i >>> 16;
    buf[o++] = i & 0xffff;
    keys[i] = toString(o);
  }
  keys.sort();
  const out = new Array<T>(n);
  for (let i = 0; i < n; i += 1) {
    const k = keys[i];
    const len = k.length;
    out[i] = items[(k.charCodeAt(len - 2) << 16) | k.charCodeAt(len - 1)];
  }
  return out;
}
