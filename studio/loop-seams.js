/* =====================================================================
 * Seamless loops — the pure half of the loop tool (app.js owns the
 * sampling, preview and bake; everything here is plain arrays so node
 * tests can drive it).
 *
 * Seam scoring. A loop of [i, j) plays frame j−1 and then frame i again,
 * so the cut is invisible when frame i looks like the frame j that would
 * have come next AND the motion matches on both sides. Each seam compares
 * a short window of frames around both ends — before the cut (incoming
 * motion) and after it (outgoing motion) — so a pose match with the wrong
 * speed or direction scores badly. That is what rejects the eased-in first
 * frames of a generated clip, which start from a still and accelerate.
 * Scores read against the clip's median frame-to-frame change: 1.00× is
 * "as smooth as ordinary playback".
 *
 * Choice. The best seam wins; length only breaks near-ties. (The old rule
 * — longest seam under the "invisible" bar — happily took a 1.2× seam over
 * a 0.3× one, and 1.2× at a loop point reads as a stitch every cycle.)
 *
 * Blend. Even a good seam carries a little residual (a pose a pixel off,
 * an exposure drift across a generated clip). Baking crossfades the last
 * K frames of the loop into the K frames that precede its start (the
 * footage's own run-up to the first frame), so the final frame flows into
 * the first by construction; with no footage before the start it fades
 * the first K frames in from the footage after the end instead.
 * =================================================================== */

export const LOOP_GOOD = 1.25;   // seams at/below this read as invisible
export const SEAM_WEIGHTS = [0.5, 1, 1, 1, 0.5];   // frames −2…+2 around the cut
const HALF = (SEAM_WEIGHTS.length - 1) / 2;
const WSUM = SEAM_WEIGHTS.reduce((a, b) => a + b, 0);

export function sadRGB(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) { const d = a[k] - b[k]; s += d < 0 ? -d : d; }
  return s;
}

/** Median frame-to-frame change, floored so a locked-off shot (≈0 motion)
 * doesn't blow ratios up: under a level per pixel nothing shows. */
export function motionBase(feats) {
  const consec = [];
  for (let k = 0; k + 1 < feats.length; k++) consec.push(sadRGB(feats[k], feats[k + 1]));
  consec.sort((x, y) => x - y);
  return Math.max(consec[consec.length >> 1] || 0, feats[0]?.length || 1);
}

/** Windowed seam score for "A[i] follows B[j−1]" (A and B may be the same
 * feature list, or fine grids around the two ends). null when the window
 * does not fit. */
export function seamScore(A, i, B, j, base) {
  if (i - HALF < 0 || j - HALF < 0 || i + HALF >= A.length || j + HALF >= B.length) return null;
  let s = 0;
  for (let k = -HALF; k <= HALF; k++) s += SEAM_WEIGHTS[k + HALF] * sadRGB(A[i + k], B[j + k]);
  return s / WSUM / base;
}

/** Score every seam (i, j), j − i ≥ minGap; returns a diverse handful,
 * best first by the choice rule. `onRow` lets a caller yield to the UI. */
export async function scoreLoopSeams(feats, minGap, onRow) {
  const N = feats.length;
  const base = motionBase(feats);
  const pairs = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + minGap; j < N; j++) {
      const r = seamScore(feats, i, feats, j, base);
      if (r !== null) pairs.push({ i, j, r });
    }
    if (onRow) await onRow(i + 1, N);
  }
  return { picked: chooseSeams(pairs, N), base };
}

/** Near-duplicate suppression + the choice rule (pure, exported for tests). */
export function chooseSeams(pairs, N, keep = 5) {
  pairs = [...pairs].sort((a, b) => a.r - b.r);
  const rad = Math.max(2, Math.round(N * 0.04));
  const picked = [];
  for (const p of pairs) {
    if (picked.length >= 12) break;
    if (picked.some((q) => Math.abs(q.i - p.i) < rad && Math.abs(q.j - p.j) < rad)) continue;
    picked.push(p);
  }
  if (!picked.length) return picked;
  // Seams about as good as the best are interchangeable to the eye: among
  // those the longer loop is worth more. Everything else ranks by quality.
  const best = picked[0].r;
  const near = (p) => p.r <= Math.min(Math.max(best * 1.5, best + 0.1), Math.max(best, LOOP_GOOD));
  picked.sort((a, b) => {
    const na = near(a);
    if (na !== near(b)) return na ? -1 : 1;
    return na ? (b.j - b.i) - (a.j - a.i) || a.r - b.r : a.r - b.r;
  });
  return picked.slice(0, keep);
}

/**
 * Crossfade plan for baking a loop of n output frames whose seam blend is
 * up to K frames long. `pre` = frames of footage available before the loop
 * start, `post` = after its end. Returns { side: 'pre'|'post'|null, K }.
 */
export function blendPlan(n, K, pre, post) {
  K = Math.max(0, Math.min(Math.floor(K), Math.floor(n / 3)));
  if (K < 2) return { side: null, K: 0 };
  if (pre >= K) return { side: 'pre', K };
  if (post >= K) return { side: 'post', K };
  const best = Math.max(pre, post);
  if (best < 2) return { side: null, K: 0 };
  return { side: pre >= post ? 'pre' : 'post', K: Math.floor(best) };
}

const smooth = (x) => x * x * (3 - 2 * x);

/**
 * For output frame k of n: which source frame offsets to mix. Returns
 * { other, w } — `other` is the frame offset (in loop frames, relative to
 * k) of the footage mixed in (−n: the run-up before the start, +n: the
 * footage after the end), w its weight (0 = the loop's own frame only).
 * The last frame is ≈ the one just before the start, so playback flows on.
 */
export function blendWeight(k, n, plan) {
  if (!plan.side) return { other: 0, w: 0 };
  const K = plan.K;
  if (plan.side === 'pre') {
    const m = k - (n - K);   // 0…K−1 inside the blend zone
    if (m < 0) return { other: 0, w: 0 };
    return { other: -n, w: smooth((m + 1) / (K + 1)) };
  }
  if (k >= K) return { other: 0, w: 0 };
  return { other: n, w: smooth(1 - (k + 1) / (K + 1)) };
}

/** Equal-power gains for an audio crossfade at position x ∈ [0, 1] of the
 * fade (x = 0: all `own`, x = 1: all `other`). */
export const equalPower = (x) => {
  const c = Math.min(1, Math.max(0, x));
  return { own: Math.cos(c * Math.PI / 2), other: Math.sin(c * Math.PI / 2) };
};

/**
 * Build the looped PCM for one channel: `src` the whole source channel,
 * [a, b) the loop in samples, F the fade length in samples on `side`.
 * The fade mirrors the video's: 'pre' fades the tail into the run-up
 * before `a`, 'post' fades the head in from the audio after `b`.
 */
export function loopChannel(src, a, b, F, side) {
  const n = b - a;
  const out = new Float32Array(n);
  out.set(src.subarray(a, b));
  if (!side || F < 2) return out;
  for (let m = 0; m < F; m++) {
    const x = (m + 0.5) / F;
    if (side === 'pre') {
      const k = n - F + m;               // tail sample
      const g = equalPower(x);
      out[k] = out[k] * g.own + src[a - F + m] * g.other;
    } else {
      const g = equalPower(1 - x);       // head: mostly the post-roll at first
      out[m] = out[m] * g.own + src[b + m] * g.other;
    }
  }
  return out;
}
