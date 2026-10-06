/*
 * slangfx studio — beat grid.
 *
 * One audio-bearing clip is nominated the SNAP TRACK. Its audio is analyzed
 * for a steady tempo (bpm) and the time of a bar line (offset), and the
 * timeline then offers a musical grid — bars, beats and finer subdivisions —
 * that every other clip edge, keyframe and drop snaps to.
 *
 * The grid is stored on the comp as plain JSON and is DERIVED from the clip
 * it names, never copied out of it:
 *
 *   comp.grid = { clipId, bpm, offset, beatsPerBar, division, show }
 *
 *   bpm / offset are in SOURCE time (the analysis's own frame of reference),
 *   so retiming the clip, trimming it, or sliding it along the timeline
 *   carries the grid with it — comp time of a source instant is
 *   clip.start + (src − clip.in) / rate, the same mapping the player uses.
 *
 * This module is UI-free (no DOM, no Web Audio) so it can run under node.
 */

/** Snap granularities. `beats` is in quarter-note beats; bars scale with the
 * time signature. Triplets divide their parent note by three. */
export const DIVISIONS = [
  { id: '2bar', label: '2 bars', beats: (bpb) => bpb * 2 },
  { id: 'bar', label: '1 bar', beats: (bpb) => bpb },
  { id: '1/2', label: '1/2', beats: () => 2 },
  { id: '1/4', label: '1/4 (beat)', beats: () => 1 },
  { id: '1/4T', label: '1/4 triplet', beats: () => 2 / 3 },
  { id: '1/8', label: '1/8', beats: () => 0.5 },
  { id: '1/8T', label: '1/8 triplet', beats: () => 1 / 3 },
  { id: '1/16', label: '1/16', beats: () => 0.25 },
  { id: '1/16T', label: '1/16 triplet', beats: () => 1 / 6 },
  { id: '1/32', label: '1/32', beats: () => 0.125 },
];
export const DEFAULT_DIVISION = '1/4';

const divisionOf = (id) => DIVISIONS.find((d) => d.id === id) ?? DIVISIONS.find((d) => d.id === DEFAULT_DIVISION);

/* ---- analysis ----------------------------------------------------------- */

const HOP = 64;                 // envelope hop, in decimated samples
const TARGET_SR = 12000;        // decimated rate — kick/snare/hat energy all survives
const BINS = 256;               // phase-histogram resolution

/** Onset strength per hop, full-band and low-band only (kick-weighted, which
 * is what picks the bar's downbeat). Both are spectral-flux style: rises in
 * log-compressed band energy, minus a local mean so a sustained loud passage
 * doesn't read as one long onset. */
function onsetEnvelopes(samples, sr) {
  const factor = Math.max(1, Math.floor(sr / TARGET_SR));
  const dsr = sr / factor;
  const hopSec = (HOP * factor) / sr;
  const frames = Math.floor(samples.length / factor / HOP);
  if (frames < 8) return null;

  // One-pole low-pass at ~150 Hz splits the signal; the remainder is "high".
  const a = 1 - Math.exp((-2 * Math.PI * 150) / dsr);
  let lp = 0;
  const lowE = new Float64Array(frames);
  const highE = new Float64Array(frames);
  for (let f = 0; f < frames; f++) {
    let le = 0, he = 0;
    for (let k = 0; k < HOP; k++) {
      let x = 0;
      const base = (f * HOP + k) * factor;
      for (let j = 0; j < factor; j++) x += samples[base + j];
      x /= factor;
      lp += a * (x - lp);
      const hi = x - lp;
      le += lp * lp;
      he += hi * hi;
    }
    lowE[f] = Math.sqrt(le / HOP);
    highE[f] = Math.sqrt(he / HOP);
  }

  const compress = (e) => {
    let max = 0;
    for (let i = 0; i < e.length; i++) if (e[i] > max) max = e[i];
    const out = new Float64Array(e.length);
    if (max < 1e-9) return out;
    const k = Math.log1p(60);
    for (let i = 0; i < e.length; i++) out[i] = Math.log1p((60 * e[i]) / max) / k;
    return out;
  };
  const low = compress(lowE), high = compress(highE);

  const flux = (c) => {
    const out = new Float64Array(frames);
    for (let i = 1; i < frames; i++) out[i] = Math.max(0, c[i] - c[i - 1]);
    return out;
  };
  const detrend = (e) => {
    // moving mean over ~0.5 s via prefix sums
    const win = Math.max(3, Math.round(0.5 / hopSec));
    const pre = new Float64Array(e.length + 1);
    for (let i = 0; i < e.length; i++) pre[i + 1] = pre[i] + e[i];
    const out = new Float64Array(e.length);
    for (let i = 0; i < e.length; i++) {
      const lo = Math.max(0, i - win), hi = Math.min(e.length, i + win + 1);
      out[i] = Math.max(0, e[i] - (pre[hi] - pre[lo]) / (hi - lo));
    }
    return out;
  };
  const fl = flux(low), fh = flux(high);
  const all = new Float64Array(frames);
  for (let i = 0; i < frames; i++) all[i] = fl[i] + fh[i];
  return { all: detrend(all), low: detrend(fl), lowE, hopSec, frames };
}

/** Peakiness of the onset envelope folded at `period` frames: how much of
 * the onset energy lands in one phase. Returns the best phase, in frames. */
function foldScore(env, period) {
  const hist = new Float64Array(BINS);
  let total = 0;
  for (let i = 0; i < env.length; i++) {
    const v = env[i];
    if (!v) continue;
    const ph = (i / period) % 1;
    hist[Math.min(BINS - 1, Math.floor(ph * BINS))] += v;
    total += v;
  }
  if (total <= 0) return { score: 0, phase: 0 };
  // Box-smooth circularly over ~1.5 frames of the period.
  const w = Math.max(1, Math.round((BINS * 1.5) / period) | 1);
  const half = (w - 1) >> 1;
  let best = -1, bestBin = 0;
  for (let b = 0; b < BINS; b++) {
    let s = 0;
    for (let k = -half; k <= half; k++) s += hist[(b + k + BINS) % BINS];
    if (s > best) { best = s; bestBin = b; }
  }
  return { score: best / total, phase: ((bestBin + 0.5) / BINS) * period };
}

/**
 * Find a steady tempo and bar line in mono audio.
 * @param {Float32Array} samples
 * @param {number} sr
 * @returns {{bpm:number, offset:number, confidence:number}|null}
 *   offset = SOURCE time (s) of a bar line, in [0, one bar).
 */
export function analyzeBeats(samples, sr, { beatsPerBar = 4, minBpm = 70, maxBpm = 180 } = {}) {
  const env = onsetEnvelopes(samples, sr);
  if (!env) return null;
  const { all, lowE, hopSec, frames } = env;

  // Coarse tempo: autocorrelation of onset strength, biased toward the
  // 100–140 range where a pulse is most often felt, with the double-period
  // peak lending support (a true beat repeats at its own multiples).
  const acf = (lag) => {
    let s = 0;
    for (let i = 0; i + lag < frames; i++) s += all[i] * all[i + lag];
    return s / Math.max(1, frames - lag);
  };
  const lagLo = Math.floor(60 / maxBpm / hopSec);
  const lagHi = Math.ceil(60 / minBpm / hopSec);
  let bestLag = 0, bestScore = -1;
  for (let lag = lagLo; lag <= lagHi; lag++) {
    const bpm = 60 / (lag * hopSec);
    const prior = Math.exp(-0.5 * (Math.log2(bpm / 120) / 0.8) ** 2);
    const s = (acf(lag) + 0.5 * acf(lag * 2)) * prior;
    if (s > bestScore) { bestScore = s; bestLag = lag; }
  }
  if (bestLag <= 0 || bestScore <= 0) return null;

  // Fine tempo + phase: scan ±3% and keep the most sharply peaked fold.
  const bpm0 = 60 / (bestLag * hopSec);
  let best = { score: -1, bpm: bpm0, phase: 0 };
  for (let pct = -3; pct <= 3.0001; pct += 0.05) {
    const bpm = bpm0 * (1 + pct / 100);
    const r = foldScore(all, 60 / bpm / hopSec);
    if (r.score > best.score) best = { score: r.score, bpm, phase: r.phase };
  }
  const period = 60 / best.bpm;                        // seconds per beat
  // A frame's flux marks a rise somewhere inside it: take its centre.
  let offset = (best.phase + 0.5) * hopSec;

  // Which beat is the "one"? Kicks (low-band onsets) usually land there.
  const bpb = Math.max(1, Math.round(beatsPerBar));
  const votes = new Float64Array(bpb);
  const beats = Math.floor((frames * hopSec - offset) / period);
  for (let n = 0; n < beats; n++) {
    const c = Math.round((offset + n * period) / hopSec - 0.5);
    // Raw (uncompressed) low-band level just after the beat: an accented
    // kick on the one is louder than the others, which flux compression flattens.
    let l = 0;
    for (let d = 0; d <= 3; d++) {
      const i = c + d;
      if (i >= 0 && i < frames) l = Math.max(l, lowE[i]);
    }
    votes[n % bpb] += l;
  }
  let down = 0;
  for (let k = 1; k < bpb; k++) if (votes[k] > votes[down] * 1.02) down = k;
  offset += down * period;
  offset %= period * bpb;

  return {
    bpm: best.bpm,
    offset,
    // uniform onsets would put ~1.5 frames/period of the energy in the window
    confidence: Math.max(0, Math.min(1, (best.score * (60 / best.bpm / hopSec) / 1.5 - 1) / 10)),
  };
}

/* ---- the grid as a function of its clip --------------------------------- */

const clipRateOf = (clip) => (Number.isFinite(clip?.rate) && clip.rate > 0 ? clip.rate : 1);

/** Comp-time geometry of the grid: seconds per beat, and the comp time of a
 * bar line. Null when the grid has no usable tempo or its clip is gone. */
export function gridGeometry(grid, clip) {
  if (!grid || !clip || !(grid.bpm > 0)) return null;
  const rate = clipRateOf(clip);
  const beat = 60 / grid.bpm / rate;
  const bar = clip.start + ((grid.offset ?? 0) - (clip.in ?? 0)) / rate;
  return { beat, bar, bpb: Math.max(1, Math.round(grid.beatsPerBar ?? 4)) };
}

/** Comp seconds between snap lines at the grid's chosen granularity. */
export function gridStep(grid, clip) {
  const g = gridGeometry(grid, clip);
  if (!g) return 0;
  return divisionOf(grid.division).beats(g.bpb) * g.beat;
}

/** The grid line nearest `t` (comp seconds), or null if there is no grid. */
export function nearestGridLine(grid, clip, t) {
  const g = gridGeometry(grid, clip);
  const step = gridStep(grid, clip);
  if (!g || !(step > 0)) return null;
  return g.bar + Math.round((t - g.bar) / step) * step;
}

/** Every grid line in [t0, t1], tagged 'bar' | 'beat' | 'sub'. `maxLines`
 * guards against a pathological zoom (the caller thins on screen density). */
export function gridLines(grid, clip, t0, t1, maxLines = 4000) {
  const g = gridGeometry(grid, clip);
  const step = gridStep(grid, clip);
  if (!g || !(step > 0) || !(t1 > t0)) return [];
  const out = [];
  const k0 = Math.ceil((t0 - g.bar) / step - 1e-9);
  const eps = 1e-6;
  for (let k = k0; out.length < maxLines; k++) {
    const t = g.bar + k * step;
    if (t > t1 + 1e-9) break;
    const beats = (k * step) / g.beat;           // distance from a bar line, in beats
    const nearInt = (x) => Math.abs(x - Math.round(x)) < eps;
    const kind = nearInt(beats / g.bpb) ? 'bar' : nearInt(beats) ? 'beat' : 'sub';
    out.push({ t, kind });
  }
  return out;
}

/** "128 BPM · 1/8" — the badge text for the snap track. */
export function gridSummary(grid) {
  if (!grid) return '';
  return `${+grid.bpm.toFixed(2)} BPM · ${divisionOf(grid.division).label.replace(/ \(beat\)$/, '')}`;
}

export function divisionLabel(id) { return divisionOf(id).label; }
