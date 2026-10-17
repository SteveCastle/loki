/*
 * slangfx studio — beat grid.
 *
 * One audio-bearing clip is nominated the SNAP TRACK. Its audio is analyzed
 * and the timeline then offers a musical grid that every other clip edge,
 * keyframe and drop snaps to. Three kinds of grid come out of one analysis,
 * and the snap track picks which one is live (`grid.mode`):
 *
 *   hits   the DRUM HITS themselves — every detected onset, tagged kick /
 *          snare / hat and weighted by how hard it lands. Snapping to a hit
 *          is sample-accurate to the attack, which no tempo grid can be.
 *   beats  the TRACKED BEATS: the beat pulse followed through the track by
 *          a dynamic-programming beat tracker (a port of librosa's), each
 *          beat pulled onto the drum hit it lands on. Follows tempo drift;
 *          subdivisions are interpolated between consecutive beats.
 *   grid   a STEADY grid from one bpm + bar offset, the classic model —
 *          what a loop or a quantized track wants, and the fallback for
 *          projects analyzed before hits and beats existed.
 *
 * The grid is stored on the comp as plain JSON and is DERIVED from the clip
 * it names, never copied out of it:
 *
 *   comp.grid = { clipId, mode, bpm, offset, beatsPerBar, division, show,
 *                 beats: [src s…], beatPhase, hits: [[src s, strength, band]…],
 *                 hitBand, hitMin }
 *
 *   Every time is SOURCE time (the analysis's own frame of reference), so
 *   retiming the clip, trimming it, or sliding it along the timeline carries
 *   the grid with it — comp time of a source instant is
 *   clip.start + (src − clip.in) / rate, the same mapping the player uses.
 *
 * This module is UI-free (no DOM, no Web Audio) so it can run under node.
 * The analysis is a port of the librosa recipe the sibling Python tooling
 * uses (onset_strength → peak_pick / beat_track), kept in this one file so
 * the test harnesses can load it from a data: URL.
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

/** What the timeline snaps to. */
export const GRID_MODES = [
  { id: 'hits', label: 'drum hits' },
  { id: 'beats', label: 'tracked beats' },
  { id: 'grid', label: 'steady grid' },
];
export const DEFAULT_MODE = 'hits';

/** Hit bands, by index as stored: 0 kick (low), 1 snare (mid), 2 hat (high). */
export const HIT_BANDS = [
  { id: 'all', label: 'all hits', bands: [0, 1, 2] },
  { id: 'kick', label: 'kicks', bands: [0] },
  { id: 'snare', label: 'snares', bands: [1] },
  { id: 'hat', label: 'hats', bands: [2] },
];
export const DEFAULT_HIT_MIN = 0.15;

const divisionOf = (id) => DIVISIONS.find((d) => d.id === id) ?? DIVISIONS.find((d) => d.id === DEFAULT_DIVISION);

/* ---- tempo + bar line (steady grid) -------------------------------------- */

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

/* ---- drum hits: mel spectral flux + peak picking (librosa port) ----------
 * librosa.onset.onset_strength: a 128-band mel power spectrogram in dB,
 * first-differenced along time, rectified and averaged over bands.
 * librosa.onset.onset_detect: peak_pick on that envelope, backtracked to the
 * preceding energy minimum. On top of that, each hit is then refined to the
 * millisecond with a fine RMS envelope — a 11.6 ms analysis frame is too
 * coarse to cut video on. */

const MEL_SR = 22050;           // analysis rate (librosa's default)
const N_FFT = 2048;
const MEL_HOP = 256;            // 11.6 ms frames (librosa uses 512; finer is cheap here)
const N_MELS = 128;

/** In-place iterative radix-2 complex FFT. */
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] += tr; im[a] += ti;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = ncr;
      }
    }
  }
}

/** Slaney mel scale, as librosa's default (`htk=False`). */
const hzToMel = (f) => (f < 1000 ? f / (200 / 3) : 15 + Math.log(f / 1000) / (Math.log(6.4) / 27));
const melToHz = (m) => (m < 15 ? m * (200 / 3) : 1000 * Math.exp((Math.log(6.4) / 27) * (m - 15)));

/** librosa.filters.mel with slaney normalization, as a sparse list of
 * {start, w} per band (the triangles only touch a few bins each). */
function melFilters(sr, nFft, nMels) {
  const bins = nFft / 2 + 1;
  const fftFreq = (k) => (k * sr) / nFft;
  const mLo = hzToMel(0), mHi = hzToMel(sr / 2);
  const melF = [];
  for (let i = 0; i < nMels + 2; i++) melF.push(melToHz(mLo + ((mHi - mLo) * i) / (nMels + 1)));
  const filters = [];
  for (let m = 0; m < nMels; m++) {
    const lo = melF[m], mid = melF[m + 1], hi = melF[m + 2];
    const enorm = 2 / (hi - lo);
    const k0 = Math.max(0, Math.ceil((lo * nFft) / sr));
    const k1 = Math.min(bins - 1, Math.floor((hi * nFft) / sr));
    const w = [];
    for (let k = k0; k <= k1; k++) {
      const f = fftFreq(k);
      const up = (f - lo) / (mid - lo), down = (hi - f) / (hi - mid);
      w.push(Math.max(0, Math.min(up, down)) * enorm);
    }
    filters.push({ start: k0, w, hz: mid, width: hi - lo });
  }
  return filters;
}

/** Linear-interpolation resample to `outRate` (the source is band-limited
 * well below MEL_SR/2 as far as drum onsets care). */
function resample(samples, sr, outRate) {
  if (sr === outRate) return samples;
  const n = Math.floor((samples.length * outRate) / sr);
  const out = new Float32Array(n);
  const step = sr / outRate;
  for (let i = 0; i < n; i++) {
    const p = i * step;
    const j = Math.floor(p);
    const f = p - j;
    const a = samples[j] ?? 0, b = samples[j + 1] ?? a;
    out[i] = a + (b - a) * f;
  }
  return out;
}

/**
 * Mel spectral-flux onset strength (librosa.onset.onset_strength, lag 1,
 * centred frames), plus the same flux split into kick / snare / hat bands.
 * Frame i is centred at i × MEL_HOP samples of the 22.05 kHz signal.
 */
function melOnsetStrength(y) {
  const frames = Math.floor(y.length / MEL_HOP) + 1;
  const filters = melFilters(MEL_SR, N_FFT, N_MELS);
  const hz = filters.map((f) => f.hz);
  const width = filters.map((f) => f.width);
  const win = new Float64Array(N_FFT);
  for (let i = 0; i < N_FFT; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  const power = new Float64Array(N_FFT / 2 + 1);
  const mel = new Float64Array(frames * N_MELS);
  // Linear mel power is kept too: the dB flux below says WHEN, the power
  // that arrives says WHAT — a kick is low power landing, a hat is high.
  const melLin = new Float32Array(frames * N_MELS);
  const lowPower = new Float64Array(frames);      // < 120 Hz: the kick's home
  let ref = 1e-10;
  for (let t = 0; t < frames; t++) {
    const centre = t * MEL_HOP;
    for (let i = 0; i < N_FFT; i++) {
      const j = centre - N_FFT / 2 + i;
      re[i] = (j >= 0 && j < y.length ? y[j] : 0) * win[i];
      im[i] = 0;
    }
    fft(re, im);
    for (let k = 0; k <= N_FFT / 2; k++) power[k] = re[k] * re[k] + im[k] * im[k];
    for (let m = 0; m < N_MELS; m++) {
      const { start, w } = filters[m];
      let s = 0;
      for (let i = 0; i < w.length; i++) s += w[i] * power[start + i];
      mel[t * N_MELS + m] = s;
      melLin[t * N_MELS + m] = s;
      if (hz[m] < 120) lowPower[t] += s;
      if (s > ref) ref = s;
    }
  }
  // power_to_db(ref=max, top_db=80)
  const floor = 10 * Math.log10(ref) - 80;
  for (let i = 0; i < mel.length; i++) mel[i] = Math.max(floor, 10 * Math.log10(Math.max(1e-10, mel[i])));
  const env = new Float64Array(frames);
  const lowDb = new Float64Array(frames);         // the same flux, < 120 Hz bands only
  let lowBands = 0;
  for (const f of hz) if (f < 120) lowBands++;
  for (let t = 1; t < frames; t++) {
    let sum = 0, low = 0;
    for (let m = 0; m < N_MELS; m++) {
      const d = mel[t * N_MELS + m] - mel[(t - 1) * N_MELS + m];
      if (d > 0) { sum += d; if (hz[m] < 120) low += d; }
    }
    env[t] = sum / N_MELS;
    lowDb[t] = low / Math.max(1, lowBands);
  }
  // Kick onset. The dB flux says precisely WHEN but can't tell a kick from
  // a hat (any sound after silence lifts every band by the same 80 dB);
  // linear low-band power ARRIVING says which, but smears over the frames
  // the window takes to swallow the attack. Gate the sharp one with the
  // honest one: a frame counts as a kick onset when both agree (the power
  // rise is allowed to trail the flux by a few frames).
  const lowOn = new Float64Array(frames);
  let lowMax = 0, dbMax = 0;
  for (let t = 1; t < frames; t++) {
    lowOn[t] = Math.max(0, lowPower[t] - lowPower[t - 1]);
    if (lowOn[t] > lowMax) lowMax = lowOn[t];
    if (lowDb[t] > dbMax) dbMax = lowDb[t];
  }
  if (lowMax > 0) for (let t = 0; t < frames; t++) lowOn[t] = Math.log1p((20 * lowOn[t]) / lowMax) / Math.log1p(20);
  const kick = new Float64Array(frames);
  if (dbMax > 0) {
    for (let t = 0; t < frames; t++) {
      let rise = 0;
      for (let d = 0; d <= 3; d++) rise = Math.max(rise, lowOn[t + d] ?? 0);
      kick[t] = (lowDb[t] / dbMax) * rise;
    }
  }
  return { env, melLin, hz, width, kick, frames, hopSec: MEL_HOP / MEL_SR };
}

/** librosa.util.peak_pick: a frame is a peak when it is the max of
 * [i−preMax, i+postMax], exceeds the mean of [i−preAvg, i+postAvg] by
 * `delta`, and sits at least `wait` frames after the previous peak. */
export function peakPick(x, { preMax, postMax, preAvg, postAvg, delta, wait }) {
  const n = x.length;
  const peaks = [];
  let last = -Infinity;
  const pre = new Float64Array(n + 1);
  for (let i = 0; i < n; i++) pre[i + 1] = pre[i] + x[i];
  for (let i = 0; i < n; i++) {
    const m0 = Math.max(0, i - preMax), m1 = Math.min(n, i + postMax + 1);
    let max = -Infinity;
    for (let j = m0; j < m1; j++) if (x[j] > max) max = x[j];
    if (x[i] < max) continue;
    const a0 = Math.max(0, i - preAvg), a1 = Math.min(n, i + postAvg + 1);
    const mean = (pre[a1] - pre[a0]) / (a1 - a0);
    if (x[i] < mean + delta) continue;
    if (i - last <= wait) continue;
    last = i;
    peaks.push(i);
  }
  return peaks;
}

/** librosa.onset.onset_backtrack: slide each event back to the preceding
 * local minimum of the envelope (the start of the rise). */
function backtrack(events, env) {
  return events.map((i) => {
    while (i > 0 && env[i - 1] <= env[i]) i--;
    return i;
  });
}

/**
 * Drum hits in mono audio: SOURCE times at millisecond precision, each with
 * a strength in (0, 1] relative to the loudest hit and a band — 0 kick,
 * 1 snare / clap / tom, 2 hat / cymbal — from where its spectral flux lands.
 * @returns {{hits: Array<[number, number, number]>, env: Float64Array,
 *            kick: Float64Array, hopSec: number}|null}
 */
export function detectHits(samples, sr, { delta = 0.07 } = {}) {
  const y = resample(samples, sr, MEL_SR);
  if (y.length < N_FFT * 2) return null;
  const { env, melLin, hz, width, kick, frames, hopSec } = melOnsetStrength(y);
  let max = 0;
  for (let i = 0; i < frames; i++) if (env[i] > max) max = env[i];
  if (!(max > 0)) return null;
  const norm = new Float64Array(frames);
  for (let i = 0; i < frames; i++) norm[i] = env[i] / max;
  // onset_detect's defaults, in frames of this hop.
  const fpsec = MEL_SR / MEL_HOP;
  const peaks = peakPick(norm, {
    preMax: Math.round(0.03 * fpsec), postMax: Math.round(0.0 * fpsec) + 1,
    preAvg: Math.round(0.1 * fpsec), postAvg: Math.round(0.1 * fpsec) + 1,
    delta, wait: Math.round(0.03 * fpsec),
  });
  // The kick / snare / hat call reads the power that ARRIVED with the hit
  // (each mel band's rise from just before the onset to just after, scaled
  // back to real power by the band's width — mel filters are unit-area):
  // a kick lands most of it under 120 Hz; of the rest, a hat's spectral
  // centroid sits high (cymbals live above 4 kHz) and a snare's, clap's
  // or tom's sits low.
  const classify = (p) => {
    let total = 0, low = 0, moment = 0;
    for (let m = 0; m < N_MELS; m++) {
      let before = Infinity, after = 0;
      for (let f = Math.max(0, p - 4); f <= p - 1; f++) before = Math.min(before, melLin[f * N_MELS + m]);
      for (let f = p; f <= Math.min(frames - 1, p + 2); f++) after = Math.max(after, melLin[f * N_MELS + m]);
      const a = Math.max(0, after - (before === Infinity ? 0 : before)) * width[m];
      total += a;
      if (hz[m] < 120) low += a;
      moment += a * hz[m];
    }
    if (!(total > 0)) return 1;
    if (low / total >= 0.35) return 0;
    return moment / total >= 4000 ? 2 : 1;
  };
  // A ~1 ms RMS envelope of the signal, for refining each hit to its attack
  // (buckets of `fine` samples; convert through the sample rate, not a
  // nominal millisecond, or the error grows along the track).
  const fine = Math.max(1, Math.round(MEL_SR / 1000));
  const fineSec = fine / MEL_SR;
  const nFine = Math.floor(y.length / fine);
  const rms = new Float64Array(nFine);
  for (let k = 0; k < nFine; k++) {
    let s = 0;
    for (let i = 0; i < fine; i++) { const v = y[k * fine + i]; s += v * v; }
    rms[k] = Math.sqrt(s / fine);
  }
  const starts = backtrack(peaks, norm);
  const hits = [];
  let lastT = -1;
  peaks.forEach((p, idx) => {
    // The flux peaks while the attack is still AHEAD of the frame centre
    // (the window reaches it first), so the attack lies between the
    // envelope's rise and half a window past the peak frame. In there,
    // the onset is where the 1 ms level first climbs a quarter of the way
    // from the quietest point to the loudest — the start of the rise,
    // whether the hit is a sample-sharp click or a sine kick swelling
    // over its first cycle.
    const t0 = Math.max(starts[idx], p - 1) * hopSec;
    const k0 = Math.max(0, Math.floor(t0 / fineSec));
    const k1 = Math.min(nFine - 1, Math.ceil((p * hopSec + ENV_HALF_WINDOW + hopSec) / fineSec));
    let floorK = k0, floorV = Infinity, peakV = 0;
    for (let k = k0; k <= k1; k++) if (rms[k] < floorV) { floorV = rms[k]; floorK = k; }
    for (let k = floorK; k <= k1; k++) if (rms[k] > peakV) peakV = rms[k];
    const thr = floorV + 0.25 * (peakV - floorV);
    let bestK = Math.round((p * hopSec + ENV_LEAD) / fineSec);
    for (let k = floorK; k <= k1; k++) if (rms[k] >= thr) { bestK = k; break; }
    const t = +(bestK * fineSec).toFixed(4);
    if (t - lastT < 0.02) return;   // two frames resolving to one attack
    lastT = t;
    hits.push([t, +norm[p].toFixed(3), classify(p)]);
  });
  return { hits, env: norm, kick, hopSec };
}

/** How far the spectral-flux peak leads the attack it reports. The window
 * reaches a transient before its centre does, so the attack can sit up to
 * half a window past the peak frame (ENV_HALF_WINDOW bounds the search);
 * on real drum hits the median is one frame (measured: 9 ms), which is
 * where an unrefined beat time is placed. */
const ENV_HALF_WINDOW = N_FFT / 2 / MEL_SR;
const ENV_LEAD = MEL_HOP / MEL_SR;

/* ---- beat tracking: dynamic programming (librosa.beat.beat_track port) ---
 * Ellis's tracker: a local score (the onset envelope smoothed over a
 * fraction of a beat) plus a transition cost that punishes intervals far
 * from the tempo period, maximized by DP; beats are read back from the best
 * final position and trimmed where the pulse fades out. */

/**
 * @param {Float64Array} env   onset envelope (frames)
 * @param {number} period      beat period in frames (not rounded)
 * @returns {number[]} beat frame indices
 */
export function trackBeatsDP(env, period, { tightness = 100 } = {}) {
  const n = env.length;
  const P = Math.max(1, Math.round(period));
  // Local score: envelope / std, convolved with a Gaussian of width P/32.
  let mean = 0;
  for (let i = 0; i < n; i++) mean += env[i];
  mean /= Math.max(1, n);
  let varc = 0;
  for (let i = 0; i < n; i++) varc += (env[i] - mean) ** 2;
  const std = Math.sqrt(varc / Math.max(1, n)) || 1;
  const win = [];
  for (let k = -P; k <= P; k++) win.push(Math.exp(-0.5 * ((k * 32) / P) ** 2));
  const local = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = -P; k <= P; k++) {
      const j = i + k;
      if (j >= 0 && j < n) s += (env[j] / std) * win[k + P];
    }
    local[i] = s;
  }
  let localMax = 0;
  for (let i = 0; i < n; i++) if (local[i] > localMax) localMax = local[i];

  const lagLo = -2 * P, lagHi = -Math.round(P / 2);
  const txwt = [];
  for (let lag = lagLo; lag <= lagHi; lag++) txwt.push(-tightness * Math.log(-lag / P) ** 2);
  const cum = new Float64Array(n);
  const back = new Int32Array(n).fill(-1);
  let first = true;
  for (let i = 0; i < n; i++) {
    let best = -Infinity, bestJ = -1;
    for (let lag = lagLo; lag <= lagHi; lag++) {
      const j = i + lag;
      if (j < 0) continue;
      const s = txwt[lag - lagLo] + cum[j];
      if (s > best) { best = s; bestJ = j; }
    }
    if (bestJ < 0) { cum[i] = local[i]; continue; }
    cum[i] = local[i] + best;
    // Leading silence: don't chain beats back into nothing.
    if (first && local[i] < 0.01 * localMax) back[i] = -1;
    else { back[i] = bestJ; first = false; }
  }
  // Last beat: the last local maximum of the cumulative score that is at
  // least half the median local maximum (so a fade-out tail doesn't count).
  const maxes = [];
  for (let i = 1; i < n - 1; i++) if (cum[i] > cum[i - 1] && cum[i] >= cum[i + 1]) maxes.push(i);
  if (!maxes.length) return [];
  const sorted = maxes.map((i) => cum[i]).sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  let last = -1;
  for (const i of maxes) if (cum[i] * 2 > med) last = i;
  if (last < 0) return [];
  const beats = [];
  for (let i = last; i >= 0; i = back[i]) beats.push(i);
  beats.reverse();
  // Trim: drop leading/trailing beats whose smoothed local score is weak.
  const boe = beats.map((b) => local[b]);
  const hann = [0, 0.5, 1, 0.5, 0];
  const smooth = boe.map((_, i) => {
    let s = 0;
    for (let k = -2; k <= 2; k++) s += (boe[i + k] ?? 0) * hann[k + 2];
    return s;
  });
  const thr = 0.5 * Math.sqrt(smooth.reduce((a, v) => a + v * v, 0) / Math.max(1, smooth.length));
  let lo = -1, hi = -1;
  smooth.forEach((v, i) => { if (v > thr) { if (lo < 0) lo = i; hi = i; } });
  return lo < 0 ? beats : beats.slice(lo, hi + 1);
}

/**
 * The whole analysis: steady tempo + bar line, drum hits, and tracked beats
 * pulled onto the hits they land on.
 * @returns {{bpm, offset, confidence, hits, beats, beatPhase}|null}
 */
export function analyzeTrack(samples, sr, { beatsPerBar = 4, minBpm = 70, maxBpm = 180, delta = 0.07 } = {}) {
  const steady = analyzeBeats(samples, sr, { beatsPerBar, minBpm, maxBpm });
  const detected = detectHits(samples, sr, { delta });
  if (!steady && !detected) return null;
  const bpb = Math.max(1, Math.round(beatsPerBar));
  if (!detected) return { ...steady, hits: [], beats: [], beatPhase: 0 };
  const { hits, env, kick, hopSec } = detected;
  let bpm = steady?.bpm ?? 120;
  let beats = [];
  let beatPhase = 0;
  if (steady) {
    // Track on the full flux plus the kick onsets: the pulse is carried by
    // kick and snare, and a hat pattern between them must not win the
    // phase (it is often the sharper transient).
    const dpEnv = new Float64Array(env.length);
    for (let i = 0; i < env.length; i++) dpEnv[i] = env[i] + kick[i];
    const frames = trackBeatsDP(dpEnv, 60 / bpm / hopSec);
    // Each tracked beat lands on the drum hit nearest it (within 40 ms),
    // which is where the ear puts the beat and where a cut should go.
    const times = hits.map((h) => h[0]);
    beats = frames.map((f) => {
      const t = f * hopSec + ENV_LEAD;
      const near = nearestIn(times, t);
      return near != null && Math.abs(near - t) <= 0.05 ? near : +t.toFixed(4);
    });
    beats = beats.filter((t, i) => i === 0 || t > beats[i - 1] + 0.05);
    if (beats.length >= 4) {
      const gaps = beats.slice(1).map((t, i) => t - beats[i]).sort((a, b) => a - b);
      bpm = 60 / gaps[Math.floor(gaps.length / 2)];
    }
    // The "one": kicks (low-band flux) vote by beat position in the bar.
    const votes = new Float64Array(bpb);
    beats.forEach((t, i) => {
      const f = Math.round((t - ENV_LEAD) / hopSec);
      let v = 0;
      for (let d = -1; d <= 2; d++) v = Math.max(v, kick[f + d] ?? 0);
      votes[i % bpb] += v;
    });
    for (let k = 1; k < bpb; k++) if (votes[k] > votes[beatPhase] * 1.02) beatPhase = k;
  }
  let offset = steady?.offset ?? 0;
  if (beats.length > beatPhase) {
    const bar = (60 / bpm) * bpb;
    offset = ((beats[beatPhase] % bar) + bar) % bar;
  }
  return { bpm, offset, confidence: steady?.confidence ?? 0, hits, beats, beatPhase };
}

/** Nearest value in a sorted array, or null when empty. */
function nearestIn(sorted, t) {
  if (!sorted.length) return null;
  let lo = 0, hi = sorted.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (sorted[mid] <= t) lo = mid;
    else hi = mid;
  }
  return Math.abs(sorted[lo] - t) <= Math.abs(sorted[hi] - t) ? sorted[lo] : sorted[hi];
}

/* ---- the grid as a function of its clip --------------------------------- */

const clipRateOf = (clip) => (Number.isFinite(clip?.rate) && clip.rate > 0 ? clip.rate : 1);

/** Which grid the snap track offers, honouring what the analysis produced:
 * a project analyzed before hits and beats existed has only the steady grid. */
export function gridMode(grid) {
  const want = grid?.mode ?? (grid?.hits?.length ? DEFAULT_MODE : 'grid');
  if (want === 'hits' && grid?.hits?.length) return 'hits';
  if (want === 'beats' && grid?.beats?.length >= 2) return 'beats';
  if (want === 'hits' && grid?.beats?.length >= 2) return 'beats';
  return 'grid';
}

/** The hits the snap track currently offers: band filter + strength floor. */
export function activeHits(grid) {
  const band = HIT_BANDS.find((b) => b.id === (grid?.hitBand ?? 'all')) ?? HIT_BANDS[0];
  const min = Number.isFinite(grid?.hitMin) ? grid.hitMin : DEFAULT_HIT_MIN;
  return (grid?.hits ?? []).filter((h) => h[1] >= min && band.bands.includes(h[2]));
}

/** Comp-time geometry of the grid: seconds per beat, and the comp time of a
 * bar line. Null when the grid has no usable tempo or its clip is gone. */
export function gridGeometry(grid, clip) {
  if (!grid || !clip || !(grid.bpm > 0)) return null;
  const rate = clipRateOf(clip);
  const beat = 60 / grid.bpm / rate;
  const bar = clip.start + ((grid.offset ?? 0) - (clip.in ?? 0)) / rate;
  return { beat, bar, bpb: Math.max(1, Math.round(grid.beatsPerBar ?? 4)) };
}

/** Comp seconds between snap lines at the grid's chosen granularity (for
 * tracked beats and hits: the typical spacing — the real lines are
 * wherever the music put them). */
export function gridStep(grid, clip) {
  const g = gridGeometry(grid, clip);
  if (!g) return 0;
  if (gridMode(grid) === 'hits') return activeHits(grid).length ? g.beat : 0;
  return divisionOf(grid.division).beats(g.bpb) * g.beat;
}

const srcToComp = (clip, src) => clip.start + (src - (clip.in ?? 0)) / clipRateOf(clip);
const compToSrc = (clip, t) => (clip.in ?? 0) + (t - clip.start) * clipRateOf(clip);

/** Tracked-beat lines in a SOURCE window: positions in beats from the bar
 * anchor at every `step` beats, interpolated between consecutive tracked
 * beats and extrapolated at the median period past either end. */
function beatLinesSrc(grid, s0, s1, maxLines) {
  const beats = grid.beats;
  const bpb = Math.max(1, Math.round(grid.beatsPerBar ?? 4));
  const step = divisionOf(grid.division).beats(bpb);
  const gaps = beats.slice(1).map((t, i) => t - beats[i]).sort((a, b) => a - b);
  const period = gaps[Math.floor(gaps.length / 2)] || 60 / (grid.bpm || 120);
  const n = beats.length;
  const at = (i) => (i < 0 ? beats[0] + i * period : i >= n ? beats[n - 1] + (i - n + 1) * period : beats[i]);
  const phase = ((grid.beatPhase ?? 0) % bpb + bpb) % bpb;
  // Beat index range covering the window (±1 for interpolation).
  const idxOf = (s) => (s < beats[0] ? (s - beats[0]) / period
    : s > beats[n - 1] ? n - 1 + (s - beats[n - 1]) / period
      : (() => { let i = 0; while (i < n - 1 && beats[i + 1] <= s) i++; return i + (s - beats[i]) / (beats[i + 1] - beats[i] || period); })());
  const p0 = Math.ceil((idxOf(s0) - phase) / step - 1e-9);
  const p1 = Math.floor((idxOf(s1) - phase) / step + 1e-9);
  const out = [];
  for (let k = p0; k <= p1 && out.length < maxLines; k++) {
    const pos = phase + k * step;
    const i = Math.floor(pos + 1e-9);
    const f = pos - i;
    const t = at(i) + f * (at(i + 1) - at(i));
    const beatsFromBar = k * step;
    const nearInt = (x) => Math.abs(x - Math.round(x)) < 1e-6;
    const kind = nearInt(beatsFromBar / bpb) ? 'bar' : nearInt(beatsFromBar) ? 'beat' : 'sub';
    out.push({ src: t, kind });
  }
  return out;
}

/** Every grid line in [t0, t1] (comp seconds), tagged 'bar' | 'beat' |
 * 'sub' — or, in hits mode, 'hit' with `s` (strength) and `band`.
 * `maxLines` guards against a pathological zoom (the caller thins on screen
 * density). */
export function gridLines(grid, clip, t0, t1, maxLines = 4000) {
  const g = gridGeometry(grid, clip);
  if (!g || !(t1 > t0)) return [];
  const mode = gridMode(grid);
  if (mode === 'hits') {
    const s0 = compToSrc(clip, t0), s1 = compToSrc(clip, t1);
    const out = [];
    for (const [src, s, band] of activeHits(grid)) {
      if (src < s0 - 1e-9 || src > s1 + 1e-9) continue;
      out.push({ t: srcToComp(clip, src), kind: 'hit', s, band });
      if (out.length >= maxLines) break;
    }
    return out;
  }
  if (mode === 'beats') {
    return beatLinesSrc(grid, compToSrc(clip, t0), compToSrc(clip, t1), maxLines)
      .map(({ src, kind }) => ({ t: srcToComp(clip, src), kind }));
  }
  const step = gridStep(grid, clip);
  if (!(step > 0)) return [];
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

/** The grid line nearest `t` (comp seconds), or null if there is no grid. */
export function nearestGridLine(grid, clip, t) {
  const g = gridGeometry(grid, clip);
  const step = gridStep(grid, clip);
  if (!g || !(step > 0)) return null;
  const mode = gridMode(grid);
  if (mode === 'grid') return g.bar + Math.round((t - g.bar) / step) * step;
  // Hits and tracked beats are irregular: look a few steps either side
  // (hits: a couple of beats — a snap only reaches a few pixels anyway).
  const reach = mode === 'hits' ? 2 * g.beat : Math.max(2 * step, g.beat);
  let best = null;
  for (const line of gridLines(grid, clip, t - reach, t + reach, 500))
    if (best == null || Math.abs(line.t - t) < Math.abs(best - t)) best = line.t;
  return best;
}

/** The first grid line at or after `t` (comp seconds), or null. */
export function nextGridLine(grid, clip, t) {
  const g = gridGeometry(grid, clip);
  const step = gridStep(grid, clip);
  if (!g || !(step > 0)) return null;
  if (gridMode(grid) === 'grid') return g.bar + Math.ceil((t - g.bar) / step - 1e-6) * step;
  for (let reach = Math.max(step, g.beat); reach < 1e4; reach *= 4) {
    const line = gridLines(grid, clip, t - 1e-6, t + reach, 2000).find((l) => l.t >= t - 1e-6);
    if (line) return line.t;
  }
  return null;
}

/** "128 BPM · 1/8" / "128 BPM · 1,204 hits (kicks)" — the badge text. */
export function gridSummary(grid) {
  if (!grid) return '';
  const bpm = `${+grid.bpm.toFixed(2)} BPM`;
  const mode = gridMode(grid);
  if (mode === 'hits') {
    const band = HIT_BANDS.find((b) => b.id === (grid.hitBand ?? 'all')) ?? HIT_BANDS[0];
    const n = activeHits(grid).length;
    return `${bpm} · ${n.toLocaleString()} ${band.id === 'all' ? 'hits' : band.label}`;
  }
  const div = divisionOf(grid.division).label.replace(/ \(beat\)$/, '');
  return mode === 'beats' ? `${bpm} · beats · ${div}` : `${bpm} · ${div}`;
}

export function divisionLabel(id) { return divisionOf(id).label; }
