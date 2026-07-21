/*
 * slangfx studio — offline audio analysis for property drivers.
 *
 * Turns a mono mixdown of the comp's audio into per-frame band envelopes
 * (one value per comp frame, normalized 0..1) plus onset ("beat") times
 * derived from those envelopes. Precomputing instead of tapping a live
 * AnalyserNode is what makes drivers deterministic: scrubbing backwards,
 * realtime playback, and the offline exporter all read identical values.
 *
 * Band split uses RBJ-cookbook biquads run in plain JS — at the analysis
 * sample rate (22.05 kHz) a whole comp filters in milliseconds:
 *
 *   level   the unfiltered mix
 *   bass    ≤ ~150 Hz   (two cascaded lowpasses — kick/bassline)
 *   mid     ~200–2000 Hz (highpass + lowpass — vocals, snare body)
 *   treble  ≥ ~4 kHz    (hi-hats, cymbals, air)
 *
 * Envelope sampling is random-access friendly: 'level' applies its release
 * as a look-back peak decay (no sequential smoothing state), and 'beat'
 * finds the last onset before t and decays a pulse from it.
 */

/** RBJ biquad coefficients. kind: 'lp' | 'hp'. */
function biquad(kind, sr, f0, q = 0.7071) {
  const w0 = 2 * Math.PI * (f0 / sr);
  const cw = Math.cos(w0);
  const alpha = Math.sin(w0) / (2 * q);
  let b0, b1, b2;
  if (kind === 'lp') {
    b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2;
  } else {
    b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2;
  }
  const a0 = 1 + alpha;
  return {
    b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
    a1: (-2 * cw) / a0, a2: (1 - alpha) / a0,
  };
}

/** Run samples through a chain of biquads (direct form 1), new array. */
function filterChain(samples, chain) {
  let cur = samples;
  for (const c of chain) {
    const out = new Float32Array(cur.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < cur.length; i++) {
      const x = cur[i];
      const y = c.b0 * x + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
      out[i] = y;
      x2 = x1; x1 = x; y2 = y1; y1 = y;
    }
    cur = out;
  }
  return cur;
}

/** Per-comp-frame RMS of samples, normalized so the loudest frame is 1. */
function frameEnvelope(samples, sr, fps, frames) {
  const hop = sr / fps;
  const env = new Float32Array(frames);
  let max = 0;
  for (let f = 0; f < frames; f++) {
    const lo = Math.floor(f * hop);
    const hi = Math.min(Math.floor((f + 1) * hop), samples.length);
    let sum = 0;
    for (let i = lo; i < hi; i++) sum += samples[i] * samples[i];
    const rms = hi > lo ? Math.sqrt(sum / (hi - lo)) : 0;
    env[f] = rms;
    if (rms > max) max = rms;
  }
  if (max > 1e-6) for (let f = 0; f < frames; f++) env[f] /= max;
  return env;
}

/**
 * Analyze a mono mixdown.
 * @returns {{fps, frames, bands: {level,bass,mid,treble: Float32Array}}}
 */
export function analyzeMix(samples, sr, fps) {
  const frames = Math.max(1, Math.ceil((samples.length / sr) * fps));
  const chains = {
    level: [],
    bass: [biquad('lp', sr, 150), biquad('lp', sr, 150)],
    mid: [biquad('hp', sr, 200), biquad('lp', sr, 2000)],
    treble: [biquad('hp', sr, 4000)],
  };
  const bands = {};
  for (const [name, chain] of Object.entries(chains))
    bands[name] = frameEnvelope(chain.length ? filterChain(samples, chain) : samples, sr, fps, frames);
  return { fps, frames, bands };
}

/**
 * Onset times (seconds) from a band envelope: fire where the envelope
 * crosses above `sensitivity ×` its ~1s moving average (with an absolute
 * floor so silence never triggers), with a 150 ms refractory gap.
 * Higher sensitivity = fewer, stronger beats.
 */
export function detectBeats(env, fps, sensitivity = 1.5) {
  const win = Math.max(2, Math.round(fps));
  const floor = 0.08;
  const minGap = 0.15;
  const beats = [];
  let sum = 0;
  let last = -Infinity;
  let wasAbove = true;   // suppress a fake onset at frame 0
  for (let i = 0; i < env.length; i++) {
    sum += env[i];
    if (i >= win) sum -= env[i - win];
    const avg = sum / Math.min(i + 1, win);
    const above = env[i] > Math.max(avg * sensitivity, floor);
    const t = i / fps;
    if (above && !wasAbove && t - last >= minGap) {
      beats.push(t);
      last = t;
    }
    wasAbove = above;
  }
  return beats;
}

/** Envelope level at time t with a release tail: instant attack, and the
 * value decays from recent peaks over `release` seconds. Random access —
 * a bounded look-back replaces sequential smoothing state. */
export function sampleLevel(env, fps, t, release = 0.25) {
  const i = Math.max(0, Math.min(env.length - 1, Math.round(t * fps)));
  if (!(release > 0.02)) return env[i];
  const back = Math.min(Math.ceil(release * fps * 5), 300);
  let v = 0;
  for (let k = 0; k <= back; k++) {
    const j = i - k;
    if (j < 0) break;
    const w = env[j] * Math.exp(-(k / fps) / release);
    if (w > v) v = w;
    if (env[j] <= v && w < 0.01) break;   // tail below visibility — done
  }
  return v;
}

/** Beat pulse at time t: 1 at the most recent onset, exp-decaying over
 * `decay` seconds. beats must be ascending (detectBeats output). */
export function samplePulse(beats, t, decay = 0.35) {
  let lo = 0, hi = beats.length - 1, tb = -Infinity;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (beats[mid] <= t + 1e-6) { tb = beats[mid]; lo = mid + 1; }
    else hi = mid - 1;
  }
  if (tb === -Infinity) return 0;
  return Math.exp(-(t - tb) / Math.max(+decay || 0, 0.02));
}
