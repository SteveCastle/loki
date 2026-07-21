/*
 * slangfx studio — property drivers.
 *
 * A driver modulates one PropTrack's value on top of (or instead of) its
 * keyframed base, from one of two input sources:
 *
 *   osc    a deterministic waveform of CLIP-RELATIVE time — sin, triangle,
 *          saw, square, pulse, bounce, or smooth value-noise. freq is in
 *          cycles per second, so frame-locked patterns fall out naturally
 *          (freq = fps/N repeats every N frames) and moving a clip carries
 *          its motion along, exactly like keyframes.
 *
 *   audio  a per-frame envelope of the comp's audio mix in a frequency
 *          band (full mix / bass / mids / treble), either followed
 *          directly ('level', with a release tail) or turned into
 *          decaying pulses at detected onsets ('beat'). The envelopes are
 *          PRECOMPUTED offline (audio-analysis.js) so scrubbing and the
 *          offline exporter see the exact same values as playback.
 *
 * The signal is mapped through `offset + amount·signal` and combined with
 * the base value by `mode`:
 *
 *   add        base + delta                    (delta in the property's units)
 *   multiply   base · (1 + delta/100)         (delta in percent — scale pulses)
 *   replace    delta                           (ignore keyframes entirely)
 *
 * The driver object is plain JSON living at `prop.driver`, so it rides
 * along with project persistence, undo snapshots, and clip duplication
 * for free. Everything here is pure; the audio lookup is injected by the
 * caller (app.js owns the analysis cache).
 */

export const DRIVER_WAVES = [
  ['sin', 'Sine'],
  ['triangle', 'Triangle'],
  ['saw', 'Saw'],
  ['square', 'Square'],
  ['pulse', 'Pulse'],
  ['bounce', 'Bounce'],
  ['noise', 'Noise'],
];

export const DRIVER_BANDS = [
  ['level', 'Full mix'],
  ['bass', 'Bass'],
  ['mid', 'Mids'],
  ['treble', 'Treble'],
];

export const DRIVER_FOLLOWS = [
  ['level', 'Level'],
  ['beat', 'Beat'],
];

export const DRIVER_MODES = [
  ['add', 'Add'],
  ['multiply', 'Multiply %'],
  ['replace', 'Replace'],
];

const frac = (x) => x - Math.floor(x);

/* Deterministic per-integer hash in [-1, 1] (classic sin-hash). Reproducible
 * across sessions and in the offline exporter — never Math.random. */
const hash = (i) => frac(Math.sin(i * 127.1 + 311.7) * 43758.5453123) * 2 - 1;

/** Waveform value for a driver at clip-relative time t (seconds).
 * sin/triangle/saw/square/noise span [-1, 1]; pulse is 0/1 and bounce is
 * 0..1 (one hop per cycle) — one-sided on purpose, so "kick up on the
 * pulse" needs no offset fiddling. */
export function waveSignal(d, t) {
  const ph = t * (+d.freq || 0) + (+d.phase || 0);
  const w = +d.width > 0 && +d.width < 1 ? +d.width : 0.5;
  switch (d.wave) {
    case 'triangle': return 1 - 4 * Math.abs(frac(ph) - 0.5);
    case 'saw': return 2 * frac(ph) - 1;
    case 'square': return frac(ph) < w ? 1 : -1;
    case 'pulse': return frac(ph) < w ? 1 : 0;
    case 'bounce': return Math.abs(Math.sin(Math.PI * ph));
    case 'noise': {
      // Smooth value noise: hash per cycle, smoothstep between cycles.
      const i = Math.floor(ph);
      const u = frac(ph);
      const s = u * u * (3 - 2 * u);
      return hash(i) * (1 - s) + hash(i + 1) * s;
    }
    default: return Math.sin(2 * Math.PI * ph);   // 'sin'
  }
}

/** Fresh driver with sensible defaults for a property whose slider range
 * is def.min..def.max (amount = a quarter of the range). */
export function newDriver(def = {}) {
  const range = Number.isFinite(def.max) && Number.isFinite(def.min)
    ? (def.max - def.min) / 4 : 25;
  return {
    enabled: true,
    source: 'osc',
    // osc
    wave: 'sin', freq: 1, phase: 0, width: 0.5,
    // audio
    band: 'bass', follow: 'level', release: 0.25, sensitivity: 1.5, decay: 0.35,
    // mapping
    amount: Math.round(range * 100) / 100, offset: 0, mode: 'add',
  };
}

/**
 * Combine a driver with the keyframed base value.
 * @param {number} base       evalProp result at tClip
 * @param {object} d          prop.driver (enabled checked by the caller)
 * @param {number} tClip      clip-relative seconds (osc time base)
 * @param {(d)=>number} audioSignal  resolves an audio driver to 0..1
 */
export function applyDriver(base, d, tClip, audioSignal) {
  const sig = d.source === 'audio' ? audioSignal(d) : waveSignal(d, tClip);
  const delta = (+d.offset || 0) + (+d.amount || 0) * sig;
  switch (d.mode) {
    case 'replace': return delta;
    case 'multiply': return base * (1 + delta / 100);
    default: return base + delta;
  }
}
