/*
 * slangfx studio — property drivers.
 *
 * A driver modulates one PropTrack's value on top of (or instead of) its
 * keyframed base, from one of two input sources:
 *
 *   osc    a deterministic waveform of CLIP-RELATIVE time — sin, triangle,
 *          saw, square, pulse, bounce, smooth value-noise, or one of the
 *          random behaviors (stepped sample-and-hold, multi-octave drift,
 *          on/off flicker, sparse decaying sparks). freq is in cycles per
 *          second, so frame-locked patterns fall out naturally
 *          (freq = fps/N repeats every N frames) and moving a clip carries
 *          its motion along, exactly like keyframes. The randoms are
 *          hash-based, never Math.random — scrubbing, playback and the
 *          offline exporter all see the same values.
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
  ['steps', 'Random steps'],
  ['drift', 'Noise drift'],
  ['flicker', 'Flicker'],
  ['sparks', 'Sparks'],
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

/* A third source, 'track', reads a channel of a tracker-null layer's
 * BAKED keyframes (motion tracking, app.js's bakeTrackerLayer) at COMP
 * time. Resolved by an injected lookup like audio — the comp lives with
 * the caller. Reading base keyframes only (never the target's own driver)
 * is what makes reference cycles impossible. */
export const DRIVER_TRACK_CHANNELS = [
  ['x', 'Position X'],
  ['y', 'Position Y'],
  ['rot', 'Rotation'],
  ['scale', 'Scale'],
];

export const DRIVER_TRACK_REFS = [
  ['abs', 'Absolute'],
  ['delta', 'Motion (Δ from start)'],
];

const frac = (x) => x - Math.floor(x);

/* Deterministic per-integer hash in [-1, 1] (classic sin-hash). Reproducible
 * across sessions and in the offline exporter — never Math.random. */
const hash = (i) => frac(Math.sin(i * 127.1 + 311.7) * 43758.5453123) * 2 - 1;

/** Waveform value for a driver at clip-relative time t (seconds).
 * sin/triangle/saw/square/noise/steps/drift span [-1, 1]; pulse and
 * flicker are 0/1, bounce and sparks are 0..1 — one-sided on purpose, so
 * "kick up on the pulse" needs no offset fiddling. */
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
    case 'steps':
      // Sample & hold: a fresh random level each cycle, held flat.
      return hash(Math.floor(ph));
    case 'drift': {
      // Three octaves of the smooth noise — organic wander with texture.
      let v = 0, a = 0.6, p = ph;
      for (let k = 0; k < 3; k++) {
        const i = Math.floor(p), u = frac(p), s = u * u * (3 - 2 * u);
        v += a * (hash(i + k * 57) * (1 - s) + hash(i + 1 + k * 57) * s);
        p *= 2.17;
        a *= 0.5;
      }
      return v;
    }
    case 'flicker':
      // Random telegraph: each cycle is fully on or off — width sets the
      // on-probability. The faulty-fluorescent wave.
      return hash(Math.floor(ph)) * 0.5 + 0.5 < w ? 1 : 0;
    case 'sparks': {
      // Sparse impulses: width of the cycles fire a spark of random
      // height that decays within the cycle. Lightning / arcing / glitch.
      const i = Math.floor(ph);
      const on = hash(i) * 0.5 + 0.5 < w ? 1 : 0;
      const h = hash(i + 917) * 0.5 + 0.5;
      return on * (0.4 + 0.6 * h) * Math.exp(-frac(ph) * 6);
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
 * @param {(d)=>number} trackSignal  resolves a tracker-channel driver
 *                                   (property units; 0 when unresolvable)
 */
export function applyDriver(base, d, tClip, audioSignal, trackSignal = () => 0) {
  const sig = d.source === 'audio' ? audioSignal(d)
    : d.source === 'track' ? trackSignal(d)
      : waveSignal(d, tClip);
  const delta = (+d.offset || 0) + (+d.amount || 0) * sig;
  switch (d.mode) {
    case 'replace': return delta;
    case 'multiply': return base * (1 + delta / 100);
    default: return base + delta;
  }
}
