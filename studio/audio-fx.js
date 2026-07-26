/*
 * slangfx studio — audio effects.
 *
 * The audio twin of the slang shader catalogue. Each effect is a small
 * Web Audio graph described as data: a parameter list (which the inspector
 * and the timeline render exactly like shader parameters, keyframes and
 * drivers included) plus a `build` that wires the nodes.
 *
 * `build` takes any BaseAudioContext, which is what lets one definition
 * serve both worlds: the realtime preview builds into the live
 * AudioContext and pokes AudioParams every frame, while the exporter
 * builds the same graph into an OfflineAudioContext and bakes the same
 * curves as scheduled ramps. Nothing in here knows about clips, time or
 * the comp — app.js owns all of that.
 *
 * EVERY parameter here is an AudioParam, without exception — so every one
 * of them keyframes, takes a driver, and bakes sample-accurately into the
 * export. That constraint shaped two of the effects:
 *
 *   reverb      a feedback comb/diffusion network, not a convolver. An
 *               impulse response is a buffer you'd have to regenerate to
 *               change the decay (impossible to automate, and ruinous at
 *               frame rate); in a comb network the decay IS the feedback
 *               gain and the damping IS a filter cutoff.
 *   distortion  a FIXED saturation curve with drive as pre-gain into it
 *               and a compensating post-gain — which is also how the real
 *               thing works. Rebuilding a waveshaper table per frame would
 *               be the only alternative.
 *
 * `map` on a control converts the human-facing value (dB, %, ms, seconds
 * of decay) into what the AudioParam wants; `also` lets one slider drive
 * several params at once, each through its own map — that's what makes a
 * single Decay control ride six comb gains.
 *
 * `scale: 'log'` on a parameter tells the UI to lay its slider out
 * logarithmically. Frequency and time are heard that way, and a linear
 * 20 Hz–20 kHz track leaves three usable pixels below 200 Hz.
 */

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

const dbToGain = (db) => 10 ** (db / 20);
const pct = (v) => v / 100;
const ms = (v) => v / 1000;

/** One parameter of an audio effect (same shape the inspector uses for
 * shader parameters, plus `scale`). */
function P(name, label, min, max, step, def, unit = '', extra = {}) {
  return { name, label, min, max, step, def, unit, ...extra };
}

/** Frequency / time parameters: log slider, always. */
const LOG = { scale: 'log' };

/* ---- shared node builders ------------------------------------------- */

/** Wet/dry pair around an effect branch: input feeds both, `mix` (0..100)
 * crossfades. Returns the branch's entry/exit gains. */
function wetDry(ctx) {
  const input = ctx.createGain();
  const output = ctx.createGain();
  const dry = ctx.createGain();
  const wet = ctx.createGain();
  input.connect(dry).connect(output);
  wet.connect(output);
  return { input, output, dry, wet };
}

/** Crossfade control: one slider drives both gains (equal-gain, not
 * equal-power — a 50% "mix" on a reverb should still sum to unity-ish). */
function mixControl(dry, wet) {
  return {
    param: wet.gain,
    map: pct,
    also: [{ param: dry.gain, map: (v) => 1 - pct(v) }],
  };
}

/* ---- reverb network --------------------------------------------------
 * Six parallel feedback combs (mutually prime delays so their echo trains
 * don't line up) into two diffusers. Decay maps to each comb's feedback
 * gain through the standard RT60 relation, which is why a single slider
 * can drive six different AudioParams and stay musically correct. */

const COMB_MS = [29.7, 37.1, 41.1, 43.7, 26.3, 33.9];

/** Feedback gain that decays a `d`-second loop by 60 dB in `rt60`. Capped
 * short of 1 — a comb loop at unity never stops ringing. */
const rt60Gain = (d, rt60) => clamp(10 ** ((-3 * d) / Math.max(rt60, 0.05)), 0, 0.985);

/** Damping 0..100 → the cutoff of the lowpass inside each comb loop, so
 * every pass round the loop loses a little more top end. */
const dampFreq = (v) => 400 + 17600 * (1 - clamp(v, 0, 100) / 100) ** 2;

/** Schroeder allpass: smears a comb's discrete echoes into diffusion.
 * Fixed coefficient — this is the reverb's character, not a control. */
function diffuser(ctx, src, ms, g = 0.7) {
  const delay = ctx.createDelay(0.1);
  delay.delayTime.value = ms / 1000;
  const fb = ctx.createGain();
  fb.gain.value = g;
  const ff = ctx.createGain();
  ff.gain.value = -g;
  const out = ctx.createGain();
  src.connect(delay);
  src.connect(ff).connect(out);
  delay.connect(fb).connect(delay);
  delay.connect(out);
  return out;
}

/** Fixed soft-clip table, normalized to unity at full scale. Drive rides
 * a gain in front of it instead of reshaping the curve. */
const SAT_K = 8;
function saturationCurve() {
  const n = 4096;
  const curve = new Float32Array(n);
  const norm = Math.tanh(SAT_K);
  for (let i = 0; i < n; i++) curve[i] = Math.tanh(((i * 2) / (n - 1) - 1) * SAT_K) / norm;
  return curve;
}
let satCurve = null;

/* ---- the catalogue --------------------------------------------------- */

/**
 * `group` is the one-word note the picker shows beside the name (level,
 * filter, tone, space, dynamics, drive) — the list has room for a label,
 * not a description.
 *
 * @type {Array<{id, label, group, params, widget?, build(ctx): {input,
 *               output, controls: Object<string,{param, map?, also?}>}}>}
 */
export const AUDIO_EFFECTS = [
  {
    id: 'gain',
    label: 'gain',
    group: 'level',
    params: [P('gain', 'Gain', -60, 24, 0.1, 0, 'dB')],
    build(ctx) {
      const g = ctx.createGain();
      return { input: g, output: g, controls: { gain: { param: g.gain, map: dbToGain } } };
    },
  },
  {
    id: 'pan',
    label: 'stereo pan',
    group: 'level',
    params: [P('pan', 'Pan', -100, 100, 1, 0, 'L↔R')],
    build(ctx) {
      const p = ctx.createStereoPanner();
      return { input: p, output: p, controls: { pan: { param: p.pan, map: pct } } };
    },
  },
  {
    id: 'highpass',
    label: 'high-pass',
    group: 'filter',
    params: [
      P('freq', 'Cutoff', 20, 18000, 1, 200, 'Hz', LOG),
      // Web Audio states a lowpass/highpass Q in dB, which happens to be
      // exactly the height of the knee at the corner — so this slider and
      // the curve's handle read the same number.
      P('q', 'Resonance', -6, 24, 0.1, -3, 'dB'),
    ],
    widget: { bands: [{ type: 'highpass', freq: 'freq', q: 'q', qMode: 'db' }] },
    build(ctx) {
      const f = ctx.createBiquadFilter();
      f.type = 'highpass';
      return { input: f, output: f, controls: { freq: { param: f.frequency }, q: { param: f.Q } } };
    },
  },
  {
    id: 'lowpass',
    label: 'low-pass',
    group: 'filter',
    params: [
      P('freq', 'Cutoff', 20, 20000, 1, 8000, 'Hz', LOG),
      P('q', 'Resonance', -6, 24, 0.1, -3, 'dB'),
    ],
    widget: { bands: [{ type: 'lowpass', freq: 'freq', q: 'q', qMode: 'db' }] },
    build(ctx) {
      const f = ctx.createBiquadFilter();
      f.type = 'lowpass';
      return { input: f, output: f, controls: { freq: { param: f.frequency }, q: { param: f.Q } } };
    },
  },
  {
    id: 'bandpass',
    label: 'band-pass',
    group: 'filter',
    params: [
      P('freq', 'Centre', 40, 16000, 1, 1200, 'Hz', LOG),
      P('q', 'Width', 0.2, 30, 0.01, 2, 'Q', LOG),
    ],
    widget: { bands: [{ type: 'bandpass', freq: 'freq', q: 'q', qMode: 'factor' }] },
    build(ctx) {
      const f = ctx.createBiquadFilter();
      f.type = 'bandpass';
      return { input: f, output: f, controls: { freq: { param: f.frequency }, q: { param: f.Q } } };
    },
  },
  {
    id: 'notch',
    label: 'notch',
    group: 'filter',
    params: [
      P('freq', 'Centre', 30, 16000, 1, 60, 'Hz', LOG),
      P('q', 'Width', 0.5, 40, 0.01, 12, 'Q', LOG),
    ],
    widget: { bands: [{ type: 'notch', freq: 'freq', q: 'q', qMode: 'factor' }] },
    build(ctx) {
      const f = ctx.createBiquadFilter();
      f.type = 'notch';
      return { input: f, output: f, controls: { freq: { param: f.frequency }, q: { param: f.Q } } };
    },
  },
  {
    id: 'eq',
    label: 'EQ (3-band)',
    group: 'tone',
    params: [
      P('low', 'Low gain', -24, 24, 0.1, 0, 'dB'),
      P('lowFreq', 'Low freq', 30, 1000, 1, 200, 'Hz', LOG),
      P('mid', 'Mid gain', -24, 24, 0.1, 0, 'dB'),
      P('midFreq', 'Mid freq', 100, 10000, 1, 1200, 'Hz', LOG),
      P('midQ', 'Mid width', 0.2, 12, 0.01, 0.9, 'Q', LOG),
      P('high', 'High gain', -24, 24, 0.1, 0, 'dB'),
      P('highFreq', 'High freq', 1000, 16000, 1, 4000, 'Hz', LOG),
    ],
    widget: {
      bands: [
        { type: 'lowshelf', freq: 'lowFreq', gain: 'low', label: 'low' },
        { type: 'peaking', freq: 'midFreq', gain: 'mid', q: 'midQ', qMode: 'factor', label: 'mid' },
        { type: 'highshelf', freq: 'highFreq', gain: 'high', label: 'high' },
      ],
    },
    build(ctx) {
      const lo = ctx.createBiquadFilter();
      lo.type = 'lowshelf';
      const mid = ctx.createBiquadFilter();
      mid.type = 'peaking';
      const hi = ctx.createBiquadFilter();
      hi.type = 'highshelf';
      lo.connect(mid).connect(hi);
      return {
        input: lo,
        output: hi,
        controls: {
          low: { param: lo.gain }, lowFreq: { param: lo.frequency },
          mid: { param: mid.gain }, midFreq: { param: mid.frequency },
          midQ: { param: mid.Q },
          high: { param: hi.gain }, highFreq: { param: hi.frequency },
        },
      };
    },
  },
  {
    id: 'reverb',
    label: 'reverb',
    group: 'space',
    params: [
      P('mix', 'Mix', 0, 100, 0.1, 25, '%'),
      P('decay', 'Decay', 0.1, 12, 0.01, 1.8, 's', LOG),
      P('damping', 'Damping', 0, 100, 1, 45, '%'),
      P('preDelay', 'Pre-delay', 0, 200, 1, 20, 'ms'),
    ],
    build(ctx) {
      const { input, output, dry, wet } = wetDry(ctx);
      const pre = ctx.createDelay(0.5);
      const merge = ctx.createGain();
      merge.gain.value = 1 / COMB_MS.length;
      input.connect(pre);

      const decayTargets = [];
      const dampTargets = [];
      for (const msDelay of COMB_MS) {
        const d = msDelay / 1000;
        const delay = ctx.createDelay(0.1);
        delay.delayTime.value = d;
        const fb = ctx.createGain();
        fb.gain.value = rt60Gain(d, 1.8);
        const lp = ctx.createBiquadFilter();
        lp.type = 'lowpass';
        // A lowpass/highpass Q is in DECIBELS in Web Audio, so the default
        // 1 is a ~1 dB peak at the corner — inside a feedback loop that
        // tips the gain past unity and the "reverb" howls instead of
        // decaying (measured: +114 dB after 6 s). −6 dB is critically
        // damped: magnitude ≤ 1 at every frequency.
        lp.Q.value = -6;
        lp.frequency.value = dampFreq(45);
        pre.connect(delay);
        delay.connect(lp).connect(fb).connect(delay);
        delay.connect(merge);
        // Same slider, six different maps — each comb needs the feedback
        // gain that gives THIS delay length the requested RT60.
        decayTargets.push({ param: fb.gain, map: (sec) => rt60Gain(d, sec) });
        dampTargets.push({ param: lp.frequency, map: dampFreq });
      }
      diffuser(ctx, diffuser(ctx, merge, 5.0), 1.7).connect(wet);

      return {
        input,
        output,
        controls: {
          mix: mixControl(dry, wet),
          preDelay: { param: pre.delayTime, map: ms },
          decay: { ...decayTargets[0], also: decayTargets.slice(1) },
          damping: { ...dampTargets[0], also: dampTargets.slice(1) },
        },
      };
    },
  },
  {
    id: 'delay',
    label: 'delay',
    group: 'space',
    params: [
      P('time', 'Time', 1, 2000, 1, 300, 'ms', LOG),
      P('feedback', 'Feedback', 0, 95, 0.1, 35, '%'),
      P('tone', 'Tone', 300, 16000, 1, 6000, 'Hz', LOG),
      P('mix', 'Mix', 0, 100, 0.1, 30, '%'),
    ],
    build(ctx) {
      const { input, output, dry, wet } = wetDry(ctx);
      const delay = ctx.createDelay(2.1);
      const fb = ctx.createGain();
      const tone = ctx.createBiquadFilter();
      tone.type = 'lowpass';
      // Repeats darken as they decay, which is what stops a long feedback
      // from turning into a howl.
      input.connect(delay);
      delay.connect(tone).connect(fb).connect(delay);
      delay.connect(wet);
      return {
        input,
        output,
        controls: {
          time: { param: delay.delayTime, map: ms },
          feedback: { param: fb.gain, map: pct },
          tone: { param: tone.frequency },
          mix: mixControl(dry, wet),
        },
      };
    },
  },
  {
    id: 'compressor',
    label: 'compressor',
    group: 'dynamics',
    params: [
      P('threshold', 'Threshold', -60, 0, 0.1, -24, 'dB'),
      P('ratio', 'Ratio', 1, 20, 0.1, 4, ':1'),
      P('attack', 'Attack', 0.1, 200, 0.1, 3, 'ms', LOG),
      P('release', 'Release', 1, 1000, 1, 250, 'ms', LOG),
      P('knee', 'Knee', 0, 40, 0.1, 6, 'dB'),
      P('makeup', 'Make-up', -12, 24, 0.1, 0, 'dB'),
    ],
    build(ctx) {
      const c = ctx.createDynamicsCompressor();
      const out = ctx.createGain();
      c.connect(out);
      return {
        input: c,
        output: out,
        controls: {
          threshold: { param: c.threshold },
          ratio: { param: c.ratio },
          attack: { param: c.attack, map: ms },
          release: { param: c.release, map: ms },
          knee: { param: c.knee },
          makeup: { param: out.gain, map: dbToGain },
        },
      };
    },
  },
  {
    id: 'distortion',
    label: 'distortion',
    group: 'drive',
    params: [
      P('drive', 'Drive', 0, 100, 0.1, 25, '%'),
      P('tone', 'Tone', 500, 18000, 1, 9000, 'Hz', LOG),
      P('mix', 'Mix', 0, 100, 0.1, 100, '%'),
    ],
    build(ctx) {
      const { input, output, dry, wet } = wetDry(ctx);
      const preGain = ctx.createGain();
      const shaper = ctx.createWaveShaper();
      shaper.oversample = '4x';
      shaper.curve = (satCurve ??= saturationCurve());
      const postGain = ctx.createGain();
      const tone = ctx.createBiquadFilter();
      tone.type = 'lowpass';
      input.connect(preGain).connect(shaper).connect(postGain).connect(tone).connect(wet);
      // Drive is how hard the signal is pushed into the curve; the post
      // gain hands most of that level back so the slider changes character
      // rather than just loudness.
      const driveGain = (v) => 1 + (clamp(v, 0, 100) / 100) * (SAT_K - 1);
      return {
        input,
        output,
        controls: {
          drive: {
            param: preGain.gain,
            map: driveGain,
            also: [{ param: postGain.gain, map: (v) => 1 / (0.35 + 0.65 * driveGain(v)) }],
          },
          tone: { param: tone.frequency },
          mix: mixControl(dry, wet),
        },
      };
    },
  },
];

export function audioEffectDef(id) {
  return AUDIO_EFFECTS.find((e) => e.id === id) ?? null;
}

/** Every control an effect's parameter touches (a mix slider rides two
 * gains), so callers can automate them uniformly. */
export function controlTargets(unit, name) {
  const c = unit.controls?.[name];
  if (!c) return [];
  return [c, ...(c.also ?? [])];
}
