/*
 * slangfx studio — composition data model.
 *
 * A Comp is plain JSON: tracks (top-to-bottom, like After Effects) holding
 * clips positioned on a shared timebase. Two clip kinds:
 *
 *   media  a video or image asset with a keyframable 2D transform
 *          (position / scale / rotation / opacity) composited into the frame
 *   fx     a shader preset (bundled or hand-written) applied to everything
 *          composited below it, with every shader parameter keyframable
 *
 * Every animatable property is a PropTrack: either a static value or a list
 * of keyframes `{t, v, e}` where `t` is CLIP-RELATIVE seconds (moving a clip
 * carries its animation along) and `e` names the easing curve applied on the
 * segment leaving that key.
 *
 * Rendering semantics (bottom track first):
 *   - media clips draw in track order, bottom → top
 *   - fx clips form the shader chain in track order, bottom → top
 *     (an fx clip on a higher track applies later, i.e. "on top")
 *
 * Everything here is UI-free and serializable with JSON.stringify.
 */

let uidCounter = Math.floor(Math.random() * 36 ** 4);
export function uid(prefix = 'id') {
  return `${prefix}_${Date.now().toString(36)}${(uidCounter++ % 36 ** 4).toString(36)}`;
}

/* ---- easing -------------------------------------------------------- */

export const EASINGS = {
  linear: (u) => u,
  in: (u) => u * u * u,
  out: (u) => 1 - (1 - u) ** 3,
  inout: (u) => (u < 0.5 ? 4 * u * u * u : 1 - (-2 * u + 2) ** 3 / 2),
  back: (u) => {
    const c1 = 1.70158, c3 = c1 + 1;
    return 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2;
  },
  elastic: (u) => {
    if (u === 0 || u === 1) return u;
    return 2 ** (-10 * u) * Math.sin((u * 10 - 0.75) * (2 * Math.PI) / 3) + 1;
  },
  bounce: (u) => {
    const n1 = 7.5625, d1 = 2.75;
    if (u < 1 / d1) return n1 * u * u;
    if (u < 2 / d1) return n1 * (u -= 1.5 / d1) * u + 0.75;
    if (u < 2.5 / d1) return n1 * (u -= 2.25 / d1) * u + 0.9375;
    return n1 * (u -= 2.625 / d1) * u + 0.984375;
  },
  hold: (u) => 0,
};

export const EASING_LABELS = [
  ['linear', 'Linear'],
  ['inout', 'Ease In-Out'],
  ['in', 'Ease In'],
  ['out', 'Ease Out'],
  ['back', 'Overshoot'],
  ['elastic', 'Elastic'],
  ['bounce', 'Bounce'],
  ['hold', 'Hold'],
];

/* ---- property tracks ----------------------------------------------- */

/** @returns {{v:number, anim:boolean, keys:Array<{t:number,v:number,e:string}>}} */
export function newProp(v) {
  return { v, anim: false, keys: [] };
}

export function sortKeys(prop) {
  prop.keys.sort((a, b) => a.t - b.t);
}

/** Evaluate a PropTrack at clip-relative time t (seconds). */
export function evalProp(prop, t) {
  if (!prop) return 0;
  if (!prop.anim || prop.keys.length === 0) return prop.v;
  const keys = prop.keys;
  if (t <= keys[0].t) return keys[0].v;
  const last = keys[keys.length - 1];
  if (t >= last.t) return last.v;
  let lo = 0, hi = keys.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (keys[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = keys[lo], b = keys[hi];
  if (a.e === 'hold' || b.t <= a.t) return a.v;
  const u = (t - a.t) / (b.t - a.t);
  const f = EASINGS[a.e] ?? EASINGS.linear;
  return a.v + (b.v - a.v) * f(u);
}

/** Add or replace a keyframe at time t (seconds, clip-relative). */
export function upsertKey(prop, t, v, e = null) {
  const EPS = 1e-4;
  const existing = prop.keys.find((k) => Math.abs(k.t - t) < EPS);
  if (existing) {
    existing.v = v;
    if (e) existing.e = e;
  } else {
    prop.keys.push({ t, v, e: e ?? 'inout' });
    sortKeys(prop);
  }
  prop.anim = true;
}

export function removeKeyNear(prop, t, eps = 1e-4) {
  const i = prop.keys.findIndex((k) => Math.abs(k.t - t) < eps);
  if (i >= 0) prop.keys.splice(i, 1);
  if (prop.keys.length === 0) prop.anim = false;
  return i >= 0;
}

export function keyNear(prop, t, eps = 1e-4) {
  return prop.keys.find((k) => Math.abs(k.t - t) < eps) ?? null;
}

/* ---- clips ---------------------------------------------------------- */

/** Media transform properties: [key, label, defaultFor(comp), unit]. */
export const MEDIA_PROPS = [
  ['x', 'Position X', (c) => c.width / 2, 'px'],
  ['y', 'Position Y', (c) => c.height / 2, 'px'],
  ['scaleX', 'Scale X', () => 100, '%'],
  ['scaleY', 'Scale Y', () => 100, '%'],
  ['rot', 'Rotation', () => 0, '°'],
  ['opacity', 'Opacity', () => 100, '%'],
  ['volume', 'Volume', () => 100, '%'],   // audible for video assets only
];

/** Upgrade older projects in place (e.g. uniform `scale` → scaleX/scaleY). */
export function migrateComp(comp) {
  for (const track of comp.tracks ?? [])
    for (const clip of track.clips) {
      if (clip.kind !== 'media' || !clip.props) continue;
      if (clip.props.scale && !clip.props.scaleX) {
        clip.props.scaleX = clip.props.scale;
        clip.props.scaleY = structuredClone(clip.props.scale);
        delete clip.props.scale;
      }
      for (const [key, , def] of MEDIA_PROPS) clip.props[key] ??= newProp(def(comp));
    }
  return comp;
}

export function newMediaClip(comp, asset, start, dur) {
  const props = {};
  for (const [key, , def] of MEDIA_PROPS) props[key] = newProp(def(comp));
  return {
    id: uid('clip'),
    kind: 'media',
    name: asset.name.replace(/\.[^.]+$/, ''),
    assetId: asset.id,
    start,
    dur,
    in: 0,                       // trim offset into the source (seconds)
    props,
  };
}

/**
 * @param {object} spec  {fxKind:'preset'|'custom', path?, source?, label}
 */
export function newFxClip(spec, start, dur) {
  return {
    id: uid('clip'),
    kind: 'fx',
    name: spec.label,
    fxKind: spec.fxKind,
    path: spec.path ?? null,      // preset path (fxKind 'preset')
    source: spec.source ?? null,  // slang source (fxKind 'custom')
    savedName: spec.savedName ?? null,
    enabled: true,
    start,
    dur,
    params: {},                   // paramName -> PropTrack (created on touch)
    mask: null,                   // {dataURL, opacity, invert}
    overlay: null,                // texName -> overlay source descriptor
  };
}

export function clipEnd(clip) { return clip.start + clip.dur; }

/** The PropTrack for a named property, creating it (fx params) on demand. */
export function clipProp(clip, name, defaultValue = 0) {
  if (clip.kind === 'media') return clip.props[name];
  return (clip.params[name] ??= newProp(defaultValue));
}

/** Evaluate a clip property at COMP time t. */
export function clipPropAt(clip, name, tComp, defaultValue = 0) {
  const prop = clip.kind === 'media' ? clip.props[name] : clip.params[name];
  if (!prop) return defaultValue;
  return evalProp(prop, tComp - clip.start);
}

/** Split a clip at comp time t; mutates `clip` into the left part and
 * returns the new right part (or null if t is outside the clip). */
export function splitClip(clip, t) {
  if (t <= clip.start + 1e-6 || t >= clipEnd(clip) - 1e-6) return null;
  const offset = t - clip.start;
  const right = structuredClone(clip);
  right.id = uid('clip');
  right.start = t;
  right.dur = clip.dur - offset;
  clip.dur = offset;
  if (clip.kind === 'media') right.in = clip.in + offset;
  // Re-anchor the right half's keys to its new start, and advance any
  // oscillator driver's phase so the waveform is continuous across the cut
  // (audio drivers sample comp time and need no adjustment).
  const eachProp = (c, fn) => {
    const bag = c.kind === 'media' ? c.props : c.params;
    for (const p of Object.values(bag)) fn(p);
  };
  eachProp(right, (p) => {
    for (const k of p.keys) k.t -= offset;
    if (p.driver && p.driver.source !== 'audio')
      p.driver.phase = (p.driver.phase ?? 0) + offset * (p.driver.freq ?? 0);
  });
  return right;
}

/* ---- comp ----------------------------------------------------------- */

export function newTrack(name) {
  return { id: uid('trk'), name, clips: [] };
}

export function newComp({ width = 1280, height = 720, fps = 30, dur = 10 } = {}) {
  return { width, height, fps, dur, tracks: [] };
}

export function findClip(comp, clipId) {
  for (const track of comp.tracks)
    for (const clip of track.clips)
      if (clip.id === clipId) return { track, clip };
  return null;
}

export function trackOf(comp, clip) {
  return comp.tracks.find((t) => t.clips.includes(clip)) ?? null;
}

/** Active clips of `kind` at time t, bottom track first, then by start.
 * Returns [{track, clip}]. */
export function activeClips(comp, t, kind = null) {
  const out = [];
  for (let i = comp.tracks.length - 1; i >= 0; i--) {
    const track = comp.tracks[i];
    const hits = track.clips
      .filter((c) => (!kind || c.kind === kind) && t >= c.start && t < clipEnd(c) - 1e-9)
      .sort((a, b) => a.start - b.start);
    for (const clip of hits) out.push({ track, clip });
  }
  return out;
}

/** All clips, bottom track first (chain / draw order). */
export function allClipsBottomUp(comp, kind = null) {
  const out = [];
  for (let i = comp.tracks.length - 1; i >= 0; i--)
    for (const clip of comp.tracks[i].clips)
      if (!kind || clip.kind === kind) out.push(clip);
  return out;
}

/** Grow the comp so no clip hangs off the end. */
export function ensureDur(comp) {
  let end = 0;
  for (const track of comp.tracks)
    for (const clip of track.clips) end = Math.max(end, clipEnd(clip));
  if (end > comp.dur) comp.dur = Math.ceil(end * comp.fps) / comp.fps;
}

export function removeEmptyTracks(comp) {
  // Tracks only exist to hold clips — empty ones are unreachable UI.
  comp.tracks = comp.tracks.filter((t) => t.clips.length > 0);
}

export function quantize(t, fps) {
  return Math.round(t * fps) / fps;
}

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/* ---- undo history ---------------------------------------------------- */

/** Snapshot-based undo/redo over the comp JSON. Cheap because the model is
 * plain data; media blobs live outside the comp and are not copied. */
export class History {
  constructor(limit = 100) {
    this.undoStack = [];
    this.redoStack = [];
    this.limit = limit;
    this.pending = null;
  }

  /** Forget everything (e.g. when switching projects). */
  reset() {
    this.undoStack.length = 0;
    this.redoStack.length = 0;
    this.pending = null;
  }

  /** Capture the pre-change state; call before mutating. */
  begin(comp) {
    this.pending = JSON.stringify(comp);
  }

  /** Push the captured state (no-op if nothing changed). */
  commit(comp) {
    if (this.pending == null) return;
    if (this.pending !== JSON.stringify(comp)) {
      this.undoStack.push(this.pending);
      if (this.undoStack.length > this.limit) this.undoStack.shift();
      this.redoStack.length = 0;
    }
    this.pending = null;
  }

  /** begin+commit convenience for single-step mutations. */
  record(comp, fn) {
    this.begin(comp);
    fn();
    this.commit(comp);
  }

  undo(comp) {
    if (!this.undoStack.length) return null;
    this.redoStack.push(JSON.stringify(comp));
    return JSON.parse(this.undoStack.pop());
  }

  redo(comp) {
    if (!this.redoStack.length) return null;
    this.undoStack.push(JSON.stringify(comp));
    return JSON.parse(this.redoStack.pop());
  }
}
