/*
 * slangfx studio — web editor UI.
 *
 * After-Effects-style editor built on the lowkey-studio engine:
 *
 *   comp.js        composition model — tracks, clips, keyframes, undo
 *   compositor.js  WebGPU compositing of media clips into the frame
 *   timeline.js    the zoomable timeline / keyframe panel
 *   app.js (this)  playback clock + multi-video sync, audio mixing, fx
 *                  chain management, inspector panel, import/export,
 *                  project persistence
 *
 * Render path per frame:
 *   1. active media clips → sync video elements to the comp clock,
 *      upload current frames, composite into the engine input texture
 *      with each clip's animated transform
 *   2. active fx clips → the engine layer chain (rebuilt only when the
 *      active set changes); every shader param is driven from its
 *      keyframe track each frame
 *   3. fx.render() runs the shader chain and presents to the canvas
 */

import { SlangFx, loadToolchain, parsePreset, dirnameOf } from './engine/index.js';
import { renameReserved } from './engine/preprocess.js';
import {
  newComp, newTrack, newMediaClip, newFxClip, clipEnd, evalProp, upsertKey,
  keyNear, activeClips, findClip, ensureDur, removeEmptyTracks, quantize,
  clamp, History, uid, newProp, migrateComp, trackOf,
} from './comp.js';
import { Compositor } from './compositor.js';
import { Timeline, fmtTimecode, showMenu } from './timeline.js';
import { makeShaderEditor, CHEAT_HTML } from './shader-editor.js';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const canvas = $('preview');
const inspectorEl = $('inspector');
const addLayerSearch = $('add-layer-search');
const addLayerList = $('add-layer-list');

const VIDEO_EXTS = /\.(mp4|mov|mkv|webm|avi|m4v)$/i;
const GIF_EXT = /\.gif$/i;
const PROJECT_KEY = 'lowkey-studio.project.v2';
const DEFAULT_VIDEO_DUR = 4;   // fallback when a video reports no duration

function setStatus(msg) { statusEl.textContent = msg; }

/* =====================================================================
 * Custom (hand-written) shader plumbing — virtual files served to the
 * engine's readFile under a reserved prefix, one directory per fx clip.
 * =================================================================== */

const CUSTOM_PREFIX = 'custom/';
const virtualFiles = new Map();
let customCounter = 0;

const CUSTOM_PRESET = `shaders = 1
shader0        = custom.slang
filter_linear0 = true
scale_type0    = viewport
scale0         = 1.0
wrap_mode0     = clamp_to_edge
`;

const CUSTOM_BOILERPLATE = `#version 450

// Hand-written slang shader — edit and hit Compile.
//
// Declare a tunable with one line and it appears in the inspector AND
// as a keyframable property lane on the timeline:
//   //@param name "Label" default min max step
//
// Inputs the engine fills in every frame:
//   Source            everything composited below this clip
//   vTexCoord         0..1 UV, (0,0) = top-left
//   params.SourceSize (w, h, 1/w, 1/h) of Source
//   params.OutputSize (w, h, 1/w, 1/h) of this pass
//   params.FrameCount frame counter (uint)
//   params.Time       seconds (comp time)

layout(push_constant) uniform Push
{
    vec4  SourceSize;
    vec4  OutputSize;
    uint  FrameCount;
    float Time;
} params;

//@param amount "Mix" 1.0 0.0 1.0 0.01
//@param wobble "Wobble (px)" 6.0 0.0 64.0 0.5
//@param speed  "Speed" 1.5 0.0 8.0 0.05

layout(std140, set = 0, binding = 0) uniform UBO { mat4 MVP; } global;

#pragma stage vertex
layout(location = 0) in vec4 Position;
layout(location = 1) in vec2 TexCoord;
layout(location = 0) out vec2 vTexCoord;
void main() { gl_Position = global.MVP * Position; vTexCoord = TexCoord; }

#pragma stage fragment
layout(location = 0) in vec2 vTexCoord;
layout(location = 0) out vec4 FragColor;
layout(set = 0, binding = 2) uniform sampler2D Source;

void main()
{
    vec2 uv = vTexCoord;
    uv.x += sin(uv.y * 24.0 + params.Time * speed * 6.2832)
            * wobble * params.SourceSize.z;

    vec3 c = texture(Source, uv).rgb;
    c *= vec3(1.05, 0.95, 1.10);              // playground: make it yours

    vec3 base = texture(Source, vTexCoord).rgb;
    FragColor = vec4(mix(base, c, amount), 1.0);
}
`;

/* ---- saved shaders (localStorage) ---------------------------------- */

const SAVED_KEY = 'lowkey-studio.saved-shaders';

function loadSaved() {
  try { return JSON.parse(localStorage.getItem(SAVED_KEY)) ?? {}; }
  catch { return {}; }
}

function storeSaved(saves) {
  localStorage.setItem(SAVED_KEY, JSON.stringify(saves));
  if (addLayerList.classList.contains('open')) rebuildAddMenu();
}

/* =====================================================================
 * Global state
 * =================================================================== */

let fx = null;
let compositor = null;
let manifest = { categories: [], effects: [] };

let comp = newComp({ width: 1280, height: 720, fps: 30, dur: 12 });
comp._autoSize = true;
let projectName = null;
const history = new History();

const assets = new Map();        // assetId -> asset record
const fxSpecs = new Map();       // clipId -> engine layer spec (persistent)
const paramMetaCache = new Map();// clipId -> [{name, desc, min, max, step, default}]

let tCur = 0;
let playing = false;
let looping = true;
let clock = { perf: 0, t: 0 };
let timeline = null;

let trimPreviewT = null;   // render-time override while trimming a clip edge

let chainKey = '';
let chainDirty = false;
let chainBuilding = false;
let chainPromise = Promise.resolve();

let recorder = null;
let exportMode = false;

/* =====================================================================
 * Boot
 * =================================================================== */

async function boot() {
  if (!navigator.gpu) {
    setStatus('WebGPU is not available. Use Chrome/Edge 113+ (or enable WebGPU).');
    return;
  }
  // Shader paths ("shaders/x/y.slangp") resolve against the studio dir.
  const ROOT = new URL('./', location.href);
  const rootUrl = (p) => new URL(p.replace(/^\/+/, ''), ROOT);
  try {
    const toolchain = await loadToolchain();
    fx = await SlangFx.create({
      canvas,
      toolchain,
      readFile: async (p) => {
        const clean = p.replace(/^\/+/, '');
        if (virtualFiles.has(clean)) return virtualFiles.get(clean);
        const res = await fetch(rootUrl(p));
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${p}`);
        return res.text();
      },
      readImage: async (p) => {
        const res = await fetch(rootUrl(p));
        if (!res.ok) throw new Error(`HTTP ${res.status} for ${p}`);
        return createImageBitmap(await res.blob());
      },
    });
  } catch (e) {
    setStatus(`init failed: ${e.message}`);
    throw e;
  }
  compositor = new Compositor(fx.device);
  // Media stacked above an effect is layered onto that effect's output so
  // only effects higher in the stack process it (see compositeFrame).
  fx.onAfterLayer = (encoder, layer) => {
    const draws = fxOverlays.get(layer.clipId);
    if (!draws?.length) return;
    const view = layer.blendView ?? layer.runtime.finalPass.fboView;
    compositor.composite(encoder, view, comp.width, comp.height, draws, { over: true });
  };

  try {
    manifest = await (await fetch('effects.json')).json();
  } catch {
    manifest = { categories: [], effects: [] };
    setStatus('effects.json missing — run: node web/tools/build-manifest.mjs');
  }

  timeline = new Timeline($('timeline'), timelineHost);
  await restoreProject();
  await applyCompSize();
  document.body.classList.add('has-media');
  refreshDropHint();
  timeline.zoomFit();
  timeline.render();
  renderInspector();
  setStatus('ready — import media or add an effect');
  window.fx = fx;                 // console/debug access
  window.comp = () => comp;
  requestAnimationFrame(tick);
}

async function applyCompSize() {
  canvas.width = comp.width;
  canvas.height = comp.height;
  await fx.setSourceSize(comp.width, comp.height);
  chainDirty = true;
  rescaleMasks();
  applyViewSizing();
}

function refreshDropHint() {
  const empty = comp.tracks.every((t) => t.clips.length === 0);
  $('drop-hint').style.display = empty ? '' : 'none';
}

/* =====================================================================
 * Playback clock + per-frame render
 * =================================================================== */

function setTime(t) {
  tCur = clamp(quantize(t, comp.fps), 0, comp.dur);
  if (playing) clock = { perf: performance.now(), t: tCur };
  timeline?.updatePlayhead();
}

function togglePlay() { playing ? pause() : play(); }

function play() {
  ensureAudio();
  if (tCur >= comp.dur - 1e-6) tCur = 0;
  playing = true;
  clock = { perf: performance.now(), t: tCur };
}

function pause() {
  playing = false;
  tCur = quantize(tCur, comp.fps);
}

function tick() {
  requestAnimationFrame(tick);
  if (!fx?.inputTexture) return;

  if (playing) {
    tCur = clock.t + (performance.now() - clock.perf) / 1000;
    if (tCur >= comp.dur) {
      if (exportMode) {
        finishExport();
        tCur = comp.dur;
        pause();
      } else if (looping) {
        tCur = 0;
        clock = { perf: performance.now(), t: 0 };
      } else {
        tCur = comp.dur;
        pause();
      }
    }
  }

  // While a trim handle is being dragged, render the frame at the cut
  // point instead of the playhead (the playhead UI itself stays put).
  const t = trimPreviewT ?? tCur;
  const activeMedia = activeClips(comp, t, 'media').filter(({ track }) => !track.hidden);
  syncMedia(t, activeMedia);
  compositeFrame(t);
  syncFxChain(t);
  applyParams(t);
  fx.render(null, t);
  timeline.updatePlayhead();
  updateInspectorLive();
  updateGizmo();
}

/* ---- media sync ---------------------------------------------------- */

function syncMedia(t, activeMedia) {
  const used = new Set();
  for (const { track, clip } of activeMedia) {
    const asset = assets.get(clip.assetId);
    if (!asset?.ready) continue;
    used.add(asset.id);
    if (asset.kind === 'gif') {
      // A GIF is a pre-decoded frame strip, not a <video>: pick the loop
      // frame for the clip's local time, upload only when it changes.
      const src = clip.in + (t - clip.start);
      const len = asset.duration || 1;
      const local = ((src % len) + len) % len;
      let idx = asset.frames.findIndex((f) => local < f.start + f.dur);
      if (idx < 0) idx = asset.frames.length - 1;
      if (idx !== asset._frameIdx) {
        asset._frameIdx = idx;
        fx.device.queue.copyExternalImageToTexture(
          { source: asset.frames[idx].bitmap }, { texture: asset.texture }, [asset.w, asset.h]);
      }
      continue;
    }
    if (asset.kind !== 'video') continue;
    const el = asset.el;
    // Audio: master prefs × track mute × the clip's animated Volume.
    el.muted = !!audioState.muted || !!track.muted;
    const clipVol = clip.props.volume
      ? clamp(evalProp(clip.props.volume, t - clip.start) / 100, 0, 1) : 1;
    el.volume = clamp((audioState.volume ?? 1) * clipVol, 0, 1);
    // Source time, wrapped so clips longer than their source loop.
    const src = clip.in + (t - clip.start);
    const len = asset.duration ?? 0;
    const desired = len > 0.02 ? ((src % len) + len) % len : 0;
    if (playing) {
      if (el.paused) {
        el.currentTime = desired;
        el.play().catch(() => {});
      } else {
        // Drift correction, measured around the loop seam.
        let drift = Math.abs(el.currentTime - desired);
        if (len > 0.02) drift = Math.min(drift, len - drift);
        if (drift > 0.15) el.currentTime = desired;
      }
    } else {
      if (!el.paused) el.pause();
      if (!el.seeking && Math.abs(el.currentTime - desired) > 0.5 / comp.fps)
        el.currentTime = desired;
    }
    if (el.readyState >= 2) uploadVideoFrame(asset);
  }
  for (const asset of assets.values())
    if (asset.kind === 'video' && asset.ready && !used.has(asset.id) && !asset.el.paused)
      asset.el.pause();
}

/* Firefox's WebGPU rejects HTMLVideoElement as a copyExternalImageToTexture
 * source (TypeError; only bitmaps/canvases are accepted), so after the first
 * rejection all video uploads reroute through a 2D scratch canvas. */
let videoNeedsCanvasHop = false;

function uploadVideoFrame(asset) {
  if (!videoNeedsCanvasHop) {
    try {
      fx.device.queue.copyExternalImageToTexture(
        { source: asset.el }, { texture: asset.texture }, [asset.w, asset.h]);
      return;
    } catch (e) {
      if (!(e instanceof TypeError)) return;   // frame not ready
      videoNeedsCanvasHop = true;
    }
  }
  const c = (asset.scratch ??= new OffscreenCanvas(asset.w, asset.h));
  const ctx2d = (asset.scratchCtx ??= c.getContext('2d'));
  ctx2d.drawImage(asset.el, 0, 0, asset.w, asset.h);
  try {
    fx.device.queue.copyExternalImageToTexture(
      { source: c }, { texture: asset.texture }, [asset.w, asset.h]);
  } catch { /* frame not ready */ }
}

function drawForClip(clip, t) {
  const asset = assets.get(clip.assetId);
  if (!asset?.ready) return null;
  const tc = t - clip.start;
  return {
    clipId: clip.id,
    view: asset.view,
    w: asset.w,
    h: asset.h,
    x: evalProp(clip.props.x, tc),
    y: evalProp(clip.props.y, tc),
    // Negative scale mirrors the media on that axis.
    scaleX: evalProp(clip.props.scaleX, tc) / 100,
    scaleY: evalProp(clip.props.scaleY, tc) / 100,
    rot: evalProp(clip.props.rot, tc),
    opacity: clamp(evalProp(clip.props.opacity, tc) / 100, 0, 1),
  };
}

/* Layer-true rendering: walking the stack bottom → top, media below the
 * first effect goes into the chain input; media sitting ABOVE an effect
 * is composited onto that effect's output (via fx.onAfterLayer), so only
 * effects higher in the stack see it. An effect therefore affects exactly
 * the tracks beneath it — the After Effects adjustment-layer rule. */
let fxOverlays = new Map();   // fx clipId -> media draws layered above it

function compositeFrame(t) {
  const base = [];
  fxOverlays = new Map();
  let curFx = null;           // nearest effect below the media being placed
  for (const { track, clip } of activeClips(comp, t)) {
    if (track.hidden) continue;
    if (clip.kind === 'fx') {
      // Broken / still-compiling effects are skipped by the engine, so
      // media above them merges down to the previous working effect.
      if (clip.enabled !== false && fxSpecs.get(clip.id)?.runtime) curFx = clip.id;
      continue;
    }
    const d = drawForClip(clip, t);
    if (!d) continue;
    if (curFx) {
      if (!fxOverlays.has(curFx)) fxOverlays.set(curFx, []);
      fxOverlays.get(curFx).push(d);
    } else {
      base.push(d);
    }
  }
  const encoder = fx.device.createCommandEncoder();
  compositor.composite(encoder, fx.inputView, comp.width, comp.height, base);
  fx.device.queue.submit([encoder.finish()]);
}

/* ---- fx chain ------------------------------------------------------- */

function specFor(clip) {
  let spec = fxSpecs.get(clip.id);
  if (!spec) {
    spec = { clipId: clip.id, enabled: true, runtime: null, error: null, savedParams: null, label: clip.name };
    if (clip.fxKind === 'custom') {
      const dir = `${CUSTOM_PREFIX}${customCounter++}/`;
      virtualFiles.set(dir + 'custom.slangp', CUSTOM_PRESET);
      virtualFiles.set(dir + 'custom.slang', clip.source ?? CUSTOM_BOILERPLATE);
      spec.dir = dir;
      spec.path = dir + 'custom.slangp';
    } else {
      spec.path = clip.path;
    }
    fxSpecs.set(clip.id, spec);
    restoreSpecExtras(clip, spec);
  }
  spec.label = clip.name;
  return spec;
}

/** Rehydrate persisted mask / overlay textures onto a fresh spec. */
async function restoreSpecExtras(clip, spec) {
  let dirty = false;
  if (clip.overlay) {
    for (const [name, o] of Object.entries(clip.overlay)) {
      if (o.kind === 'text' && o.state?.text) {
        (spec.textureOverrides ??= {})[name] = renderTitleCanvas(o.state);
        dirty = true;
      } else if (o.kind === 'image' && o.dataURL) {
        const img = new Image();
        await new Promise((res) => { img.onload = res; img.onerror = res; img.src = o.dataURL; });
        if (img.width) {
          (spec.textureOverrides ??= {})[name] = await createImageBitmap(img);
          dirty = true;
        }
      }
    }
  }
  if (clip.mask?.dataURL) {
    const img = new Image();
    await new Promise((res) => { img.onload = res; img.onerror = res; img.src = clip.mask.dataURL; });
    const c = document.createElement('canvas');
    c.width = comp.width;
    c.height = comp.height;
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    spec.maskState = { source: c, opacity: clip.mask.opacity ?? 1, invert: !!clip.mask.invert };
    dirty = true;
  }
  if (dirty) chainDirty = true;
}

function activeFxEntries(t) {
  return activeClips(comp, t, 'fx')
    .filter(({ track, clip }) => !track.hidden && clip.enabled !== false);
}

function syncFxChain(t) {
  if (chainBuilding) return chainPromise;
  const entries = activeFxEntries(t);
  const key = entries.map((e) => e.clip.id).join('|');
  if (key === chainKey && !chainDirty) return chainPromise;
  chainKey = key;
  chainDirty = false;
  chainBuilding = true;
  fx.layers = entries.map(({ clip }) => specFor(clip));
  chainPromise = fx.rebuild()
    .catch((e) => console.error('slangfx: chain rebuild failed:', e))
    .finally(() => { chainBuilding = false; });
  return chainPromise;
}

function applyParams(t) {
  for (const layer of fx.layers) {
    const rt = layer.runtime;
    if (!rt) continue;
    const hit = findClip(comp, layer.clipId);
    if (!hit) continue;
    const tc = t - hit.clip.start;
    for (const meta of rt.paramMeta) {
      const prop = hit.clip.params[meta.name];
      let v = prop ? evalProp(prop, tc) : meta.default;
      if (meta.max > meta.min) v = clamp(v, meta.min, meta.max);
      rt.paramValues.set(meta.name, v);
    }
  }
}

function activeIndexOfClip(clipId) {
  return fx.layers.findIndex((l) => l.clipId === clipId && l.runtime);
}

/** Shader parameter metadata for a clip without needing an active chain —
 * parses the preset and compiles its modules (cached), so custom-shader
 * compile errors also surface here. */
async function ensureParamMeta(clip) {
  if (clip.kind !== 'fx') return null;
  const cached = paramMetaCache.get(clip.id);
  if (cached) return cached === 'loading' ? null : cached;
  paramMetaCache.set(clip.id, 'loading');
  const spec = specFor(clip);
  try {
    const text = await fx.readFile(spec.path);
    const preset = parsePreset(text, dirnameOf(spec.path));
    const seen = new Map();
    for (const pass of preset.passes) {
      const mod = await fx.compileModule(pass.path);
      for (const p of mod.params) if (!seen.has(p.name)) seen.set(p.name, { ...p });
    }
    for (const ov of preset.params) {
      const m = seen.get(renameReserved(ov.name));
      if (m) m.default = ov.value;
    }
    const metas = [...seen.values()];
    paramMetaCache.set(clip.id, metas);
    spec.lastCompileError = null;
    timeline.render();
    renderInspector();
    return metas;
  } catch (e) {
    paramMetaCache.delete(clip.id);
    spec.lastCompileError = String(e.message ?? e);
    renderInspector();
    throw e;
  }
}

/* =====================================================================
 * Property editing policy (shared by timeline + inspector)
 * =================================================================== */

function mediaPropDefs() {
  const W = comp.width, H = comp.height;
  return [
    { key: 'x', label: 'Position X', min: -W, max: 2 * W, step: 1, unit: 'px', def: W / 2 },
    { key: 'y', label: 'Position Y', min: -H, max: 2 * H, step: 1, unit: 'px', def: H / 2 },
    { key: 'scaleX', label: 'Scale X', min: -400, max: 400, step: 0.1, unit: '%', def: 100 },
    { key: 'scaleY', label: 'Scale Y', min: -400, max: 400, step: 0.1, unit: '%', def: 100 },
    { key: 'rot', label: 'Rotation', min: -360, max: 360, step: 0.1, unit: '°', def: 0 },
    { key: 'opacity', label: 'Opacity', min: 0, max: 100, step: 0.1, unit: '%', def: 100 },
    { key: 'volume', label: 'Volume', min: 0, max: 100, step: 0.1, unit: '%', def: 100 },
  ];
}

function propDefs(clip) {
  if (clip.kind === 'media') {
    const defs = mediaPropDefs();
    // Volume only makes sense (and sound) for video assets.
    return assets.get(clip.assetId)?.kind === 'video'
      ? defs
      : defs.filter((d) => d.key !== 'volume');
  }
  const metas = paramMetaCache.get(clip.id);
  if (!metas || metas === 'loading') {
    ensureParamMeta(clip).catch(() => {});
    return [];
  }
  return metas.map((m) => ({
    key: m.name,
    label: m.desc || m.name,
    min: m.min,
    max: m.max,
    step: m.step || 0.001,
    unit: '',
    def: m.default,
  }));
}

function defFor(clip, key) {
  return propDefs(clip).find((d) => d.key === key) ?? { def: 0, min: 0, max: 0, step: 0.001 };
}

function getProp(clip, key) {
  return clip.kind === 'media' ? clip.props[key] : clip.params[key] ?? null;
}

function getOrCreateProp(clip, key) {
  if (clip.kind === 'media') return clip.props[key];
  return (clip.params[key] ??= newProp(defFor(clip, key).def));
}

function relTime(clip) {
  return clamp(quantize(tCur - clip.start, comp.fps), 0, clip.dur);
}

function valueAt(clip, key) {
  const prop = getProp(clip, key);
  return prop ? evalProp(prop, tCur - clip.start) : defFor(clip, key).def;
}

/** Set a property value at the playhead: writes a keyframe when the
 * property is animated, otherwise the static value. `tRel` pins the
 * keyframe time — slider drags pass their drag-start time so a moving
 * playhead doesn't spray a key per input event. */
function setPropValueLive(clip, key, v, tRel = null) {
  const prop = getOrCreateProp(clip, key);
  if (prop.anim) upsertKey(prop, tRel ?? relTime(clip), v);
  else prop.v = v;
}

function setPropValue(clip, key, v) {
  history.record(comp, () => setPropValueLive(clip, key, v));
  onModelChange({ structural: false });
}

function toggleAnim(clip, key) {
  history.record(comp, () => {
    const prop = getOrCreateProp(clip, key);
    if (prop.anim) {
      prop.v = evalProp(prop, tCur - clip.start);   // freeze at current value
      prop.keys = [];
      prop.anim = false;
    } else {
      upsertKey(prop, relTime(clip), prop.v);
    }
  });
  onModelChange({ structural: false });
}

function toggleKey(clip, key) {
  history.record(comp, () => {
    const prop = getOrCreateProp(clip, key);
    const t = relTime(clip);
    const eps = 0.5 / comp.fps;
    const existing = keyNear(prop, t, eps);
    if (existing && prop.anim) {
      prop.keys.splice(prop.keys.indexOf(existing), 1);
      if (prop.keys.length === 0) prop.anim = false;
    } else {
      upsertKey(prop, t, evalProp(prop, tCur - clip.start));
    }
  });
  onModelChange({ structural: false });
}

/* ---- model change fan-out ------------------------------------------ */

function onModelChange({ structural = false, transient = false } = {}) {
  ensureDur(comp);
  chainDirty = true;
  if (transient) return;
  removeEmptyTracks(comp);   // e.g. the last clip was dragged off a track
  refreshDropHint();
  timeline.render();
  renderInspector();
  scheduleSave();
}

function appUndo() {
  const prev = history.undo(comp);
  if (prev) { comp = prev; afterModelReplace('undo'); }
  else setStatus('nothing to undo');
}

function appRedo() {
  const next = history.redo(comp);
  if (next) { comp = next; afterModelReplace('redo'); }
  else setStatus('nothing to redo');
}

function afterModelReplace(what) {
  stopMaskEdit();
  syncCustomSources();
  chainKey = '';
  chainDirty = true;
  tCur = clamp(tCur, 0, comp.dur);
  refreshDropHint();
  timeline.render();
  renderInspector();
  scheduleSave();
  setStatus(what);
}

/** After undo/redo the clip's source text may differ from the virtual
 * file that the engine compiles — resync + invalidate. */
function syncCustomSources() {
  for (const track of comp.tracks)
    for (const clip of track.clips) {
      if (clip.kind !== 'fx' || clip.fxKind !== 'custom') continue;
      const spec = fxSpecs.get(clip.id);
      if (!spec) continue;
      const cur = virtualFiles.get(spec.dir + 'custom.slang');
      if (cur !== clip.source) {
        virtualFiles.set(spec.dir + 'custom.slang', clip.source);
        fx.invalidateModules(spec.dir);
        paramMetaCache.delete(clip.id);
        chainDirty = true;
      }
    }
}

/* =====================================================================
 * Timeline host interface
 * =================================================================== */

const timelineHost = {
  comp: () => comp,
  time: () => tCur,
  setTime,
  playing: () => playing,
  togglePlay,
  looping: () => looping,
  toggleLoop: () => { looping = !looping; },
  history,
  assetOf: (id) => assets.get(id) ?? null,
  propList: (clip) => propDefs(clip),
  getProp,
  valueAt,
  setPropValue,
  toggleAnim,
  toggleKey,
  onModelChange,
  onSelect: () => renderInspector(),
  addMediaAt: (files, t, trackIdx) => importFiles(files, { t, trackIdx }),
  setTrimPreview: (t) => {
    trimPreviewT = t;
    const badge = $('trim-badge');
    if (badge) badge.hidden = t == null;
  },
  status: setStatus,
  undo: appUndo,
  redo: appRedo,
};

/* =====================================================================
 * Media import + assets
 * =================================================================== */

/** Decode every frame of an animated GIF up front (WebCodecs
 * ImageDecoder — Chromium-only, which WebGPU already requires).
 * Returns { frames: [{bitmap, start, dur}], dur } in seconds. */
async function decodeGifFrames(file) {
  const dec = new ImageDecoder({ data: await file.arrayBuffer(), type: 'image/gif' });
  await dec.tracks.ready;
  const count = dec.tracks.selectedTrack.frameCount;
  const frames = [];
  let t = 0;
  for (let i = 0; i < count; i++) {
    const { image } = await dec.decode({ frameIndex: i });
    // VideoFrame.duration is µs; renderers clamp near-zero GIF delays.
    const dur = Math.max((image.duration || 100_000) / 1e6, 0.02);
    frames.push({ bitmap: await createImageBitmap(image), start: t, dur });
    image.close();
    t += dur;
  }
  dec.close();
  return { frames, dur: t };
}

async function createAsset(file, id = null) {
  const isGif = file.type === 'image/gif' || GIF_EXT.test(file.name);
  const isVideo = !isGif && (file.type.startsWith('video/') || VIDEO_EXTS.test(file.name));
  const asset = {
    id: id ?? uid('asset'),
    kind: isVideo ? 'video' : isGif ? 'gif' : 'image',
    name: file.name,
    file,
    url: URL.createObjectURL(file),
    ready: false,
    w: 0, h: 0,
    duration: null,
    el: null,
    texture: null,
    view: null,
  };
  assets.set(asset.id, asset);

  if (isVideo) {
    const el = document.createElement('video');
    el.playsInline = true;
    el.preload = 'auto';
    el.loop = true;                 // clips longer than their source loop
    el.crossOrigin = 'anonymous';
    el.src = asset.url;
    $('media-pool').appendChild(el);
    asset.el = el;
    // A stalled decode (bad codec, corrupt file) must never wedge the app —
    // give metadata a generous window, then fail this asset and move on.
    await new Promise((res, rej) => {
      const timer = setTimeout(() => rej(new Error(`timed out opening ${file.name}`)), 12_000);
      el.onloadedmetadata = () => { clearTimeout(timer); res(); };
      el.onerror = () => { clearTimeout(timer); rej(new Error(`could not open ${file.name}`)); };
    }).catch((e) => {
      assets.delete(asset.id);
      el.remove();
      throw e;
    });
    asset.w = el.videoWidth;
    asset.h = el.videoHeight;
    asset.duration = Number.isFinite(el.duration) ? el.duration : null;
    applyAudioPrefsTo(el);
    attachAudio(asset);
  } else if (asset.kind === 'gif' && typeof ImageDecoder !== 'undefined') {
    let decoded;
    try {
      decoded = await decodeGifFrames(file);
    } catch (e) {
      assets.delete(asset.id);
      throw new Error(`could not decode ${file.name}`);
    }
    if (decoded.frames.length > 1) {
      asset.frames = decoded.frames;
      asset.duration = decoded.dur;
      asset._frameIdx = 0;
    } else {
      asset.kind = 'image';               // single-frame GIF is just a still
      asset.bitmap = decoded.frames[0].bitmap;
    }
    const first = decoded.frames[0].bitmap;
    asset.w = first.width;
    asset.h = first.height;
  } else {
    // Plain images — and GIFs on the off chance ImageDecoder is missing,
    // where createImageBitmap still yields the first frame as a still.
    if (asset.kind === 'gif') asset.kind = 'image';
    const bitmap = await createImageBitmap(file);
    asset.w = bitmap.width;
    asset.h = bitmap.height;
    asset.bitmap = bitmap;
  }

  asset.texture = fx.device.createTexture({
    label: `asset ${asset.name}`,
    size: [asset.w, asset.h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  asset.view = asset.texture.createView();
  const firstFrame = asset.bitmap ?? asset.frames?.[0].bitmap;
  if (firstFrame)
    fx.device.queue.copyExternalImageToTexture({ source: firstFrame }, { texture: asset.texture }, [asset.w, asset.h]);
  asset.ready = true;
  return asset;
}

async function importFiles(files, { t = null, trackIdx = null } = {}) {
  const media = [...files].filter((f) => f.type.startsWith('video/') || f.type.startsWith('image/') || VIDEO_EXTS.test(f.name) || GIF_EXT.test(f.name));
  if (!media.length) return;
  let at = t ?? tCur;
  history.begin(comp);
  for (const file of media) {
    let asset;
    setStatus(`importing ${file.name}…`);
    try {
      asset = await createAsset(file);
    } catch (e) {
      setStatus(`import failed: ${e.message}`);
      continue;
    }
    idbSet(`asset:${asset.id}`, file).catch(() => {});

    // The very first media item defines the comp size; anything after that
    // is scaled to fit inside the existing frame.
    const noMediaYet = !comp.tracks.some((tr) => tr.clips.some((c) => c.kind === 'media'));
    if (noMediaYet && asset.w) {
      comp.width = asset.w;
      comp.height = asset.h;
      await applyCompSize();
    }

    // Videos land at the playhead with their source length; images fill
    // the whole timeline by default (trim down as needed).
    let clip;
    if (asset.kind === 'video') {
      const dur = quantize(Math.max(asset.duration ?? DEFAULT_VIDEO_DUR, 1 / comp.fps), comp.fps);
      clip = newMediaClip(comp, asset, quantize(at, comp.fps), dur);
    } else {
      clip = newMediaClip(comp, asset, 0, Math.max(1 / comp.fps, comp.dur));
    }
    if (asset.w && (asset.w !== comp.width || asset.h !== comp.height)) {
      const fit = Math.round(Math.min(comp.width / asset.w, comp.height / asset.h) * 10000) / 100;
      clip.props.scaleX.v = fit;
      clip.props.scaleY.v = fit;
    }

    let track = trackIdx != null ? comp.tracks[trackIdx] : null;
    if (!track) {
      track = newTrack(clip.name);
      comp.tracks.unshift(track);
    }
    track.clips.push(clip);
    if (comp._autoSize) comp.dur = Math.max(comp.dur, clipEnd(clip));
    // Only videos advance the drop cursor (images span the full comp).
    if (asset.kind === 'video') at = clipEnd(clip);
    timeline.selectClip(clip.id);
  }
  ensureDur(comp);
  history.commit(comp);
  onModelChange({ structural: true });
  timeline.zoomFit();
  setStatus(`imported ${media.length} item${media.length > 1 ? 's' : ''}`);
}

$('file-input').addEventListener('change', (e) => {
  if (e.target.files.length) importFiles([...e.target.files]);
  e.target.value = '';
});

document.body.addEventListener('dragover', (e) => e.preventDefault());
document.body.addEventListener('drop', (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) importFiles([...e.dataTransfer.files]);
});

/* =====================================================================
 * Audio — one WebAudio mixer so playback and recordings hear every
 * active video clip.
 * =================================================================== */

const AUDIO_KEY = 'lowkey-studio.audio';
const muteBtn = $('btn-mute');
const volumeSlider = $('volume');
let audioCtx = null;
let masterGain = null;
let recordDest = null;

function audioPrefs() {
  try { return JSON.parse(localStorage.getItem(AUDIO_KEY)) ?? {}; }
  catch { return {}; }
}

/* Master audio state — combined per frame in syncMedia with each clip's
 * animated Volume and its track's mute flag. */
const audioState = { muted: audioPrefs().muted ?? false, volume: audioPrefs().volume ?? 1 };

function applyAudioPrefsTo(el) {
  el.volume = audioState.volume;
  el.muted = audioState.muted;
}

function updateAudioUI() {
  muteBtn.textContent = (audioState.muted || audioState.volume === 0) ? '🔇' : '🔊';
  volumeSlider.value = String(audioState.volume);
}

function setAudioPrefs({ muted, volume }) {
  if (muted != null) audioState.muted = muted;
  if (volume != null) audioState.volume = volume;
  try { localStorage.setItem(AUDIO_KEY, JSON.stringify(audioState)); } catch {}
  updateAudioUI();
}

function ensureAudio() {
  if (!audioCtx) {
    try {
      audioCtx = new AudioContext();
      masterGain = audioCtx.createGain();
      masterGain.connect(audioCtx.destination);
      recordDest = audioCtx.createMediaStreamDestination();
      masterGain.connect(recordDest);
      for (const a of assets.values()) attachAudio(a);
    } catch (e) {
      console.warn('slangfx: audio mixer unavailable:', e);
    }
  }
  audioCtx?.resume().catch(() => {});
}

function attachAudio(asset) {
  if (!audioCtx || asset.kind !== 'video' || asset.audioNode) return;
  try {
    asset.audioNode = audioCtx.createMediaElementSource(asset.el);
    asset.audioNode.connect(masterGain);
  } catch (e) {
    console.warn('slangfx: could not attach audio for', asset.name, e);
  }
}

muteBtn.addEventListener('click', () => {
  const muted = !audioState.muted;
  setAudioPrefs({ muted, volume: muted ? audioState.volume : (audioState.volume || 0.5) });
});

volumeSlider.addEventListener('input', () => {
  const v = parseFloat(volumeSlider.value);
  setAudioPrefs({ volume: v, muted: v === 0 ? undefined : false });
});
updateAudioUI();

/* =====================================================================
 * Viewer sizing / zoom / fullscreen
 * =================================================================== */

const viewer = $('viewer');
const canvasStack = $('canvas-stack');
const zoomReadout = $('zoom-readout');
const VIEWMODE_KEY = 'lowkey-studio.viewmode';

/* Free view — wheel zoom / pan layered over Fit. The frame gets an
 * explicit pixel size and offset inside the pane (no CSS transforms, so
 * every getBoundingClientRect-based mapping — gizmo, snapping, mask
 * painting — keeps working untouched). zoom multiplies the fit scale;
 * panX/panY offset the frame center from the pane center in screen px. */
let freeView = null;
const ZOOM_MIN = 0.05;
const ZOOM_MAX = 32;

function fitScale() {
  return Math.min(viewer.clientWidth / canvas.width, viewer.clientHeight / canvas.height) || 1;
}

function applyViewSizing() {
  if (freeView && !document.fullscreenElement) {
    viewer.classList.add('free-view');
    const s = fitScale() * freeView.zoom;
    const w = Math.max(1, canvas.width * s);
    const h = Math.max(1, canvas.height * s);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    canvas.style.maxWidth = '';
    canvas.style.maxHeight = '';
    canvasStack.style.left = `${(viewer.clientWidth - w) / 2 + freeView.panX}px`;
    canvasStack.style.top = `${(viewer.clientHeight - h) / 2 + freeView.panY}px`;
    zoomReadout.hidden = false;
    zoomReadout.textContent = `${Math.round(s * 100)}%`;
    return;
  }
  viewer.classList.remove('free-view');
  canvasStack.style.left = '';
  canvasStack.style.top = '';
  zoomReadout.hidden = true;
  if (viewer.classList.contains('size-fit') && !document.fullscreenElement) {
    // True fit: scale the canvas up OR down to fill the pane (contain).
    const scale = fitScale();
    if (Number.isFinite(scale) && scale > 0) {
      canvas.style.width = `${Math.max(1, Math.floor(canvas.width * scale))}px`;
      canvas.style.height = `${Math.max(1, Math.floor(canvas.height * scale))}px`;
    }
    canvas.style.maxWidth = '';
    canvas.style.maxHeight = '';
  } else {
    canvas.style.width = '';
    canvas.style.height = '';
    canvas.style.maxWidth = '';
    canvas.style.maxHeight = '';
  }
}

new ResizeObserver(applyViewSizing).observe(viewer);
document.addEventListener('fullscreenchange', applyViewSizing);

function setViewMode(mode) {
  freeView = null;
  viewer.className = `size-${mode}`;
  for (const b of document.querySelectorAll('#view-controls .btn[data-mode]'))
    b.classList.toggle('active', b.dataset.mode === mode);
  applyViewSizing();
  try { localStorage.setItem(VIEWMODE_KEY, mode); } catch {}
}

for (const b of document.querySelectorAll('#view-controls .btn[data-mode]'))
  b.addEventListener('click', () => setViewMode(b.dataset.mode));
zoomReadout.addEventListener('click', () => setViewMode('fit'));
setViewMode(localStorage.getItem(VIEWMODE_KEY) ?? 'fit');

/** Seed free view from wherever the frame currently sits on screen so the
 * first wheel tick / pan continues from the current framing (works from
 * Fit, Cover, and 1:1 alike). */
function enterFreeView() {
  if (freeView) return;
  const d = canvasDisplayRect();
  const vr = viewer.getBoundingClientRect();
  freeView = {
    zoom: (d.s / fitScale()) || 1,
    panX: d.left + comp.width * d.s / 2 - (vr.left + vr.width / 2),
    panY: d.top + comp.height * d.s / 2 - (vr.top + vr.height / 2),
  };
  viewer.className = 'size-fit free-view';
  for (const b of document.querySelectorAll('#view-controls .btn[data-mode]'))
    b.classList.remove('active');
  applyViewSizing();
}

/* Keep at least a sliver of the frame reachable — a wild fling can't lose
 * it off-pane entirely. */
function clampPan() {
  const s = fitScale() * freeView.zoom;
  const mx = (viewer.clientWidth + canvas.width * s) / 2 - 24;
  const my = (viewer.clientHeight + canvas.height * s) / 2 - 24;
  freeView.panX = clamp(freeView.panX, -mx, mx);
  freeView.panY = clamp(freeView.panY, -my, my);
}

viewer.addEventListener('wheel', (e) => {
  if (document.fullscreenElement || !document.body.classList.contains('has-media')) return;
  e.preventDefault();
  enterFreeView();
  const vr = viewer.getBoundingClientRect();
  const s0 = fitScale() * freeView.zoom;
  const dy = e.deltaY * (e.deltaMode === 1 ? 40 : 1);   // line-scroll mice
  const zoom = clamp(freeView.zoom * Math.exp(-dy * 0.0015), ZOOM_MIN, ZOOM_MAX);
  const s1 = fitScale() * zoom;
  // Anchor the comp point under the cursor while the scale changes.
  const cx = e.clientX - (vr.left + vr.width / 2);
  const cy = e.clientY - (vr.top + vr.height / 2);
  freeView.panX = cx - (cx - freeView.panX) * (s1 / s0);
  freeView.panY = cy - (cy - freeView.panY) * (s1 / s0);
  freeView.zoom = zoom;
  clampPan();
  applyViewSizing();
}, { passive: false });

let panState = null;
viewer.addEventListener('pointerdown', (e) => {
  if (e.target.closest('.btn')) return;
  if (viewer.classList.contains('size-actual')) {
    if (maskEdit) return;
    panState = { kind: 'scroll', x: e.clientX, y: e.clientY, sl: viewer.scrollLeft, st: viewer.scrollTop };
  } else {
    // Free-view pan: middle-drag anywhere, left-drag on the empty space
    // around the frame.
    if (document.fullscreenElement || !document.body.classList.contains('has-media')) return;
    const emptySpace = e.target === viewer || e.target === canvasStack;
    if (!(e.button === 1 || (e.button === 0 && emptySpace && !maskEdit))) return;
    e.preventDefault();                     // middle-click autoscroll
    enterFreeView();
    panState = { kind: 'free', x: e.clientX, y: e.clientY, px: freeView.panX, py: freeView.panY };
  }
  viewer.classList.add('panning');
  try { viewer.setPointerCapture(e.pointerId); } catch {}
});
viewer.addEventListener('pointermove', (e) => {
  if (!panState) return;
  if (panState.kind === 'scroll') {
    viewer.scrollLeft = panState.sl - (e.clientX - panState.x);
    viewer.scrollTop = panState.st - (e.clientY - panState.y);
  } else {
    freeView.panX = panState.px + (e.clientX - panState.x);
    freeView.panY = panState.py + (e.clientY - panState.y);
    clampPan();
    applyViewSizing();
  }
});
viewer.addEventListener('pointerup', () => { panState = null; viewer.classList.remove('panning'); });
viewer.addEventListener('pointercancel', () => { panState = null; viewer.classList.remove('panning'); });

function toggleFullscreen() {
  if (document.fullscreenElement) document.exitFullscreen();
  else canvasStack.requestFullscreen().catch(() => {});
}

$('btn-fullscreen').addEventListener('click', toggleFullscreen);
canvasStack.addEventListener('dblclick', () => { if (!maskEdit) toggleFullscreen(); });

/* =====================================================================
 * Timeline panel resize
 * =================================================================== */

const TL_HEIGHT_KEY = 'lowkey-studio.tl-height';
const timelineEl = $('timeline');
timelineEl.style.height = `${clamp(parseInt(localStorage.getItem(TL_HEIGHT_KEY) ?? '240', 10) || 240, 120, 600)}px`;

$('tl-resize').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  const startY = e.clientY;
  const startH = timelineEl.clientHeight;
  const onMove = (ev) => {
    const h = clamp(startH + (startY - ev.clientY), 120, Math.max(140, innerHeight - 220));
    timelineEl.style.height = `${h}px`;
  };
  const onUp = () => {
    window.removeEventListener('pointermove', onMove);
    window.removeEventListener('pointerup', onUp);
    try { localStorage.setItem(TL_HEIGHT_KEY, String(timelineEl.clientHeight)); } catch {}
    timeline?.render();
  };
  window.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
});

/* =====================================================================
 * Comp settings modal
 * =================================================================== */

/* Common comp sizes shown as clickable device cards in the settings modal.
 * Each card's silhouette is drawn at the true aspect ratio. */
const SIZE_PRESETS = [
  { name: 'HD 1080p', w: 1920, h: 1080, kind: 'tv', ratio: '16:9' },
  { name: '4K UHD', w: 3840, h: 2160, kind: 'tv', ratio: '16:9' },
  { name: 'HD 720p', w: 1280, h: 720, kind: 'monitor', ratio: '16:9' },
  { name: 'Ultrawide', w: 2560, h: 1080, kind: 'monitor', ratio: '21:9' },
  { name: 'Square', w: 1080, h: 1080, kind: 'square', ratio: '1:1' },
  { name: 'Phone', w: 1080, h: 1920, kind: 'phone', ratio: '9:16' },
];

$('btn-settings').addEventListener('click', () => {
  const old = document.querySelector('.modal-wrap');
  if (old) { old.remove(); return; }
  const wrap = document.createElement('div');
  wrap.className = 'modal-wrap';
  const presetCards = SIZE_PRESETS.map((p) => {
    const scale = Math.min(60 / p.w, 38 / p.h);
    const bw = Math.max(10, Math.round(p.w * scale));
    const bh = Math.max(10, Math.round(p.h * scale));
    return `<button type="button" class="size-preset" data-w="${p.w}" data-h="${p.h}">
      <span class="sp-slot"><span class="sp-device ${p.kind}" style="width:${bw}px;height:${bh}px"></span></span>
      <span class="sp-name">${p.name}</span>
      <span class="sp-dims">${p.w}×${p.h} · ${p.ratio}</span>
    </button>`;
  }).join('');
  wrap.innerHTML = `
    <div class="modal">
      <h3>Composition settings</h3>
      <div class="size-presets">${presetCards}</div>
      <label>Width <input id="cs-w" type="number" min="16" max="7680" step="2" value="${comp.width}"></label>
      <label>Height <input id="cs-h" type="number" min="16" max="4320" step="2" value="${comp.height}"></label>
      <label>Frame rate <select id="cs-fps">
        ${[24, 25, 30, 48, 50, 60].map((f) => `<option value="${f}" ${f === comp.fps ? 'selected' : ''}>${f} fps</option>`).join('')}
      </select></label>
      <label>Duration (s) <input id="cs-dur" type="number" min="0.5" max="7200" step="0.5" value="${comp.dur}"></label>
      <div class="modal-actions">
        <button class="btn" id="cs-match">Match first media</button>
        <span style="flex:1"></span>
        <button class="btn" id="cs-cancel">Cancel</button>
        <button class="btn" id="cs-apply">Apply</button>
      </div>
    </div>`;
  document.body.appendChild(wrap);
  wrap.addEventListener('pointerdown', (e) => { if (e.target === wrap) wrap.remove(); });
  wrap.querySelector('#cs-cancel').addEventListener('click', () => wrap.remove());

  const wInput = wrap.querySelector('#cs-w');
  const hInput = wrap.querySelector('#cs-h');
  const syncActivePreset = () => {
    for (const card of wrap.querySelectorAll('.size-preset'))
      card.classList.toggle('active',
        card.dataset.w === wInput.value && card.dataset.h === hInput.value);
  };
  for (const card of wrap.querySelectorAll('.size-preset'))
    card.addEventListener('click', () => {
      wInput.value = card.dataset.w;
      hInput.value = card.dataset.h;
      syncActivePreset();
    });
  wInput.addEventListener('input', syncActivePreset);
  hInput.addEventListener('input', syncActivePreset);
  wrap.querySelector('#cs-match').addEventListener('click', syncActivePreset);
  syncActivePreset();
  wrap.querySelector('#cs-match').addEventListener('click', () => {
    const first = [...assets.values()].find((a) => a.ready);
    if (!first) return;
    wrap.querySelector('#cs-w').value = first.w;
    wrap.querySelector('#cs-h').value = first.h;
  });
  wrap.querySelector('#cs-apply').addEventListener('click', async () => {
    const w = clamp(parseInt(wrap.querySelector('#cs-w').value, 10) || comp.width, 16, 7680);
    const h = clamp(parseInt(wrap.querySelector('#cs-h').value, 10) || comp.height, 16, 4320);
    const fps = parseInt(wrap.querySelector('#cs-fps').value, 10) || comp.fps;
    const dur = clamp(parseFloat(wrap.querySelector('#cs-dur').value) || comp.dur, 0.5, 7200);
    wrap.remove();
    const sizeChanged = w !== comp.width || h !== comp.height;
    history.record(comp, () => {
      comp.width = w; comp.height = h; comp.fps = fps; comp.dur = dur;
      comp._autoSize = false;
      ensureDur(comp);
    });
    if (sizeChanged) await applyCompSize();
    setTime(tCur);
    onModelChange({ structural: true });
    setStatus(`comp: ${w}×${h} @ ${fps} fps, ${comp.dur}s`);
  });
});

/* =====================================================================
 * Project persistence — comp JSON in localStorage, media blobs in idb.
 * =================================================================== */

function idbOpen() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open('lowkey-studio', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('media');
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}

async function idbSet(key, val) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('media', 'readwrite');
    tx.objectStore('media').put(val, key);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function idbGet(key) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const rq = db.transaction('media', 'readonly').objectStore('media').get(key);
    rq.onsuccess = () => resolve(rq.result);
    rq.onerror = () => reject(rq.error);
  });
}

async function idbDelete(key) {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction('media', 'readwrite');
    tx.objectStore('media').delete(key);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

let saveTimer = null;
function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveProject, 700);
}

function projectPayload() {
  // Sync live mask canvases into the model before serializing.
  for (const track of comp.tracks)
    for (const clip of track.clips) {
      if (clip.kind !== 'fx') continue;
      const spec = fxSpecs.get(clip.id);
      if (spec?.maskState) {
        const dataURL = spec.maskState.source.toDataURL('image/png');
        clip.mask = dataURL.length > 2_000_000
          ? null
          : { dataURL, opacity: spec.maskState.opacity ?? 1, invert: !!spec.maskState.invert };
      }
    }
  const assetMeta = [...assets.values()].map((a) => ({ id: a.id, name: a.name, kind: a.kind }));
  return { comp, assets: assetMeta, t: tCur, name: projectName };
}

function saveProject() {
  try {
    localStorage.setItem(PROJECT_KEY, JSON.stringify(projectPayload()));
  } catch (e) {
    console.warn('slangfx: project save failed (quota?):', e);
  }
}

/** Drop every runtime handle for the current project's media assets. */
function unloadAssets() {
  for (const a of assets.values()) {
    if (a.el) { a.el.pause(); a.el.remove(); }
    for (const f of a.frames ?? []) f.bitmap.close();
    a.texture?.destroy();
    try { URL.revokeObjectURL(a.url); } catch {}
  }
  assets.clear();
}

/** Make `data` ({comp, assets, t, name}) the current project. */
async function applyProjectData(data) {
  stopMaskEdit();
  document.getElementById('demo-card')?.remove();
  unloadAssets();
  fxSpecs.clear();
  paramMetaCache.clear();
  comp = migrateComp(data.comp);
  removeEmptyTracks(comp);
  projectName = data.name ?? null;
  tCur = clamp(data.t ?? 0, 0, comp.dur);
  history.reset();
  chainKey = '';
  chainDirty = true;
  fx.layers = [];

  const ids = new Set();
  for (const track of comp.tracks)
    for (const clip of track.clips)
      if (clip.kind === 'media') ids.add(clip.assetId);
  // Restore assets in parallel; a missing or unloadable one must not block
  // the app — its clips simply render as offline until re-imported.
  await Promise.allSettled((data.assets ?? [])
    .filter((meta) => ids.has(meta.id))
    .map(async (meta) => {
      const file = await idbGet(`asset:${meta.id}`);
      if (!file) throw new Error(`missing media blob for ${meta.name}`);
      await createAsset(file, meta.id);
    })).then((results) => {
      const failed = results.filter((r) => r.status === 'rejected');
      for (const f of failed) console.warn('slangfx: asset restore failed:', f.reason);
      if (failed.length) setStatus(`${failed.length} media item(s) could not be restored — re-import them`);
    });

  await applyCompSize();
  setTime(tCur);
  refreshDropHint();
  updateProjectButton();
  timeline?.zoomFit();
  timeline?.render();
  renderInspector();
}

async function restoreProject() {
  let data = null;
  try { data = JSON.parse(localStorage.getItem(PROJECT_KEY)); } catch {}
  if (data?.comp) {
    setStatus('restoring project…');
    await applyProjectData(data);
  } else {
    await loadDemoProject();
  }
  if (comp._demo) showDemoCard();
}

/* =====================================================================
 * Onboarding — first boot (no saved project) lands in a live demo comp:
 * an image clip with a keyframed slow zoom and a CRT effect over it, so
 * the first thing a new user sees is media + effects + keyframes already
 * working. A card offers the jump to a fresh project.
 * =================================================================== */

const DEMO_IMAGE = 'demo/seagull.jpg';   // bundled, pre-sized to 1280×720

async function loadDemoProject() {
  let asset;
  try {
    const res = await fetch(DEMO_IMAGE);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const file = new File([await res.blob()], 'seagull.jpg', { type: 'image/jpeg' });
    asset = await createAsset(file);
    idbSet(`asset:${asset.id}`, file).catch(() => {});
  } catch (e) {
    // No demo asset (or it failed to decode) — boot into the empty comp.
    console.warn('slangfx: demo project unavailable:', e);
    return;
  }
  setStatus('loading demo project…');
  comp.width = asset.w;
  comp.height = asset.h;
  comp.dur = 12;

  const media = newMediaClip(comp, asset, 0, comp.dur);
  // Slow push-in — keyframes visible on the timeline out of the box.
  for (const k of ['scaleX', 'scaleY']) {
    upsertKey(media.props[k], 0, 100);
    upsertKey(media.props[k], comp.dur, 108);
  }
  const mediaTrack = newTrack(media.name);
  mediaTrack.clips.push(media);

  const fxClip = newFxClip(
    { fxKind: 'preset', path: 'shaders/crt/crt-tv/crt-tv.slangp', label: 'crt-tv' },
    0, comp.dur);
  const fxTrack = newTrack(fxClip.name);
  fxTrack.clips.push(fxClip);

  comp.tracks = [fxTrack, mediaTrack];   // fx above the media it styles
  comp._demo = true;
  scheduleSave();
}

function showDemoCard() {
  if (document.getElementById('demo-card')) return;
  const card = document.createElement('div');
  card.id = 'demo-card';
  card.innerHTML = `
    <h3>👋 Welcome to Lowkey Studio</h3>
    <p>This demo comp is live: an image clip with a slow keyframed zoom and
    a CRT effect layered over it. Press <b>Space</b> to play, click a clip
    to tweak it in the inspector — or start clean.</p>
    <div class="demo-actions">
      <button class="btn primary" id="demo-new">Start a new project</button>
      <button class="btn" id="demo-keep">Explore the demo</button>
    </div>`;
  $('preview-wrap').appendChild(card);
  card.querySelector('#demo-new').addEventListener('click', async () => {
    card.remove();
    await newProject();
  });
  card.querySelector('#demo-keep').addEventListener('click', () => {
    delete comp._demo;      // card stays dismissed on future reloads
    scheduleSave();
    card.remove();
  });
}

/* =====================================================================
 * Project management — named projects live in IndexedDB (media blobs are
 * already there, shared across projects by asset id); a recents index
 * lives in localStorage. The autosave slot keeps working as the "current
 * session" and restores on reload.
 * =================================================================== */

const PROJECT_INDEX_KEY = 'lowkey-studio.projects.index';

function projectIndex() {
  try { return JSON.parse(localStorage.getItem(PROJECT_INDEX_KEY)) ?? []; }
  catch { return []; }
}

function updateProjectButton() {
  $('btn-project').textContent = `☰ ${projectName ?? 'untitled'}`;
  document.title = `${projectName ?? 'untitled'} — Lowkey Studio`;
}

function relTimeLabel(ts) {
  const d = Date.now() - ts;
  if (d < 60_000) return 'just now';
  if (d < 3_600_000) return `${Math.round(d / 60_000)}m ago`;
  if (d < 86_400_000) return `${Math.round(d / 3_600_000)}h ago`;
  return new Date(ts).toLocaleDateString();
}

async function saveNamedProject(name, { silent = false } = {}) {
  if (comp._demo) {          // explicitly saved → it's the user's project now
    delete comp._demo;
    document.getElementById('demo-card')?.remove();
  }
  projectName = name;
  saveProject();                       // autosave slot follows the name too
  await idbSet(`project:${name}`, JSON.stringify(projectPayload()));
  const idx = projectIndex().filter((p) => p.name !== name);
  idx.unshift({ name, savedAt: Date.now() });
  try { localStorage.setItem(PROJECT_INDEX_KEY, JSON.stringify(idx.slice(0, 20))); } catch {}
  updateProjectButton();
  if (!silent) setStatus(`saved project '${name}'`);
}

/** Never lose work: before switching away, silently save the current comp
 * (auto-naming it if it was never saved). The onboarding demo is not the
 * user's work — leaving it must not clutter the projects list. */
async function stashCurrent() {
  if (comp._demo) return;
  if (!comp.tracks.some((t) => t.clips.length)) return;
  const name = projectName ?? `Untitled ${new Date().toLocaleString()}`;
  await saveNamedProject(name, { silent: true });
}

async function openProject(name) {
  const raw = await idbGet(`project:${name}`);
  if (!raw) { setStatus(`project '${name}' not found`); return; }
  await stashCurrent();
  setStatus(`opening '${name}'…`);
  const data = typeof raw === 'string' ? JSON.parse(raw) : raw;
  data.name = name;
  await applyProjectData(data);
  scheduleSave();
  setStatus(`opened '${name}'`);
}

async function newProject() {
  await stashCurrent();
  const fresh = newComp({ width: 1280, height: 720, fps: 30, dur: 12 });
  fresh._autoSize = true;
  await applyProjectData({ comp: fresh, assets: [], t: 0, name: null });
  scheduleSave();
  setStatus('new project');
}

/** Small name dialog (no window.prompt — it blocks the page). */
function promptName(defaultVal = '') {
  return new Promise((resolve) => {
    const wrap = document.createElement('div');
    wrap.className = 'modal-wrap';
    wrap.innerHTML = `
      <div class="modal">
        <h3>Save project</h3>
        <label>Name <input id="pn-name" type="text" spellcheck="false"></label>
        <div class="modal-actions">
          <span style="flex:1"></span>
          <button class="btn" id="pn-cancel">Cancel</button>
          <button class="btn" id="pn-save">Save</button>
        </div>
      </div>`;
    document.body.appendChild(wrap);
    const input = wrap.querySelector('#pn-name');
    input.value = defaultVal;
    const done = (v) => { wrap.remove(); resolve(v); };
    wrap.addEventListener('pointerdown', (e) => { if (e.target === wrap) done(null); });
    wrap.querySelector('#pn-cancel').addEventListener('click', () => done(null));
    wrap.querySelector('#pn-save').addEventListener('click', () => done(input.value.trim() || null));
    input.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') done(input.value.trim() || null);
      if (e.key === 'Escape') done(null);
    });
    input.focus();
    input.select();
  });
}

/** In-app confirmation dialog (window.confirm blocks the page). */
function confirmDialog({ title = 'Are you sure?', message = '', confirmLabel = 'Delete' } = {}) {
  return new Promise((resolve) => {
    const wrap = document.createElement('div');
    wrap.className = 'modal-wrap';
    wrap.innerHTML = `
      <div class="modal">
        <h3></h3>
        <p class="confirm-msg"></p>
        <div class="modal-actions">
          <span style="flex:1"></span>
          <button class="btn" data-a="cancel">Cancel</button>
          <button class="btn btn-danger" data-a="ok"></button>
        </div>
      </div>`;
    wrap.querySelector('h3').textContent = title;
    wrap.querySelector('.confirm-msg').textContent = message;
    wrap.querySelector('[data-a=ok]').textContent = confirmLabel;
    const done = (v) => {
      wrap.remove();
      document.removeEventListener('keydown', onKey);
      resolve(v);
    };
    const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); done(false); } };
    document.addEventListener('keydown', onKey);
    wrap.addEventListener('pointerdown', (e) => { if (e.target === wrap) done(false); });
    wrap.querySelector('[data-a=cancel]').addEventListener('click', () => done(false));
    wrap.querySelector('[data-a=ok]').addEventListener('click', () => done(true));
    document.body.appendChild(wrap);
  });
}

async function deleteProject(name) {
  const ok = await confirmDialog({
    title: `Delete project '${name}'?`,
    message: 'The saved comp (clips, keyframes, masks, custom shaders) is removed from this browser permanently. Imported media files stay cached for other projects. If the project is open right now it stays open, just unsaved.',
    confirmLabel: 'Delete project',
  });
  if (!ok) return;
  try { await idbDelete(`project:${name}`); } catch {}
  const idx = projectIndex().filter((p) => p.name !== name);
  try { localStorage.setItem(PROJECT_INDEX_KEY, JSON.stringify(idx)); } catch {}
  setStatus(`deleted project '${name}'`);
}

async function saveFlow(alwaysAsk) {
  let name = projectName;
  if (alwaysAsk || !name) name = await promptName(name ?? '');
  if (!name) return;
  await saveNamedProject(name);
}

$('btn-project').addEventListener('click', (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const items = [
    { label: '✚ New project', action: () => newProject() },
    { label: projectName ? `Save '${projectName}'` : 'Save…', action: () => saveFlow(false) },
    { label: 'Save as…', action: () => saveFlow(true) },
  ];
  const idx = projectIndex();
  if (idx.length) {
    items.push('-');
    for (const p of idx.slice(0, 8))
      items.push({
        label: `🗂 ${p.name} · ${relTimeLabel(p.savedAt)}`,
        checked: p.name === projectName,
        action: () => openProject(p.name),
      });
    items.push('-');
    items.push({
      label: '🗑 Delete a project…',
      danger: true,
      action: () => {
        const rows = projectIndex().map((p) => ({
          label: `🗑 ${p.name}`,
          danger: true,
          action: () => deleteProject(p.name),
        }));
        showMenu(r.left, r.bottom + 4, rows);
      },
    });
  }
  showMenu(r.left, r.bottom + 4, items);
});

/* =====================================================================
 * Mask painting (per fx clip) — ported from the live demo.
 * =================================================================== */

const maskOverlay = $('mask-overlay');
const brush = { size: 60, soft: 0.5, mode: 'hide', tool: 'brush' };
let maskEdit = null;   // { clipId } | null
let gradState = null;

function makeMaskCanvas() {
  const c = document.createElement('canvas');
  c.width = comp.width;
  c.height = comp.height;
  const ctx = c.getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, c.width, c.height);
  return c;
}

function rebuildRuby(maskCanvas) {
  maskOverlay.width = maskCanvas.width;
  maskOverlay.height = maskCanvas.height;
  const rctx = maskOverlay.getContext('2d');
  const mctx = maskCanvas.getContext('2d');
  const img = mctx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  const out = rctx.createImageData(img.width, img.height);
  for (let i = 0; i < img.data.length; i += 4) {
    out.data[i] = 255;
    out.data[i + 3] = 255 - img.data[i];
  }
  rctx.putImageData(out, 0, 0);
}

function stampBrush(ctx, x, y, erase) {
  const r = Math.max(brush.size / 2, 1);
  const g = ctx.createRadialGradient(x, y, r * (1 - brush.soft), x, y, r);
  ctx.save();
  if (erase) {
    ctx.globalCompositeOperation = 'destination-out';
    g.addColorStop(0, 'rgba(255,255,255,1)');
    g.addColorStop(1, 'rgba(255,255,255,0)');
  } else {
    g.addColorStop(0, ctx._stampColor + '1)');
    g.addColorStop(1, ctx._stampColor + '0)');
  }
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function maskSpec() {
  return maskEdit ? fxSpecs.get(maskEdit.clipId) : null;
}

function pushMaskToGpu() {
  const idx = maskEdit ? activeIndexOfClip(maskEdit.clipId) : -1;
  if (idx >= 0) fx.updateLayerMask(idx);
}

function maskStroke(x, y) {
  const spec = maskSpec();
  if (!spec?.maskState) return;
  const mctx = spec.maskState.source.getContext('2d');
  const rctx = maskOverlay.getContext('2d');
  if (brush.mode === 'hide') {
    mctx._stampColor = 'rgba(0,0,0,';
    stampBrush(mctx, x, y, false);
    rctx._stampColor = 'rgba(255,0,0,';
    stampBrush(rctx, x, y, false);
  } else {
    mctx._stampColor = 'rgba(255,255,255,';
    stampBrush(mctx, x, y, false);
    stampBrush(rctx, x, y, true);
  }
  pushMaskToGpu();
}

function applyGradient(from, to) {
  const spec = maskSpec();
  if (!spec?.maskState) return;
  if (Math.hypot(to[0] - from[0], to[1] - from[1]) < 2) return;
  const src = spec.maskState.source;
  const hide = brush.mode === 'hide';

  const ramp = (ctx, c0, c1) => {
    let g;
    if (brush.tool === 'radial') {
      const r = Math.max(Math.hypot(to[0] - from[0], to[1] - from[1]), 2);
      g = ctx.createRadialGradient(from[0], from[1], 0, from[0], from[1], r);
    } else {
      g = ctx.createLinearGradient(from[0], from[1], to[0], to[1]);
    }
    g.addColorStop(0, c0);
    g.addColorStop(1, c1);
    return g;
  };

  const mctx = src.getContext('2d');
  mctx.save();
  mctx.globalCompositeOperation = 'source-over';
  mctx.fillStyle = ramp(mctx, hide ? '#000' : '#fff', hide ? '#fff' : '#000');
  mctx.fillRect(0, 0, src.width, src.height);
  mctx.restore();

  const rctx = maskOverlay.getContext('2d');
  rctx.save();
  rctx.globalCompositeOperation = 'source-over';
  rctx.clearRect(0, 0, maskOverlay.width, maskOverlay.height);
  rctx.fillStyle = ramp(rctx,
    hide ? 'rgba(255,0,0,1)' : 'rgba(255,0,0,0)',
    hide ? 'rgba(255,0,0,0)' : 'rgba(255,0,0,1)');
  rctx.fillRect(0, 0, maskOverlay.width, maskOverlay.height);
  rctx.restore();

  pushMaskToGpu();
}

function overlayToMedia(e) {
  const rect = maskOverlay.getBoundingClientRect();
  return [
    (e.clientX - rect.left) / rect.width * comp.width,
    (e.clientY - rect.top) / rect.height * comp.height,
  ];
}

let painting = false;
let lastPt = null;

maskOverlay.addEventListener('pointerdown', (e) => {
  if (!maskEdit || e.button !== 0) return;
  try { maskOverlay.setPointerCapture(e.pointerId); } catch {}
  if (brush.tool === 'brush') {
    painting = true;
    lastPt = overlayToMedia(e);
    maskStroke(...lastPt);
  } else {
    gradState = { from: overlayToMedia(e) };
  }
});
maskOverlay.addEventListener('pointermove', (e) => {
  if (!maskEdit) return;
  if (gradState) {
    applyGradient(gradState.from, overlayToMedia(e));
    return;
  }
  if (!painting) return;
  const pt = overlayToMedia(e);
  const step = Math.max(brush.size / 4, 2);
  const d = Math.hypot(pt[0] - lastPt[0], pt[1] - lastPt[1]);
  const n = Math.max(1, Math.ceil(d / step));
  for (let i = 1; i <= n; i++)
    maskStroke(lastPt[0] + (pt[0] - lastPt[0]) * i / n, lastPt[1] + (pt[1] - lastPt[1]) * i / n);
  lastPt = pt;
});
maskOverlay.addEventListener('pointerup', (e) => {
  if (gradState && maskEdit) applyGradient(gradState.from, overlayToMedia(e));
  painting = false;
  lastPt = null;
  gradState = null;
  scheduleSave();
});

async function startMaskEdit(clip) {
  const idx = activeIndexOfClip(clip.id);
  if (idx < 0) {
    setStatus('move the playhead over this clip to edit its mask');
    return;
  }
  if (viewer.classList.contains('size-cover')) setViewMode('fit');
  const spec = specFor(clip);
  if (!spec.maskState) {
    spec.maskState = { source: makeMaskCanvas(), opacity: 1, invert: false };
    chainDirty = true;
    await syncFxChain(tCur);
  }
  maskEdit = { clipId: clip.id };
  rebuildRuby(spec.maskState.source);
  document.body.classList.add('mask-editing');
  setStatus('painting mask — red = effect hidden');
  renderInspector();
}

function stopMaskEdit() {
  maskEdit = null;
  painting = false;
  document.body.classList.remove('mask-editing');
  renderInspector();
}

function rescaleMasks() {
  for (const spec of fxSpecs.values()) {
    const src = spec.maskState?.source;
    if (!src || (src.width === comp.width && src.height === comp.height)) continue;
    const scaled = document.createElement('canvas');
    scaled.width = comp.width;
    scaled.height = comp.height;
    scaled.getContext('2d').drawImage(src, 0, 0, scaled.width, scaled.height);
    spec.maskState.source = scaled;
  }
}

/* =====================================================================
 * Transform gizmo — select a media clip, then drag it directly in the
 * player: body = move, edge handles = scale one axis, corner handles =
 * scale both (Shift = uniform). Writes keyframes at the playhead when a
 * property is animated (same policy as the sliders).
 * =================================================================== */

const gizmo = document.createElement('div');
gizmo.id = 'gizmo';
gizmo.hidden = true;
for (const h of ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w']) {
  const el = document.createElement('div');
  el.className = `gz-h gz-${h}`;
  el.dataset.h = h;
  gizmo.appendChild(el);
}
$('canvas-inner').appendChild(gizmo);

function gizmoTarget() {
  const clip = timeline?.selectedClip;
  if (!clip || clip.kind !== 'media' || maskEdit) return null;
  if (trackOf(comp, clip)?.hidden) return null;
  const asset = assets.get(clip.assetId);
  if (!asset?.ready) return null;
  if (tCur < clip.start || tCur >= clipEnd(clip)) return null;
  return { clip, asset };
}

/** Where the comp frame actually lands inside the canvas element.
 * Fit / 1:1 / fullscreen draw object-fit:contain; Cover crops (max). */
function canvasDisplayRect() {
  const r = canvas.getBoundingClientRect();
  const cover = viewer.classList.contains('size-cover') && !document.fullscreenElement;
  const s = (cover
    ? Math.max(r.width / comp.width, r.height / comp.height)
    : Math.min(r.width / comp.width, r.height / comp.height)) || 1;
  const w = comp.width * s;
  const h = comp.height * s;
  return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, s };
}

function updateGizmo() {
  const tgt = !gzDrag ? gizmoTarget() : { clip: gzDrag.clip, asset: assets.get(gzDrag.clip.assetId) };
  if (!tgt) { gizmo.hidden = true; return; }
  const { clip, asset } = tgt;
  const tc = tCur - clip.start;
  const d = canvasDisplayRect();
  const innerR = $('canvas-inner').getBoundingClientRect();
  const w = asset.w * Math.abs(evalProp(clip.props.scaleX, tc)) / 100 * d.s;
  const h = asset.h * Math.abs(evalProp(clip.props.scaleY, tc)) / 100 * d.s;
  const cx = d.left - innerR.left + evalProp(clip.props.x, tc) * d.s;
  const cy = d.top - innerR.top + evalProp(clip.props.y, tc) * d.s;
  gizmo.style.left = `${cx - w / 2}px`;
  gizmo.style.top = `${cy - h / 2}px`;
  gizmo.style.width = `${w}px`;
  gizmo.style.height = `${h}px`;
  gizmo.style.transform = `rotate(${evalProp(clip.props.rot, tc)}deg)`;
  gizmo.hidden = false;
}

let gzDrag = null;

gizmo.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  const tgt = gizmoTarget();
  if (!tgt) return;
  e.preventDefault();
  e.stopPropagation();
  try { gizmo.setPointerCapture(e.pointerId); } catch {}
  const { clip } = tgt;
  const tc = tCur - clip.start;
  const d = canvasDisplayRect();
  const rot = evalProp(clip.props.rot, tc) * Math.PI / 180;
  const startX = evalProp(clip.props.x, tc);
  const startY = evalProp(clip.props.y, tc);
  // Pointer offset from the clip center, in unrotated comp pixels.
  const toLocal = (clientX, clientY) => {
    const px = (clientX - d.left) / d.s - startX;
    const py = (clientY - d.top) / d.s - startY;
    return [
      px * Math.cos(-rot) - py * Math.sin(-rot),
      px * Math.sin(-rot) + py * Math.cos(-rot),
    ];
  };
  gzDrag = {
    clip,
    d,
    rot,
    handle: e.target.dataset?.h ?? null,
    tFrozen: clamp(quantize(tc, comp.fps), 0, clip.dur),
    startClient: [e.clientX, e.clientY],
    startX,
    startY,
    startSX: evalProp(clip.props.scaleX, tc),
    startSY: evalProp(clip.props.scaleY, tc),
    startLocal: toLocal(e.clientX, e.clientY),
    toLocal,
  };
  history.begin(comp);
});

/* Alignment guides shown while a viewport drag snaps to the frame. */
const guideV = document.createElement('div');
guideV.className = 'gz-guide v';
const guideH = document.createElement('div');
guideH.className = 'gz-guide h';
guideV.hidden = guideH.hidden = true;
$('canvas-inner').append(guideV, guideH);

function hideGuides() { guideV.hidden = guideH.hidden = true; }

/** Snap a dragged clip center per axis to the frame's edges/center using
 * the clip's rotated bounding box. Returns {v, line} or null. */
function snapAxisTargets(v, targets, thresh) {
  let best = null;
  let bestD = thresh;
  for (const t of targets) {
    const d = Math.abs(v - t.v);
    if (d < bestD) { bestD = d; best = t; }
  }
  return best;
}

function applyViewportSnap(g, nx, ny) {
  const asset = assets.get(g.clip.assetId);
  if (!asset?.w) return [nx, ny];
  const w = asset.w * Math.abs(g.startSX) / 100;
  const h = asset.h * Math.abs(g.startSY) / 100;
  const c = Math.abs(Math.cos(g.rot));
  const s = Math.abs(Math.sin(g.rot));
  const hw = (w * c + h * s) / 2;
  const hh = (w * s + h * c) / 2;
  const W = comp.width, H = comp.height;
  const thresh = 8 / g.d.s;                 // ~8 screen px of magnetism

  const bx = snapAxisTargets(nx, [
    { v: W / 2, line: W / 2 },              // center ↔ center
    { v: hw, line: 0 },                     // left edge ↔ frame left
    { v: W - hw, line: W },                 // right edge ↔ frame right
  ], thresh);
  const by = snapAxisTargets(ny, [
    { v: H / 2, line: H / 2 },
    { v: hh, line: 0 },
    { v: H - hh, line: H },
  ], thresh);

  const innerR = $('canvas-inner').getBoundingClientRect();
  if (bx) {
    nx = bx.v;
    guideV.style.left = `${g.d.left - innerR.left + bx.line * g.d.s}px`;
    guideV.style.top = `${g.d.top - innerR.top}px`;
    guideV.style.height = `${H * g.d.s}px`;
    guideV.hidden = false;
  } else {
    guideV.hidden = true;
  }
  if (by) {
    ny = by.v;
    guideH.style.top = `${g.d.top - innerR.top + by.line * g.d.s}px`;
    guideH.style.left = `${g.d.left - innerR.left}px`;
    guideH.style.width = `${W * g.d.s}px`;
    guideH.hidden = false;
  } else {
    guideH.hidden = true;
  }
  return [nx, ny];
}

gizmo.addEventListener('pointermove', (e) => {
  if (!gzDrag) return;
  const g = gzDrag;
  const live = (k, v) => setPropValueLive(g.clip, k, v, g.tFrozen);
  if (!g.handle) {
    let nx = g.startX + (e.clientX - g.startClient[0]) / g.d.s;
    let ny = g.startY + (e.clientY - g.startClient[1]) / g.d.s;
    if (timeline?.snap && !e.ctrlKey && !e.metaKey)
      [nx, ny] = applyViewportSnap(g, nx, ny);
    else
      hideGuides();
    // Keep sub-pixel precision so snapped edges sit exactly flush.
    live('x', Math.round(nx * 100) / 100);
    live('y', Math.round(ny * 100) / 100);
    return;
  }
  const local = g.toLocal(e.clientX, e.clientY);
  const factor = (axis) => {
    const from = g.startLocal[axis];
    if (Math.abs(from) < 1e-3) return 1;
    return clamp(Math.abs(local[axis] / from), 0.005, 100);
  };
  const doX = g.handle.includes('e') || g.handle.includes('w');
  const doY = g.handle.includes('n') || g.handle.includes('s');
  let sx = doX ? g.startSX * factor(0) : g.startSX;
  let sy = doY ? g.startSY * factor(1) : g.startSY;
  if (e.shiftKey && doX && doY) {
    // Uniform: follow the dominant axis.
    const f = Math.abs(factor(0) - 1) >= Math.abs(factor(1) - 1) ? factor(0) : factor(1);
    sx = g.startSX * f;
    sy = g.startSY * f;
  }
  if (doX) live('scaleX', Math.round(sx * 100) / 100);
  if (doY) live('scaleY', Math.round(sy * 100) / 100);
});

function gzFinish() {
  hideGuides();
  if (!gzDrag) return;
  gzDrag = null;
  history.commit(comp);
  onModelChange({ structural: false });
}
gizmo.addEventListener('pointerup', gzFinish);
gizmo.addEventListener('pointercancel', gzFinish);
gizmo.addEventListener('dblclick', (e) => e.stopPropagation());

/* ---- viewport selection --------------------------------------------- */

/* The player doubles as a selection surface: clicking a media layer picks
 * it (timeline selection + gizmo follow), clicking empty space clears the
 * selection. The gizmo swallows its own pointerdowns, so anything that
 * reaches the canvas is a click outside the current selection. */

function clipAtViewportPoint(clientX, clientY) {
  const d = canvasDisplayRect();
  const px = (clientX - d.left) / d.s;
  const py = (clientY - d.top) / d.s;
  const hits = activeClips(comp, tCur, 'media').filter(({ track }) => !track.hidden);
  for (let i = hits.length - 1; i >= 0; i--) {          // top-most first
    const clip = hits[i].clip;
    const asset = assets.get(clip.assetId);
    if (!asset?.ready) continue;
    const tc = tCur - clip.start;
    const sx = Math.abs(evalProp(clip.props.scaleX, tc)) / 100;
    const sy = Math.abs(evalProp(clip.props.scaleY, tc)) / 100;
    if (sx < 1e-4 || sy < 1e-4) continue;
    const dx = px - evalProp(clip.props.x, tc);
    const dy = py - evalProp(clip.props.y, tc);
    const r = evalProp(clip.props.rot, tc) * Math.PI / 180;
    const lx = (dx * Math.cos(r) + dy * Math.sin(r)) / sx;
    const ly = (-dx * Math.sin(r) + dy * Math.cos(r)) / sy;
    if (Math.abs(lx) <= asset.w / 2 && Math.abs(ly) <= asset.h / 2) return clip;
  }
  return null;
}

canvas.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || maskEdit) return;
  const hit = clipAtViewportPoint(e.clientX, e.clientY);
  timeline.selectClip(hit ? hit.id : null);
  updateGizmo();
});

// Clicking the letterbox / empty pane around the canvas also deselects.
viewer.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || maskEdit) return;
  if (e.target !== viewer && e.target !== canvasStack) return;
  timeline.selectClip(null);
  updateGizmo();
});

/* ---- viewport context menu ----------------------------------------- */

/** Set several transform values at once as ONE undo step, honoring the
 * per-property animation state (keyframe at playhead vs static). */
function applyTransformValues(clip, values, note = null) {
  history.record(comp, () => {
    for (const [k, v] of Object.entries(values)) setPropValueLive(clip, k, v);
  });
  onModelChange({ structural: false });
  if (note) setStatus(note);
}

/** Move a clip to the very top (front) or bottom (back) of the stack. */
function reorderClip(clip, toFront) {
  history.record(comp, () => {
    const track = trackOf(comp, clip);
    if (!track) return;
    track.clips.splice(track.clips.indexOf(clip), 1);
    const nt = newTrack(clip.name);
    nt.clips.push(clip);
    if (toFront) comp.tracks.unshift(nt);
    else comp.tracks.push(nt);
    removeEmptyTracks(comp);
  });
  onModelChange({ structural: true });
  setStatus(`${clip.name} ${toFront ? 'brought to front' : 'sent to back'}`);
}

gizmo.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  e.stopPropagation();
  const tgt = gizmoTarget();
  if (!tgt) return;
  const { clip, asset } = tgt;
  const W = comp.width, H = comp.height;
  const r2 = (v) => Math.round(v * 100) / 100;
  const fit = asset.w ? r2(Math.min(W / asset.w, H / asset.h) * 100) : 100;
  const fill = asset.w ? r2(Math.max(W / asset.w, H / asset.h) * 100) : 100;
  const center = { x: W / 2, y: H / 2 };
  showMenu(e.clientX, e.clientY, [
    { label: 'Fit in frame', action: () => applyTransformValues(clip, { ...center, scaleX: fit, scaleY: fit }, `${clip.name} fit inside the frame`) },
    { label: 'Fill frame (crop)', action: () => applyTransformValues(clip, { ...center, scaleX: fill, scaleY: fill }, `${clip.name} fills the frame`) },
    { label: 'Stretch to frame', action: () => applyTransformValues(clip, { ...center, scaleX: asset.w ? r2(W / asset.w * 100) : 100, scaleY: asset.h ? r2(H / asset.h * 100) : 100 }, `${clip.name} stretched to the frame`) },
    { label: 'Center in frame', action: () => applyTransformValues(clip, center) },
    '-',
    { label: 'Mirror horizontal', action: () => applyTransformValues(clip, { scaleX: r2(-valueAt(clip, 'scaleX')) }) },
    { label: 'Mirror vertical', action: () => applyTransformValues(clip, { scaleY: r2(-valueAt(clip, 'scaleY')) }) },
    { label: 'Rotate 90° cw', action: () => applyTransformValues(clip, { rot: r2(valueAt(clip, 'rot') + 90) }) },
    '-',
    { label: 'Bring to front', action: () => reorderClip(clip, true) },
    { label: 'Send to back', action: () => reorderClip(clip, false) },
    '-',
    { label: 'Reset transform', action: () => applyTransformValues(clip, { ...center, scaleX: 100, scaleY: 100, rot: 0, opacity: 100 }, `${clip.name} transform reset`) },
    {
      label: 'Delete clip',
      danger: true,
      action: () => {
        timeline.selClips = new Set([clip.id]);
        timeline.deleteSelection();
      },
    },
  ]);
});

/* =====================================================================
 * Overlay textures (stamp images + rendered titles)
 * =================================================================== */

const STAMP_PRESET_PATH = 'shaders/overlay/stamp/stamp.slangp';
const DEFAULT_TITLE = { text: 'TITLE', font: 'Arial', sizePx: 96, color: '#ffffff', outline: true };

function renderTitleCanvas({ text, font, sizePx, color, outline }) {
  const pad = Math.ceil(sizePx * 0.4);
  const c = document.createElement('canvas');
  let ctx = c.getContext('2d');
  ctx.font = `bold ${sizePx}px ${font}`;
  const w = Math.ceil(ctx.measureText(text || ' ').width);
  c.width = Math.max(2, w + pad * 2);
  c.height = Math.ceil(sizePx * 1.35) + pad;
  ctx = c.getContext('2d');
  ctx.font = `bold ${sizePx}px ${font}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  if (outline) {
    ctx.lineWidth = Math.max(2, sizePx * 0.09);
    ctx.lineJoin = 'round';
    ctx.strokeStyle = 'rgba(0,0,0,0.9)';
    ctx.strokeText(text, c.width / 2, c.height / 2);
  }
  ctx.fillStyle = color;
  ctx.fillText(text, c.width / 2, c.height / 2);
  return c;
}

function applyOverlaySource(clip, texName, source, descriptor) {
  const spec = specFor(clip);
  (spec.textureOverrides ??= {})[texName] = source;
  (clip.overlay ??= {})[texName] = descriptor;
  chainDirty = true;
  scheduleSave();
}

const stampFileInput = $('stamp-file-input');
let stampPickTarget = null;   // { clipId, texName }

stampFileInput.addEventListener('change', async () => {
  const f = stampFileInput.files[0];
  const target = stampPickTarget;
  stampFileInput.value = '';
  stampPickTarget = null;
  if (!f || !target) return;
  const hit = findClip(comp, target.clipId);
  if (!hit) return;
  const bmp = await createImageBitmap(f);
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  const dataURL = c.toDataURL('image/png');
  applyOverlaySource(hit.clip, target.texName, bmp,
    { kind: 'image', dataURL: dataURL.length > 2_000_000 ? null : dataURL });
  renderInspector();
  setStatus(`${target.texName} texture replaced with ${f.name}`);
});

/* =====================================================================
 * Add-effect menu (search + folders) → fx clips
 * =================================================================== */

const openFolders = new Set();

function closeAddMenu() {
  addLayerList.classList.remove('open');
  addLayerSearch.value = '';
}

function openAddMenu() {
  addLayerList.classList.add('open');
  rebuildAddMenu();
}

addLayerSearch.addEventListener('focus', openAddMenu);

document.addEventListener('pointerdown', (e) => {
  if (addLayerList.classList.contains('open') && !$('add-section').contains(e.target)) closeAddMenu();
});

document.addEventListener('keydown', (e) => {
  if (e.key !== '/' || e.target.matches?.('input, textarea')) return;
  e.preventDefault();
  addLayerSearch.focus();
});

addLayerSearch.addEventListener('input', () => {
  addLayerList.classList.add('open');
  rebuildAddMenu();
});
addLayerSearch.addEventListener('keydown', (e) => {
  e.stopPropagation();
  if (e.key === 'Escape') { closeAddMenu(); addLayerSearch.blur(); }
  else if (e.key === 'Enter') addLayerList.querySelector('.menu-item')?.click();
});

function categoryLabel(id) {
  return (manifest.categories ?? []).find((c) => c.id === id)?.label ?? id;
}

function rebuildAddMenu() {
  const q = addLayerSearch.value.trim().toLowerCase();
  addLayerList.replaceChildren();
  const savedList = Object.keys(loadSaved()).sort();

  const addItem = (label, onPick, note = null) => {
    const it = document.createElement('div');
    it.className = 'menu-item';
    const span = document.createElement('span');
    span.textContent = label;
    it.appendChild(span);
    if (note) {
      const n = document.createElement('span');
      n.className = 'note';
      n.textContent = note;
      it.appendChild(n);
    }
    it.addEventListener('click', () => { closeAddMenu(); addChoice(onPick); });
    addLayerList.appendChild(it);
  };

  if (q) {
    if ('custom shader write your own'.includes(q))
      addItem('✎ custom shader', '__custom__');
    if ('text title caption overlay'.includes(q))
      addItem('T text / title', '__title__');
    for (const name of savedList)
      if (name.toLowerCase().includes(q))
        addItem(`🗎 ${name}`, `__saved__:${name}`, 'saved');
    for (const eff of manifest.effects) {
      const cat = categoryLabel(eff.category);
      if (eff.name.toLowerCase().includes(q) || cat.toLowerCase().includes(q))
        addItem(eff.name, eff.path, cat);
    }
    if (!addLayerList.children.length) {
      const none = document.createElement('div');
      none.className = 'menu-empty';
      none.textContent = 'no matches';
      addLayerList.appendChild(none);
    }
    return;
  }

  addItem('✎ custom shader (write your own)', '__custom__');
  addItem('T text / title', '__title__');

  const folder = (id, label, children) => {
    if (!children.length) return;
    const open = openFolders.has(id);
    const head = document.createElement('div');
    head.className = 'menu-folder';
    const title = document.createElement('span');
    title.textContent = `${open ? '▾' : '▸'} ${label}`;
    const count = document.createElement('span');
    count.className = 'note';
    count.textContent = String(children.length);
    head.append(title, count);
    head.addEventListener('click', () => {
      if (open) openFolders.delete(id);
      else openFolders.add(id);
      rebuildAddMenu();
    });
    addLayerList.appendChild(head);
    if (open) for (const c of children) addItem(c.label, c.choice, c.note);
  };

  folder('saved', 'Saved shaders',
    savedList.map((name) => ({ label: `🗎 ${name}`, choice: `__saved__:${name}` })));
  for (const cat of manifest.categories ?? [])
    folder(cat.id, cat.label,
      manifest.effects
        .filter((e) => e.category === cat.id)
        .map((e) => ({ label: e.name, choice: e.path })));
}

/** Create an fx clip at the playhead on a new top track. */
async function addChoice(choice) {
  if (!choice || !fx) return;
  let spec;
  let overlayTitle = false;
  if (choice === '__custom__') {
    spec = { fxKind: 'custom', source: CUSTOM_BOILERPLATE, label: 'custom shader' };
  } else if (choice === '__title__') {
    spec = { fxKind: 'preset', path: STAMP_PRESET_PATH, label: 'title' };
    overlayTitle = true;
  } else if (choice.startsWith('__saved__:')) {
    const name = choice.slice('__saved__:'.length);
    const saves = loadSaved();
    if (!saves[name]) return;
    spec = { fxKind: 'custom', source: saves[name].source, label: name, savedName: name };
  } else {
    spec = { fxKind: 'preset', path: choice, label: choice.split('/').pop().replace(/\.slangp$/, '') };
  }

  // New effects cover the whole timeline; trim them down when needed.
  const clip = newFxClip(spec, 0, Math.max(1 / comp.fps, comp.dur));
  if (overlayTitle) clip.overlay = { Stamp: { kind: 'text', state: { ...DEFAULT_TITLE } } };

  history.record(comp, () => {
    const track = newTrack(clip.name);
    track.clips.push(clip);
    comp.tracks.unshift(track);
    ensureDur(comp);
  });
  timeline.selectClip(clip.id);
  timeline.expanded.add(clip.id);
  onModelChange({ structural: true });
  setStatus(`added ${clip.name} — compiling…`);
  const t0 = performance.now();
  try {
    await ensureParamMeta(clip);
    setStatus(`${clip.name} ready in ${Math.round(performance.now() - t0)} ms`);
  } catch (e) {
    setStatus(`${clip.name} failed to compile — see inspector`);
  }
}

/* =====================================================================
 * Custom shader compile
 * =================================================================== */

const editorDrafts = new Map();   // clipId -> unsaved editor text

async function compileCustomClip(clip, source) {
  history.record(comp, () => { clip.source = source; });
  const spec = specFor(clip);
  virtualFiles.set(spec.dir + 'custom.slang', source);
  fx.invalidateModules(spec.dir);
  paramMetaCache.delete(clip.id);
  chainDirty = true;
  setStatus('compiling custom shader…');
  const t0 = performance.now();
  try {
    await ensureParamMeta(clip);
    await syncFxChain(tCur);
    const err = fxSpecs.get(clip.id)?.error;
    setStatus(err ? 'custom shader failed — see inspector'
                  : `custom shader compiled in ${Math.round(performance.now() - t0)} ms`);
  } catch {
    setStatus('custom shader failed to compile — see inspector');
  }
  scheduleSave();
  timeline.render();
  renderInspector();
}

/* =====================================================================
 * Inspector — the right panel for the selected clip.
 * =================================================================== */

const inspLive = [];   // [{clip, key, slider, num}] animated bindings

function updateInspectorLive() {
  for (const b of inspLive) {
    if (document.activeElement === b.num || b.dragging) continue;
    const v = valueAt(b.clip, b.key);
    b.slider.value = String(v);
    b.num.value = fmtVal(v);
  }
}

const fmtVal = (v) => (+v).toFixed(3).replace(/\.?0+$/, '') || '0';

function renderInspector() {
  if (!timeline) return;
  inspLive.length = 0;
  const clip = timeline.selectedClip;
  inspectorEl.replaceChildren();

  if (!clip) {
    const div = document.createElement('div');
    div.className = 'insp-empty';
    div.innerHTML = `
      <p>Select a clip on the timeline to edit it.</p>
      <p class="hint">· <b>Import media…</b> or drop files to create media clips<br>
      · the <b>＋ search box</b> above adds effect clips<br>
      · ▸ on a clip twirls out its keyframable properties<br>
      · <b>⏱</b> starts animating a property; change its value at another
      time to add keyframes; right-click a ◆ for easing<br>
      · Ctrl+wheel zooms the timeline down to single frames<br>
      · wheel over the preview zooms the viewport; drag the space around
      the frame (or middle-drag anywhere) to pan</p>`;
    inspectorEl.appendChild(div);
    return;
  }

  const hit = findClip(comp, clip.id);
  const track = hit?.track;

  /* -- header: name + kind + enable -- */
  const head = document.createElement('div');
  head.className = 'insp-head';
  const kind = document.createElement('span');
  kind.className = 'insp-kind ' + clip.kind;
  kind.textContent = clip.kind === 'media'
    ? (assets.get(clip.assetId)?.kind === 'image' ? '🖼' : '🎞')
    : (clip.fxKind === 'custom' ? '✎' : 'ƒx');
  const name = document.createElement('input');
  name.className = 'insp-name';
  name.value = clip.name;
  name.title = 'clip name';
  name.addEventListener('keydown', (e) => e.stopPropagation());
  name.addEventListener('change', () => {
    history.record(comp, () => { clip.name = name.value.trim() || clip.name; });
    onModelChange({ structural: false });
  });
  head.append(kind, name);

  if (clip.kind === 'fx') {
    const en = document.createElement('label');
    en.className = 'insp-enable';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = clip.enabled !== false;
    cb.addEventListener('change', () => {
      history.record(comp, () => { clip.enabled = cb.checked; });
      onModelChange({ structural: true });
    });
    en.append(cb, 'on');
    head.appendChild(en);
  }
  const del = document.createElement('button');
  del.className = 'tl-mini insp-del';
  del.textContent = '✕';
  del.title = 'delete clip';
  del.addEventListener('click', () => {
    timeline.selClips = new Set([clip.id]);
    timeline.deleteSelection();
  });
  head.appendChild(del);
  inspectorEl.appendChild(head);

  /* -- timing -- */
  const timing = document.createElement('div');
  timing.className = 'insp-timing';
  const tItem = (label, get, set, title = '') => {
    const l = document.createElement('label');
    l.textContent = label;
    if (title) l.title = title;
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = String(1 / comp.fps);
    inp.value = fmtVal(get());
    inp.addEventListener('keydown', (e) => e.stopPropagation());
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (Number.isNaN(v)) return;
      history.record(comp, () => set(v));
      onModelChange({ structural: true });
    });
    l.appendChild(inp);
    timing.appendChild(l);
  };
  tItem('start', () => clip.start, (v) => { clip.start = Math.max(0, quantize(v, comp.fps)); ensureDur(comp); });
  tItem('length', () => clip.dur, (v) => { clip.dur = Math.max(1 / comp.fps, quantize(v, comp.fps)); ensureDur(comp); });
  if (clip.kind === 'media')
    tItem('trim-in', () => clip.in, (v) => { clip.in = Math.max(0, v); }, 'seconds trimmed from the source head');
  inspectorEl.appendChild(timing);

  /* -- media source info -- */
  if (clip.kind === 'media') {
    const asset = assets.get(clip.assetId);
    const info = document.createElement('div');
    info.className = 'insp-src';
    info.textContent = asset
      ? `${asset.name} — ${asset.w}×${asset.h}${asset.duration ? ` · ${asset.duration.toFixed(2)}s` : ''}${asset.ready ? '' : ' (loading…)'}`
      : 'media offline — re-import the file';
    inspectorEl.appendChild(info);
  }

  /* -- error surface -- */
  const spec = clip.kind === 'fx' ? fxSpecs.get(clip.id) : null;
  const err = spec?.error || spec?.lastCompileError;
  if (err) {
    const e = document.createElement('div');
    e.className = 'layer-error';
    e.textContent = err;
    inspectorEl.appendChild(e);
  }

  /* -- custom shader editor -- */
  if (clip.kind === 'fx' && clip.fxKind === 'custom') renderCustomEditor(clip);

  /* -- overlay texture controls (stamp/title presets) -- */
  if (clip.kind === 'fx' && spec?.runtime?.preset?.textures?.length)
    renderOverlayControls(clip, spec);

  /* -- mask -- */
  if (clip.kind === 'fx') renderMaskSection(clip);

  /* -- properties -- */
  const defs = propDefs(clip);
  if (clip.kind === 'fx' && !defs.length && !err) {
    const ld = document.createElement('div');
    ld.className = 'insp-src';
    ld.textContent = 'loading parameters…';
    inspectorEl.appendChild(ld);
  }
  if (defs.length) {
    const box = document.createElement('div');
    box.className = 'insp-params';
    const h = document.createElement('h3');
    h.textContent = clip.kind === 'media' ? 'Transform' : 'Parameters';
    const hint = document.createElement('span');
    hint.className = 'hint';
    hint.textContent = '⏱ = animate';
    h.appendChild(hint);
    box.appendChild(h);
    for (const def of defs) box.appendChild(paramRow(clip, def));
    inspectorEl.appendChild(box);
  }
}

function paramRow(clip, def) {
  const prop = getProp(clip, def.key);
  const anim = !!prop?.anim;
  const row = document.createElement('div');
  row.className = 'param-row kf';

  const sw = document.createElement('button');
  sw.className = 'tl-stopwatch' + (anim ? ' on' : '');
  sw.textContent = '⏱';
  sw.title = anim ? 'animating — click to freeze' : 'animate this property';
  sw.addEventListener('click', () => toggleAnim(clip, def.key));

  const label = document.createElement('label');
  label.textContent = def.label;
  label.title = `${def.key}${def.unit ? ` (${def.unit})` : ''}`;

  const slider = document.createElement('input');
  slider.type = 'range';
  slider.min = String(def.min);
  slider.max = String(def.max);
  slider.step = String(def.step || 0.001);

  const num = document.createElement('input');
  num.type = 'number';
  num.className = 'val';
  num.step = String(def.step || 0.001);

  const v0 = valueAt(clip, def.key);
  slider.value = String(v0);
  num.value = fmtVal(v0);

  const binding = { clip, key: def.key, slider, num, dragging: false };
  if (anim) inspLive.push(binding);

  slider.addEventListener('pointerdown', () => {
    binding.dragging = true;
    binding.tFrozen = relTime(clip);
    history.begin(comp);
  });
  slider.addEventListener('input', () => {
    // Param-only change — applied every frame by applyParams, no rebuild.
    const v = parseFloat(slider.value);
    setPropValueLive(clip, def.key, v, binding.dragging ? binding.tFrozen : null);
    num.value = fmtVal(v);
  });
  const commitSlider = () => {
    if (!binding.dragging) return;
    binding.dragging = false;
    history.commit(comp);
    onModelChange({ structural: false });
  };
  slider.addEventListener('pointerup', commitSlider);
  slider.addEventListener('pointercancel', commitSlider);

  num.addEventListener('keydown', (e) => e.stopPropagation());
  num.addEventListener('change', () => {
    const v = parseFloat(num.value);
    if (Number.isNaN(v)) return;
    setPropValue(clip, def.key, v);
  });

  const keyBtn = document.createElement('button');
  keyBtn.className = 'tl-mini tl-kf-toggle';
  const atKey = anim && prop && keyNear(prop, relTime(clip), 0.5 / comp.fps);
  keyBtn.textContent = '◆';
  keyBtn.classList.toggle('at-key', !!atKey);
  keyBtn.title = 'add / remove keyframe at playhead';
  keyBtn.addEventListener('click', () => toggleKey(clip, def.key));

  row.append(sw, label, slider, num, keyBtn);
  return row;
}

function renderCustomEditor(clip) {
  const editor = document.createElement('div');
  editor.className = 'layer-editor';

  const sed = makeShaderEditor({
    value: editorDrafts.get(clip.id) ?? clip.source ?? '',
    onInput: (text) => editorDrafts.set(clip.id, text),
  });
  sed.el.classList.add('sed-inspector');

  const row = document.createElement('div');
  row.className = 'editor-actions';
  const compile = document.createElement('button');
  compile.className = 'btn';
  compile.textContent = 'Compile';
  compile.onclick = () => { editorDrafts.delete(clip.id); compileCustomClip(clip, sed.getValue()); };
  const revert = document.createElement('button');
  revert.className = 'btn';
  revert.textContent = 'Revert';
  revert.title = 'discard edits since last compile';
  revert.onclick = () => {
    editorDrafts.delete(clip.id);
    sed.setValue(clip.source ?? '');
  };
  const expand = document.createElement('button');
  expand.className = 'btn';
  expand.textContent = '⛶';
  expand.title = 'open the full-screen editor (with cheat sheet)';
  expand.onclick = () => openShaderModal(clip);

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.className = 'save-name';
  nameInput.placeholder = 'name…';
  nameInput.value = clip.savedName ?? '';
  nameInput.addEventListener('keydown', (e) => e.stopPropagation());

  const save = document.createElement('button');
  save.className = 'btn';
  save.textContent = 'Save';
  save.title = 'save this shader to the browser; it appears under "saved shaders" in the add menu';
  save.onclick = () => {
    const name = nameInput.value.trim();
    if (!name) { setStatus('give the shader a name to save it'); nameInput.focus(); return; }
    const saves = loadSaved();
    saves[name] = { source: sed.getValue(), savedAt: new Date().toISOString() };
    storeSaved(saves);
    clip.savedName = name;
    scheduleSave();
    setStatus(`saved '${name}' to this browser`);
  };

  const forget = document.createElement('button');
  forget.className = 'btn';
  forget.textContent = 'Forget';
  forget.title = 'delete this saved shader from localStorage (the clip keeps running)';
  forget.hidden = !clip.savedName;
  forget.onclick = async () => {
    if (!clip.savedName) return;
    const ok = await confirmDialog({
      title: `Delete saved shader '${clip.savedName}'?`,
      message: 'It disappears from the add-effect menu in this browser. Clips already using it keep their own copy of the code.',
      confirmLabel: 'Delete shader',
    });
    if (!ok) return;
    const saves = loadSaved();
    delete saves[clip.savedName];
    storeSaved(saves);
    setStatus(`forgot saved shader '${clip.savedName}'`);
    clip.savedName = null;
    renderInspector();
  };

  row.append(compile, revert, expand, nameInput, save, forget);
  editor.append(sed.el, row);
  inspectorEl.appendChild(editor);
}

/** Full-screen shader editor modal with the slang cheat sheet. */
function openShaderModal(clip) {
  document.querySelector('.sed-modal')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'modal-wrap sed-modal';
  wrap.innerHTML = `
    <div class="sed-frame">
      <div class="sed-head">
        <span class="sed-title"></span>
        <span class="sed-status"></span>
        <button class="btn" data-a="compile">Compile</button>
        <button class="btn" data-a="cheat">? Cheat sheet</button>
        <button class="btn" data-a="close">✕ Close</button>
      </div>
      <div class="sed-main">
        <div class="sed-slot"></div>
        <aside class="sed-cheat">${CHEAT_HTML}</aside>
      </div>
    </div>`;
  const statusEl2 = wrap.querySelector('.sed-status');
  wrap.querySelector('.sed-title').textContent = `✎ ${clip.name}`;
  const sed = makeShaderEditor({
    value: editorDrafts.get(clip.id) ?? clip.source ?? '',
    onInput: (text) => editorDrafts.set(clip.id, text),
  });
  sed.el.classList.add('sed-full');
  wrap.querySelector('.sed-slot').appendChild(sed.el);

  const close = () => {
    wrap.remove();
    document.removeEventListener('keydown', onKey);
    renderInspector();          // inspector editor picks up the draft
  };
  const onKey = (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } };
  document.addEventListener('keydown', onKey);

  wrap.querySelector('[data-a=close]').addEventListener('click', close);
  wrap.querySelector('[data-a=cheat]').addEventListener('click', () => {
    wrap.querySelector('.sed-cheat').classList.toggle('hidden');
  });
  wrap.querySelector('[data-a=compile]').addEventListener('click', async () => {
    statusEl2.textContent = 'compiling…';
    editorDrafts.delete(clip.id);
    await compileCustomClip(clip, sed.getValue());
    const err = fxSpecs.get(clip.id)?.error || fxSpecs.get(clip.id)?.lastCompileError;
    statusEl2.textContent = err ? `✗ ${err.split('\n')[0].slice(0, 120)}` : '✓ compiled';
    statusEl2.classList.toggle('err', !!err);
  });
  document.body.appendChild(wrap);
  sed.textarea.focus();
}

function renderOverlayControls(clip, spec) {
  for (const tex of spec.runtime.preset.textures) {
    const texName = tex.name;
    const oc = document.createElement('div');
    oc.className = 'overlay-controls';
    const state = clip.overlay?.[texName]?.kind === 'text'
      ? clip.overlay[texName].state
      : { ...DEFAULT_TITLE, text: '' };

    const imgBtn = document.createElement('button');
    imgBtn.className = 'btn';
    imgBtn.textContent = 'Image…';
    imgBtn.title = `use an image as the ${texName} texture`;
    imgBtn.onclick = () => {
      stampPickTarget = { clipId: clip.id, texName };
      stampFileInput.click();
    };

    const textInput = document.createElement('input');
    textInput.type = 'text';
    textInput.className = 'overlay-text';
    textInput.placeholder = 'type a title…';
    textInput.value = state.text ?? '';
    textInput.addEventListener('keydown', (e) => e.stopPropagation());

    const sizeInput = document.createElement('input');
    sizeInput.type = 'number';
    sizeInput.className = 'overlay-size';
    sizeInput.min = '12'; sizeInput.max = '400'; sizeInput.step = '4';
    sizeInput.value = String(state.sizePx);
    sizeInput.title = 'font size (px)';
    sizeInput.addEventListener('keydown', (e) => e.stopPropagation());

    const colorInput = document.createElement('input');
    colorInput.type = 'color';
    colorInput.value = state.color;
    colorInput.title = 'text color';

    const fontSel = document.createElement('select');
    for (const f of ['Arial', 'Georgia', 'Impact', 'Courier New', 'Trebuchet MS']) {
      const o = document.createElement('option');
      o.value = f; o.textContent = f;
      if (f === state.font) o.selected = true;
      fontSel.appendChild(o);
    }

    const outlineLabel = document.createElement('label');
    const outline = document.createElement('input');
    outline.type = 'checkbox';
    outline.checked = state.outline;
    outlineLabel.append(outline, 'outline');

    const applyText = () => {
      const s = {
        text: textInput.value,
        font: fontSel.value,
        sizePx: Math.max(12, parseFloat(sizeInput.value) || 96),
        color: colorInput.value,
        outline: outline.checked,
      };
      if (!s.text.trim()) return;
      applyOverlaySource(clip, texName, renderTitleCanvas(s), { kind: 'text', state: s });
    };
    textInput.addEventListener('change', applyText);
    sizeInput.addEventListener('change', applyText);
    colorInput.addEventListener('change', applyText);
    fontSel.addEventListener('change', applyText);
    outline.addEventListener('change', applyText);

    oc.append(imgBtn, textInput, sizeInput, colorInput, fontSel, outlineLabel);
    inspectorEl.appendChild(oc);
  }
}

function renderMaskSection(clip) {
  const spec = fxSpecs.get(clip.id);
  const editing = maskEdit?.clipId === clip.id;
  const mc = document.createElement('div');
  mc.className = 'mask-controls';

  const maskBtn = document.createElement('button');
  maskBtn.className = 'btn' + (editing ? ' active' : '');
  maskBtn.textContent = spec?.maskState ? (editing ? 'Done' : 'Edit mask') : 'Add mask';
  maskBtn.title = 'paint where the effect should NOT apply';
  maskBtn.onclick = () => (editing ? stopMaskEdit() : startMaskEdit(clip));
  mc.appendChild(maskBtn);

  if (editing && spec?.maskState) {
    const toolBtn = (tool, icon, title) => {
      const b = document.createElement('button');
      b.className = 'btn' + (brush.tool === tool ? ' active' : '');
      b.textContent = icon;
      b.title = title;
      b.onclick = () => { brush.tool = tool; renderInspector(); };
      return b;
    };
    mc.append(
      toolBtn('brush', '🖌', 'brush'),
      toolBtn('linear', '▤', 'linear gradient — drag across the preview'),
      toolBtn('radial', '◎', 'radial gradient — drag outward from the center'),
    );

    const modeBtn = (mode, label) => {
      const b = document.createElement('button');
      b.className = 'btn' + (brush.mode === mode ? ' active' : '');
      b.textContent = label;
      b.onclick = () => { brush.mode = mode; renderInspector(); };
      return b;
    };
    mc.append(modeBtn('hide', 'Hide fx'), modeBtn('show', 'Show fx'));

    const isBrush = brush.tool === 'brush';
    const rng = (label, min, max, step, get, set, disabled = false) => {
      const l = document.createElement('label');
      l.textContent = label;
      const r = document.createElement('input');
      r.type = 'range';
      r.min = String(min); r.max = String(max); r.step = String(step);
      r.value = String(get());
      r.disabled = disabled;
      r.oninput = () => set(parseFloat(r.value));
      l.appendChild(r);
      return l;
    };
    mc.append(
      rng('size', 8, 300, 1, () => brush.size, (v) => { brush.size = v; }, !isBrush),
      rng('soft', 0, 0.9, 0.05, () => brush.soft, (v) => { brush.soft = v; }, !isBrush),
      rng('opacity', 0, 1, 0.01, () => spec.maskState.opacity ?? 1, (v) => {
        spec.maskState.opacity = v;
        const idx = activeIndexOfClip(clip.id);
        if (idx >= 0) fx.setLayerMaskOptions(idx, { opacity: v });
        scheduleSave();
      }),
    );

    const invLabel = document.createElement('label');
    const inv = document.createElement('input');
    inv.type = 'checkbox';
    inv.checked = !!spec.maskState.invert;
    inv.onchange = () => {
      spec.maskState.invert = inv.checked;
      const idx = activeIndexOfClip(clip.id);
      if (idx >= 0) fx.setLayerMaskOptions(idx, { invert: inv.checked });
      scheduleSave();
    };
    invLabel.append(inv, 'invert');
    mc.appendChild(invLabel);

    const clear = document.createElement('button');
    clear.className = 'btn';
    clear.textContent = 'Clear';
    clear.onclick = () => {
      const src = spec.maskState.source;
      const ctx2 = src.getContext('2d');
      ctx2.globalCompositeOperation = 'source-over';
      ctx2.fillStyle = '#fff';
      ctx2.fillRect(0, 0, src.width, src.height);
      maskOverlay.getContext('2d').clearRect(0, 0, maskOverlay.width, maskOverlay.height);
      pushMaskToGpu();
      scheduleSave();
    };

    const remove = document.createElement('button');
    remove.className = 'btn';
    remove.textContent = 'Remove';
    remove.onclick = () => {
      stopMaskEdit();
      spec.maskState = null;
      clip.mask = null;
      chainDirty = true;
      scheduleSave();
      renderInspector();
    };
    mc.append(clear, remove);
  }
  inspectorEl.appendChild(mc);
}

/* =====================================================================
 * Export — current frame PNG, or render the whole comp to WebM.
 * =================================================================== */

$('btn-export-png').addEventListener('click', async () => {
  if (!fx?.inputTexture) return;
  const blob = await fx.exportPNG();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'slangfx-frame.png';
  a.click();
  setStatus('frame exported');
});

const exportBtn = $('btn-export-webm');

exportBtn.addEventListener('click', () => {
  if (recorder) { finishExport(); pause(); return; }
  if (!fx?.inputTexture) return;
  ensureAudio();
  const stream = canvas.captureStream(comp.fps);
  if (recordDest)
    for (const t of recordDest.stream.getAudioTracks()) stream.addTrack(t);

  const withAudio = stream.getAudioTracks().length > 0;
  const mime = withAudio && MediaRecorder.isTypeSupported('video/webm;codecs=vp9,opus')
    ? 'video/webm;codecs=vp9,opus'
    : 'video/webm;codecs=vp9';
  recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  const chunks = [];
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(chunks, { type: 'video/webm' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'slangfx-comp.webm';
    a.click();
    setStatus('render saved');
  };
  exportMode = true;
  pause();
  setTime(0);
  recorder.start();
  play();
  exportBtn.textContent = '■ Stop';
  exportBtn.classList.add('recording');
  setStatus('rendering comp to WebM…');
});

function finishExport() {
  exportMode = false;
  if (recorder) {
    recorder.stop();
    recorder = null;
  }
  exportBtn.textContent = 'Render WebM';
  exportBtn.classList.remove('recording');
}

boot();
