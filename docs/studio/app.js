/*
 * slangfx studio — web editor UI.
 *
 * After-Effects-style editor built on the lowkey-studio engine:
 *
 *   comp.js        composition model — tracks, clips, keyframes, undo
 *   compositor.js  WebGPU compositing of media clips into the frame
 *   timeline.js    the zoomable timeline / keyframe panel
 *   audio-fx.js    the Web Audio effect catalogue (reverb, EQ, filters…)
 *   app.js (this)  playback clock + multi-video sync, audio mixing, fx
 *                  chain management, inspector panel, import/export,
 *                  project persistence
 *
 * Render path per frame:
 *   1. active media clips → sync video elements to the comp clock,
 *      upload current frames
 *   2. a media clip carrying its OWN effect stack renders in isolation
 *      first: the clip alone into a private chain's input texture, its
 *      effects run there, and the processed result is what composites —
 *      so those effects touch nothing but that clip (see mediaChains)
 *   3. every media clip composites into the engine input texture with its
 *      animated transform
 *   4. active fx clips (adjustment layers) → the engine layer chain, one
 *      engine layer per effect in each clip's stack, rebuilt only when the
 *      active set changes; every shader param is driven from its keyframe
 *      track each frame
 *   5. fx.render() runs the chain and presents to the canvas
 *
 * Sound runs on its own path: every clip with audio (a video, or an audio
 * clip) gets a Web Audio chain — source → its audio effects → its volume
 * fader → the master gain — rebuilt when its stack changes and re-driven
 * from the same keyframe/driver machinery every frame. The offline
 * exporter rebuilds the identical graph in an OfflineAudioContext with
 * those curves baked as scheduled ramps.
 */

import { SlangFx, loadToolchain, parsePreset, dirnameOf } from './engine/index.js';
import { renameReserved } from './engine/preprocess.js';
import {
  newComp, newTrack, newMediaClip, newAudioClip, newFxClip, clipEnd, evalProp, upsertKey,
  keyNear, activeClips, findClip, ensureDur, removeEmptyTracks, quantize, lastFrame,
  clamp, History, uid, newProp, migrateComp, trackOf, MEDIA_PROPS, hasSource,
  allClipsBottomUp,
  newEffect, effectsOf, findEffect, effectPropKey, parsePropKey, eachClipProp,
  isAudioEffect, visualEffectsOf, audioEffectsOf,
} from './comp.js';
import { Compositor, BLEND_MODES } from './compositor.js';
import {
  DRIVER_WAVES, DRIVER_BANDS, DRIVER_FOLLOWS, DRIVER_MODES,
  newDriver, applyDriver,
} from './driver.js';
import { analyzeMix, detectBeats, sampleLevel, samplePulse } from './audio-analysis.js';
import { AUDIO_EFFECTS, audioEffectDef, controlTargets } from './audio-fx.js';
import { responseWidget } from './audio-widgets.js';
import { LAYER_ICONS, clipIcon, shapeIconCanvas } from './icons.js';
import { Muxer as WebMMuxer, ArrayBufferTarget } from './vendor/webm-muxer.mjs';
import { Timeline, fmtTimecode, showMenu } from './timeline.js';
import { makeShaderEditor, CHEAT_HTML } from './shader-editor.js';

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const canvas = $('preview');
const inspectorEl = $('inspector');
const addLayerSearch = $('add-layer-search');
const addLayerList = $('add-layer-list');

const VIDEO_EXTS = /\.(mp4|mov|mkv|webm|avi|m4v)$/i;
const AUDIO_EXTS = /\.(mp3|wav|m4a|aac|ogg|oga|opus|flac|weba|wma|aif|aiff)$/i;
const GIF_EXT = /\.gif$/i;
const PROJECT_KEY = 'lowkey-studio.project.v2';
const DEFAULT_VIDEO_DUR = 4;   // fallback when a video reports no duration

// ?app=1 — running embedded in a host app (the Electron viewer). Hides the
// marketing site chrome; the flag is preserved across the post-import URL
// cleanup so it survives reloads.
const APP_HOST = new URLSearchParams(location.search).get('app') === '1';
if (APP_HOST) document.body.classList.add('app-host');

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
// One engine layer spec per EFFECT (an effect is a clip's stack entry), so
// compiled runtimes and their saved params survive chain rebuilds.
const fxSpecs = new Map();       // effectId -> engine layer spec (persistent)
const paramMetaCache = new Map();// effectId -> [{name, desc, min, max, step, default}]
// Live mask stacks, one per clip, for both kinds: on an fx clip the mask
// gates the clip's whole effect group; on a media clip it cuts the clip's
// alpha. Serialized back onto clip.mask when the project saves.
const clipMasks = new Map();     // clipId -> maskState {opacity, invert, nodes, ...}
// False until loadClipMasks has hydrated the current project, so a save
// racing the restore can't blank out masks it hasn't read yet.
let masksLoaded = false;

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

/* Per-media-clip effect chains. A media clip's own effects must not touch
 * anything else, so they can't live in the shared adjustment chain: each
 * such clip gets a private headless SlangFx over the same device (compiled
 * modules are shared through fx.moduleCache), fed a transparent frame
 * holding that clip alone. */
const mediaChains = new Map();   // clipId -> {fx, key, dirty, building, promise}

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
  // Media stacked above an adjustment layer is layered onto that layer's
  // output so only effects higher in the stack process it (see
  // compositeFrame). One clip contributes several engine layers, so this
  // fires once, after the last effect in the clip's group.
  fx.onAfterLayer = (encoder, layer) => {
    if (layer.maskGroup && layer.maskGroup.tail !== layer) return;
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
  const launch = await collectLaunchImports();
  if (launch.files.length) {
    // Launched by a host app (e.g. the Lowkey viewer) with media to edit:
    // preserve the previous session, then import into a fresh comp. The
    // import param is stripped (other params survive) so a reload doesn't
    // import a second copy.
    await stashAutosavedProject();
    masksLoaded = true;   // fresh comp — nothing to hydrate
    await importFiles(launch.files, { t: 0 });
    fitDurToContent();
    if (launch.saveBack) {
      comp._saveBack = launch.saveBack;   // persists via the autosave payload
      scheduleSave();
    }
    const params = new URLSearchParams(location.search);
    params.delete('import');
    const qs = params.toString();
    window.history.replaceState(null, '', location.pathname + (qs ? `?${qs}` : ''));
  } else {
    await restoreProject();
  }
  updateSaveBackButton();
  await applyCompSize();
  document.body.classList.add('has-media');
  refreshDropHint();
  timeline.zoomFit();
  timeline.render();
  renderInspector();
  setStatus('ready — import media or add an effect');
  window.fx = fx;                 // console/debug access
  window.comp = () => comp;
  // Console/test handle. Sound especially needs one: a mixdown has no
  // picture to eyeball, so renderCompAudio is the only way to see it.
  window.studio = {
    timeline, assets, onModelChange,
    importFiles, addAssetAt, setBinOpen,
    renderCompAudio, mixCompAudio, audioEntries, audioChains,
  };
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

let scrubUntil = 0;   // paused setTime marks a short scrub window

function setTime(t) {
  tCur = clamp(quantize(t, comp.fps), 0, lastFrame(comp));
  if (playing) clock = { perf: performance.now(), t: tCur };
  else scrubUntil = performance.now() + 150;
  timeline?.updatePlayhead();
}

const isScrubbing = () => !playing && performance.now() < scrubUntil;

function togglePlay() { playing ? pause() : play(); }

function play() {
  ensureAudio();
  if (tCur >= lastFrame(comp) - 1e-6) tCur = 0;
  playing = true;
  clock = { perf: performance.now(), t: tCur };
}

function pause() {
  playing = false;
  tCur = clamp(quantize(tCur, comp.fps), 0, lastFrame(comp));
}

function tick() {
  requestAnimationFrame(tick);
  if (!fx?.inputTexture) return;
  if (offlineJob) return;   // the offline render loop owns the pipeline

  if (playing) {
    tCur = clock.t + (performance.now() - clock.perf) / 1000;
    if (tCur >= comp.dur) {
      if (exportMode) {
        finishExport();
        tCur = lastFrame(comp);
        pause();
      } else if (looping) {
        tCur = 0;
        clock = { perf: performance.now(), t: 0 };
      } else {
        tCur = lastFrame(comp);
        pause();
      }
    }
  }

  // While a trim handle is being dragged, render the frame at the cut
  // point instead of the playhead (the playhead UI itself stays put).
  const t = trimPreviewT ?? tCur;
  const activeMedia = activeClips(comp, t, 'media').filter(({ track }) => !track.hidden);
  const activeAudio = activeClips(comp, t, 'audio').filter(({ track }) => !track.hidden);
  syncMedia(t, activeMedia, activeAudio);
  prepareMasks(t);       // media masks must compose before compositeFrame samples them
  prepareMediaFx(t, activeMedia);   // per-clip effect stacks render in isolation
  compositeFrame(t);
  syncFxChain(t);
  applyParams(t);
  fx.render(null, t);
  timeline.updatePlayhead();
  updateInspectorLive();
  updateGizmo();
}

/* ---- property drivers ------------------------------------------------
 * Oscillator + audio-reactive modulation of any keyframable property
 * (model + math in driver.js). Audio drivers read per-frame band
 * envelopes precomputed by audio-analysis.js from an offline mixdown of
 * EVERY video clip's audio — including muted/hidden tracks, so a muted
 * "beat track" can drive visuals without being audible. The analysis is
 * keyed on the clips' audio arrangement and recomputed in the background
 * when it drifts; until it lands, audio drivers read 0. */

const DRIVE_SR = 22050;   // analysis-only sample rate (bands top out ~11 kHz)

const audioDrive = {
  key: '',           // arrangement fingerprint the current data was built for
  data: null,        // { fps, frames, bands } from analyzeMix
  beats: new Map(),  // `${band}|${sensitivity}` -> ascending beat times
};
let audioDriveJob = null;

/** Clips whose audio participates in the mix — video clips and audio
 * clips alike. Drivers analyze all of them; the audible export path
 * excludes muted/hidden tracks. */
function audioEntries(includeMutedHidden) {
  const out = [];
  for (const track of comp.tracks) {
    if (!includeMutedHidden && (track.hidden || track.muted)) continue;
    for (const clip of track.clips) {
      if (!hasSource(clip) || clip.start >= comp.dur) continue;
      const asset = assets.get(clip.assetId);
      if (asset?.ready && (asset.kind === 'video' || asset.kind === 'audio'))
        out.push({ clip, asset });
    }
  }
  return out;
}

function audioDriveKey() {
  const parts = [comp.dur, comp.fps];
  for (const { clip, asset } of audioEntries(true)) {
    const vol = clip.props.volume;
    parts.push(asset.id, clip.start, clip.dur, clip.in,
      vol ? JSON.stringify({ v: vol.v, anim: vol.anim, keys: vol.keys }) : '',
      // Effects colour the mix the analysis hears, so they belong in its
      // fingerprint: adding a reverb has to re-derive the envelopes.
      JSON.stringify(audioEffectsOf(clip).map((e) =>
        [e.audioId, e.enabled !== false, e.params])));
  }
  return parts.join('|');
}

function compHasAudioDrivers() {
  let found = false;
  for (const track of comp.tracks)
    for (const clip of track.clips)
      eachClipProp(clip, (p) => {
        if (p?.driver?.enabled && p.driver.source === 'audio') found = true;
      });
  return found;
}

/** (Re)build the band envelopes when any audio driver needs them and the
 * comp's audio arrangement changed. Fire-and-forget from edits; awaited
 * by the offline renderer so exports never race the analysis. */
function syncAudioDrive() {
  if (!compHasAudioDrivers()) return audioDriveJob;
  const key = audioDriveKey();
  if (key === audioDrive.key) return audioDriveJob;
  audioDrive.key = key;   // claim first so concurrent calls dedupe
  setStatus('analyzing audio for drivers…');
  audioDriveJob = (async () => {
    // Analysis mixes with BASE values (evalProp, not drivenEval): a driver
    // on Volume — or on a filter cutoff — must not feed back into its own
    // input signal.
    const mix = await mixCompAudio(DRIVE_SR, audioEntries(true), 1, { driven: false });
    if (audioDrive.key !== key) return;   // superseded by a newer edit
    audioDrive.data = mix ? analyzeMix(mix.getChannelData(0), DRIVE_SR, comp.fps) : null;
    audioDrive.beats.clear();
    setStatus(mix ? 'audio analysis ready — drivers are live'
      : 'no audio in the comp — audio drivers read 0');
  })().catch((e) => console.warn('slangfx: audio analysis failed:', e));
  return audioDriveJob;
}

/** Resolve an audio driver to 0..1 at COMP time t. */
function audioSignal(d, t) {
  const data = audioDrive.data;
  if (!data) return 0;
  const env = data.bands[d.band] ?? data.bands.level;
  if (d.follow === 'beat') {
    const sens = +d.sensitivity || 1.5;
    const key = `${d.band}|${sens.toFixed(2)}`;
    let beats = audioDrive.beats.get(key);
    if (!beats) {
      beats = detectBeats(env, data.fps, sens);
      audioDrive.beats.set(key, beats);
    }
    return samplePulse(beats, t, +d.decay || 0.35);
  }
  return sampleLevel(env, data.fps, t, d.release == null ? 0.25 : +d.release || 0);
}

/** evalProp + the prop's driver (if enabled). tc is clip-relative (the
 * oscillator time base), tComp absolute (the audio timeline). */
function drivenEval(prop, tc, tComp) {
  const base = evalProp(prop, tc);
  const d = prop?.driver;
  if (!d?.enabled) return base;
  return applyDriver(base, d, tc, (drv) => audioSignal(drv, tComp));
}

/* ---- media sync ---------------------------------------------------- */

function syncMedia(t, activeMedia, activeAudio = []) {
  const used = new Set();
  for (const { track, clip } of [...activeMedia, ...activeAudio]) {
    const asset = assets.get(clip.assetId);
    if (!asset?.ready) continue;
    used.add(asset.id);
    if (asset.kind === 'gif') {
      syncGifFrame(asset, clip, t);
      continue;
    }
    if (asset.kind !== 'video' && asset.kind !== 'audio') continue;
    const el = asset.el;
    // A timed asset always builds its element before it reports ready, but
    // this runs every frame: one missing element must not become a storm
    // of exceptions that takes the whole render loop down.
    if (!el) continue;
    // Audio: master prefs × track mute × the clip's animated Volume. With
    // the Web Audio mixer up, level and effects live in the clip's chain
    // and the element runs wide open; without it, fall back to the
    // element's own volume (no effects, but still audible).
    el.muted = !!audioState.muted || !!track.muted;
    const clipVol = clip.props.volume
      ? clamp(drivenEval(clip.props.volume, t - clip.start, t) / 100, 0, 1) : 1;
    const chain = asset.audioNode ? audioChainFor(clip) : null;
    if (chain) {
      el.volume = 1;
      routeAssetToChain(asset, chain);
      applyAudioChain(chain, clip, t, clipVol);
    } else {
      el.volume = clamp((audioState.volume ?? 1) * clipVol, 0, 1);
    }
    // Source time, wrapped so clips longer than their source loop.
    const src = clip.in + (t - clip.start);
    const len = asset.duration ?? 0;
    const desired = len > 0.02 ? ((src % len) + len) % len : 0;
    let proxyScrub = false;
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
    } else if (asset.kind === 'audio') {
      // Nothing to show for sound: park the element on the scrub position
      // so hitting play starts from the right sample.
      if (!el.paused) el.pause();
      if (Math.abs(el.currentTime - desired) > 0.5 / comp.fps) el.currentTime = desired;
    } else {
      if (!el.paused) el.pause();
      if (!asset._seekedHook) {
        asset._seekedHook = true;
        // Upload the moment a seek lands (the per-tick poll below can be a
        // frame late) and clear the latest-wins target.
        el.addEventListener('seeked', () => {
          asset._seekTarget = null;
          if (asset.ready && el.readyState >= 2) uploadVideoFrame(asset);
        });
      }
      const scrubbing = isScrubbing();
      const proxyOk = scrubbing && proxiesEnabled;
      if (proxyOk) ensureScrubProxy(asset);
      if (proxyOk && asset.proxyEl?.readyState >= 2) {
        // Scrub against the all-intra proxy — its seeks land in
        // milliseconds. The full-res element stays put; once the scrub
        // settles the branch below re-seeks it and its 'seeked' upload
        // sharpens the frame.
        proxyScrub = true;
        scrubProxyTo(asset, desired);
      } else if (Math.abs(el.currentTime - desired) > 0.5 / comp.fps) {
        // Latest-wins: while scrubbing, retargeting mid-seek cancels the
        // stale seek instead of queueing behind it.
        const tgt = asset._seekTarget;
        if ((scrubbing || !el.seeking) && (tgt == null || Math.abs(tgt - desired) > 0.5 / comp.fps)) {
          asset._seekTarget = desired;
          el.currentTime = desired;
        }
      }
    }
    if (asset.kind === 'video' && !proxyScrub && el.readyState >= 2) uploadVideoFrame(asset);
  }
  for (const asset of assets.values())
    if ((asset.kind === 'video' || asset.kind === 'audio')
      && asset.ready && !used.has(asset.id) && asset.el && !asset.el.paused)
      asset.el.pause();
}

/* A GIF is a pre-decoded frame strip, not a <video>: pick the loop frame
 * for the clip's local time, upload only when it changes. */
function syncGifFrame(asset, clip, t) {
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
}

/* Firefox's WebGPU rejects HTMLVideoElement as a copyExternalImageToTexture
 * source (TypeError; only bitmaps/canvases are accepted), so after the first
 * rejection all video uploads reroute through a 2D scratch canvas. */
let videoNeedsCanvasHop = false;

function uploadVideoFrame(asset, sourceEl = asset.el, scale = false) {
  if (!videoNeedsCanvasHop && !scale) {
    try {
      fx.device.queue.copyExternalImageToTexture(
        { source: sourceEl }, { texture: asset.texture }, [asset.w, asset.h]);
      return;
    } catch (e) {
      if (!(e instanceof TypeError)) return;   // frame not ready
      videoNeedsCanvasHop = true;
    }
  }
  // Canvas hop: Firefox video uploads, and proxy frames that need scaling
  // up to the asset's full-res texture.
  const c = (asset.scratch ??= new OffscreenCanvas(asset.w, asset.h));
  const ctx2d = (asset.scratchCtx ??= c.getContext('2d'));
  ctx2d.drawImage(sourceEl, 0, 0, asset.w, asset.h);
  try {
    fx.device.queue.copyExternalImageToTexture(
      { source: c }, { texture: asset.texture }, [asset.w, asset.h]);
  } catch { /* frame not ready */ }
}

/* ---- scrub proxies ---------------------------------------------------
 * Seeking long-GOP video decodes forward from the previous keyframe —
 * often dozens of frames — so full-res scrubbing lags. Each video gets a
 * low-res ALL-INTRA VP8 proxy (every frame a keyframe → seeks land in
 * milliseconds), built once with WebCodecs at ~3x real time and cached in
 * IndexedDB. While the playhead scrubs, the proxy feeds the asset texture;
 * when it settles, the full-res seek sharpens the frame. */

const PROXY_H = 720;
const proxyKey = (asset) => `proxy:${PROXY_H}:${asset.id}`;   // height-versioned
const PROXIES_KEY = 'lowkey-studio.scrub-proxies';
let proxiesEnabled = localStorage.getItem(PROXIES_KEY) !== '0';
let proxyQueue = Promise.resolve();   // builds run one at a time

function setProxiesEnabled(on) {
  proxiesEnabled = on;
  try { localStorage.setItem(PROXIES_KEY, on ? '1' : '0'); } catch {}
  setStatus(on
    ? 'scrub proxies on — fast scrubbing, full-res sharpens on release'
    : 'scrub proxies off — scrubbing shows full-res frames directly');
}

function ensureScrubProxy(asset) {
  if (asset.kind !== 'video' || !asset.ready || asset.proxyState) return;
  if ((asset.duration ?? 0) < 1) { asset.proxyState = 'unavailable'; return; }
  if (typeof VideoEncoder === 'undefined'
      || !('requestVideoFrameCallback' in HTMLVideoElement.prototype)) {
    asset.proxyState = 'unavailable';
    return;
  }
  if (document.hidden) return;   // rVFC stalls hidden — retry on a later scrub
  asset.proxyState = 'building';
  proxyQueue = proxyQueue.then(async () => {
    try {
      let blob = await idbGet(proxyKey(asset)).catch(() => null);
      if (!blob) {
        blob = await transcodeScrubProxy(asset);
        idbSet(proxyKey(asset), blob).catch(() => {});
      }
      attachScrubProxy(asset, blob);
    } catch (e) {
      console.warn(`slangfx: scrub proxy failed for '${asset.name}':`, e);
      asset.proxyState = 'unavailable';
    }
  });
}

function attachScrubProxy(asset, blob) {
  const el = document.createElement('video');
  el.muted = true;
  el.preload = 'auto';
  el.src = URL.createObjectURL(blob);
  el.addEventListener('seeked', () => {
    if (asset.ready && el.readyState >= 2 && isScrubbing())
      uploadVideoFrame(asset, el, true);
  });
  $('media-pool').appendChild(el);
  asset.proxyEl = el;
  asset.proxyState = 'ready';
}

function scrubProxyTo(asset, desired) {
  const p = asset.proxyEl;
  const d = Math.min(desired, Number.isFinite(p.duration) ? p.duration : desired);
  if (asset._proxyTarget != null && Math.abs(asset._proxyTarget - d) < 1e-3) return;
  asset._proxyTarget = d;
  p.currentTime = d;   // all-intra: retargeting mid-seek is nearly free
}

async function transcodeScrubProxy(asset) {
  const src = document.createElement('video');
  src.muted = true;
  src.preload = 'auto';
  src.src = asset.url;
  await new Promise((res, rej) => {
    const timer = setTimeout(() => rej(new Error('metadata timeout')), 12_000);
    src.onloadedmetadata = () => { clearTimeout(timer); res(); };
    src.onerror = () => { clearTimeout(timer); rej(new Error('open failed')); };
  });
  const scale = Math.min(1, PROXY_H / src.videoHeight);
  const w = Math.max(2, Math.round((src.videoWidth * scale) / 2) * 2);
  const h = Math.max(2, Math.round((src.videoHeight * scale) / 2) * 2);
  const muxer = new WebMMuxer({
    target: new ArrayBufferTarget(),
    video: { codec: 'V_VP8', width: w, height: h },
  });
  let encError = null;
  const enc = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => { encError = e; },
  });
  enc.configure({ codec: 'vp8', width: w, height: h, bitrate: 8_000_000, latencyMode: 'realtime' });
  const cnv = new OffscreenCanvas(w, h);
  const c2 = cnv.getContext('2d');
  let last = -1;
  await new Promise((res, rej) => {
    // Watchdog so a wedged decode (or a window hidden mid-build, which
    // stops rVFC) can't hang the proxy queue forever.
    let watchdog = setTimeout(() => rej(new Error('transcode stalled')), 20_000);
    const kick = () => {
      clearTimeout(watchdog);
      watchdog = setTimeout(() => rej(new Error('transcode stalled')), 20_000);
    };
    const pump = (_now, meta) => {
      if (encError) return rej(encError);
      kick();
      if (meta.mediaTime > last + 1e-4 && enc.encodeQueueSize < 12) {
        last = meta.mediaTime;
        c2.drawImage(src, 0, 0, w, h);
        const frame = new VideoFrame(cnv, { timestamp: Math.round(meta.mediaTime * 1e6) });
        enc.encode(frame, { keyFrame: true });   // all-intra
        frame.close();
      }
      if (!src.ended) src.requestVideoFrameCallback(pump);
    };
    src.onended = () => { clearTimeout(watchdog); res(); };
    src.onerror = () => { clearTimeout(watchdog); rej(new Error('decode error')); };
    src.requestVideoFrameCallback(pump);
    src.playbackRate = 3;   // dropped frames just thin the proxy slightly
    src.play().catch(rej);
  }).finally(() => {
    src.pause();
    src.removeAttribute('src');
    src.load();
    src.remove();
  });
  if (encError) throw encError;
  await enc.flush();
  enc.close();
  muxer.finalize();
  return new Blob([muxer.target.buffer], { type: 'video/webm' });
}

/** The clip's raw draw: its source texture under its animated transform.
 * This is the geometric truth (gizmo, bounds, matte sources, and the
 * isolate pass that feeds the clip's own effects all want it). */
function drawForClip(clip, t) {
  const asset = assets.get(clip.assetId);
  if (!asset?.ready) return null;
  const tc = t - clip.start;
  const mm = mediaMaskTargets.get(clip.id);
  const maskState = clipMasks.get(clip.id);
  const masked = mm?.view && maskState?.nodes?.length;
  return {
    clipId: clip.id,
    view: asset.view,
    w: asset.w,
    h: asset.h,
    x: drivenEval(clip.props.x, tc, t),
    y: drivenEval(clip.props.y, tc, t),
    // Negative scale mirrors the media on that axis.
    scaleX: drivenEval(clip.props.scaleX, tc, t) / 100,
    scaleY: drivenEval(clip.props.scaleY, tc, t) / 100,
    rot: drivenEval(clip.props.rot, tc, t),
    opacity: clamp(drivenEval(clip.props.opacity, tc, t) / 100, 0, 1),
    blend: clip.blend ?? 'normal',
    maskView: masked ? mm.view : null,
    maskOpacity: masked ? maskState.opacity ?? 1 : 0,
    maskInvert: masked ? !!maskState.invert : false,
  };
}

/** What actually composites for a media clip: the raw draw, or — when the
 * clip carries its own effect stack — the stack's output as a full-frame
 * quad (the transform already happened in the isolate pass), still wearing
 * the clip's opacity, blend mode and mask. */
function compositeDrawForClip(clip, t) {
  const d = drawForClip(clip, t);
  if (!d) return null;
  const processed = mediaFxViews.get(clip.id);
  if (!processed) return d;
  return {
    ...d,
    clipId: `${clip.id}:fxout`,
    view: processed.view,
    covView: processed.covView,
    // The isolate composited the clip over transparent black, so the
    // chain's colour is premultiplied; the compositor undoes that against
    // the matte's coverage rather than multiplying by it a second time.
    premultiplied: true,
    w: comp.width, h: comp.height,
    x: comp.width / 2, y: comp.height / 2,
    scaleX: 1, scaleY: 1, rot: 0,
  };
}

/* Axis-aligned bounding box of every visible media clip's transformed
 * quad at comp time t, or null when nothing is on screen. */
/** AABB of one clip's transformed quad at comp time t, or null. */
function clipBounds(clip, t) {
  const d = drawForClip(clip, t);
  if (!d) return null;
  const hw = (d.w * Math.abs(d.scaleX)) / 2;
  const hh = (d.h * Math.abs(d.scaleY)) / 2;
  const r = (d.rot * Math.PI) / 180;
  const c = Math.cos(r), s = Math.sin(r);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [px, py] of [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]) {
    const x = d.x + px * c - py * s;
    const y = d.y + px * s + py * c;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  return { minX, minY, maxX, maxY };
}

function contentBounds(t) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const { track, clip } of activeClips(comp, t, 'media')) {
    if (track.hidden) continue;
    const b = clipBounds(clip, t);
    if (!b) continue;
    minX = Math.min(minX, b.minX); maxX = Math.max(maxX, b.maxX);
    minY = Math.min(minY, b.minY); maxY = Math.max(maxY, b.maxY);
  }
  return minX === Infinity ? null : { minX, minY, maxX, maxY };
}

/** Resize the comp to a bounding box and slide every clip so the box
 * lands centred in the new frame. The inverse of "fit in frame": instead
 * of moving the layer to suit the comp, the comp is cut to suit it. */
async function fitCompToBounds(b, label) {
  const bw = Math.max(1, b.maxX - b.minX);
  const bh = Math.max(1, b.maxY - b.minY);
  const W = clamp(2 * Math.round(bw / 2), 16, 7680);
  const H = clamp(2 * Math.round(bh / 2), 16, 4320);
  const dx = (W - bw) / 2 - b.minX;
  const dy = (H - bh) / 2 - b.minY;
  history.record(comp, () => {
    for (const track of comp.tracks)
      for (const c of track.clips) {
        if (c.kind !== 'media') continue;
        addToProp(c.props.x, dx);
        addToProp(c.props.y, dy);
      }
    comp.width = W;
    comp.height = H;
    comp._autoSize = false;
  });
  await applyCompSize();
  setTime(tCur);
  onModelChange({ structural: true });
  setStatus(`comp fit to ${label}: ${W}×${H}`);
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
    if (track.hidden || clip.kind === 'audio') continue;
    if (clip.kind === 'fx') {
      // Broken / still-compiling effects are skipped by the engine, so
      // media above them merges down to the previous working adjustment
      // layer. An empty (or wholly failed) stack is not a barrier.
      if (clip.enabled !== false && clipHasLiveLayers(clip.id)) curFx = clip.id;
      continue;
    }
    const d = compositeDrawForClip(clip, t);
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

/* ---- fx chain -------------------------------------------------------
 * An engine layer is one EFFECT. The layers a clip contributes carry that
 * clip's id as their engine `groupId`, which is what makes the engine treat
 * the stack as a single masked group (see SlangFx.rebuild). */

function specFor(clip, effect) {
  let spec = fxSpecs.get(effect.id);
  if (!spec) {
    spec = {
      clipId: clip.id, effectId: effect.id, groupId: clip.id,
      enabled: true, runtime: null, error: null, savedParams: null, label: effect.name,
    };
    if (effect.fxKind === 'custom') {
      const dir = `${CUSTOM_PREFIX}${customCounter++}/`;
      virtualFiles.set(dir + 'custom.slangp', CUSTOM_PRESET);
      virtualFiles.set(dir + 'custom.slang', effect.source ?? CUSTOM_BOILERPLATE);
      spec.dir = dir;
      spec.path = dir + 'custom.slangp';
    } else {
      spec.path = effect.path;
    }
    fxSpecs.set(effect.id, spec);
    restoreSpecExtras(clip, effect, spec);
  }
  spec.clipId = clip.id;      // a split/duplicate re-homes an effect
  spec.groupId = clip.id;
  spec.label = effect.name;
  return spec;
}

/** Rehydrate an effect's persisted overlay textures onto a fresh spec.
 * (Masks belong to the clip and load in loadClipMasks.) */
async function restoreSpecExtras(clip, effect, spec) {
  let dirty = false;
  for (const [name, o] of Object.entries(effect.overlay ?? {})) {
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
  if (dirty) markChainDirty(clip.id);
}

/** Enabled VISUAL effects of a clip, in application order — i.e. the ones
 * that become engine layers. Audio effects share the same stack but never
 * reach the GPU (see audioChainFor). */
function liveEffects(clip) {
  return visualEffectsOf(clip).filter((e) => e.enabled !== false);
}

/** Enabled audio effects of a clip, in signal-chain order. */
function liveAudioEffects(clip) {
  return audioEffectsOf(clip).filter((e) => e.enabled !== false);
}

/** Layer specs for one clip's stack. The FIRST spec carries the clip's mask
 * — the engine builds the group mask on the head and blends at the tail. */
function specsForStack(clip) {
  const effects = liveEffects(clip);
  const mask = clipMasks.get(clip.id) ?? null;
  return effects.map((effect, i) => {
    const spec = specFor(clip, effect);
    spec.maskState = i === 0 ? mask : null;
    return spec;
  });
}

function activeFxEntries(t) {
  return activeClips(comp, t, 'fx')
    .filter(({ track, clip }) => !track.hidden && clip.enabled !== false);
}

/** Identity of the built chain: which effects, in which order, under which
 * clip. Changing any of it forces a rebuild. */
function stackKey(clip) {
  return `${clip.id}[${liveEffects(clip).map((e) => e.id).join(',')}]`;
}

function syncFxChain(t) {
  if (chainBuilding) return chainPromise;
  const entries = activeFxEntries(t);
  const key = entries.map((e) => stackKey(e.clip)).join('|');
  if (key === chainKey && !chainDirty) return chainPromise;
  chainKey = key;
  chainDirty = false;
  chainBuilding = true;
  fx.layers = entries.flatMap(({ clip }) => specsForStack(clip));
  chainPromise = fx.rebuild()
    .catch((e) => console.error('slangfx: chain rebuild failed:', e))
    .finally(() => { chainBuilding = false; });
  return chainPromise;
}

/** Mark the chain that owns `clipId` for rebuild (the shared adjustment
 * chain, or that media clip's private one). Omit the id to dirty all. */
function markChainDirty(clipId = null) {
  if (clipId == null) {
    chainDirty = true;
    for (const entry of mediaChains.values()) entry.dirty = true;
    return;
  }
  const entry = mediaChains.get(clipId);
  if (entry) entry.dirty = true;
  else chainDirty = true;
}

/* ---- per-media effect chains ---------------------------------------- */

/** The chain for a media clip's own effects, created on first use. Returns
 * null until the (async) engine is ready, so callers fall back to the raw
 * clip for that frame. */
function newChain() {
  return SlangFx.create({
    device: fx.device,
    toolchain: fx.toolchain,
    readFile: fx.readFile,
    readImage: fx.readImage,
    moduleCache: fx.moduleCache,   // compile each shader once per device
  }).then(async (chain) => {
    await chain.setSourceSize(comp.width, comp.height);
    return chain;
  });
}

/** The chains for a media clip's own effects, created on first use. Returns
 * null until the (async) engine is ready, so callers fall back to the raw
 * clip for that frame.
 *
 * Two chains, same preset stack: `fx` carries the colour, `matte` carries
 * the coverage. They're separate engines rather than two passes through one
 * because a preset's feedback/history buffers must not interleave colour
 * and matte frames. See prepareMediaFx for why the matte exists at all. */
function mediaChainFor(clip) {
  let entry = mediaChains.get(clip.id);
  if (!entry) {
    entry = { fx: null, matte: null, key: '', dirty: true, building: true, promise: null };
    mediaChains.set(clip.id, entry);
    entry.promise = Promise.all([newChain(), newChain()]).then(([colour, matte]) => {
      entry.fx = colour;
      entry.matte = matte;
    }).catch((e) => {
      console.error('slangfx: media chain create failed:', e);
    }).finally(() => { entry.building = false; });
  }
  return entry.fx && entry.matte ? entry : null;
}

function destroyMediaChain(clipId) {
  const entry = mediaChains.get(clipId);
  if (!entry) return;
  mediaChains.delete(clipId);
  mediaFxViews.delete(clipId);
  Promise.resolve(entry.promise).finally(() => {
    // The engine owns these runtimes; null the specs' pointers so a spec
    // that outlives the chain can't hand out a destroyed runtime.
    for (const layer of entry.fx?.layers ?? []) layer.runtime = null;
    entry.fx?.destroy();
    entry.matte?.destroy();
    compositor.release(`${clipId}:iso`);
    compositor.release(`${clipId}:matte`);
    compositor.release(`${clipId}:fxout`);
  });
}

/** Resize every private chain with the comp (see applyCompSize). Forcing
 * `key` empty makes the next sync re-lay the stack over the new textures. */
function resizeMediaChains() {
  for (const entry of mediaChains.values()) {
    if (!entry.fx) continue;
    entry.key = '';
    entry.building = true;
    entry.promise = Promise.all([
      entry.fx.setSourceSize(comp.width, comp.height),
      entry.matte?.setSourceSize(comp.width, comp.height),
    ])
      .catch((e) => console.error('slangfx: media chain resize failed:', e))
      .finally(() => { entry.building = false; });
  }
}

/** Drop chains for clips that were deleted or lost their effects. */
function gcMediaChains() {
  for (const clipId of [...mediaChains.keys()]) {
    const clip = findClip(comp, clipId)?.clip;
    if (!clip || clip.kind !== 'media' || !liveEffects(clip).length)
      destroyMediaChain(clipId);
  }
}

/** Release per-effect runtime state (compiled layer, param metadata,
 * editor draft) for effects that no longer exist in the comp. Nulling
 * `runtime` is what makes this safe mid-frame: the engine skips layers
 * without one, so a stale chain built before the edit renders nothing for
 * the removed effect instead of sampling freed textures. */
function gcEffectState() {
  const live = new Set();
  for (const track of comp.tracks)
    for (const clip of track.clips)
      for (const effect of effectsOf(clip)) live.add(effect.id);
  for (const [id, spec] of fxSpecs) {
    if (live.has(id)) continue;
    spec.runtime?.destroy();
    spec.runtime = null;
    fxSpecs.delete(id);
  }
  for (const id of [...paramMetaCache.keys()]) if (!live.has(id)) paramMetaCache.delete(id);
  for (const id of [...editorDrafts.keys()]) if (!live.has(id)) editorDrafts.delete(id);
  for (const id of [...openEffects]) if (!live.has(id)) openEffects.delete(id);
}

function syncMediaChain(clip, entry) {
  if (entry.building) return entry.promise;
  const key = stackKey(clip);
  if (key === entry.key && !entry.dirty) return entry.promise;
  entry.key = key;
  entry.dirty = false;
  entry.building = true;
  // No group mask here: a media clip's mask cuts its alpha at composite
  // time (green screen), which already covers the effected result.
  const effects = liveEffects(clip);
  entry.fx.layers = effects.map((effect) => {
    const spec = specFor(clip, effect);
    spec.maskState = null;
    return spec;
  });
  // The matte chain runs the same presets over its own runtimes. Distinct
  // spec objects: the engine writes layer.runtime into them, and the two
  // chains must not fight over that pointer.
  entry.matte.layers = effects.map((effect) => ({
    ...specFor(clip, effect), maskState: null, runtime: null,
  }));
  entry.promise = Promise.all([entry.fx.rebuild(), entry.matte.rebuild()])
    .catch((e) => console.error('slangfx: media chain rebuild failed:', e))
    .finally(() => { entry.building = false; });
  return entry.promise;
}

/**
 * Render every effected media clip in isolation. For each: composite the
 * clip alone (its transform, full opacity, no mask, transparent
 * background) into its chain's input, run the stack, and remember the
 * processed view for compositeFrame.
 *
 * Coverage is the subtle part. A slang preset is written for an opaque
 * framebuffer and signs off with `FragColor = vec4(rgb, 1.0)`, so the
 * alpha coming out of a chain says nothing about where the effect
 * actually landed. Guessing costs us either way: assume the clip's
 * original silhouette and an effect can never grow (a blur has nothing
 * to soften into); trust the shader's alpha and every effect covers the
 * whole comp.
 *
 * So we don't guess. The same stack runs a second time over a white
 * silhouette, and whatever it does to those pixels is what it did to the
 * coverage — a blur blurs it, a warp warps it, a colour grade leaves it
 * alone. That output is the matte, and it costs one extra chain run per
 * effected media clip.
 */
const mediaFxViews = new Map();   // clipId -> {view, covView}

function prepareMediaFx(t, activeMedia) {
  mediaFxViews.clear();
  for (const { clip } of activeMedia) {
    if (!liveEffects(clip).length) continue;
    const entry = mediaChainFor(clip);
    if (!entry) continue;
    const d = drawForClip(clip, t);
    if (!d) continue;
    syncMediaChain(clip, entry);
    const chain = entry.fx;
    const matte = entry.matte;
    if (!chain.inputView || !matte.inputView) continue;

    const encoder = fx.device.createCommandEncoder();
    // ':iso' / ':matte' key their own compositor items so these draws
    // don't fight the clip's on-screen draw over one uniform buffer.
    const iso = {
      ...d, opacity: 1, blend: 'normal',
      maskView: null, covView: null,
    };
    compositor.composite(encoder, chain.inputView, comp.width, comp.height,
      [{ ...iso, clipId: `${clip.id}:iso` }], { transparent: true });
    compositor.composite(encoder, matte.inputView, comp.width, comp.height,
      [{ ...iso, clipId: `${clip.id}:matte`, matte: true }], { transparent: true });
    fx.device.queue.submit([encoder.finish()]);

    applyParamsFor(chain, t);
    chain.render(null, t);
    applyParamsFor(matte, t);
    matte.render(null, t);

    const view = chain.finalView;
    if (view === chain.inputView) continue;   // nothing built yet
    mediaFxViews.set(clip.id, {
      view,
      covView: matte.finalView,   // where the effect actually reached
    });
  }
}

function applyParamsFor(engine, t) {
  for (const layer of engine.layers) {
    const rt = layer.runtime;
    if (!rt) continue;
    const hit = findClip(comp, layer.clipId);
    const effect = hit && findEffect(hit.clip, layer.effectId);
    if (!effect) continue;
    const tc = t - hit.clip.start;
    for (const meta of rt.paramMeta) {
      const prop = effect.params?.[meta.name];
      let v = prop ? drivenEval(prop, tc, t) : meta.default;
      if (meta.max > meta.min) v = clamp(v, meta.min, meta.max);
      rt.paramValues.set(meta.name, v);
    }
  }
}

function applyParams(t) {
  applyParamsFor(fx, t);
}

/** True when a clip currently contributes at least one working layer. */
function clipHasLiveLayers(clipId) {
  return fx.layers.some((l) => l.clipId === clipId && l.runtime);
}

/** Shader parameter metadata for one effect without needing an active
 * chain — parses the preset and compiles its modules (cached), so
 * custom-shader compile errors also surface here. */
async function ensureParamMeta(clip, effect) {
  const cached = paramMetaCache.get(effect.id);
  if (cached) return cached === 'loading' ? null : cached;
  paramMetaCache.set(effect.id, 'loading');
  const spec = specFor(clip, effect);
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
    paramMetaCache.set(effect.id, metas);
    spec.lastCompileError = null;
    timeline.render();
    renderInspector();
    return metas;
  } catch (e) {
    paramMetaCache.delete(effect.id);
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

/** An audio clip's only intrinsic property (everything else it can do is
 * an audio effect). */
function audioClipPropDefs() {
  return [{ key: 'volume', label: 'Volume', min: 0, max: 100, step: 0.1, unit: '%', def: 100 }];
}

/**
 * Shader authors prefix parameter descriptions with the preset's own name
 * ("Exposure: mix", "MotionTrails: response curve") because in RetroArch
 * every parameter of every pass lands in one flat list. Here they're
 * already inside that effect's card, so the prefix is dead weight — and
 * it's the part that pushes the actual name out of a narrow panel.
 *
 * Only strips when something readable is left: "Saturation" on the
 * saturation effect keeps its name, and "Exposure (stops)" keeps its,
 * since "(stops)" alone says nothing.
 */
function trimParamLabel(label, effectName) {
  const words = String(effectName || '').toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  let rest = String(label).trim();
  for (const w of words) {
    const m = rest.match(new RegExp(`^${w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}[\\s:_-]*`, 'i'));
    if (!m) break;
    const next = rest.slice(m[0].length);
    // Keep going only while the remainder still reads as a name.
    if (!/^[a-z0-9]/i.test(next)) break;
    rest = next;
  }
  if (rest === String(label).trim()) return label;
  return rest.charAt(0).toUpperCase() + rest.slice(1);
}

/** Parameter defs for one effect, keyed in the clip-wide namespace. Returns
 * [] (and kicks off the compile) while the metadata is still loading. */
function effectPropDefs(clip, effect) {
  // Audio effects declare their parameters up front — nothing to compile,
  // so they're never in the "loading" state a shader can be.
  if (isAudioEffect(effect)) {
    const def = audioEffectDef(effect.audioId);
    return (def?.params ?? []).map((p) => ({
      key: effectPropKey(effect.id, p.name),
      effectId: effect.id,
      effectName: effect.name,
      label: p.label,
      min: p.min,
      max: p.max,
      step: p.step,
      unit: p.unit,
      def: p.def,
      // Frequencies and times get a logarithmic slider; the value itself
      // is unchanged, only the travel that reaches it.
      scale: p.scale ?? null,
    }));
  }
  const metas = paramMetaCache.get(effect.id);
  if (!metas || metas === 'loading') {
    if (!metas) ensureParamMeta(clip, effect).catch(() => {});
    return [];
  }
  return metas.map((m) => {
    const full = m.desc || m.name;
    return {
      key: effectPropKey(effect.id, m.name),
      effectId: effect.id,
      effectName: effect.name,
      label: trimParamLabel(full, effect.name),
      fullLabel: full,
      min: m.min,
      max: m.max,
      step: m.step || 0.001,
      unit: '',
      def: m.default,
    };
  });
}

/** Every editable property of a clip: its transform (media) followed by the
 * parameters of each effect in its stack, in stack order. */
function propDefs(clip) {
  const defs = [];
  if (clip.kind === 'media') {
    // Volume only makes sense (and sound) for video assets.
    const video = assets.get(clip.assetId)?.kind === 'video';
    for (const d of mediaPropDefs()) if (video || d.key !== 'volume') defs.push(d);
  } else if (clip.kind === 'audio') {
    defs.push(...audioClipPropDefs());
  }
  for (const effect of effectsOf(clip)) defs.push(...effectPropDefs(clip, effect));
  return defs;
}

function defFor(clip, key) {
  return propDefs(clip).find((d) => d.key === key) ?? { def: 0, min: 0, max: 0, step: 0.001 };
}

function getProp(clip, key) {
  const { effectId, name } = parsePropKey(key);
  if (!effectId) return clip.props?.[name] ?? null;
  return findEffect(clip, effectId)?.params?.[name] ?? null;
}

function getOrCreateProp(clip, key) {
  const { effectId, name } = parsePropKey(key);
  if (!effectId) return clip.props[name];
  const effect = findEffect(clip, effectId);
  if (!effect) return null;
  return ((effect.params ??= {})[name] ??= newProp(defFor(clip, key).def));
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
  if (!prop) return;
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
    if (!prop) return;
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
    if (!prop) return;
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
  markChainDirty();
  if (transient) return;
  removeEmptyTracks(comp);   // e.g. the last clip was dragged off a track
  gcEffectState();           // deleted effects release their compiled state
  gcMediaChains();           // clips that lost their effects release theirs
  gcAudioChains();           // …and their audio graphs
  reconcileShapeAssets();    // duplicated/split shape clips get their own asset
  syncAudioDrive();          // no-op unless audio drivers exist + audio changed
  refreshDropHint();
  timeline.render();
  renderInspector();
  renderMediaBin();          // clip counts on the cards track the model
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
  reconcileShapeAssets();   // undo/redo may restore stale shape settings
  gcEffectState();
  gcMediaChains();
  gcAudioChains();
  chainKey = '';
  markChainDirty();
  tCur = clamp(tCur, 0, lastFrame(comp));
  // The restored state may carry a different comp size than the canvas.
  if (canvas.width !== comp.width || canvas.height !== comp.height)
    applyCompSize();
  refreshDropHint();
  timeline.render();
  renderInspector();
  renderMediaBin();
  scheduleSave();
  setStatus(what);
}

/** After undo/redo the clip's source text may differ from the virtual
 * file that the engine compiles — resync + invalidate. */
function syncCustomSources() {
  for (const track of comp.tracks)
    for (const clip of track.clips)
      for (const effect of effectsOf(clip)) {
        if (effect.fxKind !== 'custom') continue;
        const spec = fxSpecs.get(effect.id);
        if (!spec) continue;
        const cur = virtualFiles.get(spec.dir + 'custom.slang');
        if (cur !== effect.source) {
          virtualFiles.set(spec.dir + 'custom.slang', effect.source);
          fx.invalidateModules(spec.dir);
          paramMetaCache.delete(effect.id);
          markChainDirty(clip.id);
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
  onSelect: () => { focusIsFallback = false; renderInspector(); },
  addMediaAt: (files, t, trackIdx) => importFiles(files, { t, trackIdx }),
  addAssetAt: (assetId, t, trackIdx) => addAssetAt(assetId, t, trackIdx),
  setTrimPreview: (t) => {
    trimPreviewT = t;
    const badge = $('trim-badge');
    if (badge) badge.hidden = t == null;
  },
  status: setStatus,
  undo: appUndo,
  redo: appRedo,
  addLayerMenu: (anchor) => showAddLayerMenu(anchor),
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
  const isAudio = !isGif && (file.type.startsWith('audio/') || AUDIO_EXTS.test(file.name));
  const isVideo = !isGif && !isAudio
    && (file.type.startsWith('video/') || VIDEO_EXTS.test(file.name));
  const asset = {
    id: id ?? uid('asset'),
    kind: isVideo ? 'video' : isAudio ? 'audio' : isGif ? 'gif' : 'image',
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

  if (isVideo || isAudio) {
    // Sound files ride the same <video>-shaped path as video (play, seek,
    // loop, drift correction) — only with an <audio> element and no frames.
    const el = document.createElement(isAudio ? 'audio' : 'video');
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
    asset.w = el.videoWidth ?? 0;
    asset.h = el.videoHeight ?? 0;
    asset.duration = Number.isFinite(el.duration) ? el.duration : null;
    applyAudioPrefsTo(el);
    attachAudio(asset);
    if (isAudio) {
      // No texture, no frames — just peaks for the timeline, in the
      // background so a long track doesn't stall the import.
      asset.ready = true;
      buildWaveform(asset).catch(() => {});
      return asset;
    }
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

/** An already-imported asset that is plausibly the same file — the
 * re-import guard that keeps one source from piling up in the bin. */
function findExistingAsset(file) {
  for (const a of assets.values())
    if (a.kind !== 'shape' && a.name === file.name && a.file?.size === file.size)
      return a;
  return null;
}

/** The very first media item defines the comp size; anything after that
 * is scaled to fit inside the existing frame. */
async function maybeAdoptCompSize(asset) {
  const noMediaYet = !comp.tracks.some((tr) => tr.clips.some((c) => c.kind === 'media'));
  if (noMediaYet && asset.w) {
    comp.width = asset.w;
    comp.height = asset.h;
    await applyCompSize();
  }
}

/** Build a clip for `asset` and place it on a track (a new one when
 * trackIdx is null). Timed sources land at `at` with their source length.
 * Stills span the whole comp with `imageFull` (the import default) and
 * otherwise land at `at` with a starter duration, like a drag from the
 * bin onto a specific spot. Returns the new clip. */
function placeAssetClip(asset, at, trackIdx = null, { imageFull = false } = {}) {
  let clip;
  if (asset.kind === 'audio') {
    const dur = quantize(Math.max(asset.duration ?? DEFAULT_VIDEO_DUR, 1 / comp.fps), comp.fps);
    clip = newAudioClip(asset, quantize(at, comp.fps), dur);
  } else if (asset.kind === 'video') {
    const dur = quantize(Math.max(asset.duration ?? DEFAULT_VIDEO_DUR, 1 / comp.fps), comp.fps);
    clip = newMediaClip(comp, asset, quantize(at, comp.fps), dur);
  } else if (imageFull) {
    clip = newMediaClip(comp, asset, 0, Math.max(1 / comp.fps, comp.dur));
  } else {
    // Stills placed deliberately (bin drag) — animated GIFs keep their
    // loop length, plain images get the default starter duration.
    const dur = quantize(Math.max(asset.duration ?? DEFAULT_VIDEO_DUR, 1 / comp.fps), comp.fps);
    clip = newMediaClip(comp, asset, quantize(at, comp.fps), dur);
  }
  if (clip.kind === 'media' && asset.w && (asset.w !== comp.width || asset.h !== comp.height)) {
    const fit = Math.round(Math.min(comp.width / asset.w, comp.height / asset.h) * 10000) / 100;
    clip.props.scaleX.v = fit;
    clip.props.scaleY.v = fit;
  }

  let track = trackIdx != null ? comp.tracks[trackIdx] : null;
  if (!track) {
    track = newTrack(clip.name);
    // Audio has no place in the visual stacking order, so its tracks
    // gather at the bottom (DAW-style) instead of covering the picture.
    if (clip.kind === 'audio') comp.tracks.push(track);
    else comp.tracks.unshift(track);
  }
  track.clips.push(clip);
  if (comp._autoSize) comp.dur = Math.max(comp.dur, clipEnd(clip));
  return clip;
}

async function importFiles(files, { t = null, trackIdx = null, binOnly = false } = {}) {
  const media = [...files].filter((f) =>
    f.type.startsWith('video/') || f.type.startsWith('image/') || f.type.startsWith('audio/')
    || VIDEO_EXTS.test(f.name) || GIF_EXT.test(f.name) || AUDIO_EXTS.test(f.name));
  if (!media.length) return;
  let at = t ?? tCur;
  let reused = 0;
  history.begin(comp);
  for (const file of media) {
    // The same file imported again reuses the existing asset — new clips
    // keep pointing at one copy of the media instead of stacking duplicates.
    let asset = findExistingAsset(file);
    if (asset) {
      reused++;
    } else {
      setStatus(`importing ${file.name}…`);
      try {
        asset = await createAsset(file);
      } catch (e) {
        setStatus(`import failed: ${e.message}`);
        continue;
      }
      idbSet(`asset:${asset.id}`, file).catch(() => {});
      if (asset.kind === 'video') ensureScrubProxy(asset);   // background build
    }
    if (binOnly) continue;               // into the media bin, no clip

    await maybeAdoptCompSize(asset);
    // Videos and audio land at the playhead with their source length;
    // images fill the whole timeline by default (trim down as needed).
    const clip = placeAssetClip(asset, at, trackIdx, { imageFull: true });
    // Only timed sources advance the drop cursor (images span the full comp).
    if (asset.kind === 'video' || asset.kind === 'audio') at = clipEnd(clip);
    timeline.selectClip(clip.id);
  }
  ensureDur(comp);
  history.commit(comp);
  if (binOnly) {
    renderMediaBin();
    scheduleSave();                      // the bin's asset list rides the project payload
    setStatus(`added ${media.length} item${media.length > 1 ? 's' : ''} to the media bin`);
    return;
  }
  onModelChange({ structural: true });
  timeline.zoomFit();
  setStatus(`imported ${media.length} item${media.length > 1 ? 's' : ''}`
    + (reused ? ` (${reused} reused from the media bin)` : ''));
}

/** Place a clip for an already-imported asset — a media-bin drag onto the
 * timeline, or a double-click on a bin card (lands at the playhead). */
async function addAssetAt(assetId, t, trackIdx = null) {
  const asset = assets.get(assetId);
  if (!asset?.ready) {
    setStatus('that media is still loading — try again in a moment');
    return null;
  }
  history.begin(comp);
  await maybeAdoptCompSize(asset);
  const clip = placeAssetClip(asset, Math.max(0, t), trackIdx);
  ensureDur(comp);
  history.commit(comp);
  onModelChange({ structural: true });
  timeline.selectClip(clip.id);
  setStatus(`added ${clip.name}`);
  return clip;
}

$('file-input').addEventListener('change', (e) => {
  if (e.target.files.length) importFiles([...e.target.files]);
  e.target.value = '';
});

document.body.addEventListener('dragover', (e) => e.preventDefault());
document.body.addEventListener('drop', (e) => {
  e.preventDefault();
  // A bin card dropped on the preview (the timeline handles its own drops):
  // a new clip for that asset at the playhead, on a track of its own.
  const assetId = e.dataTransfer.getData('application/x-lowkey-asset');
  if (assetId) { addAssetAt(assetId, tCur, null); return; }
  if (e.dataTransfer.files.length) importFiles([...e.dataTransfer.files]);
});

/* =====================================================================
 * Media bin — one card per imported asset. A single import can back any
 * number of clips: drag a card onto the timeline (or double-click it) to
 * cut another clip from the same underlying media.
 * =================================================================== */

const BIN_OPEN_KEY = 'lowkey-studio.bin-open';
const binEl = $('media-bin');
const binListEl = $('media-bin-list');
const binBtn = $('btn-media-bin');

const binAssets = () => [...assets.values()].filter((a) => a.kind !== 'shape');

function assetUseCount(assetId) {
  let n = 0;
  for (const track of comp.tracks)
    for (const clip of track.clips)
      if (clip.assetId === assetId) n++;
  return n;
}

function setBinOpen(open) {
  binEl.hidden = !open;
  binBtn.classList.toggle('active', open);
  try { localStorage.setItem(BIN_OPEN_KEY, open ? '1' : ''); } catch {}
  if (open) renderMediaBin();
}

binBtn.addEventListener('click', () => setBinOpen(binEl.hidden));
if (localStorage.getItem(BIN_OPEN_KEY) === '1') setBinOpen(true);

// Files dropped (or picked) directly on the bin import WITHOUT touching the
// timeline — organize first, place clips later.
binEl.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); });
binEl.addEventListener('drop', (e) => {
  e.preventDefault();
  e.stopPropagation();
  if (e.dataTransfer.files.length) importFiles([...e.dataTransfer.files], { binOnly: true });
});
$('bin-file-input').addEventListener('change', (e) => {
  if (e.target.files.length) importFiles([...e.target.files], { binOnly: true });
  e.target.value = '';
});

function renderMediaBin() {
  if (binEl.hidden) return;
  binListEl.textContent = '';
  const list = binAssets();
  if (!list.length) {
    const empty = document.createElement('div');
    empty.className = 'bin-empty';
    empty.textContent = 'nothing imported yet — drop files here, or use Import media';
    binListEl.appendChild(empty);
    return;
  }
  for (const asset of list) binListEl.appendChild(binCard(asset));
}

function binCard(asset) {
  const card = document.createElement('div');
  card.className = 'bin-card' + (asset.ready ? '' : ' loading');
  card.draggable = !!asset.ready;
  card.dataset.assetId = asset.id;
  card.title = `${asset.name}\ndrag onto the timeline — double-click adds at the playhead`;

  const thumb = document.createElement('div');
  thumb.className = 'bin-thumb';
  paintBinThumb(asset, thumb);
  card.appendChild(thumb);

  const name = document.createElement('div');
  name.className = 'bin-name';
  name.textContent = asset.name;
  card.appendChild(name);

  const uses = assetUseCount(asset.id);
  const meta = document.createElement('div');
  meta.className = 'bin-meta';
  const bits = [asset.kind];
  if (asset.duration != null) bits.push(`${(+asset.duration.toFixed(1))}s`);
  else if (asset.w) bits.push(`${asset.w}×${asset.h}`);
  bits.push(uses ? `${uses} clip${uses > 1 ? 's' : ''}` : 'unused');
  meta.textContent = bits.join(' · ');
  card.appendChild(meta);

  const del = document.createElement('button');
  del.className = 'bin-del';
  del.textContent = '✕';
  del.title = 'remove from the project';
  del.addEventListener('click', (e) => {
    e.stopPropagation();
    requestRemoveAsset(asset, del);
  });
  card.appendChild(del);

  card.addEventListener('dragstart', (e) => {
    e.dataTransfer.setData('application/x-lowkey-asset', asset.id);
    e.dataTransfer.effectAllowed = 'copy';
  });
  card.addEventListener('dblclick', () => addAssetAt(asset.id, tCur, null));
  return card;
}

/** Fill a card's thumbnail box: stills and GIFs from their bitmap, videos
 * from the media element's current frame (waiting for first data if
 * needed), audio from its waveform peaks. Falls back to a kind glyph. */
function paintBinThumb(asset, holder) {
  if (asset.kind === 'audio') {
    const img = document.createElement('img');
    img.alt = '';
    if (asset.waveform) img.src = asset.waveform;
    else buildWaveform(asset).then((url) => { if (url) img.src = url; }).catch(() => {});
    holder.textContent = '';
    holder.appendChild(img);
    return;
  }
  const paint = () => {
    const src = asset.bitmap ?? asset.frames?.[0]?.bitmap
      ?? (asset.kind === 'video' && asset.el?.readyState >= 2 ? asset.el : null);
    if (!src || !asset.w) return false;
    const W = 200, H = 64;
    const cv = document.createElement('canvas');
    cv.width = W;
    cv.height = H;
    const ctx = cv.getContext('2d');
    const s = Math.min(W / asset.w, H / asset.h);
    ctx.drawImage(src, (W - asset.w * s) / 2, (H - asset.h * s) / 2, asset.w * s, asset.h * s);
    holder.textContent = '';
    holder.appendChild(cv);
    return true;
  };
  if (!paint()) {
    holder.textContent = asset.kind === 'video' ? '🎞' : '🖼';
    if (asset.kind === 'video' && asset.el)
      asset.el.addEventListener('loadeddata', paint, { once: true });
  }
}

function requestRemoveAsset(asset, anchor) {
  const uses = assetUseCount(asset.id);
  if (!uses) { removeAsset(asset); return; }
  const r = anchor.getBoundingClientRect();
  showMenu(r.left, r.bottom + 4, [
    {
      label: `Remove "${asset.name}" and its ${uses} clip${uses > 1 ? 's' : ''}`,
      danger: true,
      action: () => removeAsset(asset),
    },
    { label: 'Keep it', action: () => {} },
  ]);
}

/** Drop an asset from the project: its clips, its runtime handles, and its
 * stored blobs. Blobs are shared across saved projects by asset id, so
 * another project still referencing this media will report it missing on
 * load — same recovery as any lost blob: re-import. */
function removeAsset(asset) {
  history.begin(comp);
  for (const track of comp.tracks)
    track.clips = track.clips.filter((c) => c.assetId !== asset.id);
  history.commit(comp);
  disposeAsset(asset);
  assets.delete(asset.id);
  idbDelete(`asset:${asset.id}`).catch(() => {});
  idbDelete(proxyKey(asset)).catch(() => {});
  onModelChange({ structural: true });
  setStatus(`removed ${asset.name}`);
}

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

/** Master level rides the mixer's output gain when the graph is up; when
 * it isn't, syncMedia folds it into each element's own volume. */
function applyMasterVolume() {
  if (masterGain) masterGain.gain.value = audioState.volume ?? 1;
}

function updateAudioUI() {
  muteBtn.textContent = (audioState.muted || audioState.volume === 0) ? '🔇' : '🔊';
  volumeSlider.value = String(audioState.volume);
}

function setAudioPrefs({ muted, volume }) {
  if (muted != null) audioState.muted = muted;
  if (volume != null) audioState.volume = volume;
  applyMasterVolume();
  try { localStorage.setItem(AUDIO_KEY, JSON.stringify(audioState)); } catch {}
  updateAudioUI();
}

function ensureAudio() {
  if (!audioCtx) {
    try {
      audioCtx = new AudioContext();
      masterGain = audioCtx.createGain();
      masterGain.gain.value = audioState.volume ?? 1;
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
  if (!audioCtx || asset.audioNode) return;
  if (asset.kind !== 'video' && asset.kind !== 'audio') return;
  try {
    asset.audioNode = audioCtx.createMediaElementSource(asset.el);
    // Left unconnected on purpose: syncMedia routes it into the chain of
    // whichever clip is playing it (routeAssetToChain).
  } catch (e) {
    console.warn('slangfx: could not attach audio for', asset.name, e);
  }
}

/* ---- per-clip audio chains -------------------------------------------
 * source → [audio effects, in stack order] → volume fader → master.
 *
 * The chain belongs to the CLIP but the source belongs to the ASSET (one
 * media element per file, so two clips of the same file can never play at
 * once anyway) — hence the routing step: whichever clip is live re-points
 * the asset's source node at its own chain input.
 *
 * Rebuilds are keyed on the stack, so dragging a parameter never touches
 * the graph — only the AudioParams it drives. */

const audioChains = new Map();   // clipId -> chain entry

function audioStackKey(clip) {
  return liveAudioEffects(clip).map((e) => `${e.id}:${e.audioId}`).join(',');
}

/** Build (or reuse) the graph for one clip. Returns null when the mixer
 * never came up. */
function audioChainFor(clip) {
  if (!audioCtx) return null;
  const key = audioStackKey(clip);
  const existing = audioChains.get(clip.id);
  if (existing && existing.key === key) return existing;
  if (existing) destroyAudioChain(clip.id);

  const input = audioCtx.createGain();
  const gain = audioCtx.createGain();     // the clip's Volume, post-effects
  const units = [];
  let cur = input;
  for (const effect of liveAudioEffects(clip)) {
    const def = audioEffectDef(effect.audioId);
    if (!def) continue;
    let unit;
    try {
      unit = def.build(audioCtx);
    } catch (e) {
      console.warn(`slangfx: audio effect '${effect.audioId}' failed to build:`, e);
      continue;
    }
    cur.connect(unit.input);
    cur = unit.output;
    units.push({ effect, def, unit });
  }
  cur.connect(gain);
  gain.connect(masterGain);
  const entry = { clipId: clip.id, key, input, gain, units };
  audioChains.set(clip.id, entry);
  return entry;
}

function destroyAudioChain(clipId) {
  const entry = audioChains.get(clipId);
  if (!entry) return;
  audioChains.delete(clipId);
  // Disconnect from the input side out; anything still routed into it (a
  // paused element) lands on a dead-end gain rather than the speakers.
  entry.input.disconnect();
  for (const { unit } of entry.units) {
    try { unit.output.disconnect(); } catch {}
    try { unit.input.disconnect(); } catch {}
  }
  entry.gain.disconnect();
  for (const asset of assets.values())
    if (asset._routedTo === clipId) asset._routedTo = null;
}

/** Drop chains whose clip is gone (or lost its audio). */
function gcAudioChains() {
  for (const clipId of [...audioChains.keys()]) {
    const clip = findClip(comp, clipId)?.clip;
    if (!clip || !hasSource(clip)) destroyAudioChain(clipId);
  }
}

function routeAssetToChain(asset, entry) {
  if (asset._routedTo === entry.clipId && asset._routedKey === entry.key) return;
  try {
    asset.audioNode.disconnect();
    asset.audioNode.connect(entry.input);
    asset._routedTo = entry.clipId;
    asset._routedKey = entry.key;
  } catch (e) {
    console.warn('slangfx: audio routing failed for', asset.name, e);
  }
}

/** Push this frame's parameter values into a live chain. Every audio
 * parameter is an AudioParam by design (see audio-fx.js), so they all
 * glide the same way — setTargetAtTime, which keeps a dragged slider or a
 * keyframe ramp from clicking. */
function applyAudioChain(entry, clip, t, clipVol) {
  const now = audioCtx.currentTime;
  const tc = t - clip.start;
  for (const { effect, def, unit } of entry.units) {
    for (const p of def.params) {
      const prop = effect.params?.[p.name];
      const v = prop ? drivenEval(prop, tc, t) : p.def;
      for (const c of controlTargets(unit, p.name))
        c.param.setTargetAtTime(c.map ? c.map(v) : v, now, 0.01);
    }
  }
  entry.gain.gain.setTargetAtTime(clipVol, now, 0.01);
}

/* ---- waveform thumbnails ---------------------------------------------
 * One peaks image per audio asset, rendered once at a fixed width and used
 * as a CSS background by the timeline. The clip stretches and offsets that
 * one image, so zooming and trimming cost nothing and a re-render never
 * touches sample data. */

const WAVE_W = 1600;
const WAVE_H = 64;

async function buildWaveform(asset) {
  if (asset.waveform) return asset.waveform;
  asset._waveJob ??= (async () => {
    const buf = await decodeAssetAudio(asset);
    if (!buf) return null;
    const cv = document.createElement('canvas');
    cv.width = WAVE_W;
    cv.height = WAVE_H;
    const c2d = cv.getContext('2d');
    const data = buf.getChannelData(0);
    const step = data.length / WAVE_W;
    c2d.fillStyle = 'rgba(255, 255, 255, 0.5)';
    for (let x = 0; x < WAVE_W; x++) {
      const lo = Math.floor(x * step);
      const hi = Math.min(data.length, Math.floor((x + 1) * step));
      let peak = 0;
      for (let i = lo; i < hi; i++) {
        const v = Math.abs(data[i]);
        if (v > peak) peak = v;
      }
      const h = Math.max(1, peak * WAVE_H);
      c2d.fillRect(x, (WAVE_H - h) / 2, 1, h);
    }
    asset.waveform = cv.toDataURL('image/png');
    timeline?.render();
    return asset.waveform;
  })().catch((e) => {
    console.warn(`slangfx: waveform failed for '${asset.name}':`, e);
    return null;
  });
  return asset._waveJob;
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
        <button class="btn" id="cs-fit" title="resize the frame to the bounding box of what's on screen at the playhead">Fit to contents</button>
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
  // Fit to contents applies immediately: besides resizing, every media
  // clip shifts so the box lands centered — Apply can't express that.
  const fitBtn = wrap.querySelector('#cs-fit');
  fitBtn.disabled = !contentBounds(tCur);
  fitBtn.addEventListener('click', async () => {
    const b = contentBounds(tCur);
    if (!b) return;
    wrap.remove();
    await fitCompToBounds(b, 'contents');
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

function serializeMaskState(m) {
  if (!m?.nodes?.length) return null;
  return {
    opacity: m.opacity ?? 1,
    invert: !!m.invert,
    expand: m.expand ?? 0,
    feather: m.feather ?? 0,
    nodes: m.nodes.map((n) => {
      const out = {
        id: n.id, kind: n.kind, enabled: n.enabled !== false,
        blend: n.blend, invert: !!n.invert, strength: n.strength ?? 1,
      };
      if (n.kind === 'paint') {
        const dataURL = n.source.toDataURL('image/png');
        if (dataURL.length <= 2_000_000) out.dataURL = dataURL;
      } else if (n.kind === 'key') {
        out.keyColor = n.keyColor;
        out.similarity = n.similarity;
        out.smoothness = n.smoothness;
        out.sourceClipId = n.sourceClipId ?? null;
      } else if (n.kind === 'layer') {
        out.sourceClipId = n.sourceClipId ?? null;
        out.channel = n.channel;
      }
      return out;
    }),
  };
}

/** Rebuild runtime mask nodes from a persisted clip.mask (legacy single
 * painted mask included). Async: paint dataURLs decode through <img>. */
async function loadMaskNodes(maskModel) {
  const saved = maskModel.nodes
    ?? (maskModel.dataURL
      ? [{ id: uid(), kind: 'paint', enabled: true, blend: 'add', invert: false, strength: 1, dataURL: maskModel.dataURL }]
      : []);
  const nodes = [];
  for (const n of saved) {
    const node = { ...n };
    delete node.dataURL;
    if (n.kind === 'paint') {
      node.source = makeMaskCanvas();
      if (n.dataURL) {
        const img = new Image();
        await new Promise((res) => { img.onload = res; img.onerror = res; img.src = n.dataURL; });
        if (img.width) {
          const ctx2d = node.source.getContext('2d');
          ctx2d.clearRect(0, 0, node.source.width, node.source.height);
          ctx2d.drawImage(img, 0, 0, node.source.width, node.source.height);
        }
      }
    }
    prepareMaskNode(node);
    nodes.push(node);
  }
  return nodes;
}

/** Hydrate every clip's persisted mask into the live clipMasks registry.
 * Both kinds load up front: an fx clip's group mask must exist before the
 * chain builds, or the engine would compile the stack without it. */
async function loadClipMasks() {
  for (const track of comp.tracks)
    for (const clip of track.clips) {
      if (!clip.mask) continue;
      // Legacy single painted mask → a one-node stack (white base + painted
      // canvas added over black composes to the identical result).
      const nodes = await loadMaskNodes(clip.mask);
      if (!nodes.length) continue;
      clipMasks.set(clip.id, {
        opacity: clip.mask.opacity ?? 1, invert: !!clip.mask.invert,
        expand: clip.mask.expand ?? 0, feather: clip.mask.feather ?? 0,
        nodes,
      });
      if (clip.kind === 'media') buildMediaMaskGpu(clip.id);
    }
  masksLoaded = true;
  chainDirty = true;
}

function projectPayload() {
  // Sync live mask state (painted canvases + node params) into the model
  // before serializing. clipMasks is authoritative for every clip that has
  // been touched this session; a clip missing from it kept whatever it
  // loaded with.
  for (const track of comp.tracks)
    for (const clip of track.clips) {
      const m = clipMasks.get(clip.id);
      if (m) clip.mask = serializeMaskState(m);
      else if (masksLoaded) clip.mask = null;
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

/** Release one asset's runtime handles (element, proxy, bitmaps, texture). */
function disposeAsset(a) {
  if (a.el) { a.el.pause(); a.el.remove(); }
  if (a.proxyEl) {
    a.proxyEl.pause();
    try { URL.revokeObjectURL(a.proxyEl.src); } catch {}
    a.proxyEl.remove();
  }
  for (const f of a.frames ?? []) f.bitmap.close();
  a.texture?.destroy();
  try { URL.revokeObjectURL(a.url); } catch {}
}

/** Drop every runtime handle for the current project's media assets. */
function unloadAssets() {
  for (const a of assets.values()) disposeAsset(a);
  assets.clear();
}

/** Make `data` ({comp, assets, t, name}) the current project. */
async function applyProjectData(data) {
  stopMaskEdit();
  document.getElementById('demo-card')?.remove();
  unloadAssets();
  for (const clipId of [...mediaChains.keys()]) destroyMediaChain(clipId);
  for (const clipId of [...audioChains.keys()]) destroyAudioChain(clipId);
  fxSpecs.clear();
  paramMetaCache.clear();
  masksLoaded = false;
  for (const clipId of [...clipMasks.keys()]) destroyClipMask(clipId);
  for (const [id, t] of matteTargets) { t.tex.destroy(); matteTargets.delete(id); }
  comp = migrateComp(data.comp);
  removeEmptyTracks(comp);
  projectName = data.name ?? null;
  tCur = clamp(data.t ?? 0, 0, lastFrame(comp));
  history.reset();
  chainKey = '';
  chainDirty = true;
  fx.layers = [];

  await loadClipMasks();

  // Shape clips regenerate their textures from the model — no stored blobs.
  reconcileShapeAssets();

  // Restore assets in parallel; a missing or unloadable one must not block
  // the app — its clips simply render as offline until re-imported. Every
  // asset the project carries comes back, clips or not: the media bin
  // holds imports that haven't been placed yet.
  await Promise.allSettled((data.assets ?? [])
    .filter((meta) => meta.kind !== 'shape')
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
  syncAudioDrive();   // restored projects may carry audio drivers
  refreshDropHint();
  updateProjectButton();
  timeline?.zoomFit();
  timeline?.render();
  renderInspector();
  renderMediaBin();
}

async function restoreProject() {
  let data = null;
  try { data = JSON.parse(localStorage.getItem(PROJECT_KEY)); } catch {}
  if (data?.comp) {
    setStatus('restoring project…');
    await applyProjectData(data);
  } else {
    await loadDemoProject();
    masksLoaded = true;   // built in memory, nothing persisted to hydrate
  }
  if (comp._demo) showDemoCard();
}

/* =====================================================================
 * External launch — a host app can open the studio with media already
 * imported via ?import=<JSON [{url, name, type?, saveUrl?}]>; each url is
 * fetched against this origin and fed through the normal import pipeline.
 * An entry with saveUrl enables edit-and-save: the offline render is PUT
 * back to that url, overwriting the original file. (The Electron viewer
 * uses this: it serves the studio over studio:// and points entries at
 * local-file routes on the same origin.)
 * =================================================================== */

/** Fit an auto-sized comp's duration to its content (never below one frame).
 * Used by the edit-and-save flow so a 2s source doesn't render as the 12s
 * default comp with trailing black — and so trimming a clip shorter also
 * shortens the saved file. A manually-set duration (⚙ Comp) is respected. */
function fitDurToContent() {
  if (!comp._autoSize) return;
  let end = 0;
  for (const track of comp.tracks)
    for (const clip of track.clips) end = Math.max(end, clipEnd(clip));
  if (end > 0) comp.dur = Math.max(1 / comp.fps, Math.round(end * comp.fps) / comp.fps);
}

async function collectLaunchImports() {
  let entries = null;
  try { entries = JSON.parse(new URLSearchParams(location.search).get('import')); } catch {}
  if (!Array.isArray(entries)) return { files: [], saveBack: null };
  const files = [];
  let saveBack = null;
  for (const entry of entries) {
    if (!entry?.url || !entry?.name) continue;
    try {
      setStatus(`fetching ${entry.name}…`);
      const res = await fetch(entry.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      files.push(new File([blob], entry.name, { type: entry.type || blob.type }));
      // saveName covers targets that differ from the import (an image saves
      // as a video alongside the original: photo.jpg → photo.mp4).
      if (entry.saveUrl && !saveBack)
        saveBack = { url: entry.saveUrl, name: entry.saveName ?? entry.name };
    } catch (e) {
      console.warn('slangfx: launch import failed:', entry.url, e);
    }
  }
  return { files, saveBack };
}

/** Boot-time counterpart of stashCurrent(): a launch import is about to
 * replace the autosave slot, but the previous session only exists in
 * storage (nothing is loaded yet), so stash the raw payload straight into
 * the named-project store. Media blobs already live in IndexedDB keyed by
 * asset id, shared across projects — only the JSON needs copying. */
async function stashAutosavedProject() {
  let data = null;
  try { data = JSON.parse(localStorage.getItem(PROJECT_KEY)); } catch {}
  if (!data?.comp || data.comp._demo) return;
  if (!data.comp.tracks?.some((t) => t.clips?.length)) return;
  const name = data.name ?? `Untitled ${new Date().toLocaleString()}`;
  try {
    await idbSet(`project:${name}`, JSON.stringify(data));
    const idx = projectIndex().filter((p) => p.name !== name);
    idx.unshift({ name, savedAt: Date.now() });
    localStorage.setItem(PROJECT_INDEX_KEY, JSON.stringify(idx.slice(0, 20)));
  } catch (e) {
    console.warn('slangfx: could not stash previous session:', e);
  }
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
  updateSaveBackButton();
}

/** Edit-and-save: visible only when the current comp was launched by a host
 * app with a writable original (comp._saveBack rides in the project data). */
function updateSaveBackButton() {
  const btn = $('btn-save-back');
  const target = comp._saveBack;
  btn.hidden = !target;
  if (target)
    btn.title = `Render the comp and save it to ${target.name}, replacing that file if it exists`;
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
  const isOpen = name === projectName;
  const ok = await confirmDialog({
    title: `Delete project '${name}'?`,
    message: 'The saved comp (clips, keyframes, masks, custom shaders) is removed from this browser permanently. Imported media files stay cached for other projects.'
      + (isOpen ? ' This project is open right now — deleting it starts a fresh empty project.' : ''),
    confirmLabel: 'Delete project',
  });
  if (!ok) return;
  try { await idbDelete(`project:${name}`); } catch {}
  const idx = projectIndex().filter((p) => p.name !== name);
  try { localStorage.setItem(PROJECT_INDEX_KEY, JSON.stringify(idx)); } catch {}
  if (isOpen) {
    // Reset without newProject(): its stashCurrent() would re-save the
    // project we just deleted.
    const fresh = newComp({ width: 1280, height: 720, fps: 30, dur: 12 });
    fresh._autoSize = true;
    await applyProjectData({ comp: fresh, assets: [], t: 0, name: null });
    scheduleSave();
    setStatus(`deleted project '${name}' — new empty project`);
  } else {
    setStatus(`deleted project '${name}'`);
  }
}

async function saveFlow(alwaysAsk) {
  let name = projectName;
  if (alwaysAsk || !name) name = await promptName(name ?? '');
  if (!name) return;
  await saveNamedProject(name);
}

/* Simple filled folder — crisp and parseable at menu size, unlike emoji. */
const FOLDER_ICON =
  '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">'
  + '<path d="M1.5 3A1.5 1.5 0 0 1 3 1.5h3l1.5 2H13A1.5 1.5 0 0 1 14.5 5v7a1.5 1.5 0 0 1-1.5 1.5H3A1.5 1.5 0 0 1 1.5 12V3z"/></svg>';

$('btn-project').addEventListener('click', (e) => {
  const r = e.currentTarget.getBoundingClientRect();
  const items = [
    { label: '✚ New project', action: () => newProject() },
    { label: projectName ? `Save '${projectName}'` : 'Save…', action: () => saveFlow(false) },
    { label: 'Save as…', action: () => saveFlow(true) },
    '-',
    {
      label: '⚡ Scrub proxies',
      checked: proxiesEnabled,
      action: () => setProxiesEnabled(!proxiesEnabled),
    },
  ];
  const idx = projectIndex();
  if (idx.length) {
    items.push('-');
    for (const p of idx.slice(0, 8))
      items.push({
        label: `${p.name} · ${relTimeLabel(p.savedAt)}`,
        icon: FOLDER_ICON,
        checked: p.name === projectName,
        action: () => openProject(p.name),
        trailing: { label: '✕', title: 'delete project', danger: true, action: () => deleteProject(p.name) },
      });
  }
  showMenu(r.left, r.bottom + 4, items);
});

/* =====================================================================
 * Mask painting (per fx clip) — ported from the live demo.
 * =================================================================== */

const maskOverlay = $('mask-overlay');
const brush = { size: 60, soft: 0.5, mode: 'hide', tool: 'brush' };
/* ---- mask node stack -------------------------------------------------
 * A layer's mask is an ordered stack of nodes composited on the GPU every
 * frame (engine MaskComposer): paint canvases, chroma keys, and other
 * layers used as mattes. Everything reduces to "a texture per frame", so a
 * future AI-roto node is just one more source that swaps its view between
 * frames — no pipeline changes needed. */

const matteTargets = new Map();   // mask node id -> { tex, view, w, h }

function newMaskNode(kind) {
  const base = { id: uid(), kind, enabled: true, blend: 'add', invert: false, strength: 1 };
  if (kind === 'paint') return { ...base, source: makeMaskCanvas() };
  // Keys default inverted: mask = everything EXCEPT the key color, so
  // adding one reads as "remove this color" (green screen) rather than
  // blanking the layer until a color is picked.
  if (kind === 'key') return { ...base, invert: true, keyColor: '#00b140', similarity: 0.18, smoothness: 0.1, sourceClipId: null };
  return { ...base, sourceClipId: null, channel: 'alpha' };   // 'layer'
}

function hexToRgb01(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex ?? '');
  const v = parseInt(m ? m[1] : '00b140', 16);
  return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
}

/** Comp-sized render target for a node whose source is another layer,
 * recreated on comp resize. The node's `view` feeds the engine bind group. */
function ensureMatteTarget(node) {
  let t = matteTargets.get(node.id);
  if (!t || t.w !== comp.width || t.h !== comp.height) {
    t?.tex.destroy();
    const tex = fx.device.createTexture({
      label: 'slangfx matte target',
      size: [comp.width, comp.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    t = { tex, view: tex.createView(), w: comp.width, h: comp.height };
    matteTargets.set(node.id, t);
  }
  node.view = t.view;
  return t;
}

/** Normalize a node's runtime fields after load or an edit. */
function prepareMaskNode(node) {
  if (node.kind === 'key') node.keyRGB = hexToRgb01(node.keyColor);
  node.useInput = node.kind === 'key' && !node.sourceClipId;
  if (node.sourceClipId) ensureMatteTarget(node);
  else node.view = null;
  if (node.kind === 'layer' && !node.channel) node.channel = 'alpha';
}

/* ---- media clip masks ------------------------------------------------
 * Media clips share the same node stack model (clipMasks) as fx clips, but
 * the result multiplies the clip's ALPHA when it composites (true
 * green-screen: keyed pixels go transparent and lower tracks show
 * through). The engine owns the GPU state for an fx clip's group mask;
 * media mask GPU state is owned here. */

const mediaMaskTargets = new Map();   // media clipId -> { tex, view, w, h }

function ensureMediaMaskTarget(clipId) {
  let entry = mediaMaskTargets.get(clipId);
  if (!entry) {
    entry = { tex: null, view: null, w: 0, h: 0 };
    mediaMaskTargets.set(clipId, entry);
  }
  ensureMediaMaskTex(entry);
  return entry;
}

function ensureMediaMaskTex(entry) {
  if (!entry.tex || entry.w !== comp.width || entry.h !== comp.height) {
    entry.tex?.destroy();
    entry.tex = fx.device.createTexture({
      label: 'slangfx media mask',
      size: [comp.width, comp.height],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    entry.view = entry.tex.createView();
    entry.w = comp.width;
    entry.h = comp.height;
  }
}

/** (Re)create GPU state for a media clip's mask nodes — the app-side twin
 * of the engine's _buildLayerMask. */
function buildMediaMaskGpu(clipId) {
  const maskState = clipMasks.get(clipId);
  if (!maskState || !fx?.device) return;
  ensureMediaMaskTarget(clipId);
  for (const node of maskState.nodes) {
    node._optsBuf?.destroy();
    node._tex?.destroy();
    node._optsBuf = fx.device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    node._tex = null;
    if (node.source) {
      node._tex = fx.device.createTexture({
        label: 'slangfx mask node source',
        size: [comp.width, comp.height],
        format: 'rgba8unorm',
        // RENDER_ATTACHMENT: required by copyExternalImageToTexture's
        // GPU-canvas blit path.
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      fx.device.queue.copyExternalImageToTexture(
        { source: node.source }, { texture: node._tex }, [comp.width, comp.height]);
    }
    const view = node._tex?.createView() ?? node.view ?? compositor.whiteView;
    node._bindGroup = fx.maskComposer.bindGroup(view, fx.inputSampler, node._optsBuf);
  }
}

function destroyMaskNodeGpu(node) {
  node._optsBuf?.destroy();
  node._tex?.destroy();
  node._optsBuf = node._tex = node._bindGroup = null;
  const t = matteTargets.get(node.id);
  if (t) { t.tex.destroy(); matteTargets.delete(node.id); }
}

/** Drop a clip's mask entirely: node GPU state (whoever owns it), the
 * media compose target, and the live stack. */
function destroyClipMask(clipId) {
  const maskState = clipMasks.get(clipId);
  for (const node of maskState?.nodes ?? []) destroyMaskNodeGpu(node);
  if (maskState) fx?.maskComposer?.destroyPost(maskState);
  clipMasks.delete(clipId);
  const target = mediaMaskTargets.get(clipId);
  if (target) { target.tex?.destroy(); mediaMaskTargets.delete(clipId); }
}

/** Clip entries for every layer-sourced mask node active at t (fed to the
 * offline exporter's exact seek alongside the visible media). */
function matteSourceClips(t) {
  const out = [];
  for (const maskState of clipMasks.values())
    for (const node of maskState?.nodes ?? []) {
      if (!node.sourceClipId || node.enabled === false) continue;
      const hit = findClip(comp, node.sourceClipId);
      if (hit && t >= hit.clip.start && t < clipEnd(hit.clip)) out.push({ clip: hit.clip });
    }
  return out;
}

/* Resolve which layer-sourced nodes are live at t and render each source
 * clip (with its transform, no mask of its own — mattes are raw content)
 * into its matte target. Matte sources render even when their track is
 * hidden — a hidden track is the natural home for matte-only footage. */
function prepareNodeSources(nodes, t, getEncoder) {
  for (const node of nodes) {
    if (!node.sourceClipId) { node.active = true; continue; }
    const hit = findClip(comp, node.sourceClipId);
    const clip = hit?.clip;
    const asset = clip && assets.get(clip.assetId);
    if (!clip || !asset?.ready || t < clip.start || t >= clipEnd(clip)) {
      node.active = false;
      continue;
    }
    if (asset.kind === 'gif') {
      syncGifFrame(asset, clip, t);
    } else if (asset.kind === 'video') {
      // Hidden-track sources never go through syncMedia — chase the comp
      // clock with paused seeks (the offline exporter seeks exactly).
      const el = asset.el;
      const src = clip.in + (t - clip.start);
      const len = asset.duration ?? 0;
      const desired = len > 0.02 ? ((src % len) + len) % len : 0;
      if (!el.seeking && Math.abs(el.currentTime - desired) > 0.5 / comp.fps)
        el.currentTime = desired;
      if (el.readyState >= 2) uploadVideoFrame(asset);
    }
    const d = drawForClip(clip, t);
    if (!d) { node.active = false; continue; }
    node.active = true;
    const tgt = ensureMatteTarget(node);
    // ':matte' keys a separate compositor item so the raw matte draw does
    // not fight the clip's on-screen draw over one uniform buffer. Mattes
    // are raw content — the clip's blend mode does not apply.
    compositor.composite(getEncoder(), tgt.view, comp.width, comp.height,
      [{ ...d, clipId: d.clipId + ':matte', maskView: null, blend: 'normal' }], { transparent: true });
  }
}

/** Per-frame mask prep for every clip carrying a mask. Media stacks
 * compose here (before compositeFrame samples them); an fx clip's group
 * mask composes inside fx.render(), so it only needs its node sources. */
function prepareMasks(t) {
  let encoder = null;
  const getEncoder = () => (encoder ??= fx.device.createCommandEncoder());
  for (const [clipId, maskState] of clipMasks) {
    const nodes = maskState?.nodes;
    if (!nodes?.length) continue;
    const clip = findClip(comp, clipId)?.clip;
    if (!clip || t < clip.start || t >= clipEnd(clip)) continue;
    prepareNodeSources(nodes, t, getEncoder);
    if (clip.kind !== 'media') continue;
    const entry = ensureMediaMaskTarget(clipId);
    fx.maskComposer.encode(getEncoder(),
      { maskState, maskView: entry.view, maskW: entry.w, maskH: entry.h });
  }
  if (encoder) fx.device.queue.submit([encoder.finish()]);
}

/* ---- key-color eyedropper -------------------------------------------- */

let pickState = null;   // { node } while waiting for a preview click

function startColorPick(node) {
  pickState = { node };
  canvasStack.style.cursor = 'crosshair';
  setStatus('click the preview to sample the key color (Esc cancels)');
}

function endColorPick() {
  pickState = null;
  canvasStack.style.cursor = '';
}

/* Capture phase on the canvas stack so the pick wins over gizmo / pan. */
canvasStack.addEventListener('pointerdown', async (e) => {
  if (!pickState || e.button !== 0) return;
  e.stopPropagation();
  e.preventDefault();
  const { node } = pickState;
  endColorPick();
  const rect = canvas.getBoundingClientRect();
  const x = clamp(Math.floor((e.clientX - rect.left) / rect.width * comp.width), 0, comp.width - 1);
  const y = clamp(Math.floor((e.clientY - rect.top) / rect.height * comp.height), 0, comp.height - 1);
  // Sample the pre-effect composite — that's what key nodes see.
  const { pixels, width } = await fx.readPixels(fx.inputTexture);
  const i = (y * width + x) * 4;
  node.keyColor = '#' + [pixels[i], pixels[i + 1], pixels[i + 2]]
    .map((v) => v.toString(16).padStart(2, '0')).join('');
  prepareMaskNode(node);
  scheduleSave();
  renderInspector();
  setStatus(`key color ${node.keyColor}`);
}, true);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && pickState) endColorPick();
});

/* ---- mask painting --------------------------------------------------- */

let maskEdit = null;   // { clipId, nodeId } | null
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

function maskStateFor(clipId) {
  return clipMasks.get(clipId) ?? null;
}

function maskEditNode() {
  if (!maskEdit) return null;
  return maskStateFor(maskEdit.clipId)?.nodes.find((n) => n.id === maskEdit.nodeId) ?? null;
}

/** Push a paint node's canvas to the GPU after a stroke or clear. */
function uploadPaintNode(clipId, node) {
  if (mediaMaskTargets.has(clipId)) {
    if (node?._tex && node.source)
      fx.device.queue.copyExternalImageToTexture(
        { source: node.source }, { texture: node._tex }, [comp.width, comp.height]);
    return;
  }
  fx.updateGroupMask(clipId);   // fx clip: the engine owns the group's mask
}

function pushMaskToGpu() {
  if (maskEdit) uploadPaintNode(maskEdit.clipId, maskEditNode());
}

function maskStroke(x, y) {
  const node = maskEditNode();
  if (!node?.source) return;
  const mctx = node.source.getContext('2d');
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
  const node = maskEditNode();
  if (!node?.source) return;
  if (Math.hypot(to[0] - from[0], to[1] - from[1]) < 2) return;
  const src = node.source;
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

/* Brush-size cursor: while the brush tool paints, the native cursor is
 * replaced by a circle matching the brush's on-screen diameter, so the
 * stroke footprint is visible before committing it. Gradient tools keep
 * the crosshair. */
const brushCursor = document.createElement('div');
brushCursor.id = 'brush-cursor';
brushCursor.hidden = true;
$('canvas-inner').appendChild(brushCursor);
let brushCursorPos = null;   // last pointer [clientX, clientY] over the overlay

function updateBrushCursor() {
  const active = maskEdit && brush.tool === 'brush';
  maskOverlay.style.cursor = active ? 'none' : '';
  if (!active || !brushCursorPos) {
    brushCursor.hidden = true;
    return;
  }
  const rect = maskOverlay.getBoundingClientRect();
  const innerR = $('canvas-inner').getBoundingClientRect();
  const d = brush.size * (rect.width / comp.width);   // comp px → screen px
  brushCursor.style.width = `${d}px`;
  brushCursor.style.height = `${d}px`;
  brushCursor.style.left = `${brushCursorPos[0] - innerR.left - d / 2}px`;
  brushCursor.style.top = `${brushCursorPos[1] - innerR.top - d / 2}px`;
  brushCursor.hidden = false;
}

maskOverlay.addEventListener('pointerleave', () => {
  brushCursorPos = null;
  updateBrushCursor();
});

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
  brushCursorPos = [e.clientX, e.clientY];
  updateBrushCursor();
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

async function startMaskEdit(clip, nodeId) {
  // The brush paints in comp space, so the clip has to be the thing on
  // screen. (An empty or still-compiling stack is fine — the canvas is
  // app-side and the GPU picks it up on the next rebuild.)
  if (tCur < clip.start || tCur >= clipEnd(clip)) {
    setStatus('move the playhead over this clip to edit its mask');
    return;
  }
  if (viewer.classList.contains('size-cover')) setViewMode('fit');
  const node = maskStateFor(clip.id)?.nodes.find((n) => n.id === nodeId);
  if (!node?.source) return;
  maskEdit = { clipId: clip.id, nodeId };
  rebuildRuby(node.source);
  document.body.classList.add('mask-editing');
  updateBrushCursor();
  setStatus(clip.kind === 'media'
    ? 'painting mask — red = clip hidden'
    : 'painting mask — red = effect hidden');
  renderInspector();
}

function stopMaskEdit() {
  maskEdit = null;
  painting = false;
  document.body.classList.remove('mask-editing');
  updateBrushCursor();
  renderInspector();
}

function rescaleMasks() {
  const rescaleNodes = (nodes) => {
    for (const node of nodes ?? []) {
      const src = node.source;
      if (src && (src.width !== comp.width || src.height !== comp.height)) {
        const scaled = document.createElement('canvas');
        scaled.width = comp.width;
        scaled.height = comp.height;
        scaled.getContext('2d').drawImage(src, 0, 0, scaled.width, scaled.height);
        node.source = scaled;
      }
      if (node.sourceClipId && fx?.device) ensureMatteTarget(node);
    }
  };
  for (const [clipId, maskState] of clipMasks) {
    rescaleNodes(maskState?.nodes);
    // Media mask node textures + compose target track the comp size; fx
    // clip masks are rebuilt by the engine on the next chain rebuild.
    if (findClip(comp, clipId)?.clip.kind === 'media') buildMediaMaskGpu(clipId);
  }
  resizeMediaChains();   // private chains render at comp resolution too
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

/* The inspector always keeps a layer in focus so the effect panel has a
 * target (see ensureFocusedLayer), which means "nothing selected" quietly
 * becomes "topmost layer selected". Handles must not follow that: a fallback
 * focus is not something the user pointed at, so it gets no gizmo. */
let focusIsFallback = false;

function gizmoTarget() {
  const clip = timeline?.selectedClip;
  if (focusIsFallback) return null;
  if (!clip || clip.kind !== 'media' || maskEdit) return null;
  // While a shape draw is armed the pointer belongs to the draw, not to the
  // selected layer's handles.
  if (shapeDraw) return null;
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
  // Crop owns the viewport while it's open; a transform gizmo on a clip
  // that's about to be re-placed would only be in the way.
  if (crop) { gizmo.hidden = true; return; }
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
  if (e.button !== 0 || maskEdit || crop) return;
  // The canvas element is bigger than the comp frame — object-fit letterboxes
  // the frame inside it — so a click here can still be "out in the workspace".
  // Anywhere outside the frame drops the selection; inside it, one click
  // switches to whatever layer is under the pointer.
  const d = canvasDisplayRect();
  const inFrame = e.clientX >= d.left && e.clientX <= d.left + comp.width * d.s
    && e.clientY >= d.top && e.clientY <= d.top + comp.height * d.s;
  if (!inFrame) { timeline.selectClip(null); updateGizmo(); return; }
  const hit = clipAtViewportPoint(e.clientX, e.clientY);
  timeline.selectClip(hit ? hit.id : null);
  focusIsFallback = !hit;
  updateGizmo();
});

// The workspace around the frame deselects. Anything in the viewer that
// isn't the frame itself counts — viewer, canvas-stack and canvas-inner all
// show through around the comp depending on the view mode, so match by
// exclusion rather than naming the two that used to be reachable.
viewer.addEventListener('pointerdown', (e) => {
  if (e.button !== 0 || maskEdit || crop) return;
  if (e.target === canvas || e.target.closest('#gizmo, .btn, #mask-overlay')) return;
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

/* =====================================================================
 * Crop & straighten
 *
 * The box is axis-aligned ON SCREEN and the contents rotate under it
 * (Lightroom's crop+straighten), so `crop` lives in "rotated space": the
 * comp turned by `angle` about its own centre. cx/cy are the box centre
 * in that space, relative to the comp centre, in comp pixels.
 *
 * Preview rotation is a CSS transform on #canvas-inner — nothing in the
 * render pipeline knows about crop mode, and Cancel is just "throw the
 * state away". Applying bakes the same transform into every media clip.
 * =================================================================== */

let crop = null;        // { cx, cy, w, h, angle } or null when not cropping
let cropBase = null;    // screen geometry captured on entry (see cropMeasure)
let cropDrag = null;

const cropOverlay = $('crop-overlay');
const cropBox = $('crop-box');
const cropBar = $('crop-bar');

/** Rotate (x, y) by `deg` about the origin. Comp space is y-down, so a
 * positive angle reads as clockwise, matching CSS and the compositor. */
function rot2(x, y, deg) {
  const a = deg * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return [x * c - y * s, x * s + y * c];
}

/** Screen placement of the comp frame, measured with the crop rotation
 * temporarily off: getBoundingClientRect on a rotated element returns the
 * bounding box of the rotation, which is not what we want to map through. */
// Crop mode pulls the comp in a little so the box has somewhere to grow
// into; without the margin the frame fills the pane and "expand" has no
// reachable target.
const CROP_VIEW = 0.74;

function cropMeasure() {
  const inner = $('canvas-inner');
  const prev = inner.style.transform;
  inner.style.transform = '';
  const d = canvasDisplayRect();
  const wrapR = $('preview-wrap').getBoundingClientRect();
  inner.style.transform = prev;
  cropBase = {
    s: d.s * CROP_VIEW,
    ccx: d.left + comp.width * d.s / 2 - wrapR.left,
    ccy: d.top + comp.height * d.s / 2 - wrapR.top,
  };
}

const CROP_MAX_W = 7680, CROP_MAX_H = 4320;   // same ceiling as comp settings

/** Largest box (in rotated space) with NO blank corners — it stays inside
 * the rotated comp. Offered as "Fit inside" rather than enforced, because
 * the crop is also allowed to grow the frame past what the comp had. */
function cropInsetBox(angle) {
  const W = comp.width, H = comp.height;
  const a = Math.abs(angle % 180) * Math.PI / 180;
  const c = Math.abs(Math.cos(a)), s = Math.abs(Math.sin(a));
  const denom = c * c - s * s;
  // Near 45° the inscribed rect degenerates; fall back to a square.
  if (Math.abs(denom) < 1e-6) {
    const side = Math.min(W, H) / Math.SQRT2;
    return { w: side, h: side };
  }
  const w = (W * c - H * s) / denom;
  const h = (H * c - W * s) / denom;
  if (w <= 16 || h <= 16) {
    const side = Math.min(W, H) / Math.SQRT2;
    return { w: side, h: side };
  }
  return { w, h };
}

/** Smallest box (in rotated space) containing the whole rotated comp — so
 * nothing is lost. What a quarter turn and Reset snap to. */
function cropCoverBox(angle) {
  const W = comp.width, H = comp.height;
  const pts = [[-W / 2, -H / 2], [W / 2, -H / 2], [W / 2, H / 2], [-W / 2, H / 2]]
    .map(([x, y]) => rot2(x, y, angle));
  return {
    w: Math.max(...pts.map((p) => Math.abs(p[0]))) * 2,
    h: Math.max(...pts.map((p) => Math.abs(p[1]))) * 2,
  };
}

/* ---- crop snapping ---------------------------------------------------
 * The same magnetism a layer gets in the viewport, on the crop box's own
 * edges. In crop space the origin is the frame centre, so the magnets are
 * the lines that mean something there: the frame itself, and — once the
 * image is straightened — the largest crop with no blank corners and the
 * smallest that loses nothing. At 0° those two are the frame, which is
 * why the common case simply snaps to the edges. */

const cropGuideV = document.createElement('div');
cropGuideV.className = 'gz-guide v cr-guide';
const cropGuideH = document.createElement('div');
cropGuideH.className = 'gz-guide h cr-guide';
cropGuideV.hidden = cropGuideH.hidden = true;
cropOverlay.append(cropGuideV, cropGuideH);

const hideCropGuides = () => { cropGuideV.hidden = cropGuideH.hidden = true; };

function cropSnapLines() {
  const ins = cropInsetBox(crop.angle);
  const cov = cropCoverBox(crop.angle);
  const half = (n) => [n / 2, -n / 2];
  return {
    x: [...new Set([...half(ins.w), ...half(cov.w), 0])],
    y: [...new Set([...half(ins.h), ...half(cov.h), 0])],
  };
}

/** Nearest magnet within `thresh`, or null. */
function nearestLine(v, lines, thresh) {
  let best = null;
  let bestD = thresh;
  for (const t of lines) {
    const d = Math.abs(v - t);
    if (d < bestD) { best = t; bestD = d; }
  }
  return best;
}

/** Draw (or hide) a guide at crop-space coordinate `v`. */
function showCropGuide(axis, v) {
  const el = axis === 'x' ? cropGuideV : cropGuideH;
  if (v == null) { el.hidden = true; return; }
  const { s, ccx, ccy } = cropBase;
  if (axis === 'x') {
    el.style.left = `${ccx + v * s}px`;
    el.style.top = '0';
    el.style.height = '100%';
  } else {
    el.style.top = `${ccy + v * s}px`;
    el.style.left = '0';
    el.style.width = '100%';
  }
  el.hidden = false;
}

function setCropBox(w, h, cx = crop.cx, cy = crop.cy) {
  crop.w = clamp(w, 16, CROP_MAX_W);
  crop.h = clamp(h, 16, CROP_MAX_H);
  // Position is free — the box may sit anywhere, including entirely off the
  // old frame — but keep it findable rather than lost in deep space.
  const reach = Math.max(comp.width, comp.height) * 4;
  crop.cx = clamp(cx, -reach, reach);
  crop.cy = clamp(cy, -reach, reach);
}

function cropRender() {
  if (!crop) return;
  if (!cropBase) cropMeasure();
  const { s, ccx, ccy } = cropBase;
  $('canvas-inner').style.transform = `rotate(${crop.angle}deg) scale(${CROP_VIEW})`;
  cropBox.style.left = `${ccx + (crop.cx - crop.w / 2) * s}px`;
  cropBox.style.top = `${ccy + (crop.cy - crop.h / 2) * s}px`;
  cropBox.style.width = `${crop.w * s}px`;
  cropBox.style.height = `${crop.h * s}px`;
  $('cr-size').textContent =
    `${Math.max(16, Math.round(crop.w))} × ${Math.max(16, Math.round(crop.h))}`;
  const slider = $('cr-slider');
  const num = $('cr-num');
  // The slider only spans ±45; quarter turns live outside it.
  const fine = ((crop.angle % 90) + 135) % 90 - 45;
  if (document.activeElement !== slider) slider.value = String(fine);
  if (document.activeElement !== num) num.value = String(Math.round(crop.angle * 10) / 10);
}

function enterCrop() {
  if (crop) return exitCrop(false);
  cropMeasure();
  crop = { cx: 0, cy: 0, w: comp.width, h: comp.height, angle: 0 };
  cropOverlay.hidden = false;
  cropBar.hidden = false;
  gizmo.hidden = true;
  $('btn-crop').classList.add('active');
  cropRender();
  setStatus('crop — drag the box, drag outside it to straighten, then Apply');
}

function exitCrop(commit) {
  if (!crop) return;
  const state = crop;
  crop = null;
  cropDrag = null;
  hideCropGuides();
  cropOverlay.hidden = true;
  cropBar.hidden = true;
  $('btn-crop').classList.remove('active');
  $('canvas-inner').style.transform = '';
  if (commit) applyCrop(state);
  else setStatus('crop cancelled');
}

/** Rewrite a clip's x/y through an arbitrary point map. A rotation makes
 * x' depend on y and vice versa, so when either track is animated both
 * have to be resampled onto the union of their key times first. Rotation
 * is linear, so this is exact wherever the two tracks share easing. */
function mapClipPosition(clip, fn) {
  const px = clip.props.x, py = clip.props.y;
  if (!px || !py) return;
  const animated = (px.anim && px.keys.length) || (py.anim && py.keys.length);
  if (!animated) {
    const [nx, ny] = fn(px.v, py.v);
    px.v = nx; py.v = ny;
    return;
  }
  const times = [...new Set([
    ...(px.anim ? px.keys.map((k) => k.t) : []),
    ...(py.anim ? py.keys.map((k) => k.t) : []),
  ])].sort((a, b) => a - b);
  const easeOf = (prop, t) => prop.keys.find((k) => Math.abs(k.t - t) < 1e-9)?.e;
  const pts = times.map((t) => {
    const [nx, ny] = fn(evalProp(px, t), evalProp(py, t));
    return { t, nx, ny };
  });
  for (const [prop, pick] of [[px, (p) => p.nx], [py, (p) => p.ny]]) {
    const keys = pts.map((p) => {
      const e = easeOf(prop, p.t);
      return e ? { t: p.t, v: pick(p), e } : { t: p.t, v: pick(p) };
    });
    prop.anim = true;
    prop.keys = keys;
    prop.v = keys.length ? keys[0].v : prop.v;
  }
}

function addToProp(prop, d) {
  if (!prop || !d) return;
  prop.v += d;
  for (const k of prop.keys) k.v += d;
}

async function applyCrop(state) {
  const W = clamp(2 * Math.round(state.w / 2), 16, 7680);
  const H = clamp(2 * Math.round(state.h / 2), 16, 4320);
  const { cx, cy, angle } = state;
  const oldCx = comp.width / 2, oldCy = comp.height / 2;
  // Comp point -> rotated space about the comp centre -> new frame origin.
  const place = (x, y) => {
    const [rx, ry] = rot2(x - oldCx, y - oldCy, angle);
    return [rx - cx + W / 2, ry - cy + H / 2];
  };

  let driven = 0, masked = 0;
  history.record(comp, () => {
    for (const track of comp.tracks)
      for (const c of track.clips) {
        if (c.kind !== 'media') continue;
        if (c.props.x?.driver?.enabled || c.props.y?.driver?.enabled) driven++;
        if (clipMasks.get(c.id)?.nodes?.length) masked++;
        mapClipPosition(c, place);
        addToProp(c.props.rot, angle);
      }
    comp.width = W;
    comp.height = H;
    comp._autoSize = false;
  });
  await applyCompSize();
  setTime(tCur);
  onModelChange({ structural: true });

  // Both caveats are real and neither is silently recoverable, so say so
  // rather than let someone find it later in a render.
  const notes = [];
  if (driven) notes.push(`${driven} clip${driven > 1 ? 's' : ''} with a driver on X/Y kept its own motion`);
  if (masked) notes.push(`${masked} mask${masked > 1 ? 's are' : ' is'} authored in the old frame`);
  setStatus(`cropped to ${W}×${H}${angle ? ` at ${Math.round(angle * 10) / 10}°` : ''}`
    + (notes.length ? ` — ${notes.join('; ')}` : ''));
}

/* ---- crop interaction ------------------------------------------------ */

$('btn-crop').addEventListener('click', enterCrop);
$('cr-cancel').addEventListener('click', () => exitCrop(false));
$('cr-apply').addEventListener('click', () => exitCrop(true));
$('cr-reset').addEventListener('click', () => {
  if (!crop) return;
  crop = { cx: 0, cy: 0, w: comp.width, h: comp.height, angle: 0 };
  cropRender();
});
$('cr-inset').addEventListener('click', () => {
  if (!crop) return;
  const b = cropInsetBox(crop.angle);
  setCropBox(b.w, b.h, 0, 0);
  cropRender();
});
$('cr-cover').addEventListener('click', () => {
  if (!crop) return;
  const b = cropCoverBox(crop.angle);
  setCropBox(b.w, b.h, 0, 0);
  cropRender();
});

function setCropAngle(deg) {
  if (!crop) return;
  crop.angle = deg;
  // The box is deliberately left alone: straightening may now expose blank
  // corners, which is allowed. "Fit inside" trims them in one click.
  cropRender();
}

$('cr-slider').addEventListener('input', () => {
  if (!crop) return;
  const quarter = Math.round(crop.angle / 90) * 90;
  setCropAngle(quarter + parseFloat($('cr-slider').value));
});
$('cr-num').addEventListener('input', () => {
  const v = parseFloat($('cr-num').value);
  if (Number.isFinite(v)) setCropAngle(clamp(v, -180, 180));
});
$('cr-num').addEventListener('keydown', (e) => e.stopPropagation());
$('cr-ccw').addEventListener('click', () => { if (crop) quarterTurn(-90); });
$('cr-cw').addEventListener('click', () => { if (crop) quarterTurn(90); });

/** A quarter turn swaps which way the frame is long. Snap to the cover
 * box so a 90° turn is lossless — nothing falls outside the new frame. */
function quarterTurn(d) {
  const angle = crop.angle + d;
  const box = cropCoverBox(angle);
  crop = { cx: 0, cy: 0, w: box.w, h: box.h, angle };
  cropRender();
}

cropOverlay.addEventListener('pointerdown', (e) => {
  if (!crop || e.button !== 0) return;
  e.preventDefault();
  try { cropOverlay.setPointerCapture(e.pointerId); } catch {}
  const handle = e.target.closest('.cr-h')?.dataset.h ?? null;
  const inBox = !!e.target.closest('#crop-box');
  const wrapR = $('preview-wrap').getBoundingClientRect();
  const px = e.clientX - wrapR.left, py = e.clientY - wrapR.top;
  cropDrag = {
    handle,
    // Outside the box with no handle = straighten instead of reframe.
    mode: handle ? 'resize' : inBox ? 'move' : 'spin',
    x0: px, y0: py,
    start: { ...crop },
    a0: Math.atan2(py - cropBase.ccy, px - cropBase.ccx) * 180 / Math.PI,
  };
});

cropOverlay.addEventListener('pointermove', (e) => {
  if (!cropDrag || !crop) return;
  const wrapR = $('preview-wrap').getBoundingClientRect();
  const px = e.clientX - wrapR.left, py = e.clientY - wrapR.top;
  const s = cropBase.s;
  const dx = (px - cropDrag.x0) / s, dy = (py - cropDrag.y0) / s;
  const st = cropDrag.start;

  if (cropDrag.mode === 'spin') {
    const a = Math.atan2(py - cropBase.ccy, px - cropBase.ccx) * 180 / Math.PI;
    setCropAngle(clamp(st.angle + (a - cropDrag.a0), -180, 180));
    return;
  }
  // Magnetism follows the timeline's 🧲 toggle, and Ctrl suspends it —
  // the same contract as dragging a layer in the viewport.
  const snapOn = timeline?.snap && !e.ctrlKey && !e.metaKey;
  const thresh = 8 / s;
  const lines = snapOn ? cropSnapLines() : null;

  if (cropDrag.mode === 'move') {
    let nx = st.cx + dx;
    let ny = st.cy + dy;
    if (snapOn) {
      // Whichever of the two edges (or the centre) is closest to a magnet
      // pulls the whole box — you aim with the edge you're watching.
      const pull = (c, half, ls) => {
        let best = null;
        for (const [probe, ref] of [[c - half, -half], [c + half, half], [c, 0]]) {
          const m = nearestLine(probe, ls, thresh);
          if (m == null) continue;
          const d = m - probe;
          if (!best || Math.abs(d) < Math.abs(best.d)) best = { d, line: m, ref };
        }
        return best;
      };
      const bx = pull(nx, crop.w / 2, lines.x);
      const by = pull(ny, crop.h / 2, lines.y);
      if (bx) nx += bx.d;
      if (by) ny += by.d;
      showCropGuide('x', bx?.line ?? null);
      showCropGuide('y', by?.line ?? null);
    } else {
      hideCropGuides();
    }
    setCropBox(crop.w, crop.h, nx, ny);
    cropRender();
    return;
  }
  // Resize: move only the edges the grabbed handle owns.
  const h = cropDrag.handle;
  let l = st.cx - st.w / 2, r = st.cx + st.w / 2;
  let t = st.cy - st.h / 2, b = st.cy + st.h / 2;
  if (h.includes('w')) l = Math.min(l + dx, r - 16);
  if (h.includes('e')) r = Math.max(r + dx, l + 16);
  if (h.includes('n')) t = Math.min(t + dy, b - 16);
  if (h.includes('s')) b = Math.max(b + dy, t + 16);
  if (snapOn) {
    // Only the dragged edges are magnetic — the opposite edge stays put.
    const snapEdge = (edge, ls, axis, set) => {
      const m = nearestLine(edge, ls, thresh);
      if (m != null) set(m);
      return m;
    };
    let gx = null;
    let gy = null;
    if (h.includes('w')) gx = snapEdge(l, lines.x, 'x', (m) => { l = Math.min(m, r - 16); });
    if (h.includes('e')) gx = snapEdge(r, lines.x, 'x', (m) => { r = Math.max(m, l + 16); }) ?? gx;
    if (h.includes('n')) gy = snapEdge(t, lines.y, 'y', (m) => { t = Math.min(m, b - 16); });
    if (h.includes('s')) gy = snapEdge(b, lines.y, 'y', (m) => { b = Math.max(m, t + 16); }) ?? gy;
    showCropGuide('x', gx);
    showCropGuide('y', gy);
  } else {
    hideCropGuides();
  }
  // No frame constraint: dragging a handle outward grows the comp.
  setCropBox(r - l, b - t, (l + r) / 2, (t + b) / 2);
  cropRender();
});

const endCropDrag = () => { cropDrag = null; hideCropGuides(); };
cropOverlay.addEventListener('pointerup', endCropDrag);
cropOverlay.addEventListener('pointercancel', endCropDrag);

window.addEventListener('resize', () => { if (crop) { cropMeasure(); cropRender(); } });

// Escape backs out, Enter commits — captured so crop mode wins over the
// timeline's own shortcuts while it owns the viewport.
window.addEventListener('keydown', (e) => {
  if (!crop) return;
  if (e.target.matches?.('input, textarea, select') && e.key !== 'Escape') return;
  if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); exitCrop(false); }
  else if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); exitCrop(true); }
}, true);

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
    {
      // The other way round: leave the layer where it is and cut the comp
      // down (or out) to exactly its bounds.
      label: 'Fit frame to this layer',
      action: () => {
        const b = clipBounds(clip, tCur);
        if (b) fitCompToBounds(b, clip.name);
      },
    },
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
 * Shape layers — media clips whose texture is a canvas-drawn vector
 * shape (preset + fill color), redrawn whenever the settings change.
 * Picked from the Add menu, then drawn onto the viewport as a new layer.
 * They stay kind:'media' so transforms, masks, mattes, the gizmo, and
 * the exporter all treat them like any other still.
 * =================================================================== */

const SHAPE_TEX_SIZE = 1024;

// New shapes reuse whatever fill you picked last, across sessions — drawing
// six shapes in one colour shouldn't mean setting it six times. White until
// you've picked anything, because an empty comp is black and the first
// shape you draw should be something you can see.
const SHAPE_COLOR_KEY = 'lowkey-studio.shape-color';
let lastShapeColor = /^#[0-9a-f]{6}$/i.test(localStorage.getItem(SHAPE_COLOR_KEY) ?? '')
  ? localStorage.getItem(SHAPE_COLOR_KEY)
  : '#ffffff';

function setLastShapeColor(hex) {
  lastShapeColor = hex;
  try { localStorage.setItem(SHAPE_COLOR_KEY, hex); } catch {}
}

/* Each preset appends its outline to the current path inside (x,y,w,h). */
const SHAPE_PRESETS = {
  rect: { label: 'Rectangle', icon: '▮', path: (c, x, y, w, h) => c.rect(x, y, w, h) },
  rounded: {
    label: 'Rounded rectangle', icon: '▢',
    path: (c, x, y, w, h) => c.roundRect(x, y, w, h, Math.min(w, h) * 0.14),
  },
  ellipse: {
    label: 'Ellipse', icon: '⬤',
    path: (c, x, y, w, h) => c.ellipse(x + w / 2, y + h / 2, w / 2, h / 2, 0, 0, Math.PI * 2),
  },
  triangle: {
    label: 'Triangle', icon: '▲',
    path: (c, x, y, w, h) => {
      c.moveTo(x + w / 2, y);
      c.lineTo(x + w, y + h);
      c.lineTo(x, y + h);
      c.closePath();
    },
  },
  diamond: {
    label: 'Diamond', icon: '◆',
    path: (c, x, y, w, h) => {
      c.moveTo(x + w / 2, y);
      c.lineTo(x + w, y + h / 2);
      c.lineTo(x + w / 2, y + h);
      c.lineTo(x, y + h / 2);
      c.closePath();
    },
  },
  star: {
    label: 'Star', icon: '★',
    path: (c, x, y, w, h) => {
      const cx = x + w / 2, cy = y + h / 2;
      for (let i = 0; i < 10; i++) {
        const r = i % 2 === 0 ? 1 : 0.42;
        const a = -Math.PI / 2 + (i * Math.PI) / 5;
        const px = cx + Math.cos(a) * r * (w / 2);
        const py = cy + Math.sin(a) * r * (h / 2);
        if (i === 0) c.moveTo(px, py);
        else c.lineTo(px, py);
      }
      c.closePath();
    },
  },
  heart: {
    label: 'Heart', icon: '♥',
    path: (c, x, y, w, h) => {
      const cx = x + w / 2;
      const top = h * 0.3;
      c.moveTo(cx, y + top);
      c.bezierCurveTo(cx, y, x, y, x, y + top);
      c.bezierCurveTo(x, y + (h + top) / 2, cx, y + (h + top) / 2, cx, y + h);
      c.bezierCurveTo(cx, y + (h + top) / 2, x + w, y + (h + top) / 2, x + w, y + top);
      c.bezierCurveTo(x + w, y, cx, y, cx, y + top);
    },
  },
  arrow: {
    label: 'Arrow', icon: '➜',
    path: (c, x, y, w, h) => {
      c.moveTo(x, y + h * 0.3);
      c.lineTo(x + w * 0.55, y + h * 0.3);
      c.lineTo(x + w * 0.55, y);
      c.lineTo(x + w, y + h * 0.5);
      c.lineTo(x + w * 0.55, y + h);
      c.lineTo(x + w * 0.55, y + h * 0.7);
      c.lineTo(x, y + h * 0.7);
      c.closePath();
    },
  },
};

/** A shape entry: a preset + fill drawn into `rect` of the layer's texture,
 * in UNIT space (0..1, y down) so it survives the layer being rescaled. */
function newShape(presetId, color, rect = { x: 0, y: 0, w: 1, h: 1 }) {
  return { id: uid('shp'), preset: presetId, color, ...rect };
}

/** True for a shape layer — a media clip whose texture we draw ourselves. */
function isShapeClip(clip) {
  return clip?.kind === 'media' && Array.isArray(clip.shapes) && clip.shapes.length > 0;
}

/** Render every shape of a layer into one texture-sized canvas. Later
 * entries paint over earlier ones, so array order IS z-order. */
function drawShapeCanvas(shapes) {
  const s = SHAPE_TEX_SIZE;
  const c = document.createElement('canvas');
  c.width = c.height = s;
  const ctx2d = c.getContext('2d');
  // Tiny inset so the anti-aliased edge isn't clipped by the texture border.
  // Applied per shape (rather than to the texture) now that a shape can sit
  // anywhere in the square — only ones touching the border need it, and 0.4%
  // is imperceptible on the ones that don't.
  const inset = s * 0.004;
  for (const shape of shapes) {
    const x = shape.x * s + inset;
    const y = shape.y * s + inset;
    const w = Math.max(1, shape.w * s - 2 * inset);
    const h = Math.max(1, shape.h * s - 2 * inset);
    ctx2d.fillStyle = shape.color;
    ctx2d.beginPath();
    (SHAPE_PRESETS[shape.preset] ?? SHAPE_PRESETS.rect).path(ctx2d, x, y, w, h);
    ctx2d.fill();
  }
  return c;
}

/** Give `clip` its own shape asset (keyed by clip id) and redraw the
 * texture when the shapes no longer match what's uploaded. */
function ensureShapeAsset(clip) {
  const id = `shape:${clip.id}`;
  clip.assetId = id;
  let asset = assets.get(id);
  if (!asset) {
    const texture = fx.device.createTexture({
      label: `shape layer ${clip.id}`,
      size: [SHAPE_TEX_SIZE, SHAPE_TEX_SIZE],
      format: 'rgba8unorm',
      // RENDER_ATTACHMENT: required by copyExternalImageToTexture's
      // GPU-canvas blit path.
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    asset = {
      id, kind: 'shape', name: '', ready: false,
      w: SHAPE_TEX_SIZE, h: SHAPE_TEX_SIZE, duration: null,
      el: null, texture, view: texture.createView(),
    };
    assets.set(id, asset);
  }
  const key = clip.shapes
    .map((s) => `${s.preset}|${s.color}|${s.x},${s.y},${s.w},${s.h}`).join(';');
  if (asset._shapeKey !== key) {
    asset._shapeKey = key;
    asset.name = clip.shapes.length === 1
      ? (SHAPE_PRESETS[clip.shapes[0].preset]?.label ?? 'Shape')
      : `${clip.shapes.length} shapes`;
    fx.device.queue.copyExternalImageToTexture(
      { source: drawShapeCanvas(clip.shapes) },
      { texture: asset.texture }, [SHAPE_TEX_SIZE, SHAPE_TEX_SIZE]);
  }
  asset.ready = true;
  return asset;
}

/* Duplicated / split / undo-restored shape clips can reference another
 * clip's asset (structuredClone copies assetId) or carry settings that no
 * longer match the uploaded texture — re-key every shape clip to its own
 * asset and redraw stale ones. Cheap: one string compare per shape clip. */
function reconcileShapeAssets() {
  if (!fx?.device) return;
  for (const track of comp.tracks)
    for (const clip of track.clips)
      if (isShapeClip(clip)) ensureShapeAsset(clip);
}

/* -- layer footprint ---------------------------------------------------
 *
 * A shape layer's texture square IS its footprint: the box the gizmo draws
 * and the transform scales. Shape rects are stored in that square's unit
 * space, so after adding or removing a shape the box no longer hugs the
 * content — and a shape drawn outside it would be clipped away entirely.
 *
 * refitShapeLayer shrink-wraps the square onto the union of the shapes and
 * applies the inverse change to the clip transform, which leaves every shape
 * exactly where it was on screen. Both halves of the compensation are affine
 * (one scale, one offset), so they apply cleanly to keyframe values as well
 * as to the static value.
 */

/** Add `d` to a property's value and to every keyframe on it. */
function offsetProp(prop, d) {
  if (!prop || !d) return;
  prop.v += d;
  for (const k of prop.keys ?? []) k.v += d;
}

/** Multiply a property's value and every keyframe on it by `k`. */
function scaleProp(prop, k) {
  if (!prop || k === 1) return;
  prop.v *= k;
  for (const key of prop.keys ?? []) key.v *= k;
}

function refitShapeLayer(clip) {
  const shapes = clip.shapes;
  if (!shapes?.length) return;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of shapes) {
    x0 = Math.min(x0, s.x); y0 = Math.min(y0, s.y);
    x1 = Math.max(x1, s.x + s.w); y1 = Math.max(y1, s.y + s.h);
  }
  const uw = x1 - x0, uh = y1 - y0;
  if (!(uw > 1e-6 && uh > 1e-6)) return;
  // Already shrink-wrapped (the common single-shape case) — don't churn the
  // transform, and don't drift it by floating-point dust.
  const EPS = 1e-6;
  if (Math.abs(x0) < EPS && Math.abs(y0) < EPS &&
      Math.abs(uw - 1) < EPS && Math.abs(uh - 1) < EPS) return;

  for (const s of shapes) {
    s.x = (s.x - x0) / uw;
    s.w /= uw;
    s.y = (s.y - y0) / uh;
    s.h /= uh;
  }

  // The footprint grows by (uw, uh) about its own centre, which also moves;
  // undo both on the transform. Scale is read at the current time, so with an
  // ANIMATED scale the offset is only exact at the playhead — an acceptable
  // corner (the alternative is refusing to refit animated layers at all).
  const t = tCur - clip.start;
  const sx = evalProp(clip.props.scaleX, t) / 100;
  const sy = evalProp(clip.props.scaleY, t) / 100;
  const rot = evalProp(clip.props.rot, t) * Math.PI / 180;
  const dxLocal = ((x0 + x1) / 2 - 0.5) * SHAPE_TEX_SIZE * sx;
  const dyLocal = ((y0 + y1) / 2 - 0.5) * SHAPE_TEX_SIZE * sy;
  offsetProp(clip.props.x, dxLocal * Math.cos(rot) - dyLocal * Math.sin(rot));
  offsetProp(clip.props.y, dxLocal * Math.sin(rot) + dyLocal * Math.cos(rot));
  scaleProp(clip.props.scaleX, uw);
  scaleProp(clip.props.scaleY, uh);
}

/** Where a shape layer's texture square lands in comp pixels, right now.
 * Signed width/height so a mirrored layer (negative scale) inverts too. */
function shapeLayerFrame(clip) {
  const t = tCur - clip.start;
  return {
    cx: evalProp(clip.props.x, t),
    cy: evalProp(clip.props.y, t),
    w: SHAPE_TEX_SIZE * (evalProp(clip.props.scaleX, t) / 100),
    h: SHAPE_TEX_SIZE * (evalProp(clip.props.scaleY, t) / 100),
    rot: evalProp(clip.props.rot, t) * Math.PI / 180,
  };
}

/** Add a shape to an EXISTING layer from a rect drawn in comp pixels. The
 * rect is pulled back through the layer's transform into unit space, then
 * the footprint is refit so a shape dropped outside the layer's current box
 * grows the box instead of falling off the edge of the texture. */
function addShapeToLayer(clip, presetId, { cx, cy, w, h }) {
  const f = shapeLayerFrame(clip);
  if (!f.w || !f.h) { setStatus('layer is scaled to zero — can’t place a shape'); return; }
  const ddx = cx - f.cx, ddy = cy - f.cy;
  const lx = ddx * Math.cos(-f.rot) - ddy * Math.sin(-f.rot);
  const ly = ddx * Math.sin(-f.rot) + ddy * Math.cos(-f.rot);
  const uw = w / Math.abs(f.w), uh = h / Math.abs(f.h);
  const shape = newShape(presetId, lastShapeColor, {
    x: lx / f.w + 0.5 - uw / 2,
    y: ly / f.h + 0.5 - uh / 2,
    w: uw,
    h: uh,
  });
  history.record(comp, () => {
    clip.shapes.push(shape);
    refitShapeLayer(clip);
  });
  ensureShapeAsset(clip);
  timeline.selectClip(clip.id);
  onModelChange({ structural: false });
  setStatus(`added a ${(SHAPE_PRESETS[presetId]?.label ?? 'shape').toLowerCase()} to ${clip.name}`);
}

/* -- draw-to-create interaction --------------------------------------- */

// { preset, label, targetId } armed; + { start, d } once dragging.
// targetId names an existing shape layer to append to; null = new layer.
let shapeDraw = null;
const shapeRect = document.createElement('div');
shapeRect.id = 'shape-draw-rect';
shapeRect.hidden = true;
$('canvas-inner').appendChild(shapeRect);

function armShapeDraw(presetId, targetId = null) {
  stopMaskEdit();
  const label = SHAPE_PRESETS[presetId]?.label ?? 'shape';
  shapeDraw = { preset: presetId, label, targetId };
  viewer.classList.add('shape-drawing');
  // The gizmo would sit under the pointer while drawing; gizmoTarget() bows
  // out for as long as a draw is armed, so just refresh it.
  updateGizmo();
  const where = targetId
    ? `onto ${findClip(comp, targetId)?.clip.name ?? 'the layer'}`
    : 'as a new layer';
  setStatus(`drag on the preview to draw a ${label.toLowerCase()} ${where} (Shift = square, click = default size) — Esc cancels`);
}

function cancelShapeDraw() {
  if (!shapeDraw) return;
  shapeDraw = null;
  shapeRect.hidden = true;
  viewer.classList.remove('shape-drawing');
  window.removeEventListener('pointermove', shapeDrawMove);
  updateGizmo();   // the selected layer gets its handles back
}

function shapeDragRect(e) {
  let [sx, sy] = shapeDraw.start;
  let ex = e.clientX, ey = e.clientY;
  if (e.shiftKey) {
    const dx = ex - sx, dy = ey - sy;
    const m = Math.max(Math.abs(dx), Math.abs(dy));
    ex = sx + Math.sign(dx || 1) * m;
    ey = sy + Math.sign(dy || 1) * m;
  }
  return { left: Math.min(sx, ex), top: Math.min(sy, ey), w: Math.abs(ex - sx), h: Math.abs(ey - sy) };
}

// Capture phase so an armed draw wins over viewport selection/deselection.
viewer.addEventListener('pointerdown', (e) => {
  if (!shapeDraw || maskEdit || e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();
  shapeDraw.start = [e.clientX, e.clientY];
  shapeDraw.d = canvasDisplayRect();
  window.addEventListener('pointermove', shapeDrawMove);
  window.addEventListener('pointerup', shapeDrawUp, { once: true });
}, { capture: true });

function shapeDrawMove(e) {
  if (!shapeDraw?.start) return;
  const r = shapeDragRect(e);
  const innerR = $('canvas-inner').getBoundingClientRect();
  shapeRect.style.left = `${r.left - innerR.left}px`;
  shapeRect.style.top = `${r.top - innerR.top}px`;
  shapeRect.style.width = `${r.w}px`;
  shapeRect.style.height = `${r.h}px`;
  shapeRect.hidden = false;
}

function shapeDrawUp(e) {
  const g = shapeDraw;
  if (!g?.start) { cancelShapeDraw(); return; }
  const r = shapeDragRect(e);
  const d = g.d;
  let w = r.w / d.s, h = r.h / d.s;
  let cx = (r.left + r.w / 2 - d.left) / d.s;
  let cy = (r.top + r.h / 2 - d.top) / d.s;
  if (w < 4 || h < 4) {
    // A plain click drops a default-sized shape at the point.
    w = h = Math.min(comp.width, comp.height) * 0.35;
    cx = (e.clientX - d.left) / d.s;
    cy = (e.clientY - d.top) / d.s;
  }
  const preset = g.preset;
  const target = g.targetId ? findClip(comp, g.targetId)?.clip : null;
  cancelShapeDraw();
  if (target && isShapeClip(target)) addShapeToLayer(target, preset, { cx, cy, w, h });
  else createShapeClip(preset, cx, cy, w, h);
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && shapeDraw) {
    cancelShapeDraw();
    setStatus('shape cancelled');
  }
});

/** New shape clip on a new top track, sized/positioned from the drawn
 * rect (comp px). Spans the whole timeline like an imported image. */
function createShapeClip(presetId, cx, cy, w, h) {
  const props = {};
  for (const [key, , def] of MEDIA_PROPS) props[key] = newProp(def(comp));
  const clip = {
    id: uid('clip'),
    kind: 'media',
    name: SHAPE_PRESETS[presetId]?.label ?? 'Shape',
    assetId: null,
    start: 0,
    dur: Math.max(1 / comp.fps, comp.dur),
    in: 0,
    // One shape filling the layer's texture square; more can be drawn into
    // the same layer later, each with its own rect (see addShapeToLayer).
    shapes: [newShape(presetId, lastShapeColor)],
    props,
    effects: [],
    mask: null,
  };
  clip.props.x.v = Math.round(cx);
  clip.props.y.v = Math.round(cy);
  clip.props.scaleX.v = Math.round((w / SHAPE_TEX_SIZE) * 10000) / 100;
  clip.props.scaleY.v = Math.round((h / SHAPE_TEX_SIZE) * 10000) / 100;
  ensureShapeAsset(clip);
  history.record(comp, () => {
    const track = newTrack(clip.name);
    track.clips.push(clip);
    comp.tracks.unshift(track);
    ensureDur(comp);
  });
  timeline.selectClip(clip.id);
  onModelChange({ structural: true });
  setStatus(`added ${clip.name.toLowerCase()} — + Shape in the inspector adds more to this layer`);
}

/* -- inspector section -------------------------------------------------
 * One row per shape (they stack in array order, last on top) plus the
 * + Shape button that arms a draw onto THIS layer. */

function renderShapeSection(clip) {
  const add = secBtn('＋ Shape', 'draw another shape into this layer', () => {
    collapsedSections.delete('shape');
    openShapePicker(add, (pid) => armShapeDraw(pid, clip.id));
  });
  const { sec, body, collapsed } = inspSection('shape', 'Shapes', {
    count: clip.shapes.length,
    actions: [add],
  });
  if (collapsed) { inspectorEl.appendChild(sec); return; }
  const box = body;

  clip.shapes.forEach((shape, i) => {
    const row = document.createElement('div');
    row.className = 'shape-row';

    const sel = document.createElement('select');
    sel.title = 'shape preset';
    for (const [pid, p] of Object.entries(SHAPE_PRESETS)) {
      const o = document.createElement('option');
      o.value = pid;
      o.textContent = `${p.icon} ${p.label}`;
      sel.appendChild(o);
    }
    sel.value = shape.preset;
    sel.onchange = () => {
      history.record(comp, () => {
        const old = SHAPE_PRESETS[shape.preset]?.label;
        shape.preset = sel.value;
        // Auto-rename a single-shape layer, and only if the user hasn't.
        if (clip.shapes.length === 1 && clip.name === old)
          clip.name = SHAPE_PRESETS[sel.value]?.label ?? clip.name;
      });
      ensureShapeAsset(clip);
      onModelChange({ structural: false });
    };

    const colLabel = document.createElement('label');
    colLabel.className = 'shape-color';
    const col = document.createElement('input');
    col.type = 'color';
    col.value = shape.color;
    col.title = 'fill color';
    let editing = false;   // one undo step per picker session
    col.oninput = () => {
      if (!editing) { history.begin(comp); editing = true; }
      shape.color = col.value;
      lastShapeColor = col.value;   // live; persisted once the picker commits
      ensureShapeAsset(clip);
    };
    col.onchange = () => {
      if (editing) { history.commit(comp); editing = false; }
      setLastShapeColor(col.value);
      scheduleSave();
    };
    colLabel.append(col, 'fill');
    row.append(sel, colLabel);

    // Draw order only matters once shapes can overlap.
    if (clip.shapes.length > 1) {
      const move = (dir) => {
        const b = document.createElement('button');
        b.className = 'tl-mini';
        b.textContent = dir < 0 ? '▲' : '▼';
        b.title = dir < 0 ? 'move down the stack' : 'move up the stack';
        b.disabled = i + dir < 0 || i + dir >= clip.shapes.length;
        b.onclick = () => {
          history.record(comp, () => {
            const [s] = clip.shapes.splice(i, 1);
            clip.shapes.splice(i + dir, 0, s);
          });
          ensureShapeAsset(clip);
          onModelChange({ structural: false });
        };
        return b;
      };
      const del = document.createElement('button');
      del.className = 'tl-mini';
      del.textContent = '✕';
      del.title = 'remove this shape';
      del.onclick = () => {
        history.record(comp, () => {
          clip.shapes.splice(i, 1);
          refitShapeLayer(clip);
        });
        ensureShapeAsset(clip);
        onModelChange({ structural: false });
      };
      row.append(move(-1), move(1), del);
    }

    box.appendChild(row);
  });

  inspectorEl.appendChild(sec);
}

/** Where to hang a popover: an element's box, or a bare {x, y} click point
 * when the menu came from a right-click and there's nothing to anchor to. */
function anchorRect(anchor) {
  if (anchor instanceof Element) return anchor.getBoundingClientRect();
  return { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y, width: 0, height: 0 };
}

/**
 * Position an already-attached popup against its anchor and keep it on
 * screen. `prefer: 'above'` for anything hung off the ＋ Layer button —
 * it sits at the bottom of the window, so a list opening downwards runs
 * straight off the edge with its tail unreachable.
 */
function placePopup(el, anchor, { prefer = 'below', gap = 4 } = {}) {
  const r = anchorRect(anchor);
  const box = el.getBoundingClientRect();
  const below = r.bottom + gap;
  const above = r.top - box.height - gap;
  const fitsBelow = below + box.height <= innerHeight - 6;
  const fitsAbove = above >= 6;
  let top = prefer === 'above'
    ? (fitsAbove || !fitsBelow ? above : below)
    : (fitsBelow || !fitsAbove ? below : above);
  top = clamp(top, 6, Math.max(6, innerHeight - box.height - 6));
  el.style.top = `${top}px`;
  el.style.left = `${clamp(r.left, 6, Math.max(6, innerWidth - box.width - 6))}px`;
}

/** Small popover listing the shape presets. Used by + Shape (adds to a
 * layer) and by + Layer (starts a new one). */
function openShapePicker(anchor, onPick, { prefer = 'below' } = {}) {
  document.querySelector('.shape-picker')?.remove();
  const pop = document.createElement('div');
  pop.className = 'shape-picker menu-pop';
  const list = document.createElement('div');
  list.className = 'add-layer-list open';
  pop.appendChild(list);

  const close = () => { pop.remove(); document.removeEventListener('pointerdown', onDown, true); };
  for (const [pid, p] of Object.entries(SHAPE_PRESETS)) {
    const it = document.createElement('div');
    it.className = 'menu-item shape-item';
    // The icon is the preset's own geometry, drawn by the same path
    // function that builds the layer — no glyph can misrepresent it.
    it.appendChild(shapeIconCanvas(p.path));
    const span = document.createElement('span');
    span.textContent = p.label;
    it.appendChild(span);
    it.onclick = () => { close(); onPick(pid); };
    list.appendChild(it);
  }
  const onDown = (e) => { if (!pop.contains(e.target) && e.target !== anchor) close(); };
  document.addEventListener('pointerdown', onDown, true);

  document.body.appendChild(pop);   // attach first: placement needs a size
  placePopup(pop, anchor, { prefer });
}

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

function applyOverlaySource(clip, effect, texName, source, descriptor) {
  const spec = specFor(clip, effect);
  (spec.textureOverrides ??= {})[texName] = source;
  (effect.overlay ??= {})[texName] = descriptor;
  markChainDirty(clip.id);
  scheduleSave();
}

const stampFileInput = $('stamp-file-input');
let stampPickTarget = null;   // { clipId, effectId, texName }

stampFileInput.addEventListener('change', async () => {
  const f = stampFileInput.files[0];
  const target = stampPickTarget;
  stampFileInput.value = '';
  stampPickTarget = null;
  if (!f || !target) return;
  const hit = findClip(comp, target.clipId);
  const effect = hit && findEffect(hit.clip, target.effectId);
  if (!effect) return;
  const bmp = await createImageBitmap(f);
  const c = document.createElement('canvas');
  c.width = bmp.width;
  c.height = bmp.height;
  c.getContext('2d').drawImage(bmp, 0, 0);
  const dataURL = c.toDataURL('image/png');
  applyOverlaySource(hit.clip, effect, target.texName, bmp,
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

/**
 * Fill `list` with the searchable effect catalogue. Shared by the toolbar's
 * ＋ box and the inspector's + Effect picker — both of which append to a
 * clip's stack; only the target differs (the focused layer vs. the row's own
 * clip). Layers themselves are made with + Layer, so nothing here creates
 * one.
 *
 * Visual and audio effects live in one list but never blur together: each
 * kind sits under its own banner, and a clip that can only take one of them
 * is never offered the other.
 *
 * @param {object} opts {query, onPick, folders:Set, rerender, header:string,
 *                       visual:boolean, audio:boolean}
 */
function renderChoiceList(list, {
  query, onPick, folders, rerender, header = null, visual = true, audio = false,
}) {
  const q = query.trim().toLowerCase();
  list.replaceChildren();
  const savedList = Object.keys(loadSaved()).sort();
  const both = visual && audio;

  if (header) {
    const h = document.createElement('div');
    h.className = 'menu-target';
    h.textContent = header;
    list.appendChild(h);
  }

  /* Only worth banners when the list actually holds both kinds. */
  const group = (label, kind) => {
    if (!both) return;
    const g = document.createElement('div');
    g.className = `menu-group ${kind}`;
    g.textContent = label;
    list.appendChild(g);
  };

  const addItem = (label, choice, note = null) => {
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
    it.addEventListener('click', () => onPick(choice));
    list.appendChild(it);
  };

  // The note is a short group word, never the effect's full hint: a long
  // note is nowrap and would eat the name it's supposed to annotate.
  const addAudioItem = (def) =>
    addItem(`♪ ${def.label}`, `__audio__:${def.id}`, def.group);

  if (q) {
    if (visual) {
      group('Visual effects', 'visual');
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
    }
    if (audio) {
      group('Audio effects', 'audio');
      for (const def of AUDIO_EFFECTS)
        if (def.label.toLowerCase().includes(q) || def.id.includes(q)
          || def.hint.toLowerCase().includes(q) || 'audio sound'.includes(q))
          addAudioItem(def);
    }
    if (!list.querySelector('.menu-item')) {
      list.replaceChildren();
      const none = document.createElement('div');
      none.className = 'menu-empty';
      none.textContent = 'no matches';
      list.appendChild(none);
    }
    return;
  }

  if (audio && !visual) {
    for (const def of AUDIO_EFFECTS) addAudioItem(def);
    return;
  }

  group('Visual effects', 'visual');
  addItem('✎ custom shader (write your own)', '__custom__');
  addItem('T text / title', '__title__');

  const folder = (id, label, children) => {
    if (!children.length) return;
    const open = folders.has(id);
    const head = document.createElement('div');
    head.className = 'menu-folder';
    const title = document.createElement('span');
    title.textContent = `${open ? '▾' : '▸'} ${label}`;
    const count = document.createElement('span');
    count.className = 'note';
    count.textContent = String(children.length);
    head.append(title, count);
    head.addEventListener('click', () => {
      if (open) folders.delete(id);
      else folders.add(id);
      rerender();
    });
    list.appendChild(head);
    if (open) for (const c of children) addItem(c.label, c.choice, c.note);
  };

  folder('saved', 'Saved shaders',
    savedList.map((name) => ({ label: `🗎 ${name}`, choice: `__saved__:${name}` })));
  for (const cat of manifest.categories ?? [])
    folder(cat.id, cat.label,
      manifest.effects
        .filter((e) => e.category === cat.id)
        .map((e) => ({ label: e.name, choice: e.path })));

  if (!audio) return;
  // Flat, not foldered: there are a handful of these and they're the ones
  // you reach for by name.
  group('Audio effects', 'audio');
  for (const def of AUDIO_EFFECTS) addAudioItem(def);
}

/* What a clip's stack can hold. An adjustment layer processes the picture
 * below it and has no sound of its own; an audio clip is the mirror image;
 * a video clip is the only thing that takes both. */
function clipTakesVisualFx(clip) {
  return clip?.kind === 'fx' || clip?.kind === 'media';
}

function clipTakesAudioFx(clip) {
  if (clip?.kind === 'audio') return true;
  if (clip?.kind !== 'media' || isShapeClip(clip)) return false;
  const asset = assets.get(clip.assetId);
  // Only a still is definitely silent. An asset that's missing or still
  // loading counts as "might have sound" — hiding the section there would
  // leave a video clip with no way to reach its own audio.
  return !asset || asset.kind === 'video' || asset.kind === 'audio';
}

/* The panel adds to the layer the inspector is focused on. Shapes are not
 * in this list — they are layers, not effects, and live on + Layer (new
 * layer) and the inspector's + Shape (another shape on this one). */
function rebuildAddMenu() {
  const target = timeline?.selectedClip ?? null;
  renderChoiceList(addLayerList, {
    query: addLayerSearch.value,
    folders: openFolders,
    rerender: rebuildAddMenu,
    visual: !target || clipTakesVisualFx(target),
    audio: !!target && clipTakesAudioFx(target),
    header: target
      ? `adding to ${target.name}`
      : 'no layer yet — this will create an adjustment layer',
    onPick: (choice) => { closeAddMenu(); addEffectToFocusedLayer(choice); },
  });
}

/** Append the chosen effect to the focused layer. With nothing to add to
 * (an empty comp) fall back to the old behaviour and make the layer. */
function addEffectToFocusedLayer(choice) {
  const clip = ensureFocusedLayer();
  if (clip) addEffectToClip(clip, choice);
  else addFxLayer(choice);
}

/** Keep the effect box's placeholder honest about where a pick will land. */
function syncAddPlaceholder() {
  if (!addLayerSearch) return;
  const target = timeline?.selectedClip;
  addLayerSearch.placeholder = target
    ? `Add an effect to ${target.name}`
    : 'Add an effect — search or browse';
}

/* ---- + Layer -------------------------------------------------------- */

/** The new-layer menu. Opened by the timeline's ＋ Layer button and by
 * right-clicking empty timeline or preview space, so `anchor` is either
 * that button or a bare {x, y} pointer position. It lives here rather
 * than in the timeline because only the app knows what a layer can be. */
function showAddLayerMenu(anchor) {
  const r = anchorRect(anchor);
  // Opened from the ＋ Layer button, both this menu and the shape list it
  // leads to open UPWARDS: the button sits on the bottom edge of the
  // window, so a downward list has its tail off screen.
  const fromButton = anchor instanceof Element;
  showMenu(r.left, fromButton ? r.top - 4 : r.bottom + 4, [
    {
      icon: LAYER_ICONS.fx,
      label: 'Adjustment layer',
      action: () => addFxLayer(null),
    },
    {
      icon: LAYER_ICONS.shape,
      label: 'Shape layer…',
      action: () => openShapePicker(anchor, (pid) => armShapeDraw(pid),
        { prefer: fromButton ? 'above' : 'below' }),
    },
    {
      icon: LAYER_ICONS.text,
      label: 'Text / title layer',
      action: () => addFxLayer('__title__'),
    },
    '-',
    {
      icon: LAYER_ICONS.import,
      label: 'Import media…',
      action: () => $('file-input').click(),
    },
  ], { above: fromButton });
}

// Right-clicking the preview outside any layer adds one too. The gizmo
// stops propagation for its own menu, so this only sees empty space.
$('preview-wrap').addEventListener('contextmenu', (e) => {
  if (e.target.closest('#view-controls')) return;
  e.preventDefault();
  showAddLayerMenu({ x: e.clientX, y: e.clientY });
});

/* ---- + Effect picker (append to the selected clip's stack) ---------- */

const pickerFolders = new Set();

function openEffectPicker(clip, anchor, { visual = null, audio = null } = {}) {
  const wantVisual = visual ?? clipTakesVisualFx(clip);
  const wantAudio = audio ?? clipTakesAudioFx(clip);
  document.querySelector('.fx-picker')?.remove();
  const pop = document.createElement('div');
  pop.className = 'fx-picker menu-pop';
  const search = document.createElement('input');
  search.type = 'search';
  search.placeholder = wantAudio && !wantVisual ? 'search audio effects…' : 'search effects…';
  search.className = 'fx-picker-search';
  const list = document.createElement('div');
  list.className = 'add-layer-list open';
  pop.append(search, list);

  const close = () => { pop.remove(); document.removeEventListener('pointerdown', onDown, true); };
  const draw = () => renderChoiceList(list, {
    query: search.value,
    folders: pickerFolders,
    rerender: draw,
    visual: wantVisual,
    audio: wantAudio,
    onPick: (choice) => { close(); addEffectToClip(clip, choice); },
  });
  search.addEventListener('input', draw);
  search.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Escape') close();
    else if (e.key === 'Enter') list.querySelector('.menu-item')?.click();
  });
  const onDown = (e) => { if (!pop.contains(e.target) && e.target !== anchor) close(); };
  document.addEventListener('pointerdown', onDown, true);

  document.body.appendChild(pop);
  draw();
  // Draw first — the list's height decides whether it fits below.
  placePopup(pop, anchor);
  search.focus();
}

/** Resolve a menu choice to an effect spec (null for non-effect choices). */
function effectSpecForChoice(choice) {
  if (choice.startsWith('__audio__:')) {
    const def = audioEffectDef(choice.slice('__audio__:'.length));
    return def ? { spec: { fxKind: 'audio', audioId: def.id, label: def.label } } : null;
  }
  if (choice === '__custom__')
    return { spec: { fxKind: 'custom', source: CUSTOM_BOILERPLATE, label: 'custom shader' } };
  if (choice === '__title__')
    return { spec: { fxKind: 'preset', path: STAMP_PRESET_PATH, label: 'title' }, title: true };
  if (choice.startsWith('__saved__:')) {
    const name = choice.slice('__saved__:'.length);
    const saved = loadSaved()[name];
    if (!saved) return null;
    return { spec: { fxKind: 'custom', source: saved.source, label: name, savedName: name } };
  }
  return { spec: { fxKind: 'preset', path: choice, label: choice.split('/').pop().replace(/\.slangp$/, '') } };
}

/** Append an effect to an existing clip's stack. */
async function addEffectToClip(clip, choice) {
  const resolved = choice && fx && effectSpecForChoice(choice);
  if (!resolved) return;
  const effect = newEffect(resolved.spec);
  if (resolved.title) effect.overlay = { Stamp: { kind: 'text', state: { ...DEFAULT_TITLE } } };
  history.record(comp, () => { (clip.effects ??= []).push(effect); });
  openEffects.add(effect.id);
  onModelChange({ structural: true });
  if (isAudioEffect(effect)) {
    // Nothing to compile — the graph rebuilds on the next frame that
    // plays this clip.
    setStatus(`added ${effect.name} to ${clip.name}`);
    return;
  }
  setStatus(`added ${effect.name} to ${clip.name} — compiling…`);
  try {
    await ensureParamMeta(clip, effect);
    setStatus(`${effect.name} ready`);
  } catch {
    setStatus(`${effect.name} failed to compile — see inspector`);
  }
}

/** New adjustment layer on a new top track. With `choice` it starts with
 * that effect; without one it's an empty layer ready for the effect panel. */
async function addFxLayer(choice) {
  if (!fx) return;
  const resolved = choice ? effectSpecForChoice(choice) : null;
  if (choice && !resolved) return;
  if (resolved?.spec.fxKind === 'audio') {
    // An adjustment layer has no sound to process.
    setStatus('audio effects go on a clip that has audio — select one first');
    return;
  }

  // New adjustment layers cover the whole timeline; trim them when needed.
  const clip = newFxClip(resolved?.spec ?? null, 0, Math.max(1 / comp.fps, comp.dur));
  const effect = clip.effects[0] ?? null;
  if (resolved?.title) effect.overlay = { Stamp: { kind: 'text', state: { ...DEFAULT_TITLE } } };

  history.record(comp, () => {
    const track = newTrack(clip.name);
    track.clips.push(clip);
    comp.tracks.unshift(track);
    ensureDur(comp);
  });
  timeline.selectClip(clip.id);
  timeline.expanded.add(clip.id);
  onModelChange({ structural: true });
  if (!effect) {
    setStatus(`added ${clip.name} — pick an effect above to fill it`);
    addLayerSearch.focus();
    return;
  }
  openEffects.add(effect.id);
  setStatus(`added ${clip.name} — compiling…`);
  const t0 = performance.now();
  try {
    await ensureParamMeta(clip, effect);
    setStatus(`${clip.name} ready in ${Math.round(performance.now() - t0)} ms`);
  } catch (e) {
    setStatus(`${clip.name} failed to compile — see inspector`);
  }
}

/* =====================================================================
 * Custom shader compile
 * =================================================================== */

const editorDrafts = new Map();   // effectId -> unsaved editor text

async function compileCustomEffect(clip, effect, source) {
  history.record(comp, () => { effect.source = source; });
  const spec = specFor(clip, effect);
  virtualFiles.set(spec.dir + 'custom.slang', source);
  fx.invalidateModules(spec.dir);
  paramMetaCache.delete(effect.id);
  markChainDirty(clip.id);
  setStatus('compiling custom shader…');
  const t0 = performance.now();
  try {
    await ensureParamMeta(clip, effect);
    await syncFxChain(tCur);
    const err = fxSpecs.get(effect.id)?.error;
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

const inspLive = [];      // [{clip, key, slider, num, scale}] animated bindings
const inspWidgets = [];   // [{redraw, live}] curve widgets on this render

function updateInspectorLive() {
  for (const b of inspLive) {
    if (document.activeElement === b.num || b.dragging) continue;
    const v = valueAt(b.clip, b.key);
    b.slider.value = String(b.scale ? b.scale.toPos(v) : v);
    b.num.value = fmtVal(v);
  }
  // A response curve tracks its parameters, so an animated filter sweep
  // draws itself moving. Static ones are left alone.
  for (const w of inspWidgets) if (w.live) w.redraw();
}

/** Redraw every curve widget now (a slider moved under one). */
function refreshInspWidgets() {
  for (const w of inspWidgets) w.redraw();
}

const fmtVal = (v) => (+v).toFixed(3).replace(/\.?0+$/, '') || '0';

/* The inspector always has a layer in focus: it's the target for the effect
 * panel, so "nothing selected" would leave picks with nowhere to go. When a
 * selection would be empty we fall back to the topmost layer that's live at
 * the playhead (else simply the topmost), which is what the eye is on.
 *
 * Modal viewport interactions are the exception — a shape draw deliberately
 * hands the pointer to the canvas, and re-selecting under it would put the
 * gizmo back in the way.
 *
 * @returns {object|null} the focused clip, or null when the comp is empty.
 */
function ensureFocusedLayer() {
  if (!timeline) return null;
  const current = timeline.selectedClip;
  if (current || shapeDraw || maskEdit) return current;

  let fallback = null;
  for (const track of comp.tracks) {
    if (track.hidden) continue;
    for (const clip of track.clips) {
      fallback ??= clip;
      if (tCur >= clip.start && tCur < clipEnd(clip)) {
        timeline.selectClip(clip.id, { quiet: true });
        focusIsFallback = true;
        return clip;
      }
    }
  }
  if (fallback) { timeline.selectClip(fallback.id, { quiet: true }); focusIsFallback = true; }
  return fallback;
}

function renderInspector() {
  if (!timeline) return;
  inspLive.length = 0;
  inspWidgets.length = 0;
  inspRows.clear();
  const clip = ensureFocusedLayer();
  syncAddPlaceholder();
  inspectorEl.replaceChildren();

  if (!clip) {
    const div = document.createElement('div');
    div.className = 'insp-empty';
    div.innerHTML = `
      <p>Nothing to edit yet — add a layer.</p>
      <p class="hint">· <b>＋ Layer</b> at the top of the timeline makes an adjustment
      layer for effects, a shape you draw straight onto the preview, or a title<br>
      · right-click empty timeline or preview space for the same menu<br>
      · <b>Import media…</b> or drop files to create media clips<br>
      · the <b>＋ search box</b> adds effects to the layer in focus<br>
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
  const asset = hasSource(clip) ? assets.get(clip.assetId) : null;

  /* -- identity: what this layer is, what it's made of, and the two
   *    destructive-ish controls (bypass, delete). Everything below is a
   *    section; this block is the panel's title. -- */
  const head = document.createElement('div');
  head.className = 'insp-head';
  const kind = document.createElement('span');
  kind.className = 'insp-kind ' + clip.kind;
  kind.innerHTML = clipIcon(clip, asset);
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

  /* Source line: the file behind the layer, or why there isn't one. Shape
   * layers draw themselves, so they say what they are instead. */
  const meta = document.createElement('div');
  meta.className = 'insp-src';
  if (isShapeClip(clip)) {
    meta.textContent = `shape layer · ${clip.shapes.length} shape${clip.shapes.length > 1 ? 's' : ''}`;
  } else if (hasSource(clip)) {
    meta.textContent = asset
      ? [asset.name,
        asset.w ? `${asset.w}×${asset.h}` : null,
        asset.duration ? `${asset.duration.toFixed(2)}s` : null,
      ].filter(Boolean).join(' · ') + (asset.ready ? '' : ' (loading…)')
      : `${clip.kind === 'audio' ? 'audio' : 'media'} offline — re-import the file`;
  } else {
    meta.textContent = 'adjustment layer · effects apply to everything below';
  }
  inspectorEl.appendChild(meta);

  /* Timing is not in here: start / length / trim-in are all easier to
   * judge against the ruler, so the timeline owns them (drag the clip,
   * drag its edges, split at the playhead). */

  /* -- shape settings (preset + fill color) -- */
  if (isShapeClip(clip)) renderShapeSection(clip);

  /* -- transform: where the layer sits and how it merges down -- */
  if (clip.kind === 'media') {
    const { sec, body } = inspSection('transform', 'Transform', { hint: '⏱ animates' });
    // Volume lives under Sound, not here — it's not a transform.
    for (const def of mediaPropDefs().filter((d) => d.key !== 'volume'))
      addParamRow(body, clip, def);
    body.appendChild(blendRow(clip));
    inspectorEl.appendChild(sec);
  }

  /* -- mask (fx: gates the whole stack; media: cuts the clip's alpha) -- */
  if (clip.kind !== 'audio') renderMaskSection(clip);

  /* -- the two effect chains -- */
  renderEffectStack(clip);

  // Curve widgets can only measure themselves once they're in the document
  // — and rAF can't be trusted to get there (it doesn't run at all while
  // the window is occluded), so paint them synchronously now.
  refreshInspWidgets();
}

/* ---- inspector sections ----------------------------------------------
 * Every block below the header is one of these: a micro-caps title, an
 * optional count, an optional hint, right-aligned actions, and a body that
 * folds away. One shape for all of them is what keeps a panel this dense
 * readable — the eye finds the boundaries without reading a word. */

const INSP_SECTIONS_KEY = 'lowkey-studio.insp-sections';

const collapsedSections = new Set((() => {
  try { return JSON.parse(localStorage.getItem(INSP_SECTIONS_KEY)) ?? []; }
  catch { return []; }
})());

function toggleSection(key) {
  if (collapsedSections.has(key)) collapsedSections.delete(key);
  else collapsedSections.add(key);
  try { localStorage.setItem(INSP_SECTIONS_KEY, JSON.stringify([...collapsedSections])); } catch {}
  renderInspector();
}

/**
 * @returns {{sec: HTMLElement, body: HTMLElement, collapsed: boolean}}
 */
function inspSection(key, title, { count = null, hint = null, actions = [] } = {}) {
  const collapsed = collapsedSections.has(key);
  const sec = document.createElement('section');
  sec.className = `insp-sec ${key}` + (collapsed ? ' collapsed' : '');

  const head = document.createElement('div');
  head.className = 'insp-sec-head';
  head.addEventListener('click', (e) => {
    if (e.target.closest('button, select, input, label')) return;
    toggleSection(key);
  });

  const tw = document.createElement('span');
  tw.className = 'insp-sec-tw';
  tw.textContent = collapsed ? '▸' : '▾';

  const h = document.createElement('h3');
  h.textContent = title;

  head.append(tw, h);
  if (count) {
    const c = document.createElement('span');
    c.className = 'insp-count';
    c.textContent = String(count);
    head.appendChild(c);
  }
  const hintEl = document.createElement('span');
  hintEl.className = 'insp-sec-hint';
  hintEl.textContent = hint ?? '';
  head.appendChild(hintEl);          // flexes, so actions stay right-aligned
  for (const a of actions) head.appendChild(a);

  const body = document.createElement('div');
  body.className = 'insp-sec-body';
  sec.append(head, body);
  return { sec, body, collapsed };
}

/** A small header button — the section's own action (+ Effect, +Paint…). */
function secBtn(label, title, onClick) {
  const b = document.createElement('button');
  b.className = 'btn insp-sec-btn';
  b.textContent = label;
  b.title = title;
  b.onclick = (e) => { e.stopPropagation(); onClick(e); };
  return b;
}

/** A property row plus its driver panel, appended together (they belong to
 * the same control and must never be separated). */
function addParamRow(parent, clip, def) {
  parent.appendChild(paramRow(clip, def));
  const dp = driverPanel(clip, def);
  if (dp) parent.appendChild(dp);
}

/** Blend mode, shaped like a param row so it lines up with Opacity above
 * it — the two answer the same question (how this layer merges down). */
function blendRow(clip) {
  const row = document.createElement('div');
  row.className = 'param-row sel';
  const label = document.createElement('label');
  label.textContent = 'Blend';
  label.title = 'how this layer combines with the layers below it';
  const sel = document.createElement('select');
  for (const [id, label2] of BLEND_MODES) {
    const o = document.createElement('option');
    o.value = id;
    o.textContent = label2;
    sel.appendChild(o);
  }
  sel.value = clip.blend ?? 'normal';
  sel.onchange = () => {
    history.record(comp, () => { clip.blend = sel.value; });
    onModelChange({ structural: false });
  };
  row.append(label, sel);
  return row;
}

/* ---- effect stack UI -------------------------------------------------
 * The heart of the adjustment-layer model: any clip owns an ordered stack.
 * On an fx clip the stack processes everything below it; on a media clip
 * it processes only that clip. Rows twirl open to reveal parameters, the
 * custom-shader editor and overlay textures for that one effect.
 *
 * Visual and audio effects live in the same stack and the same UI, but in
 * separate collapsible sections: they are independent chains (pixels vs.
 * samples), and a video clip carrying both would otherwise read as one
 * confusing pile. Each section keeps its own order — moving an effect
 * up/down steps past its own kind. */

const openEffects = new Set();   // effect ids twirled open in the inspector

/** The picture chain and the sound chain, each its own section. Sound
 * leads with the clip's Volume: level and effects are one mental group,
 * and it's where you'd look for either. */
function renderEffectStack(clip) {
  if (clipTakesVisualFx(clip)) inspectorEl.appendChild(fxSection(clip, 'visual'));
  if (clipTakesAudioFx(clip)) inspectorEl.appendChild(fxSection(clip, 'audio'));

  // No spill toggle any more: a media stack's coverage is measured by the
  // matte pass (see prepareMediaFx), so a blur spreads past the edges on
  // its own and no preset can smear itself over the whole frame. Clip an
  // effect deliberately with a mask, which is the tool for it.
}

function fxSection(clip, kind) {
  const audio = kind === 'audio';
  const effects = audio ? audioEffectsOf(clip) : visualEffectsOf(clip);
  const add = secBtn('+ Effect',
    audio ? 'add an audio effect to this clip’s chain'
      : 'add a visual effect to this clip’s stack',
    () => {
      collapsedSections.delete(kind);   // a pick that lands out of sight is a bug report
      openEffectPicker(clip, add, { visual: !audio, audio });
    });

  const { sec, body, collapsed } = inspSection(kind, audio ? 'Sound' : 'Effects', {
    count: effects.length,
    hint: audio
      ? 'level + effects on this clip'
      : (clip.kind === 'fx' ? 'applied to every layer below' : 'this clip only'),
    actions: [add],
  });
  if (collapsed) return sec;

  if (audio) {
    // A clip's own level, ahead of the chain it feeds.
    const volDef = clip.kind === 'audio'
      ? audioClipPropDefs()[0]
      : mediaPropDefs().find((d) => d.key === 'volume');
    if (volDef && clip.props?.volume) addParamRow(body, clip, volDef);
  }

  if (!effects.length) {
    const empty = document.createElement('div');
    empty.className = 'insp-note';
    empty.textContent = audio
      ? 'no audio effects yet — reverb, EQ, filters, compression…'
      : (clip.kind === 'fx'
        ? 'empty adjustment layer — add an effect to make it do something'
        : 'no effects on this clip');
    body.appendChild(empty);
  }
  for (const effect of effects) body.appendChild(effectRow(clip, effect));
  return sec;
}

function effectRow(clip, effect) {
  const audio = isAudioEffect(effect);
  const spec = audio ? null : fxSpecs.get(effect.id);
  const err = audio
    ? (audioEffectDef(effect.audioId) ? null : `unknown audio effect '${effect.audioId}'`)
    : (spec?.error || spec?.lastCompileError);
  const open = openEffects.has(effect.id);
  const wrap = document.createElement('div');
  wrap.className = 'fx-entry' + (audio ? ' audio' : '') + (open ? ' open' : '') +
    (effect.enabled === false ? ' off' : '') + (err ? ' err' : '');

  const row = document.createElement('div');
  row.className = 'fx-entry-head';

  const twirl = document.createElement('button');
  twirl.className = 'tl-mini fx-twirl';
  twirl.textContent = open ? '▾' : '▸';
  twirl.title = 'show this effect’s parameters';
  twirl.onclick = () => {
    if (open) openEffects.delete(effect.id);
    else openEffects.add(effect.id);
    renderInspector();
  };

  const on = document.createElement('input');
  on.type = 'checkbox';
  on.className = 'fx-on';
  on.checked = effect.enabled !== false;
  on.title = 'bypass this effect';
  on.onchange = () => {
    history.record(comp, () => { effect.enabled = on.checked; });
    onModelChange({ structural: true });
  };

  const badge = document.createElement('span');
  badge.className = 'fx-badge';
  badge.textContent = audio ? '♪' : effect.fxKind === 'custom' ? '✎' : 'ƒx';

  const name = document.createElement('input');
  name.className = 'fx-name';
  name.value = effect.name;
  name.title = 'effect name';
  name.addEventListener('keydown', (e) => e.stopPropagation());
  name.addEventListener('change', () => {
    history.record(comp, () => { effect.name = name.value.trim() || effect.name; });
    onModelChange({ structural: false });
  });

  const move = (dir) => {
    const b = document.createElement('button');
    b.className = 'tl-mini';
    b.textContent = dir < 0 ? '▲' : '▼';
    b.title = dir < 0 ? 'apply earlier' : 'apply later';
    // Order matters within a chain, not across them: an audio effect steps
    // past its audio neighbours and leaves the shaders where they are.
    const list = audio ? audioEffectsOf(clip) : visualEffectsOf(clip);
    const i = list.indexOf(effect);
    b.disabled = i + dir < 0 || i + dir >= list.length;
    b.onclick = () => {
      history.record(comp, () => {
        const arr = clip.effects;
        const a = arr.indexOf(effect);
        const c = arr.indexOf(list[i + dir]);
        [arr[a], arr[c]] = [arr[c], arr[a]];
      });
      onModelChange({ structural: true });
    };
    return b;
  };

  const del = document.createElement('button');
  del.className = 'tl-mini insp-del';
  del.textContent = '✕';
  del.title = 'remove this effect';
  del.onclick = () => removeEffect(clip, effect);

  row.append(twirl, on, badge, name, move(-1), move(1), del);
  wrap.appendChild(row);

  if (!open) return wrap;

  if (err) {
    const e = document.createElement('div');
    e.className = 'layer-error';
    e.textContent = err;
    wrap.appendChild(e);
  }
  const body = document.createElement('div');
  body.className = 'fx-entry-body';
  if (effect.fxKind === 'custom') renderCustomEditor(clip, effect, body);
  if (spec?.runtime?.preset?.textures?.length)
    renderOverlayControls(clip, effect, spec, body);
  if (audio) {
    const adef = audioEffectDef(effect.audioId);
    if (adef?.widget) body.appendChild(audioCurveWidget(clip, effect, adef));
  }

  const defs = effectPropDefs(clip, effect);
  if (defs.length) {
    const params = document.createElement('div');
    params.className = 'insp-params';
    for (const def of defs) addParamRow(params, clip, def);
    body.appendChild(params);
  } else if (!err && !audio) {
    const ld = document.createElement('div');
    ld.className = 'insp-note';
    ld.textContent = 'loading parameters…';
    body.appendChild(ld);
  }
  wrap.appendChild(body);
  return wrap;
}

/**
 * The draggable frequency-response plot for a filter or EQ. It reads and
 * writes the very same PropTracks the sliders below it do — so a handle
 * drag lands on a keyframe when the parameter is animated, respects the
 * driver, and undoes as one step.
 */
function audioCurveWidget(clip, effect, adef) {
  const keyOf = (name) => effectPropKey(effect.id, name);
  const metaOf = (name) => adef.params.find((p) => p.name === name) ?? { min: 0, max: 1, step: 0.001 };
  let tFrozen = null;
  const io = {
    get: (name) => valueAt(clip, keyOf(name)),
    meta: metaOf,
    // One undo step per drag, and — like a slider drag — keyframes land at
    // the time the drag started rather than smearing across a moving
    // playhead.
    begin: () => { tFrozen = relTime(clip); history.begin(comp); },
    set: (name, v) => {
      const key = keyOf(name);
      setPropValueLive(clip, key, v, tFrozen);
      syncInspRow(key, v);
    },
    commit: () => {
      tFrozen = null;
      history.commit(comp);
      onModelChange({ structural: false });
    },
  };
  const w = responseWidget(adef.widget, io);
  // Animated or driven parameters make the curve move on its own.
  const live = adef.widget.bands.some((b) => [b.freq, b.q, b.gain]
    .filter(Boolean)
    .some((n) => {
      const p = getProp(clip, keyOf(n));
      return p?.anim || p?.driver?.enabled;
    }));
  inspWidgets.push({ redraw: w.redraw, live });
  return w.el;
}

function removeEffect(clip, effect) {
  history.record(comp, () => {
    clip.effects = effectsOf(clip).filter((e) => e !== effect);
  });
  onModelChange({ structural: true });   // gcEffectState frees the runtime
  setStatus(`removed ${effect.name}`);
}

/* A log slider's travel is the exponent, not the value: 20 Hz–20 kHz on a
 * linear track spends nine tenths of its length above 2 kHz. `toPos` /
 * `fromPos` convert, and everything else (keyframes, the number box, the
 * model) keeps working in real units. */
function sliderScale(def) {
  const lin = { toPos: (v) => v, fromPos: (p) => p, log: false };
  if (def.scale !== 'log' || !(def.min > 0) || !(def.max > def.min)) return lin;
  const k = Math.log(def.max / def.min);
  return {
    log: true,
    toPos: (v) => Math.log(clamp(v, def.min, def.max) / def.min) / k,
    fromPos: (p) => {
      const v = def.min * Math.exp(clamp(p, 0, 1) * k);
      const step = def.step || 0.001;
      return clamp(Math.round(v / step) * step, def.min, def.max);
    },
  };
}

/* Live handles on the rows of the current render, so a widget dragging a
 * value can move the matching slider without a full re-render. */
const inspRows = new Map();   // propKey -> {slider, num, scale}

function syncInspRow(key, v) {
  const r = inspRows.get(key);
  if (!r) return;
  r.slider.value = String(r.scale.toPos(v));
  if (document.activeElement !== r.num) r.num.value = fmtVal(v);
}

function paramRow(clip, def) {
  const prop = getProp(clip, def.key);
  const anim = !!prop?.anim;
  const row = document.createElement('div');
  row.className = 'param-row kf';

  const label = document.createElement('label');
  label.textContent = def.label;
  // The tooltip carries what the shader author actually wrote, prefix and
  // all, for the rare case the trimmed name is ambiguous.
  label.title = `${def.fullLabel ?? def.label}${def.unit ? ` (${def.unit})` : ''}`;

  const scale = sliderScale(def);
  const slider = document.createElement('input');
  slider.type = 'range';
  if (scale.log) {
    slider.min = '0';
    slider.max = '1';
    slider.step = '0.0005';
  } else {
    slider.min = String(def.min);
    slider.max = String(def.max);
    slider.step = String(def.step || 0.001);
  }

  const num = document.createElement('input');
  num.type = 'number';
  num.className = 'val';
  num.step = String(def.step || 0.001);

  const v0 = valueAt(clip, def.key);
  slider.value = String(scale.toPos(v0));
  num.value = fmtVal(v0);
  inspRows.set(def.key, { slider, num, scale });

  const binding = { clip, key: def.key, slider, num, scale, dragging: false };
  if (anim) inspLive.push(binding);

  slider.addEventListener('pointerdown', () => {
    binding.dragging = true;
    binding.tFrozen = relTime(clip);
    history.begin(comp);
  });
  slider.addEventListener('input', () => {
    // Param-only change — applied every frame by applyParams, no rebuild.
    const v = scale.fromPos(parseFloat(slider.value));
    setPropValueLive(clip, def.key, v, binding.dragging ? binding.tFrozen : null);
    num.value = fmtVal(v);
    refreshInspWidgets();
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

  const sw = document.createElement('button');
  sw.className = 'tl-stopwatch' + (anim ? ' on' : '');
  sw.textContent = '⏱';
  sw.title = anim ? 'animating — click to freeze' : 'animate this property';
  sw.addEventListener('click', () => toggleAnim(clip, def.key));

  const keyBtn = document.createElement('button');
  keyBtn.className = 'tl-mini tl-kf-toggle';
  const atKey = anim && prop && keyNear(prop, relTime(clip), 0.5 / comp.fps);
  keyBtn.textContent = '◆';
  keyBtn.classList.toggle('at-key', !!atKey);
  keyBtn.title = 'add / remove keyframe at playhead';
  keyBtn.addEventListener('click', () => toggleKey(clip, def.key));

  const drv = document.createElement('button');
  drv.className = 'tl-mini drv-toggle' + (prop?.driver?.enabled ? ' on' : '');
  drv.textContent = '∿';
  drv.title = prop?.driver
    ? 'driver settings'
    : 'drive this value with an oscillator or audio (beats, levels)';
  drv.addEventListener('click', () => toggleDriver(clip, def));

  row.append(sw, label, slider, num, keyBtn, drv);
  return row;
}

/* ---- driver editor --------------------------------------------------- */

const openDrivers = new Set();   // `${clipId}:${propKey}` panels twirled open

function toggleDriver(clip, def) {
  const id = `${clip.id}:${def.key}`;
  const prop = getOrCreateProp(clip, def.key);
  if (!prop) return;
  if (!prop.driver) {
    history.record(comp, () => { prop.driver = newDriver(def); });
    openDrivers.add(id);
    scheduleSave();
  } else if (openDrivers.has(id)) {
    openDrivers.delete(id);
  } else {
    openDrivers.add(id);
  }
  renderInspector();
}

/** The twirled-open editor under a driven property's row. */
function driverPanel(clip, def) {
  const id = `${clip.id}:${def.key}`;
  if (!openDrivers.has(id)) return null;
  const prop = getProp(clip, def.key);
  const d = prop?.driver;
  if (!d) return null;

  const panel = document.createElement('div');
  panel.className = 'driver-panel';

  // One labeled control. `set` mutates the driver inside one undo step;
  // pass rerender for edits that change which fields are visible.
  const field = (labelText, input) => {
    const l = document.createElement('label');
    l.append(labelText, input);
    return l;
  };
  const numF = (labelText, key, step, title = '') => {
    const inp = document.createElement('input');
    inp.type = 'number';
    inp.step = String(step);
    inp.value = fmtVal(d[key] ?? 0);
    inp.title = title;
    inp.addEventListener('keydown', (e) => e.stopPropagation());
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (Number.isNaN(v)) return;
      history.record(comp, () => { d[key] = v; });
      scheduleSave();
    });
    return field(labelText, inp);
  };
  const selF = (labelText, key, options, { rerender = false, title = '' } = {}) => {
    const sel = document.createElement('select');
    for (const [val, lab] of options) {
      const o = document.createElement('option');
      o.value = val;
      o.textContent = lab;
      sel.appendChild(o);
    }
    sel.value = d[key];
    sel.title = title;
    sel.addEventListener('change', () => {
      history.record(comp, () => { d[key] = sel.value; });
      scheduleSave();
      syncAudioDrive();
      if (rerender) renderInspector();
    });
    return field(labelText, sel);
  };

  /* row 1: enable / source / combine mode / remove */
  const top = document.createElement('div');
  top.className = 'drv-row';
  const en = document.createElement('input');
  en.type = 'checkbox';
  en.checked = d.enabled !== false;
  en.addEventListener('change', () => {
    history.record(comp, () => { d.enabled = en.checked; });
    scheduleSave();
    syncAudioDrive();
    renderInspector();
  });
  top.appendChild(field(en, ' on'));
  top.appendChild(selF('src', 'source',
    [['osc', 'Oscillator'], ['audio', 'Audio']], { rerender: true }));
  top.appendChild(selF('mode', 'mode', DRIVER_MODES,
    { title: 'add: base + delta (property units)\nmultiply: base × (1 + delta %)\nreplace: ignore keyframes' }));
  const del = document.createElement('button');
  del.className = 'mn-del';
  del.textContent = '✕';
  del.title = 'remove this driver';
  del.addEventListener('click', () => {
    history.record(comp, () => { delete prop.driver; });
    openDrivers.delete(id);
    scheduleSave();
    renderInspector();
  });
  top.appendChild(del);
  panel.appendChild(top);

  /* row 2: the source's own params */
  const srcRow = document.createElement('div');
  srcRow.className = 'drv-row';
  if (d.source === 'audio') {
    srcRow.appendChild(selF('band', 'band', DRIVER_BANDS));
    srcRow.appendChild(selF('follow', 'follow', DRIVER_FOLLOWS, {
      rerender: true,
      title: 'level: track the band’s loudness\nbeat: pulse on detected onsets',
    }));
    if (d.follow === 'beat') {
      srcRow.appendChild(numF('sense', 'sensitivity', 0.1,
        'onset threshold vs. the running average — higher = fewer, stronger beats'));
      srcRow.appendChild(numF('decay', 'decay', 0.05, 'pulse fade-out (seconds)'));
    } else {
      srcRow.appendChild(numF('release', 'release', 0.05,
        'how long the value hangs after a loud moment (seconds)'));
    }
  } else {
    srcRow.appendChild(selF('wave', 'wave', DRIVER_WAVES, { rerender: true }));
    srcRow.appendChild(numF('freq', 'freq', 0.1, 'cycles per second'));
    srcRow.appendChild(numF('phase', 'phase', 0.05, 'cycle offset (1 = one full cycle)'));
    if (d.wave === 'square' || d.wave === 'pulse')
      srcRow.appendChild(numF('width', 'width', 0.05, 'duty cycle 0..1'));
  }
  panel.appendChild(srcRow);

  /* row 3: mapping */
  const mapRow = document.createElement('div');
  mapRow.className = 'drv-row';
  mapRow.appendChild(numF('amount', 'amount', def.step || 1,
    d.mode === 'multiply' ? 'signal swing in percent of the base value'
      : `signal swing in ${def.unit || 'property units'}`));
  mapRow.appendChild(numF('offset', 'offset', def.step || 1, 'constant added to the swing'));
  panel.appendChild(mapRow);

  if (d.source === 'audio') {
    const hint = document.createElement('div');
    hint.className = 'drv-hint';
    hint.textContent = audioDrive.data
      ? 'audio drivers hear every video track — muted “beat tracks” too'
      : (compHasAudioDrivers() && audioEntries(true).length
        ? 'analyzing audio…'
        : 'no video clips with audio in the comp — this driver reads 0');
    panel.appendChild(hint);
  }

  return panel;
}

function renderCustomEditor(clip, effect, parent) {
  const editor = document.createElement('div');
  editor.className = 'layer-editor';

  const sed = makeShaderEditor({
    value: editorDrafts.get(effect.id) ?? effect.source ?? '',
    onInput: (text) => editorDrafts.set(effect.id, text),
  });
  sed.el.classList.add('sed-inspector');

  const row = document.createElement('div');
  row.className = 'editor-actions';
  const compile = document.createElement('button');
  compile.className = 'btn';
  compile.textContent = 'Compile';
  compile.onclick = () => {
    editorDrafts.delete(effect.id);
    compileCustomEffect(clip, effect, sed.getValue());
  };
  const revert = document.createElement('button');
  revert.className = 'btn';
  revert.textContent = 'Revert';
  revert.title = 'discard edits since last compile';
  revert.onclick = () => {
    editorDrafts.delete(effect.id);
    sed.setValue(effect.source ?? '');
  };
  const expand = document.createElement('button');
  expand.className = 'btn';
  expand.textContent = '⛶';
  expand.title = 'open the full-screen editor (with cheat sheet)';
  expand.onclick = () => openShaderModal(clip, effect);

  const nameInput = document.createElement('input');
  nameInput.type = 'text';
  nameInput.className = 'save-name';
  nameInput.placeholder = 'name…';
  nameInput.value = effect.savedName ?? '';
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
    effect.savedName = name;
    scheduleSave();
    setStatus(`saved '${name}' to this browser`);
  };

  const forget = document.createElement('button');
  forget.className = 'btn';
  forget.textContent = 'Forget';
  forget.title = 'delete this saved shader from localStorage (the effect keeps running)';
  forget.hidden = !effect.savedName;
  forget.onclick = async () => {
    if (!effect.savedName) return;
    const ok = await confirmDialog({
      title: `Delete saved shader '${effect.savedName}'?`,
      message: 'It disappears from the add-effect menu in this browser. Clips already using it keep their own copy of the code.',
      confirmLabel: 'Delete shader',
    });
    if (!ok) return;
    const saves = loadSaved();
    delete saves[effect.savedName];
    storeSaved(saves);
    setStatus(`forgot saved shader '${effect.savedName}'`);
    effect.savedName = null;
    renderInspector();
  };

  row.append(compile, revert, expand, nameInput, save, forget);
  editor.append(sed.el, row);
  (parent ?? inspectorEl).appendChild(editor);
}

/** Full-screen shader editor modal with the slang cheat sheet. */
function openShaderModal(clip, effect) {
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
  wrap.querySelector('.sed-title').textContent = `✎ ${clip.name} › ${effect.name}`;
  const sed = makeShaderEditor({
    value: editorDrafts.get(effect.id) ?? effect.source ?? '',
    onInput: (text) => editorDrafts.set(effect.id, text),
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
    editorDrafts.delete(effect.id);
    await compileCustomEffect(clip, effect, sed.getValue());
    const spec = fxSpecs.get(effect.id);
    const err = spec?.error || spec?.lastCompileError;
    statusEl2.textContent = err ? `✗ ${err.split('\n')[0].slice(0, 120)}` : '✓ compiled';
    statusEl2.classList.toggle('err', !!err);
  });
  document.body.appendChild(wrap);
  sed.textarea.focus();
}

function renderOverlayControls(clip, effect, spec, parent) {
  for (const tex of spec.runtime.preset.textures) {
    const texName = tex.name;
    const oc = document.createElement('div');
    oc.className = 'overlay-controls';
    const state = effect.overlay?.[texName]?.kind === 'text'
      ? effect.overlay[texName].state
      : { ...DEFAULT_TITLE, text: '' };

    const imgBtn = document.createElement('button');
    imgBtn.className = 'btn';
    imgBtn.textContent = 'Image…';
    imgBtn.title = `use an image as the ${texName} texture`;
    imgBtn.onclick = () => {
      stampPickTarget = { clipId: clip.id, effectId: effect.id, texName };
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
      applyOverlaySource(clip, effect, texName, renderTitleCanvas(s), { kind: 'text', state: s });
    };
    textInput.addEventListener('change', applyText);
    sizeInput.addEventListener('change', applyText);
    colorInput.addEventListener('change', applyText);
    fontSel.addEventListener('change', applyText);
    outline.addEventListener('change', applyText);

    oc.append(imgBtn, textInput, sizeInput, colorInput, fontSel, outlineLabel);
    (parent ?? inspectorEl).appendChild(oc);
  }
}

const MASK_BLEND_MODES = ['add', 'subtract', 'multiply', 'max', 'min'];
const MASK_KIND_LABEL = { paint: 'Paint', key: 'Color key', layer: 'Layer matte' };

/** Uniform access to a clip's mask stack. Both kinds keep the live stack in
 * clipMasks; what differs is who owns the GPU state and what the mask
 * means — an fx clip's mask gates its whole effect group (engine-owned,
 * built on the group head), a media clip's cuts its alpha (app-owned). */
function maskContextFor(clip) {
  const common = {
    state: () => clipMasks.get(clip.id) ?? null,
    ensure() {
      let st = clipMasks.get(clip.id);
      if (!st) {
        st = { opacity: 1, invert: false, nodes: [] };
        clipMasks.set(clip.id, st);
      }
      return st;
    },
    clear() {
      destroyClipMask(clip.id);
      clip.mask = null;
      markChainDirty(clip.id);
    },
  };
  if (clip.kind === 'fx') {
    return {
      ...common,
      structure() { chainDirty = true; },
      setOpts(o) { fx.setGroupMaskOptions(clip.id, o); },
      keySelfDefault: null,      // key nodes sample the group's input
    };
  }
  return {
    ...common,
    structure() { buildMediaMaskGpu(clip.id); },
    setOpts() {},                // compositor reads the live maskState each frame
    keySelfDefault: clip.id,     // key nodes key the clip's own pixels
  };
}

function maskSourceOptions(selfId) {
  const out = [];
  for (const tr of comp.tracks)
    for (const c of tr.clips)
      if (c.kind === 'media')
        out.push({ id: c.id, label: c.id === selfId ? 'this clip' : (c.name ?? 'clip') });
  return out;
}

function renderMaskSection(clip) {
  const ctx = maskContextFor(clip);
  const state = ctx.state();
  const addNode = (kind, label, tip) => secBtn(label, tip, () => {
    collapsedSections.delete('mask');
    const st = ctx.ensure();
    const node = newMaskNode(kind);
    if (kind === 'key') node.sourceClipId = ctx.keySelfDefault;
    prepareMaskNode(node);
    st.nodes.push(node);
    ctx.structure();
    scheduleSave();
    renderInspector();
    if (kind === 'paint') startMaskEdit(clip, node.id);
  });

  // No hint in the header: three buttons already crowd it, and the
  // explanation reads better as a line in the empty body.
  const { sec, body, collapsed } = inspSection('mask', 'Mask', {
    count: state?.nodes?.length ?? 0,
    actions: [
      addNode('paint', '＋Paint', 'paint a mask by hand'),
      addNode('key', '＋Key', clip.kind === 'media'
        ? 'green screen — key a color out of this clip'
        : 'chroma key — build the mask from a color'),
      addNode('layer', '＋Matte', "use another layer's alpha or luma as the mask"),
    ],
  });
  if (collapsed) { inspectorEl.appendChild(sec); return; }

  const mc = body;
  if (!state?.nodes?.length) {
    const note = document.createElement('div');
    note.className = 'insp-note';
    note.textContent = clip.kind === 'fx'
      ? 'no mask — these effects cover the whole frame'
      : 'no mask — paint, key a colour, or use another layer';
    mc.appendChild(note);
  }
  for (const node of state?.nodes ?? [])
    mc.appendChild(maskNodeRow(clip, ctx, node));

  if (state?.nodes?.length) {
    // Edge post passes — composed on the GPU each frame, so the sliders
    // just write the live maskState (no rebuild needed).
    const edge = document.createElement('div');
    edge.className = 'mask-foot';
    edge.append(
      maskRange('expand', -64, 64, 1, () => state.expand ?? 0, (v) => {
        state.expand = v;
        scheduleSave();
      }),
      // 250 must match MASK_FEATHER_MAX in engine/blit.js.
      maskRange('feather', 0, 250, 1, () => state.feather ?? 0, (v) => {
        state.feather = v;
        scheduleSave();
      }));
    mc.appendChild(edge);

    const foot = document.createElement('div');
    foot.className = 'mask-foot';
    const rng = maskRange('opacity', 0, 1, 0.01, () => state.opacity ?? 1, (v) => {
      state.opacity = v;
      ctx.setOpts({ opacity: v });
      scheduleSave();
    });
    const invLabel = document.createElement('label');
    const inv = document.createElement('input');
    inv.type = 'checkbox';
    inv.checked = !!state.invert;
    inv.onchange = () => {
      state.invert = inv.checked;
      ctx.setOpts({ invert: inv.checked });
      scheduleSave();
    };
    invLabel.append(inv, 'invert');
    const removeAll = document.createElement('button');
    removeAll.className = 'btn';
    removeAll.textContent = 'Remove mask';
    removeAll.onclick = () => {
      stopMaskEdit();
      ctx.clear();
      scheduleSave();
      renderInspector();
    };
    foot.append(rng, invLabel, removeAll);
    mc.appendChild(foot);
  }
  inspectorEl.appendChild(sec);
}

function maskRange(label, min, max, step, get, set, disabled = false) {
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
}

function maskNodeRow(clip, ctx, node) {
  const row = document.createElement('div');
  row.className = 'mask-node';

  const top = document.createElement('div');
  top.className = 'mn-top';
  const en = document.createElement('input');
  en.type = 'checkbox';
  en.checked = node.enabled !== false;
  en.title = 'enable / bypass this node';
  en.onchange = () => { node.enabled = en.checked; scheduleSave(); };
  const name = document.createElement('span');
  name.className = 'mn-name';
  name.textContent = MASK_KIND_LABEL[node.kind] ?? node.kind;
  const blendSel = document.createElement('select');
  blendSel.title = 'how this node combines with the stack above it';
  for (const m of MASK_BLEND_MODES) {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = m;
    blendSel.appendChild(o);
  }
  blendSel.value = node.blend ?? 'add';
  blendSel.onchange = () => { node.blend = blendSel.value; scheduleSave(); };
  const invLabel = document.createElement('label');
  invLabel.className = 'mn-inv';
  const inv = document.createElement('input');
  inv.type = 'checkbox';
  inv.checked = !!node.invert;
  inv.onchange = () => { node.invert = inv.checked; scheduleSave(); };
  invLabel.append(inv, 'inv');
  const del = document.createElement('button');
  del.className = 'mn-del';
  del.textContent = '✕';
  del.title = 'delete this mask node';
  del.onclick = () => {
    if (maskEdit?.nodeId === node.id) stopMaskEdit();
    const st = ctx.state();
    st.nodes = st.nodes.filter((n) => n !== node);
    destroyMaskNodeGpu(node);
    if (st.nodes.length) ctx.structure();
    else ctx.clear();
    scheduleSave();
    renderInspector();
  };
  top.append(en, name, blendSel, invLabel, del);
  row.appendChild(top);

  const body = document.createElement('div');
  body.className = 'mn-body';

  if (node.kind === 'paint') {
    const editing = maskEdit?.nodeId === node.id;
    const editBtn = document.createElement('button');
    editBtn.className = 'btn' + (editing ? ' active' : '');
    editBtn.textContent = editing ? 'Done' : 'Edit';
    editBtn.onclick = () => (editing ? stopMaskEdit() : startMaskEdit(clip, node.id));
    const clearBtn = document.createElement('button');
    clearBtn.className = 'btn';
    clearBtn.textContent = 'Clear';
    clearBtn.onclick = () => {
      const ctx2 = node.source.getContext('2d');
      ctx2.globalCompositeOperation = 'source-over';
      ctx2.fillStyle = '#fff';
      ctx2.fillRect(0, 0, node.source.width, node.source.height);
      if (editing)
        maskOverlay.getContext('2d').clearRect(0, 0, maskOverlay.width, maskOverlay.height);
      uploadPaintNode(clip.id, node);
      scheduleSave();
    };
    body.append(editBtn, clearBtn);
    if (editing) {
      const toolBtn = (tool, icon, tip) => {
        const b = document.createElement('button');
        b.className = 'btn' + (brush.tool === tool ? ' active' : '');
        b.textContent = icon;
        b.title = tip;
        b.onclick = () => { brush.tool = tool; updateBrushCursor(); renderInspector(); };
        return b;
      };
      const modeBtn = (mode, label) => {
        const b = document.createElement('button');
        b.className = 'btn' + (brush.mode === mode ? ' active' : '');
        b.textContent = label;
        b.onclick = () => { brush.mode = mode; renderInspector(); };
        return b;
      };
      const isBrush = brush.tool === 'brush';
      body.append(
        toolBtn('brush', '🖌', 'brush'),
        toolBtn('linear', '▤', 'linear gradient — drag across the preview'),
        toolBtn('radial', '◎', 'radial gradient — drag outward from the center'),
        modeBtn('hide', 'Hide'), modeBtn('show', 'Show'),
        maskRange('size', 8, 300, 1, () => brush.size, (v) => { brush.size = v; updateBrushCursor(); }, !isBrush),
        maskRange('soft', 0, 0.9, 0.05, () => brush.soft, (v) => { brush.soft = v; }, !isBrush),
      );
    }
  } else if (node.kind === 'key') {
    const colorWrap = document.createElement('span');
    colorWrap.className = 'mn-colorwrap';
    const color = document.createElement('input');
    color.type = 'color';
    color.value = node.keyColor ?? '#00b140';
    color.oninput = () => {
      node.keyColor = color.value;
      prepareMaskNode(node);
      scheduleSave();
    };
    const pick = document.createElement('button');
    pick.className = 'btn';
    pick.textContent = '⌖';
    pick.title = 'pick the key color from the preview';
    pick.onclick = () => startColorPick(node);
    colorWrap.append(color, pick);
    body.append(
      colorWrap,
      maskRange('similar', 0.005, 0.6, 0.005, () => node.similarity ?? 0.18, (v) => {
        node.similarity = v;
        scheduleSave();
      }),
      maskRange('soften', 0, 0.5, 0.005, () => node.smoothness ?? 0.1, (v) => {
        node.smoothness = v;
        scheduleSave();
      }),
      maskSourceSelect(clip, ctx, node, { allowInput: clip.kind === 'fx' }),
    );
  } else {   // layer matte
    const chanSel = document.createElement('select');
    for (const [v, l] of [['alpha', 'alpha'], ['luma', 'luma']]) {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = l;
      chanSel.appendChild(o);
    }
    chanSel.value = node.channel ?? 'alpha';
    chanSel.onchange = () => { node.channel = chanSel.value; scheduleSave(); };
    body.append(maskSourceSelect(clip, ctx, node, { allowInput: false }), chanSel);
  }
  row.appendChild(body);
  return row;
}

function maskSourceSelect(clip, ctx, node, { allowInput }) {
  const sel = document.createElement('select');
  sel.title = 'mask source';
  const opts = maskSourceOptions(clip.id);
  if (allowInput) {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = 'layer input';
    sel.appendChild(o);
  } else if (!node.sourceClipId) {
    const o = document.createElement('option');
    o.value = '';
    o.textContent = opts.length ? '(choose a layer)' : '(no media clips)';
    sel.appendChild(o);
  }
  for (const { id, label } of opts) {
    const o = document.createElement('option');
    o.value = id;
    o.textContent = label;
    sel.appendChild(o);
  }
  sel.value = node.sourceClipId ?? '';
  sel.onchange = () => {
    node.sourceClipId = sel.value || null;
    prepareMaskNode(node);
    ctx.structure();
    scheduleSave();
  };
  return sel;
}

/* =====================================================================
 * Export — current frame PNG, or render the whole comp to WebM.
 *
 * Two WebM modes, both kept on purpose:
 *   Record — plays the comp once in real time via MediaRecorder,
 *            capturing the live audio mix (good for perf-y captures).
 *   Render — offline loop: seeks every source frame-exactly, renders as
 *            fast as the GPU/encoder allow via WebCodecs, so it can beat
 *            real time and never drops or stutters. Video only.
 * =================================================================== */

function saveBlob(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
}

/* ---- export file names ----------------------------------------------
 * Exports are named after the comp's bottom-most media layer (the shot
 * you are working on), falling back to the project name and finally to a
 * random human-readable slug — anything but one fixed name every time. */

const SLUG_ADJ = [
  'amber', 'bold', 'brisk', 'calm', 'chrome', 'crisp', 'dusky', 'eager',
  'electric', 'fuzzy', 'gentle', 'glassy', 'golden', 'hazy', 'ivory', 'lucid',
  'mellow', 'neon', 'nimble', 'plush', 'quiet', 'rapid', 'rusty', 'silent',
  'silver', 'solar', 'stark', 'sunlit', 'velvet', 'wild',
];
const SLUG_NOUN = [
  'anchor', 'atlas', 'beacon', 'cascade', 'cinder', 'comet', 'delta', 'drift',
  'ember', 'falcon', 'harbor', 'lantern', 'meadow', 'mirage', 'monsoon',
  'orbit', 'otter', 'prism', 'quartz', 'ripple', 'shutter', 'signal', 'summit',
  'tempest', 'thicket', 'tundra', 'vertex', 'willow', 'zenith', 'zephyr',
];

/** e.g. "neon-otter-4f2" — readable, and unique enough to not collide. */
function humanSlug() {
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  const tail = Math.floor(Math.random() * 4096).toString(16).padStart(3, '0');
  return `${pick(SLUG_ADJ)}-${pick(SLUG_NOUN)}-${tail}`;
}

/** Strip the bits a download name can't (or shouldn't) carry. */
function safeFileBase(s) {
  return (s ?? '')
    .replace(/\.[^.\\/]+$/, '')            // drop the extension
    .replace(/[\\/:*?"<>|]+/g, '_')        // path / reserved characters
    .replace(/\s+/g, '_')
    .replace(/^[._]+|[._]+$/g, '')
    .slice(0, 72);
}

function exportBaseName() {
  for (const clip of allClipsBottomUp(comp, 'media')) {
    const asset = assets.get(clip.assetId);
    // Shape layers carry a synthetic asset with no real filename.
    if (!asset || asset.kind === 'shape') continue;
    const base = safeFileBase(asset.name);
    if (base) return base;
  }
  return safeFileBase(projectName) || humanSlug();
}

$('btn-export-png').addEventListener('click', async () => {
  if (!fx?.inputTexture || offlineJob) return;
  const blob = await fx.exportPNG();
  const frame = String(Math.round(tCur * comp.fps)).padStart(5, '0');
  saveBlob(blob, `${exportBaseName()}-${frame}.png`);
  setStatus('frame exported');
});

const exportBtn = $('btn-export-webm');

exportBtn.addEventListener('click', () => {
  if (offlineJob) return;
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
  // Named from the comp as it stands now, not as it stands when you stop.
  const outName = `${exportBaseName()}.webm`;
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    saveBlob(new Blob(chunks, { type: 'video/webm' }), outName);
    setStatus(`recording saved as ${outName}`);
  };
  exportMode = true;
  pause();
  setTime(0);
  recorder.start();
  play();
  exportBtn.textContent = '■ Stop';
  exportBtn.classList.add('recording');
  setStatus('recording comp to WebM…');
});

function finishExport() {
  exportMode = false;
  if (recorder) {
    recorder.stop();
    recorder = null;
  }
  exportBtn.textContent = 'Record';
  exportBtn.classList.remove('recording');
}

/* ---- audio-only export ----------------------------------------------
 * The same offline mix the WebM render muxes in, written out on its own.
 * WAV rather than Opus: it needs no encoder support, and an audio-only
 * export exists to be dropped into something else. */

function audioBufferToWav(buf) {
  const ch = buf.numberOfChannels;
  const bytes = buf.length * ch * 2;   // interleaved 16-bit PCM
  const out = new ArrayBuffer(44 + bytes);
  const view = new DataView(out);
  const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  str(0, 'RIFF');
  view.setUint32(4, 36 + bytes, true);
  str(8, 'WAVEfmt ');
  view.setUint32(16, 16, true);                        // fmt chunk size
  view.setUint16(20, 1, true);                         // PCM
  view.setUint16(22, ch, true);
  view.setUint32(24, buf.sampleRate, true);
  view.setUint32(28, buf.sampleRate * ch * 2, true);   // byte rate
  view.setUint16(32, ch * 2, true);                    // block align
  view.setUint16(34, 16, true);                        // bits per sample
  str(36, 'data');
  view.setUint32(40, bytes, true);
  const chans = Array.from({ length: ch }, (_, c) => buf.getChannelData(c));
  let off = 44;
  for (let i = 0; i < buf.length; i++)
    for (let c = 0; c < ch; c++) {
      const v = clamp(chans[c][i], -1, 1);
      view.setInt16(off, v < 0 ? v * 0x8000 : v * 0x7fff, true);
      off += 2;
    }
  return new Blob([out], { type: 'audio/wav' });
}

const audioExportBtn = $('btn-export-audio');

audioExportBtn.addEventListener('click', async () => {
  if (offlineJob || recorder || audioExportBtn.disabled) return;
  audioExportBtn.disabled = true;
  setStatus('mixing audio…');
  const started = performance.now();
  try {
    // Driven volumes are sampled from the analysis data, same as the
    // video export — without this a fresh audio driver bakes as flat.
    await syncAudioDrive();
    const buf = await renderCompAudio();
    if (!buf) {
      setStatus('nothing to export — the comp has no audible audio');
      return;
    }
    const outName = `${exportBaseName()}.wav`;
    saveBlob(audioBufferToWav(buf), outName);
    const secs = ((performance.now() - started) / 1000).toFixed(1);
    setStatus(`saved ${outName} — ${comp.dur}s of audio in ${secs}s`);
  } catch (e) {
    console.error('slangfx: audio export failed:', e);
    setStatus(`audio export failed: ${e.message ?? e}`);
  } finally {
    audioExportBtn.disabled = false;
  }
});

/* ---- offline (faster-than-real-time) render ------------------------- */

const fastBtn = $('btn-export-webm-fast');
let offlineJob = null;   // { cancel, error } while an offline render runs

fastBtn.addEventListener('click', () => {
  if (offlineJob) { offlineJob.cancel = true; return; }
  if (!fx?.inputTexture || recorder) return;
  if (typeof VideoEncoder === 'undefined') {
    setStatus('offline render needs WebCodecs (Chrome/Edge) — use Record instead');
    return;
  }
  runOfflineRender();
});

/* Edit-and-save: offline render, then PUT the result back to the host's
 * save url (which overwrites the original file, transcoding as needed). */
const saveBackBtn = $('btn-save-back');
saveBackBtn.addEventListener('click', () => {
  if (offlineJob) { offlineJob.cancel = true; return; }
  if (!comp._saveBack || !fx?.inputTexture || recorder) return;
  if (typeof VideoEncoder === 'undefined') {
    setStatus('save needs WebCodecs (Chrome/Edge)');
    return;
  }
  fitDurToContent();
  timeline.render();
  runOfflineRender({ saveBack: comp._saveBack });
});

async function runOfflineRender({ saveBack = null } = {}) {
  pause();
  // Audio drivers must have their envelopes before frames render — the
  // export samples the same precomputed data as the preview.
  await syncAudioDrive();
  const job = (offlineJob = { cancel: false, error: null });
  const tRestore = tCur;
  const btn = saveBack ? saveBackBtn : fastBtn;
  btn.textContent = '■ Cancel';
  btn.classList.add('recording');
  const started = performance.now();
  try {
    const blob = await renderCompOffline(job);
    if (!blob) {
      setStatus('render cancelled');
    } else if (saveBack) {
      setStatus(`saving to ${saveBack.name}…`);
      const res = await fetch(saveBack.url, { method: 'PUT', body: blob });
      const out = await res.json().catch(() => ({}));
      if (!res.ok || !out.ok) throw new Error(out.error ?? `HTTP ${res.status}`);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      setStatus(`saved to ${saveBack.name} — ${comp.dur}s comp in ${secs}s`
        + (job.hasAudio ? '' : ' (comp has no audio)'));
    } else {
      const outName = `${exportBaseName()}.webm`;
      saveBlob(blob, outName);
      const secs = ((performance.now() - started) / 1000).toFixed(1);
      setStatus(`saved ${outName} — ${comp.dur}s comp in ${secs}s`
        + (job.hasAudio ? '' : ' (comp has no audio)'));
    }
  } catch (e) {
    console.error('slangfx: offline render failed:', e);
    setStatus(`${saveBack ? 'save' : 'render'} failed: ${e.message ?? e}`);
  } finally {
    offlineJob = null;
    fastBtn.textContent = 'Render';
    fastBtn.classList.remove('recording');
    saveBackBtn.textContent = '💾 Save';
    saveBackBtn.classList.remove('recording');
    setTime(tRestore);
  }
}

/* Throttling-proof macrotask yield (setTimeout is clamped to ~1s in
 * occluded windows; MessageChannel messages are not). */
function nextTask() {
  return new Promise((resolve) => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => { ch.port1.close(); resolve(); };
    ch.port2.postMessage(0);
  });
}

/* Pause every active video and seek it frame-exactly to the comp time,
 * resolving when the browser has the frame ready (the preview clock's
 * drift-tolerant sync in syncMedia is what causes recording stutter). */
function seekMediaExact(t, activeMedia) {
  const waits = [];
  for (const { clip } of activeMedia) {
    const asset = assets.get(clip.assetId);
    if (!asset?.ready || asset.kind !== 'video') continue;
    const el = asset.el;
    if (!el.paused) el.pause();
    const src = clip.in + (t - clip.start);
    const len = asset.duration ?? 0;
    const desired = len > 0.02 ? ((src % len) + len) % len : 0;
    if (Math.abs(el.currentTime - desired) < 1e-4) continue;
    waits.push(new Promise((resolve) => {
      // Stuck-seek guard so one bad source can't stall the whole render.
      const timer = setTimeout(done, 1000);
      function done() {
        clearTimeout(timer);
        el.removeEventListener('seeked', done);
        resolve();
      }
      el.addEventListener('seeked', done);
      el.currentTime = desired;
    }));
  }
  return Promise.all(waits);
}

function uploadMediaFrames(t, activeMedia) {
  for (const { clip } of activeMedia) {
    const asset = assets.get(clip.assetId);
    if (!asset?.ready) continue;
    if (asset.kind === 'gif') syncGifFrame(asset, clip, t);
    else if (asset.kind === 'video' && asset.el.readyState >= 2) uploadVideoFrame(asset);
  }
}

/* syncFxChain returns early while a rebuild is in flight, which can mask a
 * rebuild this frame needs — re-run until the chain matches the frame. */
async function syncFxChainSettled(t) {
  for (let i = 0; i < 10; i++) {
    await syncFxChain(t);
    const key = activeFxEntries(t).map((e) => stackKey(e.clip)).join('|');
    if (!chainBuilding && !chainDirty && key === chainKey) return;
  }
}

/* Same for the per-clip chains: prepareMediaFx kicks off creation and
 * rebuilds asynchronously, so the exporter waits for them and re-runs the
 * isolate pass once every stack is live. */
async function prepareMediaFxSettled(t, activeMedia) {
  for (let i = 0; i < 10; i++) {
    prepareMediaFx(t, activeMedia);
    const pending = [...mediaChains.values()].filter((e) => e.building || e.dirty);
    if (!pending.length) return;
    await Promise.allSettled(pending.map((e) => e.promise));
  }
}

const AUDIO_SR = 48000;   // Opus native rate

/** Decode an asset's audio once and cache it on the asset (null = no
 * decodable audio stream). decodeAudioData demuxes the audio straight out
 * of the video container; a video with no audio track simply rejects. */
async function decodeAssetAudio(asset) {
  if (asset._audioBuf !== undefined) return asset._audioBuf;
  try {
    const file = asset.file ?? await idbGet(`asset:${asset.id}`);
    const decodeCtx = new OfflineAudioContext(2, 1, AUDIO_SR);
    asset._audioBuf = await decodeCtx.decodeAudioData(await file.arrayBuffer());
  } catch (e) {
    console.warn(`slangfx: no audio for '${asset.name}':`, e);
    asset._audioBuf = null;
  }
  return asset._audioBuf;
}

/* Mix clip audio offline: each entry becomes a buffer source in an
 * OfflineAudioContext, with clip trim/looping matching syncMedia, the
 * clip's audio effect chain rebuilt node-for-node from the same catalogue
 * the preview uses, and every animated value baked as per-frame ramps.
 * Shared by the export path (audible clips, driven values) and the driver
 * analysis (every clip, BASE values — a driver must never feed back into
 * the signal it listens to). Returns null when no entry has decodable
 * audio. */
async function mixCompAudio(sampleRate, entries, channels, { driven = true } = {}) {
  if (!entries.length) return null;
  for (const { asset } of entries) await decodeAssetAudio(asset);
  if (!entries.some(({ asset }) => asset._audioBuf)) return null;

  const ctx = new OfflineAudioContext(channels, Math.max(1, Math.ceil(comp.dur * sampleRate)), sampleRate);
  const step = 1 / comp.fps;

  for (const { clip, asset } of entries) {
    const buf = asset._audioBuf;
    if (!buf) continue;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;   // clips longer than their source wrap, like syncMedia
    const gain = ctx.createGain();
    const start = clip.start;
    const len = Math.min(clipEnd(clip), comp.dur) - start;
    if (len <= 0) continue;

    /** A property's value at clip-relative tc, drivers included or not. */
    const at = (prop, tc, fallback) => {
      if (!prop) return fallback;
      return driven ? drivenEval(prop, tc, start + tc) : evalProp(prop, tc);
    };
    /** Schedule one AudioParam over the clip: a value at the start, plus
     * per-frame ramps when the property actually moves. */
    const bake = (param, prop, fallback, map) => {
      const val = (tc) => {
        const v = at(prop, tc, fallback);
        return map ? map(v) : v;
      };
      param.setValueAtTime(val(0), start);
      if (!(prop?.anim && prop.keys.length) && !prop?.driver?.enabled) return;
      for (let tc = step; tc <= len + 1e-9; tc += step)
        param.linearRampToValueAtTime(val(tc), start + tc);
    };

    // source → effects → volume fader → out, exactly as syncMedia wires it.
    let cur = src;
    for (const effect of liveAudioEffects(clip)) {
      const def = audioEffectDef(effect.audioId);
      if (!def) continue;
      let unit;
      try {
        unit = def.build(ctx);
      } catch (e) {
        console.warn(`slangfx: offline audio effect '${effect.audioId}' failed:`, e);
        continue;
      }
      for (const p of def.params) {
        const prop = effect.params?.[p.name];
        for (const c of controlTargets(unit, p.name)) bake(c.param, prop, p.def, c.map);
      }
      cur.connect(unit.input);
      cur = unit.output;
    }
    cur.connect(gain).connect(ctx.destination);
    bake(gain.gain, clip.props.volume, 100, (v) => clamp(v / 100, 0, 1));

    const offset = buf.duration > 0.02
      ? ((clip.in % buf.duration) + buf.duration) % buf.duration : 0;
    src.start(start, offset);
    // Reverb and delay tails are part of the clip's sound: the source
    // stops, the graph keeps ringing until the render ends.
    src.stop(start + len);
  }
  return ctx.startRendering();
}

/* The export mix: audible clips only, driven volume included. Preview-only
 * master volume/mute is deliberately NOT baked — a muted preview should
 * not produce a silent export. */
async function renderCompAudio() {
  return mixCompAudio(AUDIO_SR, audioEntries(false), 2, { driven: true });
}

/** Encode a rendered AudioBuffer to Opus chunks straight into the muxer. */
async function encodeAudioTrack(muxer, buf, job) {
  const enc = new AudioEncoder({
    output: (chunk, meta) => muxer.addAudioChunk(chunk, meta),
    error: (e) => { job.error = e; job.cancel = true; },
  });
  enc.configure({
    codec: 'opus',
    sampleRate: buf.sampleRate,
    numberOfChannels: buf.numberOfChannels,
    bitrate: 128_000,
  });
  const CH = buf.numberOfChannels;
  const CHUNK = 9600;   // 200ms at 48k
  for (let off = 0; off < buf.length && !job.cancel; off += CHUNK) {
    const n = Math.min(CHUNK, buf.length - off);
    const data = new Float32Array(n * CH);
    for (let c = 0; c < CH; c++)
      data.set(buf.getChannelData(c).subarray(off, off + n), c * n);
    const ad = new AudioData({
      format: 'f32-planar',
      sampleRate: buf.sampleRate,
      numberOfFrames: n,
      numberOfChannels: CH,
      timestamp: Math.round(off / buf.sampleRate * 1e6),
      data,
    });
    enc.encode(ad);
    ad.close();
    while (enc.encodeQueueSize > 16)
      await new Promise((r) => enc.addEventListener('dequeue', r, { once: true }));
  }
  await enc.flush();
  enc.close();
}

async function renderCompOffline(job) {
  const { width, height, fps } = comp;
  const totalFrames = Math.max(1, Math.round(comp.dur * fps));
  const frameUs = 1e6 / fps;

  setStatus('rendering audio…');
  let audioBuf = null;
  if (typeof AudioEncoder !== 'undefined') {
    try {
      audioBuf = await renderCompAudio();
    } catch (e) {
      console.warn('slangfx: offline audio mix failed, rendering video only:', e);
    }
  }

  let codec = 'vp09.00.10.08';   // VP9 profile 0 level 1.0 8-bit
  const config = { width, height, bitrate: 12_000_000, framerate: fps };
  if (!(await VideoEncoder.isConfigSupported({ codec, ...config })).supported)
    codec = 'vp8';
  const muxer = new WebMMuxer({
    target: new ArrayBufferTarget(),
    video: { codec: codec === 'vp8' ? 'V_VP8' : 'V_VP9', width, height, frameRate: fps },
    ...(audioBuf ? {
      audio: { codec: 'A_OPUS', numberOfChannels: audioBuf.numberOfChannels, sampleRate: audioBuf.sampleRate },
    } : {}),
  });
  if (audioBuf) {
    await encodeAudioTrack(muxer, audioBuf, job);
    if (job.error) throw job.error;
  }
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => { job.error = e; job.cancel = true; },
  });
  encoder.configure({ codec, ...config });
  job.hasAudio = !!audioBuf;

  for (let f = 0; f < totalFrames && !job.cancel; f++) {
    const t = f / fps;
    tCur = t;
    const activeMedia = activeClips(comp, t, 'media').filter(({ track }) => !track.hidden);
    await seekMediaExact(t, [...activeMedia, ...matteSourceClips(t)]);
    uploadMediaFrames(t, activeMedia);
    prepareMasks(t);   // media masks must compose before compositeFrame samples them
    await prepareMediaFxSettled(t, activeMedia);
    compositeFrame(t);
    await syncFxChainSettled(t);
    applyParams(t);
    fx.render(null, t);
    // Snapshot synchronously after render() — the WebGPU canvas only holds
    // this frame until the current task yields.
    const frame = new VideoFrame(canvas, {
      timestamp: Math.round(f * frameUs),
      duration: Math.round(frameUs),
    });
    encoder.encode(frame, { keyFrame: f % (2 * fps) === 0 });
    frame.close();
    // No setTimeout anywhere in this loop: Chrome throttles timers to ~1/s
    // in occluded/background windows, which turns a 4s render into minutes.
    // Event- and message-based waits are exempt, so an offline render keeps
    // full speed even with the tab in the background.
    while (encoder.encodeQueueSize > 8)
      await new Promise((r) => encoder.addEventListener('dequeue', r, { once: true }));
    if (f % 8 === 0) {
      setStatus(`rendering frame ${f + 1}/${totalFrames}…`);
      timeline.updatePlayhead();
      await nextTask();   // let the UI paint
    }
  }

  if (job.error) { try { encoder.close(); } catch {} throw job.error; }
  if (job.cancel) { try { encoder.close(); } catch {} return null; }
  await encoder.flush();
  encoder.close();
  muxer.finalize();
  return new Blob([muxer.target.buffer], { type: 'video/webm' });
}

boot();
