import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import { JSDOM } from 'jsdom';

const read = (name) => readFileSync(new URL(name, import.meta.url), 'utf8');
const moduleURL = (source) => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const modelURL = moduleURL(read('./comp.js'));
const model = await import(modelURL);
const dom = new JSDOM('<body></body>');
Object.assign(globalThis, {
  document: dom.window.document, window: dom.window,
  innerWidth: 1280, innerHeight: 720,
  ResizeObserver: class { observe() {} },
});
const { Timeline } = await import(moduleURL(read('./timeline.js')
  .replace("'./comp.js'", JSON.stringify(modelURL))
  .replace("'./icons.js'", JSON.stringify(moduleURL(read('./icons.js'))))));

// Exercise the app's functions without booting its GPU renderer and asset store.
const app = read('./app.js');
function appFunction(name, context) {
  const start = app.indexOf(`function ${name}(`);
  assert.ok(start >= 0);
  const end = app.indexOf('\n}', start) + 2;
  return vm.runInNewContext(`(${app.slice(start, end)})`, context);
}
const noop = () => {};
function setup() {
  const comp = model.newComp({ dur: 30 });
  const history = new model.History();
  let time = 25;
  const host = {
    comp: () => comp, history, time: () => time, setTime: (t) => { time = t; },
    assetOf: () => null, status: noop,
    onModelChange: () => model.ensureDur(comp),
  };
  const root = document.createElement('div');
  document.body.replaceChildren(root);
  const timeline = new Timeline(root, host);
  timeline.zoomFit = noop;
  return { comp, history, host, timeline };
}

test('dropping a bin asset over an occupied row creates a separate track at the drop time', () => {
  for (const kind of ['video', 'audio', 'image']) {
    const { comp, host, timeline } = setup();
    const existing = model.newTrack('Existing');
    existing.clips.push({ id: 'old', start: 0, dur: 30 });
    comp.tracks.push(existing);
    const asset = { id: 'asset', name: 'Imported', kind, duration: 5, w: 1280, h: 720 };
    const place = appFunction('placeAssetClip', { ...model, comp, DEFAULT_VIDEO_DUR: 5 });
    let added;
    host.addAssetAt = (id, t, trackIdx) => {
      assert.equal(id, asset.id);
      added = place(asset, t, trackIdx);
    };
    timeline.rowsEl.innerHTML = '<div class="tl-row track"></div>';
    const drop = new dom.window.MouseEvent('drop', { clientX: 120, clientY: 0, bubbles: true });
    Object.defineProperty(drop, 'dataTransfer', {
      value: { getData: () => asset.id, files: [] },
    });
    timeline.scrollEl.dispatchEvent(drop);
    assert.equal(comp.tracks.length, 2);
    assert.equal(existing.clips.length, 1);
    assert.notEqual(model.trackOf(comp, added), existing);
    assert.equal(added.start, 2);
  }
});

test('trim uses the right-clicked clip endpoint and survives updates, saving, undo and redo', () => {
  const { comp, history, host, timeline } = setup();
  const short = { id: 'short', kind: 'audio', start: 2, dur: 4, props: {} };
  const long = { id: 'long', kind: 'audio', start: 0, dur: 30, props: {} };
  comp._autoSize = true;
  comp.tracks.push({ clips: [short] }, { clips: [long] });
  timeline.selClips.add(long.id);
  timeline._clipMenu(10, 10, short);
  [...document.querySelectorAll('.ctx-item')]
    .find((row) => row.textContent === 'Trim comp length to clip').click();
  assert.equal(comp.dur, 6);
  assert.equal(host.time(), 6);
  assert.equal(long.dur, 30);
  host.onModelChange();
  appFunction('fitDurToContent', { comp })();
  assert.equal(comp.dur, 6);
  const restored = JSON.parse(JSON.stringify(comp));
  model.ensureDur(restored);
  assert.equal(restored.dur, 6);
  const undone = history.undo(comp);
  assert.equal(undone.dur, 30);
  assert.equal(undone.autoDuration, undefined);
  const redone = history.redo(undone);
  model.ensureDur(redone);
  assert.equal(redone.dur, 6);
});

test('ordinary compositions still grow to fit their clips', () => {
  const comp = model.newComp({ dur: 5 });
  comp.tracks.push({ clips: [{ start: 3, dur: 12 }] });
  model.ensureDur(comp);
  assert.equal(comp.dur, 15);
});

test('hide and mute independently control preview visuals and preview/export audio', () => {
  for (const kind of ['media', 'audio']) for (const hidden of [false, true]) for (const muted of [false, true]) {
    const clip = { id: 'clip', assetId: 'asset', kind, start: 0, dur: 5, in: 0, props: {} };
    const comp = model.newComp({ dur: 5 });
    comp.tracks.push({ hidden, muted, clips: [clip] });
    const el = { paused: true, currentTime: 0, play() { this.paused = false; return Promise.resolve(); }, pause() { this.paused = true; } };
    const assets = new Map([['asset', { id: 'asset', kind: kind === 'media' ? 'video' : 'audio', ready: true, duration: 5, el }]]);
    const context = {
      ...model, comp, assets, playing: true, audioState: {}, drivenEval: noop,
      loopSrc: (t, len) => t % len,
    };
    const syncMedia = appFunction('syncMedia', context);
    let visualClips;
    const tick = appFunction('tick', {
      ...model, comp, syncMedia, tCur: 1, playing: true,
      clock: { t: 1, perf: 0 }, performance: { now: () => 0 },
      requestAnimationFrame: noop, fx: { inputTexture: {}, render: noop },
      offlineJob: null, rotoJob: null, loopJob: null, trimPreviewT: null, rotoEdit: null,
      prepareMasks: noop, prepareMediaFx: (t, clips) => { visualClips = clips; },
      compositeFrame: noop, syncFxChain: noop, applyParams: noop,
      timeline: { updatePlayhead: noop }, updateInspectorLive: noop, updateGizmo: noop,
    });
    tick();
    assert.equal(el.paused, false);
    assert.equal(el.muted, muted);
    assert.equal(visualClips.length, kind === 'media' && !hidden ? 1 : 0);
    const audioEntries = appFunction('audioEntries', context);
    assert.equal(audioEntries(false).length, muted ? 0 : 1);
    assert.equal(audioEntries(true).length, 1);
  }
});
