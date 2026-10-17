// Studio end-to-end smoke test — run it through the Playwright MCP tool
// (browser_run_code_unsafe with `filename: studio/tools/smoke.playwright.js`).
//
// Setup, from the repo root:
//   node studio/publish.mjs && node studio/serve.mjs 8794      (separate origin:
//     the test wipes its storage, so never point it at your working 8790)
//   ffmpeg -f lavfi -i testsrc=size=320x240:rate=30 -f lavfi -i sine=frequency=440 \
//     -t 6 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest .playwright-mcp/tone.mp4
//   ffmpeg -f lavfi -i "sine=frequency=440:duration=4:sample_rate=48000" \
//     -af "volume=enable='not(between(t,1,1.1))':volume=0" .playwright-mcp/burst.wav
//   ffmpeg -f lavfi -i "sine=frequency=60:duration=6:sample_rate=48000" -af \
//     "volume=enable='lt(mod(t-0.25\,0.5)\,0.08)':volume=1,volume=enable='gte(mod(t-0.25\,0.5)\,0.08)+lt(t\,0.25)':volume=0" \
//     .playwright-mcp/basshits.wav        (60 Hz bass hits, 80 ms long, every 0.5 s from 0.25 s)
//
// Returns { results: {check: 'PASS' | 'FAIL …'}, errors: [...] }. Exports land
// in .playwright-mcp/ for ffprobe.
async (page) => {
  const BASE = 'http://localhost:8794/studio/';
  const FIXTURE = 'C:/Users/steph/dev/loki/.playwright-mcp/tone.mp4';
  const OUT = 'C:/Users/steph/dev/loki/.playwright-mcp/smoke-export.webm';
  const results = {};
  const errors = [];
  const check = (name, ok, info) => { results[name] = ok ? 'PASS' : `FAIL ${JSON.stringify(info ?? '')}`; };
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => {
    // Resource 404s are the site header's images when serving a partial tree.
    if (m.type() === 'error' && !m.text().startsWith('Failed to load resource'))
      errors.push(`console: ${m.text().slice(0, 300)}`);
  });

  // Tap the master bus (first node wired to the speakers) and count seeks
  // per media element, before any app code runs.
  await page.addInitScript(() => {
    // Tap every LIVE context's speaker feed (the driver analysis renders
    // through OfflineAudioContexts, and a page can hold more than one live
    // context over its life) and read whichever are running.
    const connect = AudioNode.prototype.connect;
    window.__ans = [];
    AudioNode.prototype.connect = function (dst, ...rest) {
      if (dst instanceof AudioDestinationNode && !(this.context instanceof OfflineAudioContext) && !this.__tapped) {
        this.__tapped = true;
        const an = this.context.createAnalyser();
        an.fftSize = 512;
        connect.call(this, an);
        window.__ans.push(an);
      }
      return connect.call(this, dst, ...rest);
    };
    window.__level = () => {
      let m = 0;
      for (const an of window.__ans) {
        if (an.context.state !== 'running') continue;
        const b = new Float32Array(an.fftSize);
        an.getFloatTimeDomainData(b);
        for (const v of b) m = Math.max(m, Math.abs(v));
      }
      return m;
    };
    document.addEventListener('seeking', (e) => { e.target.__seeks = (e.target.__seeks ?? 0) + 1; }, true);
    // Count GPU textures still alive (created minus destroyed) for leak checks.
    window.__texLive = 0;
    if (window.GPUDevice) {
      const create = GPUDevice.prototype.createTexture;
      GPUDevice.prototype.createTexture = function (...a) {
        const tex = create.apply(this, a);
        window.__texLive++;
        const destroy = tex.destroy.bind(tex);
        let gone = false;
        tex.destroy = () => { if (!gone) { gone = true; window.__texLive--; } return destroy(); };
        return tex;
      };
    }
  });

  // Fresh origin state: no autosave, no stored assets. A background window
  // gets rAF throttled to ~1 Hz, which skews every timing check.
  await page.goto(BASE);
  await page.bringToFront();
  await page.evaluate(() => new Promise((r) => {
    localStorage.clear();
    const q = indexedDB.deleteDatabase('lowkey-studio');
    q.onsuccess = q.onerror = q.onblocked = () => r();
  }));
  await page.reload();
  await page.waitForFunction(() => window.studio && window.fx?.inputTexture, null, { timeout: 30000 });
  check('boot', true);

  // Import the fixture and reduce the comp to it, 3 s long.
  await page.setInputFiles('#file-input', FIXTURE);
  await page.waitForFunction(() => [...window.studio.assets.values()].some((a) => a.name === 'tone.mp4' && a.ready),
    null, { timeout: 20000 });
  await page.evaluate(() => {
    const c = window.comp();
    const asset = [...window.studio.assets.values()].find((a) => a.name === 'tone.mp4');
    c.tracks = c.tracks.filter((t) => t.clips.some((k) => k.assetId === asset.id)).slice(0, 1);
    const clip = c.tracks[0].clips[0];
    clip.start = 0; clip.in = 0; clip.dur = 3;
    c.dur = 3;
    window.studio.onModelChange({ structural: true });
  });
  check('import', await page.evaluate(() => window.comp().tracks.length === 1));

  const host = (fn, arg) => page.evaluate(([f, a]) => window.studio.timeline.host[f](a), [fn, arg]);
  // The real button: audio only starts from a user gesture (autoplay policy).
  const playBtn = page.locator('[data-act=play]');
  const play = async (from) => {
    await host('setTime', from);
    if (!await host('playing')) await playBtn.click();
    await page.waitForFunction(() => window.studio.timeline.host.playing(), null, { timeout: 5000 });
  };
  // Poll rather than sample once: a fresh AudioContext and a just-created
  // copy instance take a few hundred ms to sound.
  const levelWithin = (ms, pred) => page.waitForFunction(pred, null, { timeout: ms, polling: 50 })
    .then(() => true, () => false);
  const pause = async () => { if (await host('playing')) await playBtn.click(); };

  // The scrub proxy builds in the background after import.
  await page.waitForFunction(() => {
    const a = [...window.studio.assets.values()].find((x) => x.name === 'tone.mp4');
    return a.proxyState === 'ready' || a.proxyState === 'unavailable';
  }, null, { timeout: 30000 }).catch(() => {});
  const proxy = await page.evaluate(() => [...window.studio.assets.values()].find((x) => x.name === 'tone.mp4').proxyState);
  check('scrub proxy builds', proxy === 'ready', { proxy });

  // 1. A hidden, unmuted copy of a file is audible under a muted visible original.
  await page.evaluate(() => {
    const c = window.comp();
    const orig = c.tracks[0].clips[0];
    c.tracks[0].muted = true;
    c.tracks.push({ name: 'copy', hidden: true, muted: false,
      clips: [{ ...JSON.parse(JSON.stringify(orig)), id: 'smoke_copy' }] });
    window.studio.onModelChange({ structural: true });
  });
  await play(0.2);
  const audible = await levelWithin(3000, () => window.__level() > 0.01);
  check('hidden copy audible', audible, { lvl: await page.evaluate(() => window.__level()) });
  await page.evaluate(() => { window.comp().tracks[1].muted = true; });
  const silent = await levelWithin(2000, () => window.__level() < 0.001);
  check('both muted is silent', silent, { lvl: await page.evaluate(() => window.__level()) });
  await pause();

  // 2. Ctrl+S must not split.
  const clipCount = () => page.evaluate(() => window.comp().tracks.reduce((n, t) => n + t.clips.length, 0));
  const before = await clipCount();
  await page.evaluate(() => document.activeElement?.blur());
  await host('setTime', 1.0);
  await page.keyboard.press('Control+s');
  check('ctrl+s does not split', await clipCount() === before, { before, after: await clipCount() });

  // 3. An offset copy as the original's matte plays without stutter: no
  // element is re-seeked during steady playback. (That the matte path never
  // seeks at all is pinned by a unit test; this catches seek storms.)
  await page.evaluate(() => {
    const c = window.comp();
    c.tracks[0].muted = false;
    c.tracks[1].clips[0].in = 1;   // out of step -> its own instance
    window.studio.onModelChange({ structural: true });
    window.studio.timeline.selectClip(c.tracks[0].clips[0].id);
  });
  await page.getByRole('button', { name: '＋Matte' }).click();
  const picked = await page.evaluate(() => {
    const sel = [...document.querySelectorAll('select')].find((s) => [...s.options].some((o) => o.value === 'smoke_copy'));
    if (!sel) return false;
    sel.value = 'smoke_copy';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  });
  check('matte source picked', picked);
  await play(0.1);
  await page.waitForTimeout(300);   // let the start-up seeks settle (comp is 3 s — stay clear of its end)
  const seeks0 = await page.evaluate(() => [...document.querySelectorAll('#media-pool video')].map((v) => v.__seeks ?? 0));
  await page.waitForTimeout(1000);
  const seeks1 = await page.evaluate(() => [...document.querySelectorAll('#media-pool video')].map((v) => v.__seeks ?? 0));
  const extra = seeks1.map((n, i) => n - (seeks0[i] ?? 0));
  check('matte source does not re-seek playing elements', extra.every((n) => n <= 1), { extra });
  await pause();

  // 4. Blur + screen blend on the top layer renders.
  await page.getByRole('textbox', { name: /Add an effect to/ }).click();
  await page.keyboard.type('gaussian');
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => window.comp().tracks[0].clips[0].effects?.length > 0, null, { timeout: 15000 });
  await page.locator('select').filter({ has: page.locator('option[value="screen"]') }).first().selectOption('screen');
  check('blur + screen applied', await page.evaluate(() => window.comp().tracks[0].clips[0].blend === 'screen'));

  // 5. Offline render completes and downloads.
  await host('setTime', 0);
  const download = page.waitForEvent('download', { timeout: 180000 });
  await page.getByRole('button', { name: /Export/ }).click();
  await page.getByText('Render video · WebM').click();
  const dl = await download;
  await dl.saveAs(OUT);
  const size = (await (await dl.createReadStream()).toArray()).reduce((n, b) => n + b.length, 0);
  check('offline render', size > 10000, { size });

  // ---- performance ------------------------------------------------------
  // Count effect-chain rebuilds (every SlangFx: the adjustment chain and each
  // clip's private colour + matte chains) and pipeline runs.
  await page.evaluate(() => {
    const proto = Object.getPrototypeOf(window.fx);
    if (!proto.__counted) {
      const rebuild = proto.rebuild;
      proto.rebuild = function (...a) { window.__rebuilds = (window.__rebuilds ?? 0) + 1; return rebuild.apply(this, a); };
      const render = window.fx.render;
      window.fx.render = function (...a) { window.__renders = (window.__renders ?? 0) + 1; return render.apply(this, a); };
      proto.__counted = true;
    }
  });
  await page.waitForTimeout(1500);   // let the blur stack finish building

  // 11. Drag-style transient edits and a property change rebuild nothing.
  const rebuilds = await page.evaluate(async () => {
    window.__rebuilds = 0;
    const clip = window.comp().tracks[0].clips[0];
    for (let i = 0; i < 30; i++) {
      clip.start = (i % 2) * 0.01;
      window.studio.onModelChange({ structural: false, transient: true });
      await new Promise((r) => requestAnimationFrame(r));
    }
    clip.start = 0;
    clip.props.opacity.v = 80;
    window.studio.onModelChange({ propKey: 'opacity' });
    for (let i = 0; i < 10; i++) await new Promise((r) => requestAnimationFrame(r));
    return window.__rebuilds;
  });
  check('edits do not rebuild effect chains', rebuilds === 0, { rebuilds });

  // 12. Paused and quiet: the pipeline idles; input brings it back.
  await pause();
  await page.mouse.move(5, 5);
  await page.waitForTimeout(1200);   // past the idle threshold
  const idleRuns = await page.evaluate(async () => {
    window.__renders = 0;
    await new Promise((r) => setTimeout(r, 1000));
    return window.__renders;
  });
  check('idle paused frames skip the pipeline', idleRuns <= 6, { idleRuns });
  const activeRuns = await page.evaluate(async () => {
    window.__renders = 0;
    const end = performance.now() + 500;
    while (performance.now() < end) {
      window.dispatchEvent(new PointerEvent('pointermove', { clientX: Math.random() * 100, clientY: 10 }));
      await new Promise((r) => requestAnimationFrame(r));
    }
    return window.__renders;
  });
  check('input renders at full rate', activeRuns >= 15, { activeRuns });
  // A change no input event reports still reaches the screen (keep-alive).
  await page.waitForTimeout(1200);
  const healed = await page.evaluate(async () => {
    window.__renders = 0;
    window.comp().tracks[0].clips[0].props.opacity.v = 60;
    await new Promise((r) => setTimeout(r, 400));
    return window.__renders;
  });
  check('silent changes render within the keep-alive', healed >= 1, { healed });

  // 13. A burst of drag pointermoves re-renders the timeline once per frame.
  const tlRenders = await page.evaluate(async () => {
    const tl = window.studio.timeline;
    const clipEl = document.querySelector('.tl-clip');
    if (!clipEl) return -1;
    const r = clipEl.getBoundingClientRect();
    let n = 0;
    const render = tl.render;
    tl.render = function (...a) { n++; return render.apply(this, a); };
    const opts = (x) => ({ clientX: x, clientY: r.top + r.height / 2, bubbles: true, pointerId: 1, button: 0, buttons: 1 });
    clipEl.dispatchEvent(new PointerEvent('pointerdown', opts(r.left + 20)));
    for (let i = 0; i < 60; i++) window.dispatchEvent(new PointerEvent('pointermove', opts(r.left + 20 + (i % 10) + 6)));
    const during = n;
    await new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(res)));
    const afterFrame = n;
    window.dispatchEvent(new PointerEvent('pointerup', opts(r.left + 20)));
    tl.render = render;
    return { during, afterFrame };
  });
  check('timeline drag renders once per frame', tlRenders.during <= 1 && tlRenders.afterFrame <= 3, tlRenders);
  // 13b. FrameCount is the comp frame: paused re-renders don't advance it.
  await page.evaluate(() => {
    const c = window.comp();
    c.tracks.unshift({ id: 'smoke_fx_track', name: 'adjust', clips: [{
      id: 'smoke_fx', kind: 'fx', name: 'blur', enabled: true, start: 0, dur: 3, mask: null,
      effects: [{ id: 'smoke_fx_eff', name: 'gaussian-blur', fxKind: 'preset',
        path: 'shaders/blur-bloom/gaussian-blur/gaussian-blur.slangp', enabled: true, params: {}, overlay: null }],
    }] });
    window.studio.onModelChange({ structural: true });
    window.studio.timeline.host.setTime(1.0);
  });
  await page.waitForFunction(() => window.fx.layers?.[0]?.runtime, null, { timeout: 15000 }).catch(() => {});
  const fc = await page.evaluate(async () => {
    const rt = () => window.fx.layers?.[0]?.runtime;
    for (let i = 0; i < 20; i++) {   // keep it active: renders at full rate
      window.dispatchEvent(new PointerEvent('pointermove', { clientX: i, clientY: 1 }));
      await new Promise((r) => requestAnimationFrame(r));
    }
    const a = rt()?.frameCount;
    for (let i = 0; i < 20; i++) {
      window.dispatchEvent(new PointerEvent('pointermove', { clientX: i, clientY: 2 }));
      await new Promise((r) => requestAnimationFrame(r));
    }
    return { a, b: rt()?.frameCount };
  });
  check('FrameCount is the comp frame and holds while paused', fc.a === 30 && fc.b === 30, fc);
  await page.evaluate(() => {
    const c = window.comp();
    c.tracks = c.tracks.filter((tr) => tr.id !== 'smoke_fx_track');
    window.studio.onModelChange({ structural: true });
  });

  // 13c. The comp holds still while an offline render runs; Escape cancels.
  const count0 = await clipCount();
  let downloaded = false;
  const onDl = () => { downloaded = true; };
  page.on('download', onDl);
  await page.evaluate(() => window.studio.timeline.selectClip(window.comp().tracks[0].clips[0].id));
  await page.getByRole('button', { name: /Export/ }).click();
  await page.getByText('Render video · WebM').click();
  const rendering = () => document.querySelector('#btn-export').textContent.includes('Cancel');
  await page.waitForFunction(rendering, null, { timeout: 5000 });
  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press('Delete');
  await page.keyboard.press('Control+z');
  await page.locator('.tl-clip').first().click({ force: true }).catch(() => {});
  const midCount = await clipCount();
  await page.keyboard.press('Escape');
  await page.waitForFunction(() => !document.querySelector('#btn-export').textContent.includes('Cancel'), null, { timeout: 30000 });
  await page.waitForTimeout(500);
  page.off('download', onDl);
  check('edits are locked during a render; Escape cancels it',
    midCount === count0 && (await clipCount()) === count0 && !downloaded, { count0, midCount, downloaded });

  // put the clip back where the persistence steps expect it
  await page.evaluate(() => { const c = window.comp().tracks[0].clips[0]; c.start = 0; c.props.opacity.v = 100; window.studio.onModelChange({}); });

  // ---- persistence ------------------------------------------------------
  const readIdb = (key) => page.evaluate((k) => new Promise((res) => {
    const r = indexedDB.open('lowkey-studio', 1);
    r.onsuccess = () => {
      const q = r.result.transaction('media', 'readonly').objectStore('media').get(k);
      q.onsuccess = () => { r.result.close(); res(q.result ?? null); };
      q.onerror = () => { r.result.close(); res(null); };
    };
    r.onerror = () => res(null);
  }), key);
  const autosave = async () => { const raw = await readIdb('autosave'); return raw ? JSON.parse(raw) : null; };
  const reloadStudio = async () => {
    await page.reload();
    await page.waitForFunction(() => window.studio && window.fx?.inputTexture
      && [...window.studio.assets.values()].some((a) => a.name === 'tone.mp4' && a.ready), null, { timeout: 30000 });
  };
  const origId = await page.evaluate(() => window.comp().tracks[0].clips[0].id);

  // 6a. An edit inside the 700 ms debounce survives an immediate reload
  // (the unload journal), and the autosave lives in IndexedDB.
  await page.evaluate(() => { window.comp().tracks[0].name = 'flushed-on-pagehide'; window.studio.onModelChange({}); });
  await reloadStudio();
  const flushed = await page.evaluate(() => ({ name: window.comp().tracks[0].name,
    legacy: localStorage.getItem('lowkey-studio.project.v2') !== null }));
  check('debounced edit flushed on reload', flushed.name === 'flushed-on-pagehide', flushed);
  check('autosave moved off localStorage', !flushed.legacy && !!(await autosave())?.comp, flushed);

  // 6. A painted mask far past the old 2 MB cap (incompressible noise).
  await page.evaluate((id) => window.studio.timeline.selectClip(id), origId);
  await page.getByRole('button', { name: '＋Paint' }).click();
  await page.keyboard.press('Escape');   // leave paint mode
  const paintBytes = await page.evaluate((id) => {
    const node = window.studio.clipMasks.get(id).nodes.find((n) => n.kind === 'paint');
    const c = node.source.getContext('2d');
    const img = c.createImageData(node.source.width, node.source.height);
    // Opaque random RGB: incompressible, and lossless through PNG (partial
    // alpha would be premultiplied and come back altered).
    for (let i = 0; i < img.data.length; i++) img.data[i] = (i & 3) === 3 ? 255 : (Math.random() * 256) | 0;
    c.putImageData(img, 0, 0);
    let sum = 0;
    const d = c.getImageData(0, 0, 64, 64).data;
    for (let i = 0; i < d.length; i++) sum = (sum * 31 + d[i]) >>> 0;
    return { bytes: node.source.toDataURL('image/png').length, sum };
  }, origId);
  check('paint mask is past the old cap', paintBytes.bytes > 2_000_000, paintBytes);

  // 7. Split and duplicate keep the mask stack on the new clip.
  await page.evaluate((id) => { window.studio.timeline.selectClip(id); window.studio.timeline.host.setTime(1.5); document.activeElement?.blur(); }, origId);
  await page.keyboard.press('s');
  const splitMask = await page.evaluate((id) => {
    const right = window.comp().tracks[0].clips.find((c) => c.id !== id);
    const m = right && window.studio.clipMasks.get(right.id);
    return { right: right?.id, kinds: m?.nodes.map((n) => n.kind) ?? [] };
  }, origId);
  check('split keeps the mask', splitMask.kinds.includes('paint') && splitMask.kinds.includes('layer'), splitMask);
  await page.evaluate((id) => { window.studio.timeline.selectClip(id); document.activeElement?.blur(); }, origId);
  await page.keyboard.press('Control+d');
  const dupMask = await page.evaluate((id) => {
    const sel = [...window.studio.timeline.selClips][0];
    return { copy: sel, kinds: (sel !== id && window.studio.clipMasks.get(sel)?.nodes.map((n) => n.kind)) || [] };
  }, origId);
  check('duplicate keeps the mask', dupMask.kinds.includes('paint'), dupMask);

  // 8. The big masks (far past localStorage's quota) persist via the
  // IndexedDB autosave.
  await page.waitForTimeout(2000);   // past the debounce: a normal save lands
  await reloadStudio();
  const after = await page.evaluate((id) => {
    const c = window.comp();
    const clip = c.tracks.flatMap((t) => t.clips).find((k) => k.id === id);
    const node = window.studio.clipMasks.get(id)?.nodes.find((n) => n.kind === 'paint');
    let sum = null;
    if (node) {
      sum = 0;
      const d = node.source.getContext('2d').getImageData(0, 0, 64, 64).data;
      for (let i = 0; i < d.length; i++) sum = (sum * 31 + d[i]) >>> 0;
    }
    const right = c.tracks.flatMap((t) => t.clips).find((k) => k.id !== id && window.studio.clipMasks.get(k.id));
    return { hasClip: !!clip, sum, splitMaskKept: !!right };
  }, origId);
  check('big paint mask survives reload (exact pixels)', after.sum === paintBytes.sum && after.splitMaskKept,
    { ...after, expected: paintBytes.sum });

  // 16. Undo snapshots don't carry painted-mask PNGs (saves used to write
  // them into the live comp, so every later snapshot copied them).
  await page.evaluate(() => window.studio.onModelChange({}));
  await page.waitForTimeout(1500);   // a save lands
  const snap = await page.evaluate(() => {
    const h = window.studio.timeline.host.history;
    const c = window.comp();
    h.record(c, () => { c.tracks[0].name = 'snapshot-probe'; });
    const last = h.undoStack[h.undoStack.length - 1] ?? '';
    return { bytes: last.length, png: last.includes('data:image/png') };
  });
  check('undo snapshots carry no mask PNGs', !snap.png && snap.bytes < 200_000, snap);

  // 17. Deleting a masked clip frees its mask GPU state (parked for undo);
  // undo brings the mask back.
  const parked = await page.evaluate(async (id) => {
    window.studio.timeline.selectClip(id);
    document.activeElement?.blur();
    window.studio.timeline.deleteSelection();
    const gone = !window.studio.clipMasks.has(id) && window.studio.parkedMasks.has(id);
    window.studio.timeline.host.undo();
    await new Promise((r) => setTimeout(r, 100));
    const back = window.studio.clipMasks.has(id) && !window.studio.parkedMasks.has(id)
      && window.comp().tracks.some((tr) => tr.clips.some((k) => k.id === id));
    return { gone, back };
  }, origId);
  check('a deleted clip parks its mask; undo restores it', parked.gone && parked.back, parked);

  // 9. Removing media is undoable: the clips AND the media come back.
  const assetId = await page.evaluate(() => [...window.studio.assets.values()].find((a) => a.name === 'tone.mp4').id);
  const clipsOf = () => page.evaluate((a) => window.comp().tracks.flatMap((t) => t.clips).filter((c) => c.assetId === a).length, assetId);
  const before9 = await clipsOf();
  if (!await page.evaluate(() => typeof window.studio.removeAsset === 'function'))
    throw new Error('this build does not expose window.studio.removeAsset');
  await page.evaluate((a) => window.studio.removeAsset(window.studio.assets.get(a)), assetId);
  const removed = await clipsOf();
  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press('Control+z');
  const back = await page.waitForFunction((a) => window.studio.assets.get(a)?.ready, assetId, { timeout: 10000 })
    .then(() => true, () => false);
  check('undo remove-media restores clips and media', removed === 0 && (await clipsOf()) === before9 && back,
    { before9, removed, after: await clipsOf(), back });

  // 10. Media whose blob is gone stays in the project as offline.
  await page.evaluate((a) => new Promise((res) => {
    const r = indexedDB.open('lowkey-studio', 1);
    r.onsuccess = () => {
      const tx = r.result.transaction('media', 'readwrite');
      tx.objectStore('media').delete(`asset:${a}`);
      tx.oncomplete = () => { r.result.close(); res(); };
    };
  }), assetId);
  await page.reload();
  await page.waitForFunction(() => window.studio && window.fx?.inputTexture, null, { timeout: 30000 });
  await page.waitForTimeout(1500);
  await page.evaluate(() => window.studio.onModelChange({}));
  await page.waitForTimeout(1500);
  const saved = await autosave();
  const kept = saved?.assets?.find((m) => m.id === assetId);
  const clipsKept = saved?.comp?.tracks.flatMap((t) => t.clips).filter((c) => c.assetId === assetId).length;
  check('offline media kept in the project', !!kept?.offline && clipsKept === before9, { kept, clipsKept });

  // 14. Audio drivers follow an audio clip that slides — and its undo.
  // burst.wav is silent but for a 100 ms tone at exactly 1.0 s.
  if (!await page.evaluate(() => typeof window.studio.audioSignal === 'function')) {
    check('audio drivers follow a slid clip and its undo', false, 'this build does not expose window.studio.audioSignal');
    return { results, errors };
  }
  await page.setInputFiles('#file-input', 'C:/Users/steph/dev/loki/.playwright-mcp/burst.wav');
  await page.waitForFunction(() => [...window.studio.assets.values()].some((a) => a.name === 'burst.wav' && a.ready),
    null, { timeout: 20000 });
  const burstId = await page.evaluate(() => {
    const c = window.comp();
    const asset = [...window.studio.assets.values()].find((a) => a.name === 'burst.wav');
    const clip = c.tracks.flatMap((t) => t.clips).find((k) => k.assetId === asset.id);
    clip.start = 0; clip.in = 0;
    const host = c.tracks.flatMap((t) => t.clips).find((k) => k.kind === 'media') ?? clip;
    host.props.opacity ??= { v: 100, anim: false, keys: [] };
    (host.props.volume ??= { v: 100, anim: false, keys: [] }).driver =
      { enabled: true, source: 'audio', band: 'level', follow: 'level', release: 0, amount: 0, mode: 'add' };
    window.studio.onModelChange({ structural: true });
    return clip.id;
  });
  const drvPeak = () => page.evaluate(() => {
    const d = { source: 'audio', band: 'level', follow: 'level', release: 0 };
    let best = -1, at = null;
    for (let i = 0; i < 120; i++) { const t = i / 30; const v = window.studio.audioSignal(d, t); if (v > best) { best = v; at = t; } }
    return Math.round(at * 30);   // comp frame of the loudest envelope sample
  });
  const settle = (frame) => page.waitForFunction(async (f) => {
    const d = { source: 'audio', band: 'level', follow: 'level', release: 0 };
    let best = -1, at = null;
    for (let i = 0; i < 120; i++) { const t = i / 30; const v = window.studio.audioSignal(d, t); if (v > best) { best = v; at = t; } }
    return Math.abs(Math.round(at * 30) - f) <= 1;
  }, frame, { timeout: 8000, polling: 200 }).then(() => true, () => false);
  const atStart = await settle(30);
  await page.evaluate((id) => {
    const clip = window.comp().tracks.flatMap((t) => t.clips).find((k) => k.id === id);
    window.studio.timeline.host.history.record(window.comp(), () => { clip.start = 0.5; });
    window.studio.onModelChange({ structural: true });
  }, burstId);
  const slid = await settle(45);
  await page.evaluate(() => document.activeElement?.blur());
  await page.keyboard.press('Control+z');
  const undone = await settle(30);
  check('audio drivers follow a slid clip and its undo', atStart && slid && undone,
    { atStart, slid, undone, peakFrame: await drvPeak() });

  // 18. The storage sweep frees records nothing references and keeps the rest.
  const swept = await page.evaluate(async () => {
    const tx = (mode, fn) => new Promise((res) => {
      const r = indexedDB.open('lowkey-studio', 1);
      r.onsuccess = () => {
        const t = r.result.transaction('media', mode);
        const out = fn(t.objectStore('media'));
        t.oncomplete = () => { r.result.close(); res(out?.result ?? null); };
      };
    });
    await tx('readwrite', (s) => { s.put(new Blob(['x']), 'asset:orphan_smoke'); s.put({ blobs: [] }, 'roto:orphan_node'); });
    await window.studio.sweepStorage();
    const keys = await tx('readonly', (s) => s.getAllKeys());
    const burst = [...window.studio.assets.values()].find((a) => a.name === 'burst.wav');
    return { orphanGone: !keys.includes('asset:orphan_smoke') && !keys.includes('roto:orphan_node'),
      liveKept: keys.includes(`asset:${burst.id}`) };
  });
  check('storage sweep frees unreferenced records only', swept.orphanGone && swept.liveKept, swept);

  // 19. Re-importing the same file reuses its stored blob (no second copy).
  const dedupe = await page.evaluate(async () => {
    const keyCount = () => new Promise((res) => {
      const r = indexedDB.open('lowkey-studio', 1);
      r.onsuccess = () => {
        const q = r.result.transaction('media').objectStore('media').getAllKeys();
        q.onsuccess = () => { r.result.close(); res(q.result.filter((k) => String(k).startsWith('asset:')).length); };
      };
    });
    const before = await keyCount();
    const old = [...window.studio.assets.values()].find((a) => a.name === 'burst.wav');
    window.studio.removeAsset(old);   // gone from the project, blob kept
    return { before, oldId: old.id };
  });
  await page.setInputFiles('#file-input', 'C:/Users/steph/dev/loki/.playwright-mcp/burst.wav');
  await page.waitForFunction(() => [...window.studio.assets.values()].some((a) => a.name === 'burst.wav' && a.ready),
    null, { timeout: 20000 });
  await page.waitForTimeout(500);
  const dedupe2 = await page.evaluate(async () => {
    const ids = [...window.studio.assets.values()].filter((a) => a.name === 'burst.wav').map((a) => a.id);
    const n = await new Promise((res) => {
      const r = indexedDB.open('lowkey-studio', 1);
      r.onsuccess = () => {
        const q = r.result.transaction('media').objectStore('media').getAllKeys();
        q.onsuccess = () => { r.result.close(); res(q.result.filter((k) => String(k).startsWith('asset:')).length); };
      };
    });
    return { ids, after: n };
  });
  check('re-importing a file reuses its stored copy',
    dedupe2.ids.length === 1 && dedupe2.ids[0] === dedupe.oldId && dedupe2.after === dedupe.before,
    { ...dedupe, ...dedupe2 });

  // 15. The camera-shake case: shake on the VISUAL clip, its hit driven by
  // the bass band of a separate audio track that was trimmed at the front
  // and slid to the start. The hit uniform must rise on the hits you hear,
  // and the release tail must ring on after each hit.
  await page.evaluate(() => new Promise((r) => {
    localStorage.clear();
    const q = indexedDB.deleteDatabase('lowkey-studio');
    q.onsuccess = q.onerror = q.onblocked = () => r();
  }));
  await page.reload();
  await page.waitForFunction(() => window.studio && window.fx?.inputTexture, null, { timeout: 30000 });
  for (const f of ['tone.mp4', 'basshits.wav']) {
    await page.setInputFiles('#file-input', `C:/Users/steph/dev/loki/.playwright-mcp/${f}`);
    await page.waitForFunction((n) => [...window.studio.assets.values()].some((a) => a.name === n && a.ready), f, { timeout: 20000 });
  }
  const videoId = await page.evaluate(() => {
    const c = window.comp();
    const byName = (n) => [...window.studio.assets.values()].find((a) => a.name === n);
    const tone = byName('tone.mp4');
    const bass = byName('basshits.wav');
    c.tracks = c.tracks.filter((tr) => tr.clips.some((k) => k.assetId === tone.id || k.assetId === bass.id));
    const video = c.tracks.flatMap((tr) => tr.clips).find((k) => k.assetId === tone.id);
    const audio = c.tracks.flatMap((tr) => tr.clips).find((k) => k.assetId === bass.id);
    c.tracks.find((tr) => tr.clips.includes(video)).muted = true;   // hear only the bass
    video.start = 0; video.in = 0; video.dur = 5;
    video.effects = [{ id: 'fx_shake', name: 'camera-shake', fxKind: 'preset',
      path: 'shaders/motion/camera-shake/camera-shake.slangp', enabled: true, overlay: null,
      params: { hit: { v: 0, anim: false, keys: [], driver: { enabled: true, source: 'audio', band: 'bass',
        follow: 'level', release: 0.25, sensitivity: 1.5, decay: 0.35, amount: 1, offset: 0, mode: 'add',
        wave: 'sin', freq: 1, phase: 0, width: 0.5 } } } }];
    audio.in = 0.37; audio.dur = 4.63; audio.start = 0;   // 0.37 s trimmed off, slid to 0
    c.dur = 5;
    window.studio.onModelChange({ structural: true });
    return video.id;
  });
  await page.waitForFunction((vid) => window.studio.mediaChains.get(vid)?.fx?.layers?.[0]?.runtime
    && window.studio.audioSignal({ source: 'audio', band: 'bass', follow: 'level', release: 0 }, 0.4) > 0.3,
  videoId, { timeout: 15000, polling: 200 }).catch(() => {});
  await host('setTime', 0);
  await playBtn.click();
  const trace = await page.evaluate((vid) => new Promise((res) => {
    const out = []; const t0 = performance.now();
    const poll = () => {
      const rt = window.studio.mediaChains.get(vid)?.fx?.layers?.[0]?.runtime;
      out.push([window.studio.timeline.host.time(), window.__level(), rt?.paramValues?.get('hit') ?? 0]);
      if (performance.now() - t0 > 2300) return res(out);
      requestAnimationFrame(poll);
    };
    poll();
  }), videoId);
  await pause();
  const edges = (col, thr) => {
    const o = []; let prev = 0;
    for (const r of trace) { if (r[col] > thr && prev <= thr) o.push(r[0]); prev = r[col]; }
    return o;
  };
  const expectHits = [0.38, 0.88, 1.38, 1.88];   // 0.25 - 0.37 + 0.5k
  const heard = edges(1, 0.03).slice(0, 4);
  const shook = edges(2, 0.5).slice(0, 4);
  const near = (got, tol) => got.length === 4 && got.every((v, k) => Math.abs(v - expectHits[k]) <= tol);
  // tail: 120 ms after a hit ENDS (hit lasts 80 ms) the 0.25 s release still reads well above 0
  const tail = trace.filter((r) => r[0] > 0.38 + 0.08 + 0.1 && r[0] < 0.38 + 0.08 + 0.14).map((r) => r[2]);
  check('camera-shake hit lines up with a trimmed, slid bass track',
    near(heard, 0.08) && near(shook, 0.08),
    { heard: heard.map((v) => +v.toFixed(3)), shook: shook.map((v) => +v.toFixed(3)), expectHits });
  check('bass release tail rings on after each hit', tail.length > 0 && tail.every((v) => v > 0.3),
    { tail: tail.map((v) => +v.toFixed(2)) });

  // 20. Adding and removing an effect returns every GPU texture it made
  // (torn-down per-clip chains used to leak their runtimes, preset
  // textures and mask state).
  const settleChains = () => page.waitForFunction((vid) => {
    const e = window.studio.mediaChains.get(vid);
    return !e || (!e.building && !e.dirty);
  }, videoId, { timeout: 15000, polling: 100 }).catch(() => {});
  const cycleShake = async (on) => {
    await page.evaluate(([vid, on]) => {
      const clip = window.comp().tracks.flatMap((tr) => tr.clips).find((k) => k.id === vid);
      clip.effects = on ? [{ id: `fx_${Math.random().toString(36).slice(2)}`, name: 'camera-shake', fxKind: 'preset',
        path: 'shaders/motion/camera-shake/camera-shake.slangp', enabled: true, overlay: null, params: {} }] : [];
      window.studio.onModelChange({ structural: true });
    }, [videoId, on]);
    await settleChains();
    await page.waitForTimeout(400);
  };
  await cycleShake(false);
  const texBefore = await page.evaluate(() => window.__texLive);
  for (let i = 0; i < 3; i++) { await cycleShake(true); await cycleShake(false); }
  const texAfter = await page.evaluate(() => window.__texLive);
  check('adding and removing an effect frees its GPU textures', texAfter - texBefore <= 2, { texBefore, texAfter });

  check('no page errors', errors.length === 0, errors.slice(0, 5));
  return { results, errors };
}
