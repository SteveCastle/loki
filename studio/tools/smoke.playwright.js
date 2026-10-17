// Studio end-to-end smoke test — run it through the Playwright MCP tool
// (browser_run_code_unsafe with `filename: studio/tools/smoke.playwright.js`).
//
// Setup, from the repo root:
//   node studio/publish.mjs && node studio/serve.mjs 8794      (separate origin:
//     the test wipes its storage, so never point it at your working 8790)
//   ffmpeg -f lavfi -i testsrc=size=320x240:rate=30 -f lavfi -i sine=frequency=440 \
//     -t 6 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest .playwright-mcp/tone.mp4
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
    const connect = AudioNode.prototype.connect;
    AudioNode.prototype.connect = function (dst, ...rest) {
      if (dst instanceof AudioDestinationNode && !window.__an) {
        const an = this.context.createAnalyser();
        connect.call(this, an);
        window.__an = an;
      }
      return connect.call(this, dst, ...rest);
    };
    window.__level = () => {
      if (!window.__an) return 0;
      const b = new Float32Array(window.__an.fftSize);
      window.__an.getFloatTimeDomainData(b);
      let m = 0; for (const v of b) m = Math.max(m, Math.abs(v));
      return m;
    };
    document.addEventListener('seeking', (e) => { e.target.__seeks = (e.target.__seeks ?? 0) + 1; }, true);
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

  check('no page errors', errors.length === 0, errors.slice(0, 5));
  return { results, errors };
}
