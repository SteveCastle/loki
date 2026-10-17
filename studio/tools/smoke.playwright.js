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

  check('no page errors', errors.length === 0, errors.slice(0, 5));
  return { results, errors };
}
