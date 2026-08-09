#!/usr/bin/env node
/**
 * startup-report — read app-log.jsonl and print what a launch actually did.
 *
 * "Opening a file from the filesystem should be instant" is the viewer's
 * headline performance claim, and this is how it gets checked against reality:
 * run the installed app, then run this. It joins the main process's marks
 * (measured from process start) with the renderer's (measured from document
 * start) via the shared bootId, and prints the phase costs, the things that
 * blocked each process, and how the media itself was delivered.
 *
 * Usage:
 *   node scripts/startup-report.js              # the last launch
 *   node scripts/startup-report.js -n 5         # the last 5 launches
 *   node scripts/startup-report.js --json       # machine-readable
 *   node scripts/startup-report.js --log <path> # a log shipped from elsewhere
 */

const fs = require('fs');
const os = require('os');
const path = require('path');

const APP_NAME = 'Lowkey Media Viewer';

function defaultLogPath() {
  if (process.platform === 'win32') {
    return path.join(
      process.env.APPDATA || path.join(os.homedir(), 'AppData', 'Roaming'),
      APP_NAME,
      'app-log.jsonl'
    );
  }
  if (process.platform === 'darwin') {
    return path.join(
      os.homedir(),
      'Library',
      'Application Support',
      APP_NAME,
      'app-log.jsonl'
    );
  }
  return path.join(os.homedir(), '.config', APP_NAME, 'app-log.jsonl');
}

function parseArgs(argv) {
  const opts = { count: 1, json: false, log: defaultLogPath() };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '-n' || a === '--count') opts.count = Number(argv[++i]) || 1;
    else if (a === '--json') opts.json = true;
    else if (a === '--log') opts.log = argv[++i];
    else if (a === '-h' || a === '--help') opts.help = true;
  }
  return opts;
}

// The log is append-only and can be several MB; only the tail is ever relevant.
const TAIL_BYTES = 4 * 1024 * 1024;

function readTail(file) {
  const { size } = fs.statSync(file);
  const start = Math.max(0, size - TAIL_BYTES);
  const fd = fs.openSync(file, 'r');
  try {
    const buf = Buffer.alloc(size - start);
    fs.readSync(fd, buf, 0, buf.length, start);
    const text = buf.toString('utf8');
    // A mid-line start would produce one unparseable record; drop it.
    return start > 0 ? text.slice(text.indexOf('\n') + 1) : text;
  } finally {
    fs.closeSync(fd);
  }
}

function loadLaunches(file) {
  const byBoot = new Map();
  for (const line of readTail(file).split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue;
    let rec;
    try {
      rec = JSON.parse(trimmed);
    } catch {
      continue;
    }
    const scope = rec.scope || '';
    if (scope !== 'startup' && scope !== 'renderer:startup') continue;
    const bootId = rec.data?.bootId || 'unknown';
    if (!byBoot.has(bootId)) byBoot.set(bootId, []);
    byBoot.get(bootId).push({ ...rec, side: scope.startsWith('renderer') ? 'r' : 'm' });
  }
  return [...byBoot.entries()]
    .map(([bootId, events]) => ({ bootId, events }))
    .filter((l) => l.bootId !== 'unknown')
    .sort((a, b) => (a.events[0].ts < b.events[0].ts ? -1 : 1));
}

const ms = (n) => (typeof n === 'number' && n >= 0 ? `${n}ms` : '—');
const bytes = (n) => {
  if (typeof n !== 'number' || n < 0) return '—';
  if (n < 1024) return `${n}B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)}KB`;
  return `${(n / 1024 / 1024).toFixed(1)}MB`;
};

function printLaunch(launch) {
  const { bootId, events } = launch;
  const find = (msg) => events.find((e) => e.message === msg);
  const summary = find('startup-summary')?.data;
  const target = find('launch-target')?.data;
  const preload = find('preload')?.data;
  const warm = events.filter((e) => e.message === 'media-warm').map((e) => e.data);
  const stalls = find('main-loop-stalls')?.data;

  console.log('='.repeat(74));
  console.log(
    `launch ${bootId}   ${events[0].ts}   v${summary?.appVersion ?? '?'}`
  );
  if (target) {
    console.log(
      `opened: ${target.kind}  ${target.initialFile || '(nothing)'}` +
        (target.sizeBytes >= 0 ? `  ${bytes(target.sizeBytes)}` : '') +
        (target.statMs >= 0 ? `  stat ${ms(target.statMs)}` : '')
    );
  }
  if (preload && !preload.hasFile && target?.kind === 'file') {
    console.log('  !! preload resolved NO file though one was opened');
  }

  // The two processes measure from different origins: main from process start,
  // the renderer from when its document began loading. The renderer's clock
  // starts when the window is created, so that mark is the offset that puts
  // everything on one axis — which is the only way to see, say, a DB import in
  // main landing in the middle of the renderer's module evaluation.
  const windowCreatedAt = find('window-created')?.data?.sinceLaunchMs;
  const offset = typeof windowCreatedAt === 'number' ? windowCreatedAt : 0;

  const rows = [];
  for (const e of events) {
    if (e.side === 'm' && typeof e.data?.sinceLaunchMs === 'number') {
      if (e.message === 'main-loop-stalls') continue;
      rows.push({ at: e.data.sinceLaunchMs, who: 'main', name: e.message });
    }
  }
  // Renderer marks recorded outside the bundle (the preload) plus the bundle's
  // own timeline, so nothing is missed if the summary never fires.
  const rendererMarks = new Map();
  for (const e of events) {
    if (e.side === 'r' && typeof e.data?.at === 'number' && e.message !== 'media-warm') {
      rendererMarks.set(e.message, e.data.at);
    }
  }
  for (const m of summary?.timeline ?? []) rendererMarks.set(m.name, m.at);
  for (const [name, at] of Object.entries(summary?.paint ?? {})) {
    rendererMarks.set(name, at);
  }
  for (const [name, at] of rendererMarks) {
    rows.push({ at: at + offset, who: 'rend', name });
  }
  rows.sort((a, b) => a.at - b.at);

  console.log('\n-- timeline (ms from process start) --');
  let prev = 0;
  for (const r of rows) {
    const gap = r.at - prev;
    // Flag the gaps worth investigating rather than making the reader subtract.
    const flag = gap >= 100 ? `  <-- +${gap}ms` : '';
    console.log(
      `  ${String(r.at).padStart(6)}  ${r.who.padEnd(4)}  ${r.name}${flag}`
    );
    prev = r.at;
  }

  // A launch that opened a native picker spent part of its wall time waiting
  // for a human. Reporting that as app time sends you chasing nothing.
  const pickerAt = (summary?.timeline ?? []).find((m) =>
    m.name.startsWith('waiting-on-')
  );
  if (pickerAt) {
    const mounted = (summary?.timeline ?? []).find(
      (m) => m.name === 'media-mounted'
    );
    const thinking = mounted ? mounted.at - pickerAt.at : -1;
    console.log(
      `\n  NOTE: a file/folder picker opened at ${ms(pickerAt.at)}. ` +
        `~${ms(thinking)} of this launch is the user deciding, not the app.`
    );
  }

  if (summary?.phases) {
    const p = summary.phases;
    console.log('\n-- phases --');
    console.log(`  before our first JS line ......... ${ms(p.toBundleStart)}`);
    console.log(`  evaluating the module graph ...... ${ms(p.moduleGraphEval)}`);
    console.log(`  to react render .................. ${ms(p.toReactRender)}`);
    console.log(`  to providers ready ............... ${ms(p.toProvidersReady)}`);
    console.log(`  to media element mounted ......... ${ms(p.toMediaMounted)}`);
    console.log(`  media fetch + decode ............. ${ms(p.mediaLoad)}`);
    console.log(`  TOTAL to first media ............. ${ms(p.total)}  (${summary.reason})`);
  }

  if (warm.length) {
    console.log('\n-- preload warm-up --');
    for (const w of warm) {
      console.log(
        `  ${String(w.at).padStart(6)}ms  ${w.outcome}` +
          (w.elapsedMs >= 0 ? `  after ${ms(w.elapsedMs)}` : '') +
          (w.width ? `  ${w.width}x${w.height}` : '') +
          (w.error ? `  ${w.error}` : '')
      );
    }
  }

  const reads = events.filter(
    (e) => e.message === 'media-read' || e.message === 'media-read-respond'
  );
  if (reads.length) {
    console.log('\n-- gsm:// reads (main process) --');
    for (const e of reads) {
      const d = e.data;
      console.log(
        `  #${d.seq} ${e.message === 'media-read' ? d.outcome : `HTTP ${d.status}`}` +
          `  stat ${ms(d.statMs)}  respond ${ms(d.respondMs)}` +
          (d.totalMs >= 0 ? `  total ${ms(d.totalMs)}` : '') +
          `  ${bytes(d.size)}  ${path.basename(String(d.filePath || ''))}`
      );
    }
  }

  const nav = summary?.navigation;
  if (nav && Object.keys(nav).length) {
    console.log('\n-- document --');
    console.log(
      `  responseEnd ${ms(nav.responseEnd)}  domInteractive ${ms(
        nav.domInteractive
      )}  DCL ${ms(nav.domContentLoaded)}  load ${ms(nav.loadEvent)}`
    );
  }

  if (summary?.serverCalls?.length) {
    console.log('\n-- media-server calls during startup --');
    for (const c of summary.serverCalls) {
      console.log(
        `  start ${String(c.startMs).padStart(5)}ms  took ${ms(
          c.durationMs
        )}  ${c.name}`
      );
    }
  }

  const lt = summary?.longTasks;
  if (lt?.count) {
    console.log(
      `\n-- renderer long tasks: ${lt.count}, ${ms(lt.totalMs)} total --`
    );
    for (const t of lt.worst) {
      console.log(`  at ${String(t.atMs).padStart(6)}ms  blocked ${ms(t.durationMs)}`);
    }
  }

  if (stalls?.count) {
    console.log(
      `\n-- main-process event-loop stalls: ${stalls.count}, ${ms(
        stalls.totalStalledMs
      )} total --`
    );
    for (const s of stalls.worst) {
      console.log(`  at ${String(s.atMs).padStart(6)}ms  stalled ${ms(s.stallMs)}`);
    }
  }
  console.log('');
}

function main() {
  const opts = parseArgs(process.argv);
  if (opts.help) {
    console.log(
      'Usage: node scripts/startup-report.js [-n COUNT] [--json] [--log PATH]'
    );
    return;
  }
  if (!fs.existsSync(opts.log)) {
    console.error(`No log at ${opts.log}`);
    console.error('Run the app once, then try again.');
    process.exitCode = 1;
    return;
  }
  const launches = loadLaunches(opts.log).slice(-opts.count);
  if (!launches.length) {
    console.error(
      'No instrumented launches found. This needs a build that includes ' +
        'startup tracing (bootId on the startup marks).'
    );
    process.exitCode = 1;
    return;
  }
  if (opts.json) {
    console.log(JSON.stringify(launches, null, 2));
    return;
  }
  console.log(`log: ${opts.log}`);
  launches.forEach(printLaunch);
}

main();
