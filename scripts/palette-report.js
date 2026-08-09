#!/usr/bin/env node
/**
 * palette-report — right-click → command palette ready, per open.
 *
 * The palette's first open of a session is reliably slower than the rest, and
 * the only way to see why is to compare open #1 against open #2 in the same
 * session: whatever shrinks between them is a one-time cost. This prints both,
 * phase by phase, plus the main-thread blocks that landed inside each open.
 *
 * Usage:
 *   node scripts/palette-report.js            # the last 6 opens
 *   node scripts/palette-report.js -n 20
 *   node scripts/palette-report.js --json
 *   node scripts/palette-report.js --log PATH
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
  const opts = { count: 6, json: false, log: defaultLogPath() };
  for (let i = 2; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === '-n' || a === '--count') opts.count = Number(argv[++i]) || 6;
    else if (a === '--json') opts.json = true;
    else if (a === '--log') opts.log = argv[++i];
    else if (a === '-h' || a === '--help') opts.help = true;
  }
  return opts;
}

const TAIL_BYTES = 4 * 1024 * 1024;

function readTail(file) {
  const { size } = fs.statSync(file);
  const start = Math.max(0, size - TAIL_BYTES);
  const fd = fs.openSync(file, 'r');
  try {
    const buf = Buffer.alloc(size - start);
    fs.readSync(fd, buf, 0, buf.length, start);
    const text = buf.toString('utf8');
    return start > 0 ? text.slice(text.indexOf('\n') + 1) : text;
  } finally {
    fs.closeSync(fd);
  }
}

function loadOpens(file) {
  const out = [];
  for (const line of readTail(file).split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('{')) continue;
    let rec;
    try {
      rec = JSON.parse(trimmed);
    } catch {
      continue;
    }
    // Renderer events arrive prefixed by the main-process log forwarder.
    if (rec.scope !== 'palette' && rec.scope !== 'renderer:palette') continue;
    if (rec.message !== 'palette-open') continue;
    out.push(rec);
  }
  return out;
}

const ms = (n) => (typeof n === 'number' && n >= 0 ? `${n}ms` : '—');

function printOpen(rec) {
  const d = rec.data || {};
  const p = d.phases || {};
  console.log('='.repeat(70));
  console.log(
    `open #${d.seq}${d.firstOpen ? '  (FIRST OPEN)' : ''}   ${rec.ts}   ` +
      `total ${ms(d.totalMs)}   [${d.reason}]`
  );

  console.log('\n  marks (ms from right-click)');
  for (const m of d.marks || []) {
    console.log(`    ${String(m.at).padStart(5)}  ${m.name}`);
  }

  console.log('\n  phases');
  console.log(`    right-click → rendered ......... ${ms(p.toRender)}`);
  console.log(`    → measured + positioned ........ ${ms(p.toPositioned)}`);
  console.log(`    → painted ...................... ${ms(p.toPainted)}`);
  console.log(`    → search engine mounted ........ ${ms(p.toEngine)}`);
  console.log(`    → data ready ................... ${ms(p.toDataReady)}`);

  const lt = d.longTasks;
  if (lt?.count) {
    console.log(`\n  main-thread blocks: ${lt.count}, ${ms(lt.totalMs)} total`);
    for (const t of lt.worst) {
      console.log(
        `    at ${String(t.atMs).padStart(5)}ms  blocked ${ms(t.durationMs)}`
      );
    }
  }
  console.log('');
}

function printComparison(opens) {
  const first = opens.find((o) => o.data?.firstOpen);
  const later = opens.filter((o) => !o.data?.firstOpen);
  if (!first || later.length === 0) return;
  const avg = (xs) =>
    xs.length ? Math.round(xs.reduce((a, b) => a + b, 0) / xs.length) : -1;
  const f = first.data;
  const laterTotal = avg(later.map((o) => o.data.totalMs));

  console.log('='.repeat(70));
  console.log('FIRST OPEN vs LATER OPENS — the difference is the one-time cost');
  console.log(
    `  total:            ${ms(f.totalMs)}  vs  ${ms(laterTotal)} (avg of ${
      later.length
    })`
  );
  for (const key of [
    'toRender',
    'toPositioned',
    'toPainted',
    'toEngine',
    'toDataReady',
  ]) {
    const l = avg(
      later.map((o) => o.data.phases?.[key]).filter((v) => typeof v === 'number' && v >= 0)
    );
    console.log(`  ${key.padEnd(16)}  ${ms(f.phases?.[key])}  vs  ${ms(l)}`);
  }
  console.log('');
}

function main() {
  const opts = parseArgs(process.argv);
  if (opts.help) {
    console.log('Usage: node scripts/palette-report.js [-n N] [--json] [--log PATH]');
    return;
  }
  if (!fs.existsSync(opts.log)) {
    console.error(`No log at ${opts.log}`);
    process.exitCode = 1;
    return;
  }
  const opens = loadOpens(opts.log).slice(-opts.count);
  if (!opens.length) {
    console.error('No traced palette opens found. Open the palette and retry.');
    process.exitCode = 1;
    return;
  }
  if (opts.json) {
    console.log(JSON.stringify(opens, null, 2));
    return;
  }
  console.log(`log: ${opts.log}`);
  opens.forEach(printOpen);
  printComparison(opens);
}

main();
