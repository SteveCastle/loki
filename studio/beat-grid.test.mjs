import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const src = readFileSync(new URL('./beat-grid.js', import.meta.url), 'utf8');
const { analyzeBeats, gridLines, nearestGridLine, gridStep } =
  await import(`data:text/javascript;base64,${Buffer.from(src).toString('base64')}`);

/** Kick on every beat (accented on the one), hats on the off-beats. */
function drumLoop(bpm, offset, secs = 40, sr = 44100) {
  const out = new Float32Array(secs * sr);
  const period = 60 / bpm;
  for (let n = 0; ; n++) {
    const t = offset + n * period;
    if (t >= secs) break;
    const kick = (n % 4 === 0 ? 1 : 0.6);
    for (let i = 0; i < 0.12 * sr; i++) {
      const k = Math.floor(t * sr) + i;
      if (k < out.length) out[k] += kick * Math.sin(2 * Math.PI * (55 + 90 * Math.exp(-i / 800)) * i / sr) * Math.exp(-i / 2500);
    }
    const th = t + period / 2;
    let seed = n * 7919 + 1;
    for (let i = 0; i < 0.03 * sr; i++) {
      const k = Math.floor(th * sr) + i;
      seed = (seed * 16807) % 2147483647;
      if (k < out.length) out[k] += 0.25 * ((seed / 2147483647) * 2 - 1) * Math.exp(-i / 400);
    }
  }
  return out;
}

for (const [bpm, offset] of [[128, 0.23], [100, 0.5], [90, 0.07], [150, 0.31]]) {
  test(`finds ${bpm} bpm @ ${offset}s`, () => {
    const r = analyzeBeats(drumLoop(bpm, offset), 44100);
    assert.ok(Math.abs(r.bpm - bpm) < 0.4, `bpm ${r.bpm}`);
    const bar = (60 / bpm) * 4;
    let d = Math.abs(((r.offset - offset) % bar + bar) % bar);
    d = Math.min(d, bar - d);
    assert.ok(d < 0.02, `offset ${r.offset} vs ${offset} (off by ${d})`);
  });
}

test('grid lines, steps and nearest', () => {
  const grid = { bpm: 120, offset: 0.1, beatsPerBar: 4, division: '1/8' };
  const clip = { start: 2, in: 0, rate: 1 };
  assert.ok(Math.abs(gridStep(grid, clip) - 0.25) < 1e-9);
  assert.ok(Math.abs(nearestGridLine(grid, clip, 2.31) - 2.35) < 1e-9);
  const lines = gridLines(grid, clip, 2.1, 4.2);
  assert.deepEqual(lines.map((l) => l.kind).slice(0, 5), ['bar', 'sub', 'beat', 'sub', 'beat']);
  // retimed 2x: beats twice as close
  assert.ok(Math.abs(gridStep(grid, { ...clip, rate: 2 }) - 0.125) < 1e-9);
});

/* ---- drum hits + tracked beats ------------------------------------------ */

const { detectHits, analyzeTrack, trackBeatsDP, peakPick, gridMode, activeHits, nextGridLine } =
  await import(`data:text/javascript;base64,${Buffer.from(src).toString('base64')}`);

/** Kick / snare / hat pattern at explicit times, with an optional tempo
 * ramp: beats drift from bpm0 to bpm1 across the clip. */
function drumKit({ bpm0, bpm1 = bpm0, secs = 30, sr = 44100, offset = 0.2 }) {
  const out = new Float32Array(secs * sr);
  const kicks = [], snares = [], hats = [];
  let t = offset, n = 0;
  while (t < secs - 0.2) {
    const hit = (time, kind) => {
      const k0 = Math.round(time * sr);
      if (kind === 'kick') {
        for (let i = 0; i < 0.15 * sr; i++) out[k0 + i] += 0.9 * Math.sin(2 * Math.PI * (50 + 80 * Math.exp(-i / 700)) * i / sr) * Math.exp(-i / 3000);
      } else if (kind === 'snare') {
        let seed = k0 * 31 + 7;
        for (let i = 0; i < 0.12 * sr; i++) {
          seed = (seed * 16807) % 2147483647;
          out[k0 + i] += (0.35 * ((seed / 2147483647) * 2 - 1) + 0.4 * Math.sin(2 * Math.PI * 190 * i / sr)) * Math.exp(-i / 1800);
        }
      } else {
        let seed = k0 * 13 + 3;
        for (let i = 0; i < 0.03 * sr; i++) {
          seed = (seed * 16807) % 2147483647;
          // bandpass-ish white noise: difference of successive samples pushes it up high
          const w = (seed / 2147483647) * 2 - 1;
          out[k0 + i] += 0.3 * w * Math.exp(-i / 350);
        }
      }
    };
    const beat = n % 4;
    if (beat === 0 || beat === 2) { hit(t, 'kick'); kicks.push(t); }
    if (beat === 1 || beat === 3) { hit(t, 'snare'); snares.push(t); }
    const period = 60 / (bpm0 + (bpm1 - bpm0) * (t / secs));
    hit(t + period / 2, 'hat'); hats.push(t + period / 2);
    t += period;
    n++;
  }
  return { samples: out, kicks, snares, hats, beats: [...kicks, ...snares].sort((a, b) => a - b) };
}

const nearest = (arr, t) => arr.reduce((best, v) => (Math.abs(v - t) < Math.abs(best - t) ? v : best), arr[0]);

test('detectHits finds every kick, snare and hat to within 3 ms and tags the band', () => {
  const kit = drumKit({ bpm0: 124 });
  const { hits } = detectHits(kit.samples, 44100);
  const byBand = [0, 1, 2].map((b) => hits.filter((h) => h[2] === b).map((h) => h[0]));
  // A sine kick ramps up over its first cycle, so its steepest rise is a
  // few ms in; the noise-based snare and hat are sample-sharp.
  for (const [truth, band, name, tol] of [[kit.kicks, 0, 'kick', 0.006], [kit.snares, 1, 'snare', 0.003], [kit.hats, 2, 'hat', 0.003]]) {
    for (const t of truth) {
      const got = nearest(byBand[band], t);
      assert.ok(Math.abs(got - t) < tol, `${name} at ${t.toFixed(3)} → ${got?.toFixed(3)}`);
    }
    // No spurious extra hits of that band (within a tolerant count).
    assert.ok(byBand[band].length <= truth.length + 2, `${name}: ${byBand[band].length} hits for ${truth.length} events`);
  }
  for (const h of hits) assert.ok(h[1] > 0 && h[1] <= 1);
});

test('the tracked beats follow a tempo ramp and land on the hits', () => {
  const kit = drumKit({ bpm0: 118, bpm1: 130, secs: 40 });
  const r = analyzeTrack(kit.samples, 44100);
  assert.ok(r.beats.length >= kit.beats.length * 0.9, `only ${r.beats.length} of ${kit.beats.length} beats`);
  for (const b of r.beats) {
    const t = nearest(kit.beats, b);
    assert.ok(Math.abs(t - b) < 0.006, `beat ${b.toFixed(3)} vs ${t.toFixed(3)}`);
  }
  assert.ok(r.hits.length > kit.beats.length);
  // The downbeat is a kick: the bar anchor sits on a kick, not a snare.
  const anchor = r.beats[r.beatPhase];
  assert.ok(Math.abs(nearest(kit.kicks, anchor) - anchor) < 0.004, `bar anchor ${anchor} is not on a kick`);
});

test('grid modes: hits snap to hits, beats interpolate subdivisions, steady grid is unchanged', () => {
  const clip = { start: 1, in: 0, rate: 1 };
  const grid = {
    bpm: 120, offset: 0.5, beatsPerBar: 4, division: '1/8', mode: 'hits',
    beats: [0.5, 1.0, 1.52, 2.0, 2.5], beatPhase: 1,
    hits: [[0.5, 1, 0], [0.75, 0.3, 2], [1.0, 0.8, 1], [1.52, 0.9, 0], [2.0, 0.1, 1]],
  };
  assert.equal(gridMode(grid), 'hits');
  assert.equal(activeHits(grid).length, 4);                     // the 0.1 snare is under hitMin
  assert.ok(Math.abs(nearestGridLine(grid, clip, 1.8) - 1.75) < 1e-9);
  assert.ok(Math.abs(nearestGridLine(grid, clip, 2.6) - 2.52) < 1e-9);
  assert.deepEqual(gridLines(grid, clip, 1.4, 2.1).map((l) => [l.t, l.kind, l.band]), [[1.5, 'hit', 0], [1.75, 'hit', 2], [2.0, 'hit', 1]]);
  grid.hitBand = 'kick';
  assert.deepEqual(gridLines(grid, clip, 1, 4).map((l) => l.t), [1.5, 2.52]);

  grid.mode = 'beats';
  assert.equal(gridMode(grid), 'beats');
  const lines = gridLines(grid, clip, 2.0, 3.0);
  // beats at comp 2.0, 2.52, 3.0 (phase 1 → the bar is at 1.0 comp); 1/8 halves each interval
  assert.deepEqual(lines.map((l) => [+l.t.toFixed(3), l.kind]),
    [[2, 'bar'], [2.26, 'sub'], [2.52, 'beat'], [2.76, 'sub'], [3, 'beat']]);
  assert.ok(Math.abs(nearestGridLine(grid, clip, 2.3) - 2.26) < 1e-9);
  assert.ok(Math.abs(nextGridLine(grid, clip, 2.3) - 2.52) < 1e-9);
  // Past the last tracked beat the grid carries on at the median period.
  assert.ok(Math.abs(nearestGridLine(grid, clip, 4.52) - 4.5) < 1e-6);

  grid.mode = 'grid';
  assert.ok(Math.abs(nearestGridLine(grid, clip, 2.3) - 2.25) < 1e-9);
  // An old project without hits or beats stays on the steady grid.
  assert.equal(gridMode({ bpm: 120, offset: 0, mode: 'hits' }), 'grid');
});

test('peakPick and the DP tracker behave on a clean pulse', () => {
  const env = new Float64Array(400);
  for (let i = 10; i < 400; i += 20) env[i] = 1;
  assert.deepEqual(peakPick(env, { preMax: 3, postMax: 1, preAvg: 5, postAvg: 5, delta: 0.1, wait: 3 }).slice(0, 3), [10, 30, 50]);
  const beats = trackBeatsDP(env, 20);
  assert.ok(beats.length >= 18, `${beats.length} beats`);
  for (const b of beats) assert.equal((b - 10) % 20, 0, `beat at frame ${b}`);
});
