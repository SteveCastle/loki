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
