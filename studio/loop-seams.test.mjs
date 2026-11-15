import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// Studio runs as browser ES modules inside a CommonJS repository.
const src = readFileSync(new URL('./loop-seams.js', import.meta.url), 'utf8');
const { scoreLoopSeams, blendPlan, blendWeight, loopChannel, seamScore, motionBase } =
  await import(`data:text/javascript;base64,${Buffer.from(src).toString('base64')}`);

/** A generated "living photo": eases in from a still over the first frames,
 * then cycles with a 30-frame period, a couple of levels darker than the
 * still (the exposure drift seen in real generations). */
function generatedClip(N = 124, period = 30) {
  const feats = [];
  let phase = 0;
  for (let f = 0; f < N; f++) {
    const speed = Math.min(1, 0.2 + f / 8);       // ease-in
    phase += speed * (2 * Math.PI / period);
    const drift = Math.min(1, f / 10) * 2.5;      // darker once it settles
    const v = new Uint8Array(32 * 18 * 3);
    for (let p = 0; p < v.length; p++) {
      const x = (p / 3) % 32;
      v[p] = Math.round(128 + 60 * Math.sin(phase + x / 5) - drift);
    }
    feats.push(v);
  }
  return feats;
}

test('the chosen seam sits inside the steady cycle, not on the eased-in head', async () => {
  const feats = generatedClip();
  const { picked, base } = await scoreLoopSeams(feats, 10);
  const best = picked[0];
  assert.ok(best.i >= 8, `seam starts in the ease-in (frame ${best.i})`);
  assert.ok(best.r < 0.5, `seam ${best.r} should be near-invisible`);
  // The head seam the old rule took (frame ~1 → end) scores far worse.
  const head = seamScore(feats, 2, feats, feats.length - 3, base);
  assert.ok(head > best.r * 2, `head seam ${head} vs ${best.r}`);
  // The loop length is a whole number of cycles (±1 frame).
  const len = best.j - best.i;
  assert.ok(Math.abs(len / 30 - Math.round(len / 30)) < 0.05, `loop of ${len} frames`);
});

test('a longer loop wins only when it is about as good', async () => {
  const feats = generatedClip(200);
  const { picked } = await scoreLoopSeams(feats, 10);
  const top = picked[0];
  for (const p of picked.slice(1)) assert.ok(p.r >= top.r * 0.5 || p.j - p.i <= top.j - top.i);
});

test('a steady shot has no motion to normalise by, but stays finite', () => {
  const still = Array.from({ length: 10 }, () => new Uint8Array(30).fill(9));
  assert.equal(motionBase(still), 30);
});

test('blend plan prefers the run-up before the start and shrinks to fit', () => {
  assert.deepEqual(blendPlan(60, 8, 30, 0), { side: 'pre', K: 8 });
  assert.deepEqual(blendPlan(60, 8, 0, 20), { side: 'post', K: 8 });
  assert.deepEqual(blendPlan(60, 8, 3, 5), { side: 'post', K: 5 });
  assert.deepEqual(blendPlan(60, 8, 1, 1), { side: null, K: 0 });
  assert.deepEqual(blendPlan(9, 8, 30, 30), { side: 'pre', K: 3 });   // ≤ a third of the loop
});

test('the last baked frame is almost the frame before the start (pre) / the first is almost the frame after the end (post)', () => {
  const n = 60;
  const pre = blendPlan(n, 8, 30, 0);
  assert.equal(blendWeight(n - 9, n, pre).w, 0);
  const last = blendWeight(n - 1, n, pre);
  assert.equal(last.other, -n);
  assert.ok(last.w > 0.95);
  for (let k = n - 8; k < n - 1; k++) assert.ok(blendWeight(k + 1, n, pre).w > blendWeight(k, n, pre).w);
  const post = blendPlan(n, 8, 0, 30);
  const first = blendWeight(0, n, post);
  assert.equal(first.other, n);
  assert.ok(first.w > 0.95);
  assert.equal(blendWeight(8, n, post).w, 0);
});

test('looped audio wraps without a jump', () => {
  // A drifting tone: a hard loop of [a, b) jumps at the wrap.
  const sr = 48000;
  const srcCh = new Float32Array(sr * 3);
  for (let i = 0; i < srcCh.length; i++) srcCh[i] = Math.sin(i * 0.0137) * (0.3 + i / srcCh.length);
  const a = sr, b = 2 * sr + 777, F = 0.3 * sr;
  const hard = loopChannel(srcCh, a, b, 0, null);
  const jumpHard = Math.abs(hard[hard.length - 1] - hard[0]);
  for (const side of ['pre', 'post']) {
    const out = loopChannel(srcCh, a, b, F, side);
    assert.equal(out.length, b - a);
    const jump = Math.abs(out[out.length - 1] - out[0]);
    assert.ok(jump < 0.02, `${side}: wrap jump ${jump} (hard cut ${jumpHard})`);
  }
  assert.ok(jumpHard > 0.1, `the test tone should need blending (${jumpHard})`);
});
