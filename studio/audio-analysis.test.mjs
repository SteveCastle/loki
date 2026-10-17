import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const src = readFileSync(new URL('./audio-analysis.js', import.meta.url), 'utf8');
const { sampleLevel } = await import(`data:text/javascript;base64,${Buffer.from(src).toString('base64')}`);

test('a release tail decays through silence instead of cutting off', () => {
  const fps = 30;
  const env = new Float32Array(60);
  env[10] = 1;   // one hit, then silence
  const release = 0.25;
  for (let i = 10; i < 40; i++) {
    const want = Math.exp(-((i - 10) / fps) / release);
    const got = sampleLevel(env, fps, i / fps, release);
    if (want >= 0.01) assert.ok(Math.abs(got - want) < 1e-6, `frame ${i}: ${got} vs ${want}`);
    else assert.ok(got < 0.011, `frame ${i}: tail should be gone, got ${got}`);
  }
  assert.equal(sampleLevel(env, fps, 5 / fps, release), 0);    // nothing before the hit
  assert.equal(sampleLevel(env, fps, 10 / fps, 0), 1);         // no release: the raw envelope
  assert.equal(sampleLevel(env, fps, 11 / fps, 0), 0);
});

test('the louder of the current level and an earlier tail wins', () => {
  const fps = 30;
  const env = new Float32Array(30);
  env[0] = 1;
  env[3] = 0.2;
  const tail = Math.exp(-(3 / fps) / 0.25);
  assert.ok(Math.abs(sampleLevel(env, fps, 3 / fps, 0.25) - tail) < 1e-6);
});
