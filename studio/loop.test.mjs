import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// Studio runs as browser ES modules inside a CommonJS repository.
const model = readFileSync(new URL('./comp.js', import.meta.url), 'utf8');
const { srcTime, clipPlayingBackward, clipPingPong, retimeClip } =
  await import(`data:text/javascript;base64,${Buffer.from(model).toString('base64')}`);
const makeClip = (extra = {}) => ({
  kind: 'media', in: 0, start: 3, dur: 16, rate: 1,
  loopSpan: 4, loopMode: 'pingpong', props: {}, effects: [], ...extra,
});
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-8,
  `${actual} should equal ${expected}`);

test('whole source repeats forward and backward with continuous turnarounds', () => {
  const clip = makeClip();
  for (const [elapsed, source] of [[0, 0], [2, 2], [4, 4], [6, 2], [8, 0], [10, 2], [16, 0]])
    near(srcTime(clip, clip.start + elapsed), source);
  for (const elapsed of [4, 8, 12]) {
    near(srcTime(clip, clip.start + elapsed - 1e-9), srcTime(clip, clip.start + elapsed + 1e-9));
  }
  assert.equal(clipPlayingBackward(clip, 3), false);
  assert.equal(clipPlayingBackward(clip, 7), true);
  assert.equal(clipPlayingBackward(clip, 11), false);
});

test('retiming preserves source positions across both legs', () => {
  const clip = makeClip();
  const positions = [0, 2, 4, 6, 8, 12].map(t => srcTime(clip, clip.start + t));
  retimeClip(clip, 8);
  [0, 2, 4, 6, 8, 12].forEach((t, i) => near(srcTime(clip, clip.start + t / 2), positions[i]));
});

test('reversed partial cycles use the correct direction at each turnaround', () => {
  const clip = makeClip({ reversed: true, dur: 10 });
  for (const [elapsed, source, backward] of [[0, 2, true], [2, 0, false], [4, 2, false], [6, 4, true], [8, 2, true]]) {
    near(srcTime(clip, clip.start + elapsed), source);
    assert.equal(clipPlayingBackward(clip, clip.start + elapsed), backward);
  }
});

test('saved mode survives JSON and clearing it restores ordinary looping', () => {
  const clip = JSON.parse(JSON.stringify(makeClip()));
  assert.equal(clipPingPong(clip), true);
  near(srcTime(clip, 9), 2);
  delete clip.loopMode;
  near(srcTime(clip, 10), 3);
  delete clip.loopSpan;
  near(srcTime(clip, 10), 7);
  assert.equal(clipPingPong(makeClip({ loopSpan: 0 })), false);
});

test('existing seamless regions and reversed clips retain their mapping', () => {
  near(srcTime(makeClip({ loopMode: undefined, in: 1 }), 10), 4);
  near(srcTime(makeClip({ loopMode: undefined, loopSpan: undefined, reversed: true }), 5), 14);
});
