import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// Studio runs as browser ES modules inside a CommonJS repository.
const model = readFileSync(new URL('./comp.js', import.meta.url), 'utf8');
const {
  srcTime, clipTimeRemap, enableTimeRemap, disableTimeRemap, freezeClipAt,
  clipSpeedAt, clipPlayingBackward, splitClip, retimeClip, upsertKey, evalProp,
  clipRate, clipReversed, clipLoopSpan, TIME_PROP,
} = await import(`data:text/javascript;base64,${Buffer.from(model).toString('base64')}`);

const FPS = 30;
const makeClip = (extra = {}) => ({
  id: 'c1', kind: 'media', in: 0, start: 2, dur: 4, rate: 1, props: {}, effects: [], ...extra,
});
const near = (actual, expected, eps = 1e-6) => assert.ok(Math.abs(actual - expected) < eps,
  `${actual} should equal ${expected}`);
/** Compare the mapping at every comp frame of the clip against a reference. */
const framesMatch = (clip, ref) => {
  const n = Math.round(clip.dur * FPS);
  for (let i = 0; i < n; i++) {
    const t = clip.start + i / FPS;
    near(srcTime(clip, t), ref(t), 1e-6);
  }
};

test('enabling remap is picture-identical for forward, retimed and reversed clips', () => {
  for (const extra of [{}, { rate: 2, in: 1.5 }, { rate: 0.5, reversed: true, in: 0.25 }]) {
    const plain = makeClip(extra);
    const clip = makeClip(extra);
    const prop = enableTimeRemap(clip, FPS);
    assert.equal(clipTimeRemap(clip), prop);
    assert.equal(prop.keys.length, 2);
    assert.equal(prop.ease, 'linear');
    framesMatch(clip, (t) => srcTime(plain, t));
    // The plain fields are neutral now — the keys carry the timing.
    assert.equal(clipRate(clip), 1);
    assert.equal(clipReversed(clip), false);
  }
});

test('enabling remap bakes a loop region and a ping-pong loop into keys', () => {
  for (const extra of [{ loopSpan: 1.1, in: 0.3 }, { loopSpan: 0.7, loopMode: 'pingpong', rate: 1.5 },
    { loopSpan: 0.9, reversed: true }]) {
    const plain = makeClip(extra);
    const clip = makeClip(extra);
    enableTimeRemap(clip, FPS);
    framesMatch(clip, (t) => srcTime(plain, t));
    assert.equal(clipLoopSpan(clip), 0);
    assert.equal(clip.loopMode, undefined);
  }
});

test('enabling twice keeps the existing track', () => {
  const clip = makeClip();
  const a = enableTimeRemap(clip, FPS);
  upsertKey(a, 1, 3.3);
  const b = enableTimeRemap(clip, FPS);
  assert.equal(a, b);
  assert.equal(b.keys.length, 3);
});

test('dragging a key pins that source frame to a new comp time (compress then extend)', () => {
  const clip = makeClip({ dur: 4 });
  const prop = enableTimeRemap(clip, FPS);
  // Make source second 2 land at clip time 1 instead of 2: the first
  // second of the clip plays at 2×, the remaining three at 2/3×.
  upsertKey(prop, 2, 2);
  prop.keys.find((k) => k.t === 2).t = 1;
  near(srcTime(clip, clip.start + 1), 2);
  near(srcTime(clip, clip.start + 0.5), 1);
  near(srcTime(clip, clip.start + 2.5), 3);
  near(clipSpeedAt(clip, clip.start + 0.25), 2, 1e-3);
  near(clipSpeedAt(clip, clip.start + 2), 2 / 3, 1e-3);
  assert.equal(clipPlayingBackward(clip, clip.start + 2), false);
});

test('hold keys freeze, descending keys play backward', () => {
  const clip = makeClip();
  const prop = enableTimeRemap(clip, FPS);
  upsertKey(prop, 1, 1, 'hold');
  upsertKey(prop, 2, 1);
  upsertKey(prop, 4, 0);   // replaces the baked end key: 2 → 4 runs down to 0
  near(srcTime(clip, clip.start + 1.5), 1);
  near(clipSpeedAt(clip, clip.start + 1.5), 0);
  near(srcTime(clip, clip.start + 3), 0.5);
  assert.equal(clipPlayingBackward(clip, clip.start + 3), true);
  near(clipSpeedAt(clip, clip.start + 3), -0.5, 1e-3);
});

test('a static time track is a freeze frame', () => {
  const clip = makeClip({ rate: 2, in: 0.5 });
  const prop = freezeClipAt(clip, clip.start + 1, FPS);
  assert.equal(prop.anim, false);
  near(prop.v, 2.5);
  framesMatch(clip, () => 2.5);
  near(clipSpeedAt(clip, clip.start + 2), 0);
});

test('splitting pins the cut and both halves keep the original mapping', () => {
  const clip = makeClip({ rate: 2 });
  const plain = makeClip({ rate: 2 });
  const prop = enableTimeRemap(clip, FPS);
  upsertKey(prop, 1, 1, 'hold');
  upsertKey(prop, 3, 1);
  const ref = structuredClone(prop);
  const right = splitClip(clip, clip.start + 2);
  assert.ok(right);
  // The cut landed inside the hold, so the new key holds too.
  const cutKey = prop.keys.find((k) => Math.abs(k.t - 2) < 1e-6);
  assert.equal(cutKey.e, 'hold');
  const n = Math.round(4 * FPS);
  for (let i = 0; i < n; i++) {
    const t = plain.start + i / FPS;
    const expected = evalProp(ref, t - plain.start);
    const c = t < right.start ? clip : right;
    near(srcTime(c, t), expected);
  }
  assert.equal(clip.in, 0);   // `in` is not touched under a remap
});

test('retiming stretches the time curve even with animation scaling off', () => {
  const clip = makeClip({ dur: 4 });
  const prop = enableTimeRemap(clip, FPS);
  upsertKey(prop, 1, 3);
  retimeClip(clip, 8, { scaleAnim: false });
  near(prop.keys[1].t, 2);
  near(prop.keys[2].t, 8);
  near(srcTime(clip, clip.start + 2), 3);
  assert.equal(clipRate(clip), 1);
});

test('disabling fits a constant speed through the first and last keys', () => {
  const clip = makeClip({ dur: 4 });
  const prop = enableTimeRemap(clip, FPS);
  upsertKey(prop, 0, 1);
  upsertKey(prop, 4, 3);
  disableTimeRemap(clip);
  assert.equal(clipTimeRemap(clip), null);
  assert.equal(clip.props[TIME_PROP], undefined);
  near(clipRate(clip), 0.5);
  near(clip.in, 1);
  near(srcTime(clip, clip.start + 2), 2);

  const rev = makeClip({ dur: 2 });
  const p2 = enableTimeRemap(rev, FPS);
  upsertKey(p2, 0, 5);
  upsertKey(p2, 2, 1);
  disableTimeRemap(rev);
  assert.equal(clipReversed(rev), true);
  near(clipRate(rev), 2);
  near(srcTime(rev, rev.start), 5);
  near(srcTime(rev, rev.start + 2), 1);

  const frozen = makeClip();
  freezeClipAt(frozen, frozen.start + 1, FPS);
  disableTimeRemap(frozen);
  near(clip.rate, 0.5);
  near(frozen.in, 1);
  near(clipRate(frozen), 1);
});

test('the remap survives JSON round-trips and reads as the mapping afterwards', () => {
  const clip = makeClip();
  const prop = enableTimeRemap(clip, FPS);
  upsertKey(prop, 2, 0.5);
  const copy = JSON.parse(JSON.stringify(clip));
  framesMatch(copy, (t) => srcTime(clip, t));
  assert.equal(copy.props.time.ease, 'linear');
  upsertKey(copy.props.time, 3, 1);
  assert.equal(copy.props.time.keys.find((k) => k.t === 3).e, 'linear');
});
