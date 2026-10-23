import { maskBounds, regionFor, rectContains, simplifyPath, maskSize, type MaskOp } from '../renderer/components/scene/mask';
import { mergeServer, sceneBusy, type Scene, type SceneLayer } from '../renderer/components/scene/scene-api';
import { canStartScene } from '../renderer/components/scene/launch';

const W = 2000;
const H = 1000;

describe('scene masks', () => {
  it('bounds cover additive ops only, strokes by their radius', () => {
    const ops: MaskOp[] = [
      { t: 'rect', x: 100, y: 100, w: 50, h: 40 },
      { t: 'stroke', pts: [300, 300, 320, 310], r: 10 },
      { t: 'rect', x: 0, y: 0, w: 1999, h: 999, erase: true },
    ];
    expect(maskBounds(ops)).toEqual({ x: 100, y: 100, w: 230, h: 220 });
    expect(maskBounds([])).toBeNull();
    expect(maskBounds([{ t: 'rect', x: 5, y: 5, w: 400, h: 400, erase: true }])).toBeNull();
  });

  it('region pads the mask by the context and keeps the feather inside the crop', () => {
    const ops: MaskOp[] = [{ t: 'rect', x: 900, y: 400, w: 200, h: 200 }];
    const tight = regionFor(ops, W, H, { context: 'tight', feather: 0 }) as NonNullable<ReturnType<typeof regionFor>>;
    const wide = regionFor(ops, W, H, { context: 'wide', feather: 0 }) as NonNullable<ReturnType<typeof regionFor>>;
    expect(rectContains(tight, { x: 900, y: 400, w: 200, h: 200 })).toBe(true);
    expect(wide.w).toBeGreaterThan(tight.w);
    // A big feather needs at least 1.5× its width of margin, whatever the context.
    const soft = regionFor(ops, W, H, { context: 'tight', feather: 100 }) as NonNullable<ReturnType<typeof regionFor>>;
    expect(soft.x).toBeLessThanOrEqual(900 - 150);
    expect(regionFor(ops, W, H, { context: 'full', feather: 10 })).toEqual({ x: 0, y: 0, w: W, h: H });
  });

  it('region stays in the frame, grows to the minimum side and a sane aspect', () => {
    const corner = regionFor([{ t: 'rect', x: 0, y: 0, w: 20, h: 20 }], W, H, { context: 'normal', feather: 4, minSide: 320 });
    expect(corner).toEqual({ x: 0, y: 0, w: 320, h: 320 });
    const sliver = regionFor([{ t: 'rect', x: 100, y: 500, w: 1600, h: 10 }], W, H, { context: 'tight', feather: 0, maxAspect: 2.5 });
    expect(sliver).not.toBeNull();
    const s = sliver as NonNullable<typeof sliver>;
    expect(s.w / s.h).toBeLessThanOrEqual(2.51);
    expect(s.x).toBeGreaterThanOrEqual(0);
    expect(s.y + s.h).toBeLessThanOrEqual(H);
  });

  it('mask raster is capped at 2048 on the long edge', () => {
    expect(maskSize(4000, 3000)).toEqual({ w: 2048, h: 1536, scale: 0.512 });
    expect(maskSize(800, 600)).toEqual({ w: 800, h: 600, scale: 1 });
  });

  it('simplifies freehand paths but keeps corners', () => {
    const line: number[] = [];
    for (let i = 0; i <= 100; i += 1) line.push(i, 0);
    line.push(100, 100);
    const out = simplifyPath(line, 0.5);
    expect(out).toEqual([0, 0, 100, 0, 100, 100]);
  });
});

function layer(id: string, extra: Partial<SceneLayer> = {}): SceneLayer {
  return { id, name: id, kind: 'still', region: { x: 0, y: 0, w: 10, h: 10 }, mask: '', feather: 4, prompt: '', versions: [], active: -1, ...extra };
}

function scene(layers: SceneLayer[], extra: Partial<Scene> = {}): Scene {
  return { id: 's', name: 's', source: 'C:\\a.png', base: 'b.png', width: 100, height: 100, layers, duration: 0, created: '', updated: '', ...extra };
}

describe('scene documents', () => {
  const v = (path: string) => ({ path, region: { x: 0, y: 0, w: 10, h: 10 }, prompt: '', created: '' });

  it('merging keeps local edits and takes the server’s generations', () => {
    const local = scene([layer('a', { prompt: 'typed since', active: 0, versions: [v('1.png')] }), layer('new')]);
    const server = scene([layer('a', { prompt: 'old', versions: [v('1.png'), v('2.png')], active: 1, job: null })], { render: { jobId: 'r', format: 'video', state: 'pending', done: 0, total: 1 } });
    const m = mergeServer(local, server);
    expect(m.layers.map((l) => l.id)).toEqual(['a', 'new']);
    expect(m.layers[0].prompt).toBe('typed since');
    expect(m.layers[0].versions).toHaveLength(2);
    expect(m.layers[0].active).toBe(1); // a new version arrived: show it
    expect(m.render?.state).toBe('pending');
    expect(sceneBusy(m)).toBe(true);
  });

  it('merging keeps a locally picked version when nothing new arrived', () => {
    const local = scene([layer('a', { active: 0, versions: [v('1.png'), v('2.png')] })]);
    const server = scene([layer('a', { active: 1, versions: [v('1.png'), v('2.png')] })]);
    expect(mergeServer(local, server).layers[0].active).toBe(0);
  });

  it('a scene starts from an image or a video frame', () => {
    expect(canStartScene('C:\\x\\photo.JPG')).toBe(true);
    expect(canStartScene('/x/clip.mp4')).toBe(true);
    expect(canStartScene('/x/song.mp3')).toBe(false);
    expect(canStartScene(undefined)).toBe(false);
  });
});

describe('scene editor additions', () => {
  // eslint-disable-next-line global-require
  const { alphaBox } = require('../renderer/components/scene/mask');
  // eslint-disable-next-line global-require
  const { loopSpeed, sceneLength } = require('../renderer/components/scene/scene-api');

  it('protect paint and erasing never grow the region; image pieces count by what they cover', () => {
    const ops: MaskOp[] = [
      { t: 'rect', x: 100, y: 100, w: 50, h: 50 },
      { t: 'stroke', pts: [900, 900, 950, 950], r: 20, keep: true },
      { t: 'image', src: 'data:x', x: 0, y: 0, w: 2000, h: 1000, bx: 400, by: 300, bw: 100, bh: 60 },
    ];
    expect(maskBounds(ops)).toEqual({ x: 100, y: 100, w: 400, h: 260 });
  });

  it('finds the covered box of an alpha buffer', () => {
    const w = 8;
    const h = 4;
    const data = new Uint8ClampedArray(w * h * 4);
    data[(1 * w + 2) * 4 + 3] = 200;
    data[(2 * w + 5) * 4 + 3] = 90;
    data[(3 * w + 7) * 4 + 3] = 10; // below threshold
    expect(alphaBox(data, w, h, 32)).toEqual({ x: 2, y: 1, w: 4, h: 2 });
    expect(alphaBox(new Uint8ClampedArray(w * h * 4), w, h, 32)).toBeNull();
  });

  it('synced loops: a whole number of loops per scene length, like the server', () => {
    expect(loopSpeed(10, 5)).toBe(1);
    expect(loopSpeed(12, 5)).toBeCloseTo(1.2);
    expect(loopSpeed(10, 4.8)).toBeCloseTo(10 / 9.6);
    expect(loopSpeed(0, 5)).toBe(1);
    const v = (duration: number) => ({ path: 'p', region: { x: 0, y: 0, w: 1, h: 1 }, prompt: '', created: '', duration });
    const s = {
      id: 's', name: 's', source: 's', base: 'b', width: 1, height: 1, duration: 0, created: '', updated: '',
      layers: [
        { id: 'a', name: 'a', kind: 'motion', region: { x: 0, y: 0, w: 1, h: 1 }, mask: '', feather: 0, prompt: '', versions: [v(5.2)], active: 0 },
        { id: 'b', name: 'b', kind: 'motion', hidden: true, region: { x: 0, y: 0, w: 1, h: 1 }, mask: '', feather: 0, prompt: '', versions: [v(9)], active: 0 },
      ],
    };
    expect(sceneLength(s)).toBe(5.2); // hidden layers don't count
    expect(sceneLength({ ...s, duration: 8 })).toBe(8);
    expect(sceneLength({ ...s, layers: [] })).toBe(5);
  });
});

describe('scene layer timing', () => {
  // eslint-disable-next-line global-require
  const { layerTiming, layerRate, sceneLength } = require('../renderer/components/scene/scene-api');
  const v = (duration: number) => ({ path: 'p', region: { x: 0, y: 0, w: 1, h: 1 }, prompt: '', created: '', duration });
  const motion = (id: string, d: number, options: Record<string, unknown>) => ({
    id, name: id, kind: 'motion', region: { x: 0, y: 0, w: 1, h: 1 }, mask: '', feather: 0, prompt: '', versions: [v(d)], active: 0, options,
  });
  const scene = (layers: unknown[], extra = {}) => ({ id: 's', name: 's', source: 's', base: 'b', width: 1, height: 1, duration: 0, created: '', updated: '', layers, ...extra });

  it('applies defaults and limits like the server', () => {
    expect(layerTiming(undefined)).toEqual({ speed: 1, offset: 0, volume: 1 });
    expect(layerTiming({ speed: 9, offset: -1, volume: 3 })).toEqual({ speed: 4, offset: 0, volume: 2 });
    expect(layerTiming({ volume: 0 }).volume).toBe(0);
  });

  it('own speed first, then fitted to the scene (same cases as TestSceneTimingInExportAndLength)', () => {
    const s = scene([motion('a', 6, { speed: 2 }), motion('b', 2, { speed: 0.5 })]);
    expect(sceneLength(s)).toBe(4); // b: 2 s at half speed
    expect(layerRate(s, s.layers[0])).toBeCloseTo(0.5 * (4 / 3)); // 3 s loops once in 4 s: slowed to fit
    expect(layerRate(s, s.layers[1])).toBeCloseTo(2);
    const free = scene(s.layers, { freeLoops: true });
    expect(layerRate(free, free.layers[0])).toBe(0.5);
  });
});

describe('scene dialogue', () => {
  // eslint-disable-next-line global-require
  const { dialogueSeconds } = require('../renderer/components/scene/scene-api');
  it('estimates speech length from words (characters for unspaced languages)', () => {
    expect(dialogueSeconds([])).toBe(0);
    expect(dialogueSeconds([{ text: '   ' }])).toBe(0);
    // 10 words at 2.5/s + 0.6 s lead-in
    expect(dialogueSeconds([{ text: 'one two three four five six seven eight nine ten' }])).toBeCloseTo(4.6);
    // two lines add a pause between them
    expect(dialogueSeconds([{ text: 'a b c d e' }, { text: 'f g h i j' }])).toBeCloseTo(2 + 2 + 0.4 + 0.6);
    // 12 Japanese characters at ~6/s
    expect(dialogueSeconds([{ text: 'こんにちは、元気ですか？今日', lang: 'Japanese' }])).toBeCloseTo(14 / 6 + 0.6);
  });
});

describe('scene job phases', () => {
  // eslint-disable-next-line global-require
  const { jobPhase } = require('../renderer/components/scene/scene-api');
  const job = (p: Record<string, unknown>) => ({ id: 'j', state: 'in_progress', done: 0, total: 0, region: { x: 0, y: 0, w: 1, h: 1 }, prompt: '', ...p });
  it('names the engine and the finish step differently', () => {
    expect(jobPhase(job({ state: 'pending', step: 'reshoot' }))).toEqual({ label: 'Queued', percent: null });
    expect(jobPhase(job({ step: 'reshoot', done: 4, total: 10 }))).toEqual({ label: 'Generating', percent: 40 });
    expect(jobPhase(job({ state: 'pending', step: 'scene-lock' }))).toEqual({ label: 'Finishing', percent: null });
    expect(jobPhase(job({ step: 'scene-lock', done: 1, total: 2 }))).toEqual({ label: 'Finishing', percent: 50 });
    expect(jobPhase(job({ state: 'paused', done: 3, total: 10 }))).toEqual({ label: 'Paused', percent: 30 });
  });
});

describe('scene references', () => {
  // eslint-disable-next-line global-require
  const { refTokens, remapRefTokens, tokenIssues, hasVoice } = require('../renderer/components/scene/refs');
  // eslint-disable-next-line global-require
  const { pickable } = require('../renderer/components/scene/pick-bridge');
  // eslint-disable-next-line global-require
  const { refKind } = require('../renderer/components/scene/refs');
  const r = (id: string, path: string) => ({ id, path });
  const refs = [r('a', 'C:/x/jacket.png'), r('b', 'C:/x/walk.mp4'), r('c', 'C:/x/rain.mp3'), r('d', 'C:/x/hat.jpg')];

  it('numbers references like the server (TestSceneRefTokensAndChecks)', () => {
    expect(refTokens('motion', refs, false)).toEqual(['<Picture 2>', '<Video 1>', '<Audio 1>', '<Picture 3>']);
    expect(refTokens('motion', refs, true)[2]).toBe('<Audio 2>');
    expect(refTokens('still', refs, false)).toEqual(['<image2>', null, null, '<image3>']);
  });

  it('keeps the prompt pointing at the same files when references move or go', () => {
    const prompt = 'put the hat from <Picture 3> on the man, he wears <Picture 2>';
    const swapped = [refs[3], refs[1], refs[2], refs[0]];
    expect(remapRefTokens(prompt, 'motion', refs, swapped, false)).toBe('put the hat from <Picture 2> on the man, he wears <Picture 3>');
    // Removing the jacket renumbers the hat to <Picture 2>; the jacket's own token must not silently become the hat.
    const removed = refs.filter((x) => x.id !== 'a');
    const after = remapRefTokens(prompt, 'motion', refs, removed, false);
    expect(after).toBe('put the hat from <Picture 2> on the man, he wears <removed: jacket.png>');
    expect(tokenIssues(after, 'motion', removed, false)).toEqual(['<removed: jacket.png>']);
    const win = [{ id: 'w', path: 'C:\\lib\\coats\\parka.jpg' }];
    expect(remapRefTokens('wear <image2>', 'still', win, [], false)).toBe('wear <removed: parka.jpg>');
    expect(remapRefTokens('wear <image2>', 'still', [{ ...win[0], note: 'the parka' }], [], false)).toBe('wear <removed: the parka>');
    expect(tokenIssues('use <image4> and <image2>', 'still', refs, false)).toEqual(['<image4>']);
    expect(tokenIssues('<Picture 1> with <Audio 1>', 'motion', [], false)).toEqual(['<Audio 1>']);
    expect(tokenIssues('<Picture 1> with <Audio 1>', 'motion', [], true)).toEqual([]);
  });

  it('only counts the dialogue voice when there is dialogue', () => {
    expect(hasVoice({ voiceRef: 'v.wav' })).toBe(false);
    expect(hasVoice({ voiceRef: 'v.wav', dialogue: [{ text: 'hi' }] })).toBe(true);
  });

  it('a pick keeps what the layer can use, in order, once', () => {
    const req = { id: 'p', kinds: ['image'], label: 'x' };
    expect(pickable(['a.png', 'b.mp4', 'a.png', 'c.jpg', ''], req, refKind)).toEqual(['a.png', 'c.jpg']);
    expect(pickable(['a.png'], null, refKind)).toEqual([]);
  });
});
