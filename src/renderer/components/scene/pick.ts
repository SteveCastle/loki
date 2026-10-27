// Selecting objects with loki-segment (SAM 3.1, through the media server):
// the pending selection of the Select tool and of a roto, and how it becomes
// mask shapes. A pick is clicks (this / not this), a box, a name ("person"),
// or every object; the server answers with masks, and nothing changes in the
// layer until the selection is added (or cut) as labelled mask pieces, which
// then work like any other shape (expand, invert, links, fit).
//
// Picks are serialized: a click made while one is in flight replaces the
// waiting one, so a fast series of clicks costs at most two requests.
import { useCallback, useEffect, useRef, useState } from 'react';
import { pickedOp, type MaskOp, type Rect } from './mask';
import { sceneApi, type SegmentObject, type SegmentRequest } from './scene-api';

/** What is being picked on: the picture (frame pixels) or a motion version's clip frame (clip pixels). */
export type PickTarget = { kind: 'picture' } | { kind: 'clip'; layer: string; version: number; frame: number };

export interface PickState {
  target: PickTarget;
  points: Array<[number, number, number]>;
  box?: [number, number, number, number];
  text?: string;
  /** what the quick chips asked for: People, Foreground, Background */
  preset?: 'people' | 'foreground' | 'background';
  objects: SegmentObject[];
  /** which objects are in the selection (multi-object answers: names, everything) */
  chosen: boolean[];
  /** clicks/boxes: which reading of the object (part / object / whole) */
  candidate: number;
  /** what was segmented, in its own pixels */
  width: number;
  height: number;
  busy: boolean;
  error?: string;
  ms?: number;
}

export const emptyPick = (target: PickTarget): PickState => ({ target, points: [], objects: [], chosen: [], candidate: 0, width: 0, height: 0, busy: false });

/** An object's reading `candidate` (clicks and boxes have several; names one). */
export function reading(o: SegmentObject, candidate: number): SegmentObject {
  const c = o.candidates || [];
  // Readings carry no name of their own: they are the object's.
  return c.length ? { ...c[Math.min(candidate, c.length - 1)], label: o.label } : o;
}

/** What a picked object is called in the mask's shape list. */
function nameOf(o: SegmentObject): string {
  return !o.label || o.label === 'click' || o.label === 'box' ? 'object' : o.label;
}

/** The objects in the selection now (a click's chosen reading, or the chosen objects of a name / everything). */
export function selectedObjects(p: PickState): SegmentObject[] {
  if (!p.text && !p.preset && p.objects.length) return [reading(p.objects[0], p.candidate)];
  return p.objects.filter((_, i) => p.chosen[i]);
}

/**
 * Foreground, for the quick chips: the objects of an "everything" answer that are things rather than the setting
 * (sky, ground, a wall run across the frame): not too large, and not spanning two opposite edges of the frame.
 */
export function isForeground(o: SegmentObject, w: number, h: number): boolean {
  const b = o.bbox;
  const share = (b.w * b.h) / (w * h);
  const spansX = b.x <= 2 && b.x + b.w >= w - 2;
  const spansY = b.y <= 2 && b.y + b.h >= h - 2;
  return share < 0.35 && !spansX && !spansY;
}

/**
 * The selection as mask shapes in frame pixels: added (or with `cut`, cut) pieces. Background is the inverse of
 * the foreground: the whole frame added, the foreground cut. map takes rects from the segmented pixels to frame px.
 */
export function pickOps(p: PickState, cut: boolean, frameW: number, frameH: number, map: (r: Rect) => Rect = (r) => r): MaskOp[] {
  const objs = selectedObjects(p);
  if (p.preset === 'background') {
    const all: MaskOp = { t: 'rect', x: 0, y: 0, w: frameW, h: frameH, label: 'background', erase: cut || undefined };
    return [all, ...objs.map((o) => pickedOp(o.mask, map(o.rect), map(o.bbox), nameOf(o), !cut))];
  }
  return objs.map((o) => pickedOp(o.mask, map(o.rect), map(o.bbox), p.preset === 'foreground' ? 'foreground' : nameOf(o), cut));
}

/** The pending pick of one scene: ask the server, keep the answer, cycle readings, choose objects. */
export function usePick(sceneId: string, onError: (msg: string) => void) {
  const [pick, setPick] = useState<PickState | null>(null);
  const pickRef = useRef(pick);
  pickRef.current = pick;
  const inFlight = useRef(false);
  const waiting = useRef<PickState | null>(null);

  const run = useCallback(
    async (next: PickState) => {
      if (inFlight.current) {
        waiting.current = next;
        return;
      }
      inFlight.current = true;
      setPick({ ...next, busy: true, error: undefined });
      const t = next.target;
      const req: SegmentRequest = { multimask: true };
      if (t.kind === 'clip') Object.assign(req, { layer: t.layer, version: t.version, frame: t.frame });
      if (next.preset === 'people') req.text = 'person';
      else if (next.preset) req.everything = true;
      else if (next.text) req.text = next.text;
      else {
        if (next.points.length) req.points = next.points;
        if (next.box) req.box = next.box;
      }
      try {
        const res = await sceneApi.segment(sceneId, req);
        let objects = res.objects || [];
        if (next.preset === 'foreground' || next.preset === 'background') objects = objects.filter((o) => isForeground(o, res.width, res.height));
        const cur = pickRef.current;
        // Dropped (cleared, or another target) while it ran.
        if (cur && JSON.stringify(cur.target) === JSON.stringify(next.target)) {
          setPick({ ...next, objects, chosen: objects.map(() => true), candidate: 0, width: res.width, height: res.height, busy: false, ms: res.ms });
          if (!objects.length) setPick((p) => p && { ...p, error: next.text || next.preset ? 'Nothing like that found here.' : 'Nothing found there.' });
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setPick((p) => p && { ...p, busy: false, error: msg });
        onError(msg);
      } finally {
        inFlight.current = false;
        const w = waiting.current;
        waiting.current = null;
        if (w) run(w);
      }
    },
    [sceneId, onError]
  );

  /** Start (or move) a pick on target, keeping nothing of the last one. */
  const begin = useCallback((target: PickTarget) => setPick(emptyPick(target)), []);
  const clear = useCallback(() => {
    waiting.current = null;
    setPick(null);
  }, []);

  /** A click (keep = this, else not this): adds to the clicks of a click pick, starts one otherwise. */
  const click = useCallback(
    (target: PickTarget, x: number, y: number, keep: boolean) => {
      const p = pickRef.current;
      const same = p && JSON.stringify(p.target) === JSON.stringify(target) && !p.text && !p.preset;
      const points = [...(same ? p.points : []), [Math.round(x), Math.round(y), keep ? 1 : 0] as [number, number, number]];
      run({ ...emptyPick(target), points, box: same ? p.box : undefined });
    },
    [run]
  );

  const box = useCallback(
    (target: PickTarget, r: Rect) => {
      const p = pickRef.current;
      const same = p && JSON.stringify(p.target) === JSON.stringify(target) && !p.text && !p.preset;
      run({ ...emptyPick(target), points: same ? p.points : [], box: [Math.round(r.x), Math.round(r.y), Math.round(r.w), Math.round(r.h)] });
    },
    [run]
  );

  const find = useCallback((target: PickTarget, text: string) => run({ ...emptyPick(target), text: text.trim() }), [run]);
  const preset = useCallback((target: PickTarget, which: 'people' | 'foreground' | 'background') => run({ ...emptyPick(target), preset: which }), [run]);

  /** Tab: the next reading of a click / box pick. */
  const cycle = useCallback(() => {
    setPick((p) => {
      if (!p || !p.objects.length) return p;
      const n = (p.objects[0].candidates || []).length;
      return n > 1 ? { ...p, candidate: (p.candidate + 1) % n } : p;
    });
  }, []);

  const toggle = useCallback((i: number) => setPick((p) => p && { ...p, chosen: p.chosen.map((c, j) => (j === i ? !c : c)) }), []);

  // A pick on another scene never carries over.
  useEffect(() => () => clear(), [sceneId, clear]);

  return { pick, begin, clear, click, box, find, preset, cycle, toggle };
}
