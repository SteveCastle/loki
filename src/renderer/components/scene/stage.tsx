import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../platform';
import { mapOp, moveVertex, opBox, pointInRect, simplifyPath, type MaskOp, type Rect } from './mask';
import { jobPhase, layerRate, layerTiming, shownStitch, type Scene, type SceneLayer, type SceneVersion } from './scene-api';
import { selectedObjects, type PickState } from './pick';
import { POSE_COLORS, POSE_JOINTS, POSE_LIMBS, POSE_ROOT, poseHit, posed, restPose, type PoseDrag, type PoseOp } from './pose';

// The canvas of the Scene editor: the base frame with every generated layer
// composited live the way the export will be (each result placed over the
// region it was made for, seen through its feathered mask: CSS masks over the
// same PNG the server composites with), plus the drawing tools for the
// selected layer's mask, the Shape tool that moves, scales and reshapes
// one of its shapes, and the Pose tool that places and poses a mannequin on a
// paint layer (pose.ts). Coordinates handed out are base-frame pixels.

export type Tool = 'select' | 'pick' | 'box' | 'brush' | 'lasso' | 'pen' | 'erase' | 'protect' | 'shape' | 'pose' | 'hand';

/** A Pose tool drag: joint j of pose op i (-1 = a new figure, dragged out from (x0, y0)) as the drag started. */
type PoseDragState = { i: number; j: number; mode: PoseDrag; op: PoseOp; x0: number; y0: number };

/**
 * Whether a key event goes to a field the user types in: text inputs, text areas, selects, editable content. Sliders,
 * colour pickers, checkboxes and buttons are not, so the tool shortcuts keep working after using them.
 */
export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const el = target.closest('input, textarea, select, [contenteditable]');
  if (!el) return false;
  return !(el instanceof HTMLInputElement) || !['range', 'color', 'checkbox', 'radio', 'button', 'submit', 'reset', 'file'].includes(el.type);
}

/**
 * A roto in progress: the motion layer's version shown paused on one frame of its clip (clip time = (frame + 0.5) /
 * fps), with the tracked matte of that frame tinted over it when there is one.
 */
export interface RotoView {
  layerId: string;
  version: SceneVersion;
  frame: number;
  fps: number;
  /** The matte PNG of this frame (a gray image: white = kept), or '' */
  matte: string;
}

/** How the selected layer's mask is shown: tinted over the picture, as a black-and-white matte, or not at all. */
export type MaskView = 'tint' | 'matte' | 'off';

/** A Shape tool drag: moving the shape, scaling it from a corner (0 nw, 1 ne, 2 se, 3 sw), or moving one vertex. */
type ShapeDrag = { mode: 'move' | 'scale' | 'vertex'; i: number; k: number; x0: number; y0: number; op: MaskOp; box: Rect };

interface View {
  zoom: number;
  x: number;
  y: number;
}

type Draft =
  | { t: 'rect'; x0: number; y0: number; x1: number; y1: number; erase: boolean; keep?: boolean }
  | { t: 'lasso' | 'stroke'; pts: number[]; erase: boolean; keep?: boolean }
  | { t: 'pen'; pts: number[]; erase: boolean; keep?: undefined }
  | { t: 'pick'; x0: number; y0: number; x1: number; y1: number; erase: boolean; keep?: undefined }
  | null;

/**
 * One clock for every motion layer, so the preview plays the way the export renders: with synced loops each layer
 * runs at 1/loopSpeed and is held to its place in the scene's loop.
 */
interface SceneClock {
  t0: number;
  pausedAt: number | null;
}

function clockSeconds(c: SceneClock): number {
  return ((c.pausedAt ?? performance.now()) - c.t0) / 1000;
}

export interface StageProps {
  scene: Scene;
  selected: SceneLayer | null;
  tool: Tool;
  brush: number;
  /** Brush hardness 0-1 (1 = hard edge). */
  hardness: number;
  /** The paint colour while a paint layer is selected (strokes are drawn in it), else null. */
  ink: string | null;
  maskView: MaskView;
  /** Index of the selected layer's shape being edited (Shape tool), or null. */
  selectedShape: number | null;
  onSelectShape: (i: number | null) => void;
  /** Replace shape i of the selected layer (a Shape tool edit). */
  onReplaceOp: (i: number, op: MaskOp) => void;
  /** The pending Select-tool pick, and where its pixels sit in the frame (a clip's region; null = frame pixels). */
  pick: PickState | null;
  pickArea: Rect | null;
  /** Mannequin tool, Ctrl+drag: a box round a character to read its pose from (frame pixels). */
  onPoseBox: (r: Rect) => void;
  /** Select tool: a click (keep = this, else not this) or a dragged box, in frame pixels. */
  onPickClick: (x: number, y: number, keep: boolean) => void;
  onPickBox: (r: Rect) => void;
  roto: RotoView | null;
  /** Hide every layer (hold-to-compare with the base). */
  compare: boolean;
  /** A stitched layer whose seam is drawn over the picture (the panel's Show seam). */
  seamOf: SceneLayer | null;
  playing: boolean;
  soundOn: boolean;
  /** Bumped to restart every motion layer from its first frame together. */
  syncKey: number;
  fitKey: number;
  /** The selected layer's Protect paint (a full-frame alpha PNG), tinted on its own. */
  protectMask: string;
  /** Where a version moves (alpha PNG over `region`), tinted while shown. */
  motionOverlay: { dataUrl: string; region: Rect } | null;
  onCommit: (op: MaskOp) => void;
  onSelect: (id: string | null) => void;
}

const pct = (v: number, of: number) => `${(v / of) * 100}%`;

function regionStyle(r: Rect, s: Scene): React.CSSProperties {
  return { left: pct(r.x, s.width), top: pct(r.y, s.height), width: pct(r.w, s.width), height: pct(r.h, s.height) };
}

function maskStyle(mask: string): React.CSSProperties {
  const url = `url("${mask}")`;
  return {
    maskImage: url,
    WebkitMaskImage: url,
    maskSize: '100% 100%',
    WebkitMaskSize: '100% 100%',
    maskRepeat: 'no-repeat',
    WebkitMaskRepeat: 'no-repeat',
  } as React.CSSProperties;
}

interface MotionMediaProps {
  src: string;
  playing: boolean;
  muted: boolean;
  syncKey: number;
  style: React.CSSProperties;
  /** PTS multiplier it plays at (layerRate: 2 = half speed), its loop offset (scene seconds) and the scene clock. */
  rate: number;
  offset: number;
  clock: React.MutableRefObject<SceneClock>;
  /** 0-1 (the browser can't preview gain above 100%; the export applies it). */
  volume: number;
}

function MotionMedia({ src, playing, muted, syncKey, style, rate, offset, clock: clockRef, volume }: MotionMediaProps) {
  const ref = useRef<HTMLVideoElement>(null);
  const speed = rate > 0 ? rate : 1;
  useEffect(() => {
    const v = ref.current;
    if (v) v.playbackRate = 1 / speed;
  }, [speed, src]);
  useEffect(() => {
    const v = ref.current;
    if (v) v.volume = Math.min(1, Math.max(0, volume));
  }, [volume, src]);
  // Hold the clip to its place in the scene: scene time t shows clip time (t + offset) / rate, looping. It
  // drifts, and seeks/restarts/offset changes break step, so it is checked a few times a second.
  useEffect(() => {
    if (!playing) return undefined;
    const t = window.setInterval(() => {
      const v = ref.current;
      if (!v || !(v.duration > 0) || v.seeking) return;
      const want = ((clockSeconds(clockRef.current) + offset) / speed) % v.duration;
      let drift = v.currentTime - want;
      if (drift > v.duration / 2) drift -= v.duration;
      if (drift < -v.duration / 2) drift += v.duration;
      if (Math.abs(drift) > 0.12) v.currentTime = want;
    }, 250);
    return () => window.clearInterval(t);
  }, [clockRef, offset, speed, playing]);
  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    if (playing) v.play().catch(() => undefined);
    else v.pause();
  }, [playing, src]);
  useEffect(() => {
    const v = ref.current;
    if (v && syncKey) {
      v.currentTime = 0;
      if (playing) v.play().catch(() => undefined);
    }
  }, [syncKey]);
  return (
    <video
      ref={ref}
      className="sc-media"
      src={src}
      style={style}
      muted={muted}
      loop
      playsInline
      autoPlay={playing}
      preload="auto"
      onLoadedMetadata={(e) => {
        e.currentTarget.playbackRate = 1 / speed;
      }}
    />
  );
}

export default function Stage(props: StageProps) {
  const { scene, selected, tool, brush, hardness, ink, maskView, selectedShape, compare, playing, soundOn, syncKey, fitKey, protectMask, motionOverlay, onCommit, onSelect } = props;
  const { pick, pickArea, roto, seamOf } = props;
  const clock = useRef<SceneClock>({ t0: performance.now(), pausedAt: null });
  useEffect(() => {
    const c = clock.current;
    const now = performance.now();
    if (!playing && c.pausedAt === null) c.pausedAt = now;
    if (playing && c.pausedAt !== null) {
      c.t0 += now - c.pausedAt;
      c.pausedAt = null;
    }
  }, [playing]);
  useEffect(() => {
    if (!syncKey) return;
    clock.current = { t0: performance.now(), pausedAt: playing ? null : performance.now() };
    // only an explicit restart resets the clock
  }, [syncKey]);
  const wrapRef = useRef<HTMLDivElement>(null);
  const draftRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<View | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [spaceDown, setSpaceDown] = useState(false);
  const drag = useRef<
    | { kind: 'pan'; sx: number; sy: number; vx: number; vy: number }
    | { kind: 'draw' }
    | { kind: 'posebox' }
    | ({ kind: 'shape' } & ShapeDrag)
    | ({ kind: 'pose' } & PoseDragState)
    | null
  >(null);
  const draft = useRef<Draft>(null);
  const cursorRef = useRef<{ x: number; y: number } | null>(null);
  // The shape being edited as it is dragged (committed on release).
  const [shapeLive, setShapeLive] = useState<{ i: number; op: MaskOp } | null>(null);
  // The figure being posed (i = -1: a new one), drawn live over the layer's last raster until release.
  const [poseLive, setPoseLive] = useState<{ i: number; op: PoseOp } | null>(null);

  // Container size.
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return undefined;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  const fit = useCallback((): View | null => {
    if (!size.w || !size.h) return null;
    const zoom = Math.min((size.w - 48) / scene.width, (size.h - 48) / scene.height);
    return { zoom, x: (size.w - scene.width * zoom) / 2, y: (size.h - scene.height * zoom) / 2 };
  }, [size, scene.width, scene.height]);

  // Fit on first layout and on request (F); keep the picture centred on resize until the user zooms.
  const userZoomed = useRef(false);
  useEffect(() => {
    if (!userZoomed.current || !view) setView(fit());
  }, [fit]);
  useEffect(() => {
    if (!fitKey) return;
    userZoomed.current = false;
    setView(fit());
  }, [fitKey]);

  // Space held = pan.
  useEffect(() => {
    const typing = (e: KeyboardEvent) => isTyping(e.target);
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !typing(e)) {
        e.preventDefault();
        setSpaceDown(true);
      }
    };
    const up = (e: KeyboardEvent) => e.code === 'Space' && setSpaceDown(false);
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
    };
  }, []);

  const toImage = useCallback(
    (clientX: number, clientY: number) => {
      const el = wrapRef.current;
      if (!el || !view) return { x: 0, y: 0 };
      const r = el.getBoundingClientRect();
      return { x: (clientX - r.left - view.x) / view.zoom, y: (clientY - r.top - view.y) / view.zoom };
    },
    [view]
  );

  // Draft drawing (container pixels, device-pixel sharp).
  const paintDraft = useCallback(() => {
    const c = draftRef.current;
    if (!c || !view) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(size.w * dpr) || c.height !== Math.round(size.h * dpr)) {
      c.width = Math.round(size.w * dpr);
      c.height = Math.round(size.h * dpr);
    }
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    const d = draft.current;
    if (!d) return;
    const X = (v: number) => view.x + v * view.zoom;
    const Y = (v: number) => view.y + v * view.zoom;
    let fill = 'rgba(0, 200, 150, 0.35)';
    let line = '#00c896';
    if (ink && !d.erase) {
      fill = ink;
      line = ink;
      ctx.globalAlpha = 0.75;
    }
    if (d.erase) {
      fill = 'rgba(255, 90, 120, 0.35)';
      line = 'rgb(255, 120, 160)';
    } else if (d.keep) {
      fill = 'rgba(80, 160, 255, 0.4)';
      line = '#50a0ff';
    }
    ctx.fillStyle = fill;
    ctx.strokeStyle = line;
    ctx.lineWidth = 1.5;
    if (d.t === 'pick') {
      ctx.strokeStyle = '#2ec5ff';
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(X(Math.min(d.x0, d.x1)), Y(Math.min(d.y0, d.y1)), Math.abs(d.x1 - d.x0) * view.zoom, Math.abs(d.y1 - d.y0) * view.zoom);
      return;
    }
    if (d.t === 'rect') {
      const x = Math.min(d.x0, d.x1);
      const y = Math.min(d.y0, d.y1);
      ctx.fillRect(X(x), Y(y), Math.abs(d.x1 - d.x0) * view.zoom, Math.abs(d.y1 - d.y0) * view.zoom);
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(X(x), Y(y), Math.abs(d.x1 - d.x0) * view.zoom, Math.abs(d.y1 - d.y0) * view.zoom);
      return;
    }
    const p = d.pts;
    if (p.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(X(p[0]), Y(p[1]));
    for (let i = 2; i + 1 < p.length; i += 2) ctx.lineTo(X(p[i]), Y(p[i + 1]));
    if (d.t === 'pen') {
      // The points so far, a rubber band to the pointer, and the first point to click to close.
      const c = cursorRef.current;
      if (c) ctx.lineTo(X(c.x), Y(c.y));
      if (p.length >= 6) {
        ctx.save();
        ctx.closePath();
        ctx.fill();
        ctx.restore();
      }
      ctx.stroke();
      for (let i = 0; i + 1 < p.length; i += 2) {
        ctx.beginPath();
        ctx.arc(X(p[i]), Y(p[i + 1]), i === 0 ? 5 : 3, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? line : '#fff';
        ctx.fill();
      }
      return;
    }
    if (d.t === 'lasso') {
      ctx.closePath();
      ctx.fill();
      ctx.setLineDash([5, 4]);
      ctx.stroke();
    } else {
      ctx.strokeStyle = fill;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = brush * 2 * view.zoom;
      if (p.length === 2) ctx.lineTo(X(p[0]) + 0.01, Y(p[1]));
      ctx.stroke();
    }
  }, [view, size, brush, ink]);

  useEffect(() => paintDraft(), [paintDraft]);

  const drawTool = tool === 'box' || tool === 'brush' || tool === 'lasso' || tool === 'pen' || tool === 'erase' || tool === 'protect';
  const panning = spaceDown || tool === 'hand';

  // A pen shape in progress is dropped when the tool or layer changes.
  useEffect(() => {
    if (draft.current?.t === 'pen') {
      draft.current = null;
      paintDraft();
    }
  }, [tool, selected?.id]);

  const commitPen = useCallback(() => {
    const d = draft.current;
    if (!d || d.t !== 'pen') return;
    draft.current = null;
    paintDraft();
    if (d.pts.length >= 6) onCommit({ t: 'lasso', pts: d.pts.map((v) => Math.round(v * 10) / 10), erase: d.erase || undefined });
  }, [onCommit, paintDraft]);

  // Pen: Enter closes the shape, Backspace takes back the last point.
  useEffect(() => {
    if (tool !== 'pen') return undefined;
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target)) return;
      const d = draft.current;
      if (!d || d.t !== 'pen') return;
      if (e.key === 'Enter') {
        e.preventDefault();
        commitPen();
      } else if (e.key === 'Backspace') {
        e.preventDefault();
        d.pts.splice(-2, 2);
        if (!d.pts.length) draft.current = null;
        paintDraft();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [tool, commitPen, paintDraft]);

  const shapes = selected?.shapes || [];
  const editOp = selectedShape !== null && selectedShape < shapes.length ? shapes[selectedShape] : null;

  /** Shape tool: what is under the pointer (handles of the selected shape first, then any shape, topmost first). */
  const shapeHit = (x: number, y: number): ShapeDrag | null => {
    if (!view) return null;
    const tol = 8 / view.zoom;
    const near = (px: number, py: number) => Math.abs(px - x) <= tol && Math.abs(py - y) <= tol;
    if (editOp && selectedShape !== null) {
      const box = opBox(editOp);
      if (box) {
        if (editOp.t === 'lasso') {
          for (let k = 0; k * 2 + 1 < editOp.pts.length; k += 1) {
            if (near(editOp.pts[k * 2], editOp.pts[k * 2 + 1])) return { mode: 'vertex', i: selectedShape, k, x0: x, y0: y, op: editOp, box };
          }
        }
        const corners = [
          [box.x, box.y],
          [box.x + box.w, box.y],
          [box.x + box.w, box.y + box.h],
          [box.x, box.y + box.h],
        ];
        const c = corners.findIndex(([cx, cy]) => near(cx, cy));
        if (c >= 0) return { mode: 'scale', i: selectedShape, k: c, x0: x, y0: y, op: editOp, box };
        if (pointInRect(box, x, y)) return { mode: 'move', i: selectedShape, k: 0, x0: x, y0: y, op: editOp, box };
      }
    }
    for (let i = shapes.length - 1; i >= 0; i -= 1) {
      const box = opBox(shapes[i]);
      if (box && !shapes[i].off && pointInRect(box, x, y)) return { mode: 'move', i, k: 0, x0: x, y0: y, op: shapes[i], box };
    }
    return null;
  };

  /** The shape as a Shape tool drag to (x, y) leaves it. */
  const dragged = (d: ShapeDrag, x: number, y: number, keepAspect: boolean): MaskOp => {
    const dx = x - d.x0;
    const dy = y - d.y0;
    if (d.mode === 'vertex') return moveVertex(d.op, d.k, x, y);
    if (d.mode === 'move') return mapOp(d.op, d.box, { ...d.box, x: d.box.x + dx, y: d.box.y + dy });
    // Scale from the opposite corner.
    const b = d.box;
    const fx = d.k === 1 || d.k === 2 ? b.x : b.x + b.w;
    const fy = d.k === 2 || d.k === 3 ? b.y : b.y + b.h;
    let w = Math.max(2, Math.abs(x - fx));
    let h = Math.max(2, Math.abs(y - fy));
    if (keepAspect && b.w > 0 && b.h > 0) {
      const f = Math.max(w / b.w, h / b.h);
      w = b.w * f;
      h = b.h * f;
    }
    const nx = d.k === 1 || d.k === 2 ? fx : fx - w;
    const ny = d.k === 2 || d.k === 3 ? fy : fy - h;
    return mapOp(d.op, b, { x: nx, y: ny, w, h });
  };

  /** The figure a Pose tool drag to (x, y) leaves. A new one spans the drag from head to feet; a click makes one 45% of the frame tall. */
  const posedTo = (d: PoseDragState, x: number, y: number): PoseOp => {
    if (d.i >= 0) return posed(d.op, d.j, x, y, d.mode);
    if (Math.abs(y - d.y0) * (view?.zoom || 1) < 12) {
      const h = scene.height * 0.45;
      return restPose(d.x0, d.y0 - h / 2, h);
    }
    return restPose(d.x0, Math.min(y, d.y0), Math.abs(y - d.y0));
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (!view) return;
    (e.target as Element).setPointerCapture?.(e.pointerId);
    const p = toImage(e.clientX, e.clientY);
    if (e.button === 1 || panning) {
      drag.current = { kind: 'pan', sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y };
      return;
    }
    if (e.button !== 0) return;
    if (tool === 'select') {
      // Topmost layer whose generated region (or, before any, its mask) is under the pointer.
      const hit = [...scene.layers].reverse().find((l) => {
        if (l.hidden) return false;
        const v = l.versions[l.active];
        const r = v ? v.region : l.region;
        return r && r.w > 0 && pointInRect(r, p.x, p.y);
      });
      onSelect(hit ? hit.id : null);
      drag.current = { kind: 'pan', sx: e.clientX, sy: e.clientY, vx: view.x, vy: view.y };
      return;
    }
    if (tool === 'pick') {
      drag.current = { kind: 'draw' };
      draft.current = { t: 'pick', x0: p.x, y0: p.y, x1: p.x, y1: p.y, erase: e.altKey };
      return;
    }
    if (!selected) return;
    if (tool === 'shape') {
      const hit = shapeHit(p.x, p.y);
      props.onSelectShape(hit ? hit.i : null);
      if (hit) {
        drag.current = { kind: 'shape', ...hit };
        setShapeLive({ i: hit.i, op: hit.op });
      }
      return;
    }
    if (tool === 'pose') {
      if (selected.kind !== 'paint') return;
      // Ctrl+drag: a box round a character, to start a figure from its detected pose.
      if (e.ctrlKey || e.metaKey) {
        drag.current = { kind: 'posebox' };
        draft.current = { t: 'pick', x0: p.x, y0: p.y, x1: p.x, y1: p.y, erase: false };
        paintDraft();
        return;
      }
      // The topmost figure with a joint under the pointer: drag = swing that limb (lengths kept), Alt = move the joint
      // alone, Shift = scale the figure, the neck = move it. Anywhere else: a new figure.
      const shapes = selected.shapes || [];
      for (let i = shapes.length - 1; i >= 0; i -= 1) {
        const op = shapes[i];
        if (op.t === 'pose' && !op.off) {
          const j = poseHit(op, p.x, p.y, 10 / view.zoom);
          if (j >= 0) {
            let mode: PoseDrag = 'swing';
            if (e.altKey) mode = 'free';
            else if (e.shiftKey) mode = 'scale';
            else if (j === POSE_ROOT) mode = 'move';
            drag.current = { kind: 'pose', i, j, mode, op, x0: p.x, y0: p.y };
            setPoseLive({ i, op });
            return;
          }
        }
      }
      const d: PoseDragState = { i: -1, j: POSE_ROOT, mode: 'move', op: restPose(p.x, p.y, 1), x0: p.x, y0: p.y };
      drag.current = { kind: 'pose', ...d };
      setPoseLive({ i: -1, op: posedTo(d, p.x, p.y) });
      return;
    }
    if (!drawTool) return;
    if (tool === 'pen') {
      const d = draft.current;
      if (d && d.t === 'pen') {
        const closeTol = 9 / view.zoom;
        if (d.pts.length >= 6 && Math.abs(p.x - d.pts[0]) <= closeTol && Math.abs(p.y - d.pts[1]) <= closeTol) commitPen();
        else d.pts.push(p.x, p.y);
      } else {
        draft.current = { t: 'pen', pts: [p.x, p.y], erase: e.altKey };
      }
      paintDraft();
      return;
    }
    const keep = tool === 'protect';
    const erase = !keep && (tool === 'erase' || e.altKey);
    drag.current = { kind: 'draw' };
    if (tool === 'box') draft.current = { t: 'rect', x0: p.x, y0: p.y, x1: p.x, y1: p.y, erase };
    else draft.current = { t: tool === 'lasso' ? 'lasso' : 'stroke', pts: [p.x, p.y], erase, keep };
    paintDraft();
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const p = toImage(e.clientX, e.clientY);
    setCursor(p);
    cursorRef.current = p;
    if (draft.current?.t === 'pen') paintDraft();
    const d = drag.current;
    if (!d || !view) return;
    if (d.kind === 'shape') {
      setShapeLive({ i: d.i, op: dragged(d, p.x, p.y, e.shiftKey) });
      return;
    }
    if (d.kind === 'pose') {
      setPoseLive({ i: d.i, op: posedTo(d, p.x, p.y) });
      return;
    }
    if (d.kind === 'posebox') {
      const dr = draft.current;
      if (dr && dr.t === 'pick') {
        dr.x1 = p.x;
        dr.y1 = p.y;
        paintDraft();
      }
      return;
    }
    if (d.kind === 'pan') {
      userZoomed.current = true;
      setView({ ...view, x: d.vx + e.clientX - d.sx, y: d.vy + e.clientY - d.sy });
      return;
    }
    const dr = draft.current;
    if (!dr || dr.t === 'pen') return;
    if (dr.t === 'pick') {
      dr.x1 = p.x;
      dr.y1 = p.y;
      paintDraft();
      return;
    }
    if (dr.t === 'rect') {
      dr.x1 = p.x;
      dr.y1 = p.y;
      if (e.shiftKey) {
        const s = Math.max(Math.abs(dr.x1 - dr.x0), Math.abs(dr.y1 - dr.y0));
        dr.x1 = dr.x0 + Math.sign(dr.x1 - dr.x0 || 1) * s;
        dr.y1 = dr.y0 + Math.sign(dr.y1 - dr.y0 || 1) * s;
      }
    } else {
      const n = dr.pts.length;
      // Skip sub-pixel moves (on screen): keeps long strokes small.
      if (Math.hypot(p.x - dr.pts[n - 2], p.y - dr.pts[n - 1]) * view.zoom < 1.5) return;
      dr.pts.push(p.x, p.y);
    }
    paintDraft();
  };

  const finish = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (d?.kind === 'shape') {
      const p = toImage(e.clientX, e.clientY);
      setShapeLive(null);
      // A click without a drag only selects.
      if (Math.abs(p.x - d.x0) * (view?.zoom || 1) >= 1 || Math.abs(p.y - d.y0) * (view?.zoom || 1) >= 1) props.onReplaceOp(d.i, dragged(d, p.x, p.y, e.shiftKey));
      return;
    }
    if (d?.kind === 'posebox') {
      const dr = draft.current;
      draft.current = null;
      paintDraft();
      if (dr && dr.t === 'pick' && Math.abs(dr.x1 - dr.x0) * (view?.zoom || 1) > 5 && Math.abs(dr.y1 - dr.y0) * (view?.zoom || 1) > 5) {
        props.onPoseBox({ x: Math.min(dr.x0, dr.x1), y: Math.min(dr.y0, dr.y1), w: Math.abs(dr.x1 - dr.x0), h: Math.abs(dr.y1 - dr.y0) });
      }
      return;
    }
    if (d?.kind === 'pose') {
      const p = toImage(e.clientX, e.clientY);
      setPoseLive(null);
      const op = posedTo(d, p.x, p.y);
      if (d.i < 0) onCommit(op);
      else if (Math.hypot(p.x - d.x0, p.y - d.y0) * (view?.zoom || 1) >= 1) props.onReplaceOp(d.i, op);
      return;
    }
    // A pen shape lives across clicks.
    if (draft.current?.t === 'pen') return;
    const dr = draft.current;
    draft.current = null;
    paintDraft();
    if (!d || d.kind !== 'draw' || !dr || !view) return;
    if (dr.t === 'pick') {
      // A drag is a box round the object; a click is a click (Alt: not this).
      if (Math.abs(dr.x1 - dr.x0) * view.zoom > 5 && Math.abs(dr.y1 - dr.y0) * view.zoom > 5) {
        props.onPickBox({ x: Math.min(dr.x0, dr.x1), y: Math.min(dr.y0, dr.y1), w: Math.abs(dr.x1 - dr.x0), h: Math.abs(dr.y1 - dr.y0) });
      } else {
        props.onPickClick(dr.x0, dr.y0, !dr.erase);
      }
      return;
    }
    const clampX = (v: number) => Math.min(scene.width, Math.max(0, v));
    const clampY = (v: number) => Math.min(scene.height, Math.max(0, v));
    if (dr.t === 'rect') {
      const x = clampX(Math.min(dr.x0, dr.x1));
      const y = clampY(Math.min(dr.y0, dr.y1));
      const w = clampX(Math.max(dr.x0, dr.x1)) - x;
      const h = clampY(Math.max(dr.y0, dr.y1)) - y;
      if (w * view.zoom < 4 || h * view.zoom < 4) return;
      onCommit({ t: 'rect', x: Math.round(x), y: Math.round(y), w: Math.round(w), h: Math.round(h), erase: dr.erase || undefined });
      return;
    }
    const pts = simplifyPath(dr.pts, 0.75 / view.zoom);
    if (dr.t === 'lasso') {
      if (pts.length >= 6) onCommit({ t: 'lasso', pts, erase: dr.erase || undefined });
      return;
    }
    onCommit({ t: 'stroke', pts, r: brush, h: hardness < 1 ? hardness : undefined, erase: dr.erase || undefined, keep: dr.keep || undefined });
  };

  const onWheel = (e: React.WheelEvent) => {
    if (!view) return;
    const el = wrapRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const cx = e.clientX - r.left;
    const cy = e.clientY - r.top;
    const f = fit();
    const min = f ? f.zoom * 0.25 : 0.05;
    const zoom = Math.min(16, Math.max(min, view.zoom * Math.exp(-e.deltaY * 0.0015)));
    userZoomed.current = true;
    setView({ zoom, x: cx - ((cx - view.x) * zoom) / view.zoom, y: cy - ((cy - view.y) * zoom) / view.zoom });
  };

  const stageStyle: React.CSSProperties = view
    ? { left: view.x, top: view.y, width: scene.width * view.zoom, height: scene.height * view.zoom }
    : { visibility: 'hidden' };

  let cursorClass = 'default';
  if (panning) cursorClass = drag.current?.kind === 'pan' ? 'grabbing' : 'grab';
  else if ((drawTool && selected) || tool === 'pick') cursorClass = 'crosshair';
  else if (tool === 'shape' && selected) cursorClass = 'move';
  else if (tool === 'pose' && selected?.kind === 'paint') cursorClass = 'crosshair';
  const shown = shapeLive && shapeLive.i === selectedShape ? shapeLive.op : editOp;
  const sel = selected;
  const selVersion = sel ? sel.versions[sel.active] : undefined;

  return (
    <div
      ref={wrapRef}
      className={`sc-stage-wrap cursor-${cursorClass}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={finish}
      onPointerCancel={finish}
      onPointerLeave={() => {
        setCursor(null);
        cursorRef.current = null;
        if (draft.current?.t === 'pen') paintDraft();
      }}
      onDoubleClick={() => tool === 'pen' && commitPen()}
      onWheel={onWheel}
      onContextMenu={(e) => e.preventDefault()}
    >
      <div className="sc-stage" style={stageStyle}>
        <img className="sc-base" src={mediaUrl(scene.base)} alt="" draggable={false} />
        {!compare &&
          scene.layers.map((l) => {
            if (l.kind === 'paint') {
              // A guide shows while you work (dimmed); it is never in the picture the engines or the export see.
              if (l.hidden || !l.mask) return null;
              // Once a finish layer shows a result made from it, the guide is spent: drawn over the picture it read
              // as part of the result wherever that layer's mask is cut away. It shows again while it is selected.
              const spent = sel?.id !== l.id && scene.layers.some((x) => x.options?.paint === l.id && x.options?.finish && x.versions[x.active]);
              if (l.options?.guide && spent) return null;
              // While one of its figures is being posed the live pose is drawn on top, so its old raster fades back.
              const posing = !!poseLive && sel?.id === l.id;
              return (
                <img
                  key={l.id}
                  className={`sc-paint${l.options?.guide ? ' guide' : ''}`}
                  src={l.mask}
                  alt=""
                  draggable={false}
                  style={posing ? { opacity: 0.2 } : undefined}
                />
              );
            }
            const v = l.versions[l.active];
            if (l.hidden || !v || !l.mask || (roto && roto.layerId === l.id)) return null;
            // A tracked roto plays as the clip with its matte as alpha.
            const matted = l.kind === 'motion' && v.track?.state === 'completed' && v.track.video && l.options?.matte !== false;
            const src = mediaUrl(matted ? (v.track?.video as string) : v.path);
            const style = regionStyle(v.region, scene);
            // Stitched: the patch carries the blend in its own alpha (no mask over it).
            const stitch = shownStitch(l);
            if (stitch) {
              return (
                <div key={l.id} className="sc-layer">
                  <img className="sc-media" src={mediaUrl(stitch.path)} style={style} alt="" draggable={false} />
                </div>
              );
            }
            return (
              <div key={l.id} className="sc-layer" style={maskStyle(l.mask)}>
                {l.kind === 'motion' ? (
                  <MotionMedia
                    src={src}
                    playing={playing}
                    muted={!soundOn || !l.options?.sound}
                    syncKey={syncKey}
                    style={style}
                    rate={layerRate(scene, l)}
                    offset={layerTiming(l.options).offset}
                    clock={clock}
                    volume={layerTiming(l.options).volume}
                  />
                ) : (
                  <img className="sc-media" src={src} style={style} alt="" draggable={false} />
                )}
              </div>
            );
          })}
        {seamOf && !compare && (() => {
          const st = shownStitch(seamOf);
          const v = seamOf.versions[seamOf.active];
          return st?.seam && v ? <img className="sc-seam" src={mediaUrl(st.seam)} style={regionStyle(v.region, scene)} alt="" draggable={false} /> : null;
        })()}
        {sel && maskView === 'matte' && !compare && (
          <div className="sc-matte">{sel.mask && <div className="sc-matte-white" style={maskStyle(sel.mask)} />}</div>
        )}
        {sel && sel.kind !== 'paint' && maskView === 'tint' && sel.mask && !compare && <div className={`sc-mask-tint${sel.kind === 'motion' ? ' motion' : ''}`} style={maskStyle(sel.mask)} />}
        {sel && maskView === 'tint' && protectMask && !compare && <div className="sc-mask-tint protect" style={maskStyle(protectMask)} />}
        {motionOverlay && !compare && (
          <div className="sc-motion-tint" style={{ ...regionStyle(motionOverlay.region, scene), ...maskStyle(motionOverlay.dataUrl) }} />
        )}
        {sel && sel.kind !== 'paint' && sel.region.w > 0 && !compare && (
          <div className={`sc-region${sel.kind === 'motion' ? ' motion' : ''}`} style={regionStyle(sel.region, scene)}>
            <span className="sc-region-label">{selVersion && !sameRect(selVersion.region, sel.region) ? 'next generation sees this' : 'the engine sees this'}</span>
          </div>
        )}
        {scene.layers.map(
          (l) =>
            l.job &&
            !compare && (
              <div key={`job-${l.id}`} className="sc-job" style={regionStyle(l.job.region, scene)}>
                <span>
                  {jobPhase(l.job).label}
                  {jobPhase(l.job).percent !== null ? ` ${jobPhase(l.job).percent}%` : '…'}
                  {l.queued?.length ? ` · ${l.queued.length} queued` : ''}
                </span>
              </div>
            )
        )}
        {sel && shown && view && !compare && <ShapeOutline op={shown} scene={scene} zoom={view.zoom} handles={tool === 'shape'} />}
        {sel && sel.kind === 'paint' && view && !compare && (tool === 'pose' || poseLive) && (
          <PoseOverlay ops={sel.shapes || []} live={poseLive} scene={scene} zoom={view.zoom} />
        )}
        {roto && !compare && <RotoClip roto={roto} scene={scene} mask={scene.layers.find((l) => l.id === roto.layerId)?.mask || ''} />}
        {pick && !compare && view && <PickPreview pick={pick} area={pickArea} scene={scene} zoom={view.zoom} />}
      </div>
      <canvas ref={draftRef} className="sc-draft" style={{ width: size.w, height: size.h }} />
      {cursor && view && sel && (tool === 'brush' || tool === 'erase' || tool === 'protect') && !panning && (
        <div
          className={`sc-brush-cursor${tool === 'erase' ? ' erase' : ''}${tool === 'protect' ? ' protect' : ''}`}
          style={{
            left: view.x + cursor.x * view.zoom,
            top: view.y + cursor.y * view.zoom,
            width: brush * 2 * view.zoom,
            height: brush * 2 * view.zoom,
            ...(ink && tool === 'brush' ? { borderColor: ink } : {}),
          }}
        />
      )}
      {view && (
        <div className="sc-zoom-badge">
          {Math.round(view.zoom * 100)}%{cursor ? ` · ${Math.round(cursor.x)}, ${Math.round(cursor.y)}` : ''}
        </div>
      )}
      {!sel && scene.layers.length === 0 && (
        <div className="sc-empty-hint">
          Add an <strong>Edit</strong> or <strong>Motion</strong> layer on the right, then paint or box the part of the picture it should change.
        </div>
      )}
    </div>
  );
}

/** A roto's frame: the version's clip paused on it (through the layer's mask), and that frame's matte tinted. */
function RotoClip({ roto, scene, mask }: { roto: RotoView; scene: Scene; mask: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  const v = roto.version;
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const t = (roto.frame + 0.5) / roto.fps;
    if (Math.abs(el.currentTime - t) > 0.5 / roto.fps) el.currentTime = t;
  }, [roto.frame, roto.fps, v.path]);
  const r = regionStyle(v.region, scene);
  const lum = (url: string) =>
    ({
      maskImage: `url("${url}")`,
      WebkitMaskImage: `url("${url}")`,
      maskSize: '100% 100%',
      WebkitMaskSize: '100% 100%',
      maskMode: 'luminance',
      WebkitMaskSourceType: 'luminance',
    } as React.CSSProperties);
  return (
    <>
      <div className="sc-layer" style={mask ? maskStyle(mask) : undefined}>
        <video ref={ref} className="sc-media" src={mediaUrl(v.path)} style={r} muted playsInline preload="auto" />
      </div>
      {roto.matte && <div className="sc-roto-matte" style={{ ...r, ...lum(mediaUrl(roto.matte)) }} />}
    </>
  );
}

/** The pending pick: what it would add (tinted), its clicks (+ this, − not this) and its box. */
function PickPreview({ pick, area, scene, zoom }: { pick: PickState; area: Rect | null; scene: Scene; zoom: number }) {
  // Pick pixels -> frame pixels (a clip's pixels sit in its region).
  const sx = area && pick.width ? area.w / pick.width : 1;
  const sy = area && pick.height ? area.h / pick.height : 1;
  const map = (r: Rect): Rect => (area ? { x: area.x + r.x * sx, y: area.y + r.y * sy, w: r.w * sx, h: r.h * sy } : r);
  const pt = (x: number, y: number) => (area ? [area.x + x * (area.w / Math.max(1, pick.width || area.w)), area.y + y * (area.h / Math.max(1, pick.height || area.h))] : [x, y]);
  const objs = selectedObjects(pick);
  const dot = 7 / zoom;
  return (
    <>
      {objs.map((o) => (
        <div key={`${o.id}-${o.rect.x}-${o.rect.y}`} className={`sc-pick-tint${pick.preset === 'background' ? ' inverse' : ''}`} style={{ ...regionStyle(map(o.rect), scene), ...maskStyle(o.mask) }} />
      ))}
      <svg className="sc-pick-marks" viewBox={`0 0 ${scene.width} ${scene.height}`} preserveAspectRatio="none" aria-hidden>
        {pick.box &&
          (() => {
            const b = map({ x: pick.box[0], y: pick.box[1], w: pick.box[2], h: pick.box[3] });
            return <rect x={b.x} y={b.y} width={b.w} height={b.h} className="sc-pick-box" />;
          })()}
        {pick.points.map(([x, y, keep]) => {
          const [px, py] = pt(x, y);
          return <circle key={`${x},${y}`} cx={px} cy={py} r={dot} className={keep ? 'sc-pick-yes' : 'sc-pick-no'} />;
        })}
      </svg>
    </>
  );
}

/** The selected shape drawn over the stage: its outline, and with the Shape tool its corner and vertex handles. */
function ShapeOutline({ op, scene, zoom, handles }: { op: MaskOp; scene: Scene; zoom: number; handles: boolean }) {
  const box = opBox(op);
  if (!box) return null;
  const hs = 8 / zoom;
  const ve = { vectorEffect: 'non-scaling-stroke' } as React.SVGProps<SVGElement>;
  const pts = op.t === 'lasso' || op.t === 'stroke' ? op.pts : [];
  const pairs: Array<[number, number]> = [];
  for (let i = 0; i + 1 < pts.length; i += 2) pairs.push([pts[i], pts[i + 1]]);
  const poly = pairs.map(([x, y]) => `${x},${y}`).join(' ');
  return (
    <svg className={`sc-shape-edit${op.erase ? ' erase' : ''}${op.keep ? ' keep' : ''}`} viewBox={`0 0 ${scene.width} ${scene.height}`} preserveAspectRatio="none" aria-hidden>
      {op.t === 'lasso' && <polygon points={poly} className="sc-shape-line" {...(ve as object)} />}
      {op.t === 'stroke' && (
        <>
          <polyline points={poly} className="sc-shape-band" strokeWidth={op.r * 2} />
          <polyline points={poly} className="sc-shape-line" {...(ve as object)} />
        </>
      )}
      {(op.t === 'rect' || op.t === 'image') && <rect x={op.x} y={op.y} width={op.w} height={op.h} className="sc-shape-line" {...(ve as object)} />}
      <rect x={box.x} y={box.y} width={box.w} height={box.h} className="sc-shape-box" {...(ve as object)} />
      {handles &&
        [
          [box.x, box.y],
          [box.x + box.w, box.y],
          [box.x + box.w, box.y + box.h],
          [box.x, box.y + box.h],
        ].map(([x, y]) => <rect key={`c${x},${y}`} x={x - hs / 2} y={y - hs / 2} width={hs} height={hs} className="sc-shape-handle" {...(ve as object)} />)}
      {handles &&
        op.t === 'lasso' &&
        // eslint-disable-next-line react/no-array-index-key
        pairs.map(([x, y], k) => <circle key={k} cx={x} cy={y} r={hs / 2.4} className="sc-shape-vertex" {...(ve as object)} />)}
    </svg>
  );
}

/**
 * The Pose tool's view of a paint layer's figures: every joint as a handle in its OpenPose colour, and the figure being
 * dragged drawn whole (the layer's raster only catches up when the drag ends).
 */
function PoseOverlay({ ops, live, scene, zoom }: { ops: MaskOp[]; live: { i: number; op: PoseOp } | null; scene: Scene; zoom: number }) {
  const figures: Array<{ op: PoseOp; live: boolean }> = [];
  ops.forEach((op, i) => {
    if (op.t !== 'pose' || op.off) return;
    figures.push(live && live.i === i ? { op: live.op, live: true } : { op, live: false });
  });
  if (live && live.i < 0) figures.push({ op: live.op, live: true });
  const r = 5 / zoom;
  return (
    <svg className="sc-pose" viewBox={`0 0 ${scene.width} ${scene.height}`} preserveAspectRatio="none" aria-hidden>
      {figures.map(({ op, live: moving }, f) => (
        // eslint-disable-next-line react/no-array-index-key
        <g key={f}>
          {moving &&
            POSE_LIMBS.map(([a, b], k) => (
              <line
                // eslint-disable-next-line react/no-array-index-key
                key={k}
                x1={op.kp[a * 2]}
                y1={op.kp[a * 2 + 1]}
                x2={op.kp[b * 2]}
                y2={op.kp[b * 2 + 1]}
                stroke={POSE_COLORS[k]}
                strokeWidth={op.w * 2}
                strokeLinecap="round"
                opacity={0.8}
              />
            ))}
          {POSE_JOINTS.map((name, j) => (
            <circle key={name} cx={op.kp[j * 2]} cy={op.kp[j * 2 + 1]} r={j === POSE_ROOT ? r * 1.4 : r} fill={POSE_COLORS[j]} className="sc-pose-joint">
              <title>{j === POSE_ROOT ? 'neck: drag to move the figure' : name}</title>
            </circle>
          ))}
        </g>
      ))}
    </svg>
  );
}

function sameRect(a: Rect, b: Rect): boolean {
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}
