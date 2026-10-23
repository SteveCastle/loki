import React, { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { mediaUrl } from '../../platform';
import { pointInRect, simplifyPath, type MaskOp, type Rect } from './mask';
import { jobPhase, layerRate, layerTiming, type Scene, type SceneLayer } from './scene-api';

// The canvas of the Scene editor: the base frame with every generated layer
// composited live the way the export will be (each result placed over the
// region it was made for, seen through its feathered mask: CSS masks over the
// same PNG the server composites with), plus the drawing tools for the
// selected layer's mask. Coordinates handed out are base-frame pixels.

export type Tool = 'select' | 'box' | 'brush' | 'lasso' | 'erase' | 'protect' | 'hand';

interface View {
  zoom: number;
  x: number;
  y: number;
}

type Draft =
  | { t: 'rect'; x0: number; y0: number; x1: number; y1: number; erase: boolean; keep?: boolean }
  | { t: 'lasso' | 'stroke'; pts: number[]; erase: boolean; keep?: boolean }
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
  showMask: boolean;
  /** Hide every layer (hold-to-compare with the base). */
  compare: boolean;
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
  const { scene, selected, tool, brush, showMask, compare, playing, soundOn, syncKey, fitKey, protectMask, motionOverlay, onCommit, onSelect } = props;
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
  const drag = useRef<{ kind: 'pan'; sx: number; sy: number; vx: number; vy: number } | { kind: 'draw' } | null>(null);
  const draft = useRef<Draft>(null);

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
    const typing = (e: KeyboardEvent) => e.target instanceof HTMLElement && !!e.target.closest('input, textarea, select, [contenteditable]');
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
  }, [view, size, brush]);

  useEffect(() => paintDraft(), [paintDraft]);

  const drawTool = tool === 'box' || tool === 'brush' || tool === 'lasso' || tool === 'erase' || tool === 'protect';
  const panning = spaceDown || tool === 'hand';

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
    if (!selected || !drawTool) return;
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
    const d = drag.current;
    if (!d || !view) return;
    if (d.kind === 'pan') {
      userZoomed.current = true;
      setView({ ...view, x: d.vx + e.clientX - d.sx, y: d.vy + e.clientY - d.sy });
      return;
    }
    const dr = draft.current;
    if (!dr) return;
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

  const finish = () => {
    const d = drag.current;
    drag.current = null;
    const dr = draft.current;
    draft.current = null;
    paintDraft();
    if (!d || d.kind !== 'draw' || !dr || !view) return;
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
    onCommit({ t: 'stroke', pts, r: brush, erase: dr.erase || undefined, keep: dr.keep || undefined });
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

  const cursorClass = panning ? (drag.current?.kind === 'pan' ? 'grabbing' : 'grab') : drawTool && selected ? 'crosshair' : 'default';
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
      onPointerLeave={() => setCursor(null)}
      onWheel={onWheel}
      onContextMenu={(e) => e.preventDefault()}
    >
      <div className="sc-stage" style={stageStyle}>
        <img className="sc-base" src={mediaUrl(scene.base)} alt="" draggable={false} />
        {!compare &&
          scene.layers.map((l) => {
            const v = l.versions[l.active];
            if (l.hidden || !v || !l.mask) return null;
            const src = mediaUrl(v.path);
            const style = regionStyle(v.region, scene);
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
        {sel && showMask && sel.mask && !compare && <div className={`sc-mask-tint${sel.kind === 'motion' ? ' motion' : ''}`} style={maskStyle(sel.mask)} />}
        {sel && showMask && protectMask && !compare && <div className="sc-mask-tint protect" style={maskStyle(protectMask)} />}
        {motionOverlay && !compare && (
          <div className="sc-motion-tint" style={{ ...regionStyle(motionOverlay.region, scene), ...maskStyle(motionOverlay.dataUrl) }} />
        )}
        {sel && sel.region.w > 0 && !compare && (
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
                </span>
              </div>
            )
        )}
      </div>
      <canvas ref={draftRef} className="sc-draft" style={{ width: size.w, height: size.h }} />
      {cursor && view && sel && (tool === 'brush' || tool === 'erase' || tool === 'protect') && !panning && (
        <div
          className={`sc-brush-cursor${tool === 'erase' ? ' erase' : ''}${tool === 'protect' ? ' protect' : ''}`}
          style={{ left: view.x + cursor.x * view.zoom, top: view.y + cursor.y * view.zoom, width: brush * 2 * view.zoom, height: brush * 2 * view.zoom }}
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

function sameRect(a: Rect, b: Rect): boolean {
  return a.x === b.x && a.y === b.y && a.w === b.w && a.h === b.h;
}
