import { drawPose, poseBox, type PoseOp } from './pose';

// Scene editor masks: the vector strokes a layer's mask is drawn with, the
// region (crop) a generation is run on, and the raster the server composites
// through. All coordinates are base-frame pixels.
//
// The engines take no masks: the server crops the layer's REGION from the
// flattened frame, the engine edits/animates the whole crop, and the result is
// composited back through the MASK. So the region is the mask's bounds plus
// context around it, and the feathered mask must fade out inside the region
// (the crop's own edges must never show).

// An op adds to the mask, or with `erase` takes away from it. `keep` ops are
// the Protect brush: areas the layer must leave as the picture below. They
// cut the mask like an erase but are their own paint (shown in their own
// colour, cleared on their own) and never grow the region the engine sees.
// An `image` op is a raster piece of mask (alpha = coverage) placed over a
// rectangle, e.g. a version's motion map ("Grow mask to motion"); b* is the
// bounding box of what it actually covers. A `layer` op is another layer's
// mask, live: what that layer covers (its own expand and invert applied,
// before its feather) is added or, with `erase`, cut. That is how layers
// that overlap are kept apart cleanly (the sky minus the people in front).
// Any op can be switched `off` without losing it. On a paint layer ops carry
// a colour `c` (CSS hex) and opacity `a` (0-1), and erase cuts the paint. An
// image op with `fit` is the "Fit mask to result" piece: where version `fit`
// (its path) changed the picture, kept in step with the shown version. A `pose`
// op is the mannequin (pose.ts): on a paint layer it draws in OpenPose's colours.
// An op with `subject` is a Repose finish's cover of the character as they
// stand now (finishShapes adds it and takes it away with the mode).
export type MaskOp = (
  | { t: 'rect'; x: number; y: number; w: number; h: number }
  | { t: 'lasso'; pts: number[] }
  /** h: hardness 0-1 (default 1 = a hard edge). */
  | { t: 'stroke'; pts: number[]; r: number; h?: number }
  | { t: 'image'; src: string; x: number; y: number; w: number; h: number; bx: number; by: number; bw: number; bh: number }
  | { t: 'layer'; id: string }
  | PoseOp
) & { erase?: boolean; keep?: boolean; off?: boolean; c?: string; a?: number; fit?: string; label?: string; subject?: boolean };

/** Whole-mask settings, applied after the ops in this order: expand (px; < 0 chokes), invert, feather, opacity. */
export interface MaskOpts {
  invert?: boolean;
  expand?: number;
  /** 0-1, default 1 */
  opacity?: number;
}

/** What a `layer` op needs of the layer it names; null when it is gone. */
export type MaskLookup = (id: string) => { shapes?: MaskOp[]; maskOpts?: MaskOpts } | null;

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** How much of the picture around the mask the engine sees. */
export type ContextSize = 'tight' | 'normal' | 'wide' | 'full';

export const CONTEXT_RATIO: Record<Exclude<ContextSize, 'full'>, number> = {
  tight: 0.15,
  normal: 0.4,
  wide: 0.8,
};

/** Long edge of the full-frame mask raster (the server scales it up). */
export const MASK_MAX = 2048;

/** Bounds of what the additive ops cover (erasing never grows a mask); null when nothing is painted. */
export function maskBounds(ops: MaskOp[]): Rect | null {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  const add = (x: number, y: number, pad = 0) => {
    x0 = Math.min(x0, x - pad);
    y0 = Math.min(y0, y - pad);
    x1 = Math.max(x1, x + pad);
    y1 = Math.max(y1, y + pad);
  };
  ops.forEach((op) => {
    if (op.erase || op.keep || op.off || op.t === 'layer') return;
    if (op.t === 'rect') {
      add(op.x, op.y);
      add(op.x + op.w, op.y + op.h);
    } else if (op.t === 'image') {
      add(op.bx, op.by);
      add(op.bx + op.bw, op.by + op.bh);
    } else if (op.t === 'pose') {
      const b = poseBox(op);
      add(b.x, b.y);
      add(b.x + b.w, b.y + b.h);
    } else {
      const pad = op.t === 'stroke' ? op.r : 0;
      for (let i = 0; i + 1 < op.pts.length; i += 2) add(op.pts[i], op.pts[i + 1], pad);
    }
  });
  if (!Number.isFinite(x0) || x1 - x0 < 1 || y1 - y0 < 1) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Clamp a rect to a w×h frame (integer pixels). */
export function clampRect(r: Rect, w: number, h: number): Rect {
  const x0 = Math.max(0, Math.floor(r.x));
  const y0 = Math.max(0, Math.floor(r.y));
  const x1 = Math.min(w, Math.ceil(r.x + r.w));
  const y1 = Math.min(h, Math.ceil(r.y + r.h));
  return { x: x0, y: y0, w: Math.max(0, x1 - x0), h: Math.max(0, y1 - y0) };
}

/** Grow r around its centre to at least w×h, sliding it back inside the frame rather than cutting it. */
function growTo(r: Rect, w: number, h: number, fw: number, fh: number): Rect {
  const nw = Math.min(fw, Math.max(r.w, w));
  const nh = Math.min(fh, Math.max(r.h, h));
  let x = r.x - (nw - r.w) / 2;
  let y = r.y - (nh - r.h) / 2;
  x = Math.min(Math.max(0, x), fw - nw);
  y = Math.min(Math.max(0, y), fh - nh);
  return { x, y, w: nw, h: nh };
}

export interface RegionOptions {
  context: ContextSize;
  feather: number;
  /** Smallest crop side the engine should get (motion wants >= 256). */
  minSide?: number;
  /** Most elongated crop allowed (long / short side). */
  maxAspect?: number;
}

/**
 * The crop a layer's next generation runs on: the mask's bounds padded by the
 * context size (never less than the feather, so the soft edge fades out inside
 * the crop), grown to a usable size and aspect, inside the frame.
 */
export function regionFor(ops: MaskOp[], frameW: number, frameH: number, o: RegionOptions): Rect | null {
  return regionAround(maskBounds(ops), frameW, frameH, o);
}

/** regionFor around bounds b (frame px) of what a mask covers. */
export function regionAround(b: Rect | null, frameW: number, frameH: number, o: RegionOptions): Rect | null {
  if (!b) return null;
  if (o.context === 'full') return { x: 0, y: 0, w: frameW, h: frameH };
  const pad = Math.max(CONTEXT_RATIO[o.context] * Math.max(b.w, b.h), o.feather * 1.5 + 8);
  let r: Rect = { x: b.x - pad, y: b.y - pad, w: b.w + 2 * pad, h: b.h + 2 * pad };
  r = clampRect(r, frameW, frameH);
  const min = Math.min(o.minSide || 64, frameW, frameH);
  r = growTo(r, min, min, frameW, frameH);
  const maxAspect = o.maxAspect || 2.5;
  if (r.w / r.h > maxAspect) r = growTo(r, r.w, r.w / maxAspect, frameW, frameH);
  else if (r.h / r.w > maxAspect) r = growTo(r, r.h / maxAspect, r.h, frameW, frameH);
  return clampRect(r, frameW, frameH);
}

/** Whether rect inner lies within outer (1 px slack). */
export function rectContains(outer: Rect, inner: Rect): boolean {
  return inner.x >= outer.x - 1 && inner.y >= outer.y - 1 && inner.x + inner.w <= outer.x + outer.w + 1 && inner.y + inner.h <= outer.y + outer.h + 1;
}

export function pointInRect(r: Rect, x: number, y: number): boolean {
  return x >= r.x && y >= r.y && x <= r.x + r.w && y <= r.y + r.h;
}

/** The raster size of a frame's mask. */
export function maskSize(frameW: number, frameH: number): { w: number; h: number; scale: number } {
  const scale = Math.min(1, MASK_MAX / Math.max(frameW, frameH));
  return { w: Math.max(1, Math.round(frameW * scale)), h: Math.max(1, Math.round(frameH * scale)), scale };
}

// Decoded images of `image` ops, by src (data URLs). Rasterizing is
// synchronous, so they are decoded ahead (preloadMaskImages).
const maskImages = new Map<string, HTMLImageElement>();

function maskImage(src: string): HTMLImageElement | null {
  const img = maskImages.get(src);
  return img && img.complete && img.naturalWidth > 0 ? img : null;
}

/** Decode the images of every `image` op (resolves when all are ready or failed). */
export function preloadMaskImages(ops: MaskOp[]): Promise<void> {
  const waits: Array<Promise<void>> = [];
  ops.forEach((op) => {
    if (op.t !== 'image' || maskImage(op.src)) return;
    let img = maskImages.get(op.src);
    if (!img) {
      img = new Image();
      img.src = op.src;
      maskImages.set(op.src, img);
    }
    waits.push(img.decode().catch(() => undefined));
  });
  return Promise.all(waits).then(() => undefined);
}

/** Whether every `image` op can be drawn now. */
export function maskImagesReady(ops: MaskOp[]): boolean {
  return ops.every((op) => op.t !== 'image' || !!maskImage(op.src));
}

/**
 * Draw ops (white, alpha = coverage) onto ctx, whose pixels are frame pixels × scale. `erase` and `keep` ops cut
 * what is under them; with `keepAsPaint` only the keep ops are drawn, as paint (the Protect overlay). `layer` ops
 * draw what layerCoverage returns for them (a canvas the size of ctx's). `colored` draws each op in its own colour
 * and opacity (a paint layer).
 */
export function drawOps(
  ctx: CanvasRenderingContext2D,
  ops: MaskOp[],
  scale: number,
  keepAsPaint = false,
  layerCoverage?: (id: string) => HTMLCanvasElement | null,
  colored = false
): void {
  ctx.save();
  ctx.fillStyle = '#fff';
  ctx.strokeStyle = '#fff';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ops.forEach((op) => {
    if (op.off || (keepAsPaint && !op.keep)) return;
    ctx.globalCompositeOperation = !keepAsPaint && (op.erase || op.keep) ? 'destination-out' : 'source-over';
    ctx.filter = 'none';
    const ink = colored && !op.erase ? op.c || '#fff' : '#fff';
    ctx.fillStyle = ink;
    ctx.strokeStyle = ink;
    ctx.globalAlpha = colored && op.a !== undefined ? Math.min(1, Math.max(0, op.a)) : 1;
    if (op.t === 'layer') {
      const c = layerCoverage?.(op.id);
      if (c) ctx.drawImage(c, 0, 0);
      return;
    }
    if (op.t === 'rect') {
      ctx.fillRect(op.x * scale, op.y * scale, op.w * scale, op.h * scale);
      return;
    }
    if (op.t === 'image') {
      const img = maskImage(op.src);
      if (img) ctx.drawImage(img, op.x * scale, op.y * scale, op.w * scale, op.h * scale);
      return;
    }
    if (op.t === 'pose') {
      // Its own colours, never the ink: they are what tells a model which limb is which.
      if (ctx.globalCompositeOperation === 'destination-out') return;
      drawPose(ctx, op, scale, colored);
      return;
    }
    const p = op.pts;
    if (p.length < 2) return;
    if (op.t === 'lasso') {
      if (p.length < 6) return;
      ctx.beginPath();
      ctx.moveTo(p[0] * scale, p[1] * scale);
      for (let i = 2; i + 1 < p.length; i += 2) ctx.lineTo(p[i] * scale, p[i + 1] * scale);
      ctx.closePath();
      ctx.fill();
      return;
    }
    // A soft brush: blurred by (1 - hardness) of its radius, so its edge fades across that band.
    if (op.h !== undefined && op.h < 1) ctx.filter = `blur(${Math.max(0.5, (op.r * scale * (1 - op.h)) / 2)}px)`;
    if (p.length === 2) {
      ctx.beginPath();
      ctx.arc(p[0] * scale, p[1] * scale, op.r * scale, 0, Math.PI * 2);
      ctx.fill();
      return;
    }
    ctx.lineWidth = op.r * 2 * scale;
    ctx.beginPath();
    ctx.moveTo(p[0] * scale, p[1] * scale);
    for (let i = 2; i + 1 < p.length; i += 2) ctx.lineTo(p[i] * scale, p[i + 1] * scale);
    ctx.stroke();
  });
  ctx.restore();
}

function canvas(w: number, h: number): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
}

/**
 * Grow (px > 0) or choke (px < 0) a coverage canvas by |px| raster pixels, in place: a two-pass chamfer distance
 * transform (close enough to round for masks, linear time at any radius) with a one-pixel soft edge. Coverage that
 * was already there is kept as it was, so a grow keeps the soft edges inside.
 */
export function morphCoverage(c: HTMLCanvasElement, px: number): void {
  const r = Math.abs(px);
  if (r < 0.5) return;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  const { width: w, height: h } = c;
  const img = ctx.getImageData(0, 0, w, h);
  const a = img.data;
  const grow = px > 0;
  // Distance to the nearest pixel of the other side: covered pixels when growing, uncovered ones when choking.
  const d = new Float32Array(w * h);
  for (let i = 0; i < w * h; i += 1) d[i] = a[i * 4 + 3] >= 128 === grow ? 0 : 1e9;
  const D = Math.SQRT2;
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      let v = d[i];
      if (v !== 0) {
        if (x > 0) v = Math.min(v, d[i - 1] + 1);
        if (y > 0) {
          v = Math.min(v, d[i - w] + 1);
          if (x > 0) v = Math.min(v, d[i - w - 1] + D);
          if (x < w - 1) v = Math.min(v, d[i - w + 1] + D);
        }
        d[i] = v;
      }
    }
  }
  for (let y = h - 1; y >= 0; y -= 1) {
    for (let x = w - 1; x >= 0; x -= 1) {
      const i = y * w + x;
      let v = d[i];
      if (v !== 0) {
        if (x < w - 1) v = Math.min(v, d[i + 1] + 1);
        if (y < h - 1) {
          v = Math.min(v, d[i + w] + 1);
          if (x < w - 1) v = Math.min(v, d[i + w + 1] + D);
          if (x > 0) v = Math.min(v, d[i + w - 1] + D);
        }
        d[i] = v;
      }
    }
  }
  for (let i = 0; i < w * h; i += 1) {
    const k = i * 4 + 3;
    if (grow) {
      // Uncovered pixels within r of the coverage join it.
      const t = Math.min(1, Math.max(0, r + 0.5 - d[i]));
      if (t > 0) a[k] = Math.max(a[k], Math.round(t * 255));
    } else if (d[i] > 0) {
      // Covered pixels within r of the outside leave it.
      const t = Math.min(1, Math.max(0, d[i] - r - 0.5));
      a[k] = Math.min(a[k], Math.round(t * 255));
    }
    a[k - 3] = 255;
    a[k - 2] = 255;
    a[k - 1] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/**
 * What a mask covers at raster size (w×h, frame px × scale): its ops, then expand and invert, before feather and
 * opacity. Layer ops resolve through lookup, each layer at most once along a path (a cycle draws nothing).
 */
function coverage(
  ops: MaskOp[],
  opts: MaskOpts | undefined,
  w: number,
  h: number,
  scale: number,
  lookup: MaskLookup | undefined,
  seen: Set<string>,
  cache: Map<string, HTMLCanvasElement | null>,
  colored = false
): HTMLCanvasElement | null {
  const hard = canvas(w, h);
  const ctx = hard.getContext('2d', { willReadFrequently: true });
  if (!ctx) return null;
  drawOps(ctx, ops, scale, false, (id) => {
    if (!lookup || seen.has(id)) return null;
    if (cache.has(id)) return cache.get(id) || null;
    const ref = lookup(id);
    const c = ref ? coverage((ref.shapes || []).filter((op) => !op.off), ref.maskOpts, w, h, scale, lookup, new Set([...seen, id]), cache) : null;
    cache.set(id, c);
    return c;
  }, colored);
  if (opts?.expand) {
    morphCoverage(hard, opts.expand * scale);
    // Expand grows what was drawn, never into what was erased: the cuts (and whatever is drawn after the first one)
    // are replayed as drawn, so an erased area keeps its exact shape instead of filling back in from its edges.
    const firstCut = ops.findIndex((op) => op.erase || op.keep);
    if (firstCut >= 0) drawOps(ctx, ops.slice(firstCut), scale, false, (id) => cache.get(id) || null, colored);
  }
  if (!opts?.invert) return hard;
  const inv = canvas(w, h);
  const ictx = inv.getContext('2d', { willReadFrequently: true });
  if (!ictx) return hard;
  ictx.fillStyle = '#fff';
  ictx.fillRect(0, 0, w, h);
  ictx.globalCompositeOperation = 'destination-out';
  ictx.drawImage(hard, 0, 0);
  return inv;
}

/** Every op of ops and of the layers they name, at any depth (to preload their image pieces). */
export function opsDeep(ops: MaskOp[], lookup: MaskLookup | undefined, seen = new Set<string>()): MaskOp[] {
  const out = [...ops];
  ops.forEach((op) => {
    if (op.t !== 'layer' || op.off || seen.has(op.id) || !lookup) return;
    seen.add(op.id);
    const ref = lookup(op.id);
    if (ref) out.push(...opsDeep(ref.shapes || [], lookup, seen));
  });
  return out;
}

export interface MaskRaster {
  /** The full-frame PNG the server composites through ('' = covers nothing). */
  mask: string;
  /** Bounds of what it covers, frame px (null = nothing). */
  bounds: Rect | null;
}

/**
 * The mask the server composites through: a full-frame PNG (long edge <= MASK_MAX) whose alpha is the layer's
 * coverage (ops, expand, invert), feathered (frame pixels, as a blur) and scaled by its opacity, plus the bounds of
 * what it covers. null while an image piece is still decoding (try again once preloadMaskImages is done).
 */
export function rasterizeLayerMask(
  ops: MaskOp[],
  opts: MaskOpts | undefined,
  frameW: number,
  frameH: number,
  feather: number,
  lookup?: MaskLookup,
  selfId?: string,
  /** A paint layer: the raster is the painting in its colours, not white coverage. */
  colored = false
): MaskRaster | null {
  const live = ops.filter((op) => !op.off);
  if (!live.length && !opts?.invert) return { mask: '', bounds: null };
  // An image piece not decoded yet would silently drop out of the mask.
  if (!maskImagesReady(opsDeep(live, lookup))) return null;
  const { w, h, scale } = maskSize(frameW, frameH);
  const hard = coverage(live, opts, w, h, scale, lookup, new Set(selfId ? [selfId] : []), new Map(), colored);
  const hctx = hard?.getContext('2d', { willReadFrequently: true });
  const box = hard && hctx ? alphaBox(hctx.getImageData(0, 0, w, h).data, w, h, 8) : null;
  if (!hard || !box) return { mask: '', bounds: null };
  const bounds = { x: box.x / scale, y: box.y / scale, w: box.w / scale, h: box.h / scale };
  const opacity = opts?.opacity === undefined ? 1 : Math.min(1, Math.max(0, opts.opacity));
  if (feather <= 0 && opacity >= 1) return { mask: hard.toDataURL('image/png'), bounds };
  const soft = feathered(hard, feather * scale, opacity);
  return { mask: (soft || hard).toDataURL('image/png'), bounds };
}

/**
 * hard blurred by a feather of px raster pixels (and scaled by opacity). A gaussian's visible falloff is ~2σ, so
 * σ = px/2 fades over about the feather width. The frame's edge pixels are carried outward first: the picture goes
 * on past the frame, so a mask reaching the edge stays solid there instead of fading out (blurring against the
 * transparent outside left a half-strength border round every full-frame layer).
 */
function feathered(hard: HTMLCanvasElement, px: number, opacity: number): HTMLCanvasElement | null {
  const { width: w, height: h } = hard;
  const soft = canvas(w, h);
  const sctx = soft.getContext('2d');
  if (!sctx) return null;
  sctx.globalAlpha = opacity;
  if (px <= 0) {
    sctx.drawImage(hard, 0, 0);
    return soft;
  }
  const sigma = Math.max(0.5, px / 2);
  const pad = Math.ceil(sigma * 3);
  const big = canvas(w + 2 * pad, h + 2 * pad);
  const bctx = big.getContext('2d');
  if (!bctx) return null;
  bctx.drawImage(hard, pad, pad);
  // Edges stretched outward, then the corners.
  bctx.drawImage(hard, 0, 0, 1, h, 0, pad, pad, h);
  bctx.drawImage(hard, w - 1, 0, 1, h, w + pad, pad, pad, h);
  bctx.drawImage(hard, 0, 0, w, 1, pad, 0, w, pad);
  bctx.drawImage(hard, 0, h - 1, w, 1, pad, h + pad, w, pad);
  bctx.drawImage(hard, 0, 0, 1, 1, 0, 0, pad, pad);
  bctx.drawImage(hard, w - 1, 0, 1, 1, w + pad, 0, pad, pad);
  bctx.drawImage(hard, 0, h - 1, 1, 1, 0, h + pad, pad, pad);
  bctx.drawImage(hard, w - 1, h - 1, 1, 1, w + pad, h + pad, pad, pad);
  sctx.filter = `blur(${sigma}px)`;
  sctx.drawImage(big, -pad, -pad);
  return soft;
}

/** The mask of plain ops (no layer references or whole-mask settings): '' when nothing is painted, null while an image piece decodes. */
export function rasterizeMask(ops: MaskOp[], frameW: number, frameH: number, feather: number): string | null {
  const r = rasterizeLayerMask(ops, undefined, frameW, frameH, feather);
  return r ? r.mask : null;
}

/** The Protect paint alone, for its overlay (same raster size and feather as the mask); '' when there is none. */
export function rasterizeProtect(ops: MaskOp[], frameW: number, frameH: number, feather: number): string {
  if (!ops.some((op) => op.keep)) return '';
  const { w, h, scale } = maskSize(frameW, frameH);
  const hard = document.createElement('canvas');
  hard.width = w;
  hard.height = h;
  const hctx = hard.getContext('2d');
  if (!hctx) return '';
  drawOps(hctx, ops, scale, true);
  return (feathered(hard, feather * scale, 1) || hard).toDataURL('image/png');
}

/** Bounding box of the pixels whose alpha is at least `threshold` in an RGBA buffer, or null. */
export function alphaBox(data: Uint8ClampedArray, width: number, height: number, threshold: number): Rect | null {
  let x0 = width;
  let y0 = height;
  let x1 = -1;
  let y1 = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (data[(y * width + x) * 4 + 3] >= threshold) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

/**
 * A mask piece from an alpha image (a data URL, so never tainted) placed over `rect`, covering where its alpha is
 * at least `threshold`; null when it covers nothing.
 */
export async function imageOp(src: string, rect: Rect, threshold = 32): Promise<MaskOp | null> {
  const img = new Image();
  img.src = src;
  await img.decode();
  const c = document.createElement('canvas');
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext('2d');
  if (!ctx) return null;
  ctx.drawImage(img, 0, 0);
  const box = alphaBox(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height, threshold);
  if (!box) return null;
  const sx = rect.w / c.width;
  const sy = rect.h / c.height;
  maskImages.set(src, img);
  return { t: 'image', src, x: rect.x, y: rect.y, w: rect.w, h: rect.h, bx: rect.x + box.x * sx, by: rect.y + box.y * sy, bw: box.w * sx, bh: box.h * sy };
}

/** Ramer–Douglas–Peucker: thin a freehand path to the points that matter (tol in frame px). */
export function simplifyPath(pts: number[], tol: number): number[] {
  const n = pts.length / 2;
  if (n <= 2) return pts.slice();
  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;
  const stack: Array<[number, number]> = [[0, n - 1]];
  while (stack.length) {
    const [a, b] = stack.pop() as [number, number];
    const ax = pts[a * 2];
    const ay = pts[a * 2 + 1];
    const dx = pts[b * 2] - ax;
    const dy = pts[b * 2 + 1] - ay;
    const len = Math.hypot(dx, dy) || 1;
    let worst = -1;
    let dist = 0;
    for (let i = a + 1; i < b; i += 1) {
      const d = Math.abs((pts[i * 2] - ax) * dy - (pts[i * 2 + 1] - ay) * dx) / len;
      if (d > dist) {
        dist = d;
        worst = i;
      }
    }
    if (worst > 0 && dist > tol) {
      keep[worst] = 1;
      stack.push([a, worst], [worst, b]);
    }
  }
  const out: number[] = [];
  for (let i = 0; i < n; i += 1) if (keep[i]) out.push(Math.round(pts[i * 2] * 10) / 10, Math.round(pts[i * 2 + 1] * 10) / 10);
  return out;
}

// ---- editing single ops (the Shape tool and the mask panel's list) ----

/** The box an op occupies (frame px), or null for a layer op (it has no shape of its own). */
export function opBox(op: MaskOp): Rect | null {
  if (op.t === 'rect' || op.t === 'image') return { x: op.x, y: op.y, w: op.w, h: op.h };
  if (op.t === 'layer') return null;
  if (op.t === 'pose') return poseBox(op);
  const pad = op.t === 'stroke' ? op.r : 0;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (let i = 0; i + 1 < op.pts.length; i += 2) {
    x0 = Math.min(x0, op.pts[i] - pad);
    y0 = Math.min(y0, op.pts[i + 1] - pad);
    x1 = Math.max(x1, op.pts[i] + pad);
    y1 = Math.max(y1, op.pts[i + 1] + pad);
  }
  return Number.isFinite(x0) ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : null;
}

/** op mapped so that box `from` lands on box `to` (move = same size; scale = corner drag). */
export function mapOp(op: MaskOp, from: Rect, to: Rect): MaskOp {
  const sx = from.w > 0 ? to.w / from.w : 1;
  const sy = from.h > 0 ? to.h / from.h : 1;
  const X = (x: number) => Math.round((to.x + (x - from.x) * sx) * 10) / 10;
  const Y = (y: number) => Math.round((to.y + (y - from.y) * sy) * 10) / 10;
  switch (op.t) {
    case 'rect':
      return { ...op, x: X(op.x), y: Y(op.y), w: op.w * sx, h: op.h * sy };
    case 'image':
      return { ...op, x: X(op.x), y: Y(op.y), w: op.w * sx, h: op.h * sy, bx: X(op.bx), by: Y(op.by), bw: op.bw * sx, bh: op.bh * sy };
    case 'lasso':
    case 'stroke': {
      const pts = op.pts.map((v, i) => (i % 2 ? Y(v) : X(v)));
      return op.t === 'stroke' ? { ...op, pts, r: Math.max(1, op.r * Math.sqrt(sx * sy)) } : { ...op, pts };
    }
    case 'pose':
      return { ...op, kp: op.kp.map((v, i) => (i % 2 ? Y(v) : X(v))), w: Math.max(1, Math.round(op.w * Math.sqrt(sx * sy))) };
    default:
      return op;
  }
}

/** A lasso/pen shape with vertex k moved to (x, y). */
export function moveVertex(op: MaskOp, k: number, x: number, y: number): MaskOp {
  if (op.t !== 'lasso' || k < 0 || k * 2 + 1 >= op.pts.length) return op;
  const pts = op.pts.slice();
  pts[k * 2] = Math.round(x * 10) / 10;
  pts[k * 2 + 1] = Math.round(y * 10) / 10;
  return { ...op, pts };
}

/** What an op is called in the mask panel. */
export function opLabel(op: MaskOp, layerName?: (id: string) => string | null): string {
  switch (op.t) {
    case 'rect':
      return 'Box';
    case 'lasso':
      return `Shape · ${Math.floor(op.pts.length / 2)} points`;
    case 'stroke':
      return op.h !== undefined && op.h < 1 ? 'Soft brush' : 'Brush';
    case 'image':
      if (op.fit) return 'Fitted to the result';
      return op.label ? `Selected · ${op.label}` : 'Painted piece';
    case 'pose':
      return 'Mannequin pose';
    case 'layer': {
      const name = layerName?.(op.id);
      return name === null || name === undefined ? 'Layer (deleted)' : `Layer · ${name}`;
    }
    default:
      return 'Shape';
  }
}

// The frame shapes loki-reshoot's model is trained on (it pads anything else with black bars it then keeps).
const MOTION_ASPECTS = [1, 4 / 3, 3 / 4, 16 / 9, 9 / 16];

/**
 * A motion layer's region (the mask's bounds plus context) grown to the nearest trained shape that fits the frame,
 * centred where it can be and slid inside the frame where it can't. When none grown from r fits, the context gives
 * way: the largest trained shape that fits the frame and still holds keep (the mask's own bounds), centred on r. Any
 * other shape makes the engine reframe the shot. Mirrors sceneMotionRegion in media-server/tasks/scene.go.
 */
export function motionRegion(r: Rect, frameW: number, frameH: number, keep?: Rect | null): Rect {
  const a0 = Math.max(1, r.w) / Math.max(1, r.h);
  const order = [...MOTION_ASPECTS].sort((p, q) => Math.abs(Math.log(a0 / p)) - Math.abs(Math.log(a0 / q)));
  for (const a of order) {
    let w = r.w;
    let h = r.h;
    if (w / h < a) w = h * a;
    else h = w / a;
    const W = Math.round(w);
    const H = Math.round(h);
    if (W <= frameW && H <= frameH) {
      const x = Math.min(Math.max(0, r.x - Math.trunc((W - r.w) / 2)), frameW - W);
      const y = Math.min(Math.max(0, r.y - Math.trunc((H - r.h) / 2)), frameH - H);
      return { x, y, w: W, h: H };
    }
  }
  const k = keep && keep.w > 0 && keep.h > 0 ? keep : null;
  for (const a of order) {
    const W = Math.min(frameW, Math.floor(frameH * a));
    const H = Math.min(frameH, Math.round(W / a));
    if (k && (W < k.w || H < k.h)) continue;
    let x = Math.min(Math.max(0, r.x + Math.trunc(r.w / 2) - Math.trunc(W / 2)), frameW - W);
    let y = Math.min(Math.max(0, r.y + Math.trunc(r.h / 2) - Math.trunc(H / 2)), frameH - H);
    if (k) {
      x = Math.min(Math.max(x, k.x + k.w - W), k.x);
      y = Math.min(Math.max(y, k.y + k.h - H), k.y);
    }
    return { x, y, w: W, h: H };
  }
  return r;
}

/** A picked object (its mask covering rect, its box; frame pixels) as a mask piece, labelled with what it is. */
export function pickedOp(mask: string, rect: Rect, bbox: Rect, label: string, erase = false): MaskOp {
  return { t: 'image', src: mask, x: rect.x, y: rect.y, w: rect.w, h: rect.h, bx: bbox.x, by: bbox.y, bw: bbox.w, bh: bbox.h, label, erase: erase || undefined };
}
