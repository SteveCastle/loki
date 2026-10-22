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
// bounding box of what it actually covers.
export type MaskOp =
  | { t: 'rect'; x: number; y: number; w: number; h: number; erase?: boolean; keep?: boolean }
  | { t: 'lasso'; pts: number[]; erase?: boolean; keep?: boolean }
  | { t: 'stroke'; pts: number[]; r: number; erase?: boolean; keep?: boolean }
  | {
      t: 'image';
      src: string;
      x: number;
      y: number;
      w: number;
      h: number;
      bx: number;
      by: number;
      bw: number;
      bh: number;
      erase?: boolean;
      keep?: boolean;
    };

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
    if (op.erase || op.keep) return;
    if (op.t === 'rect') {
      add(op.x, op.y);
      add(op.x + op.w, op.y + op.h);
    } else if (op.t === 'image') {
      add(op.bx, op.by);
      add(op.bx + op.bw, op.by + op.bh);
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
  const b = maskBounds(ops);
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
 * what is under them; with `keepAsPaint` only the keep ops are drawn, as paint (the Protect overlay).
 */
export function drawOps(ctx: CanvasRenderingContext2D, ops: MaskOp[], scale: number, keepAsPaint = false): void {
  ctx.save();
  ctx.fillStyle = '#fff';
  ctx.strokeStyle = '#fff';
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ops.forEach((op) => {
    if (keepAsPaint && !op.keep) return;
    ctx.globalCompositeOperation = !keepAsPaint && (op.erase || op.keep) ? 'destination-out' : 'source-over';
    if (op.t === 'rect') {
      ctx.fillRect(op.x * scale, op.y * scale, op.w * scale, op.h * scale);
      return;
    }
    if (op.t === 'image') {
      const img = maskImage(op.src);
      if (img) ctx.drawImage(img, op.x * scale, op.y * scale, op.w * scale, op.h * scale);
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

/**
 * The mask the server composites through: a full-frame PNG (long edge <= MASK_MAX) whose alpha is the layer's
 * coverage, with the feather (frame pixels) as a blur. '' when nothing is painted.
 */
export function rasterizeMask(ops: MaskOp[], frameW: number, frameH: number, feather: number): string | null {
  if (!maskBounds(ops)) return '';
  // An image piece not decoded yet would silently drop out of the mask: null = try again once preloaded.
  if (!maskImagesReady(ops)) return null;
  const { w, h, scale } = maskSize(frameW, frameH);
  const hard = document.createElement('canvas');
  hard.width = w;
  hard.height = h;
  const hctx = hard.getContext('2d');
  if (!hctx) return '';
  drawOps(hctx, ops, scale);
  if (feather <= 0) return hard.toDataURL('image/png');
  const soft = document.createElement('canvas');
  soft.width = w;
  soft.height = h;
  const sctx = soft.getContext('2d');
  if (!sctx) return hard.toDataURL('image/png');
  // A gaussian's visible falloff is ~2σ: σ = feather/2 fades over about the feather width.
  sctx.filter = `blur(${Math.max(0.5, (feather * scale) / 2)}px)`;
  sctx.drawImage(hard, 0, 0);
  return soft.toDataURL('image/png');
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
  if (feather <= 0) return hard.toDataURL('image/png');
  const soft = document.createElement('canvas');
  soft.width = w;
  soft.height = h;
  const sctx = soft.getContext('2d');
  if (!sctx) return hard.toDataURL('image/png');
  sctx.filter = `blur(${Math.max(0.5, (feather * scale) / 2)}px)`;
  sctx.drawImage(hard, 0, 0);
  return soft.toDataURL('image/png');
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
