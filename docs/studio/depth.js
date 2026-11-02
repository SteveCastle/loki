/*
 * slangfx studio — AI depth maps (model layer + GPU grade).
 *
 * Video Depth Anything (https://github.com/DepthAnything/Video-Depth-Anything,
 * ByteDance, CVPR 2025), Small variant (Apache-2.0), running in the browser
 * through onnxruntime-web on WebGPU. Chosen over per-frame depth models
 * because it is built for VIDEO: a temporal DPT head looks at 32 frames at
 * once, so depth does not flicker frame to frame — the property a depth
 * MATTE lives or dies by (a matte that breathes makes every masked effect
 * shimmer).
 *
 * The export is split at the ViT-token boundary into two graphs:
 *
 *   encoder:  pixels [1, 3, 518, 910] float32, ImageNet mean/std, RGB
 *             (squish-resized) → feat1..feat4 [1, 2405, 384]
 *   head:     feat1..feat4 [32, 2405, 384] → depth [32, 1, 518, 910]
 *
 * so per-frame features are computed once and reused across overlapping
 * windows. Output is relative DISPARITY (inverse depth): bigger = nearer,
 * ≥ 0, arbitrary scale — which is also the quantity defocus blur is
 * proportional to, so it maps straight onto a blur radius.
 *
 * Long clips are stitched the way the reference implementation does it
 * (video_depth.py infer_video_depth): windows of 32 advance 22 frames; the
 * first 10 slots of every window after the first are KEYFRAMES carried
 * from the previous window (its slot 0 — ultimately frame 0, a global
 * anchor — its slot 12, and its last 8), each window's scale/shift is
 * least-squares aligned to the previous through the two keyframe slots,
 * and the 8 overlapping frames are cross-faded. DepthStream below is that
 * algorithm, streaming: frames come in through a callback and aligned depth
 * goes out in frame order, holding only one window of features.
 *
 * Everything (runtime + models) loads on demand from CDN/HF URLs into the
 * Cache API; override via localStorage['lowkey-studio.depth-config'].
 */

import { cachedFetch } from './roto.js';

export const VDA_W = 910;
export const VDA_H = 518;
const T = 32;                  // head window
const OVERLAP = 10;            // carried keyframe slots per window
const STEP = T - OVERLAP;      // 22 new frames per window
const INTERP = 8;              // cross-faded frames between windows
const KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31];
const TOKENS = 2405;           // 37 × 65 patches of 14 px
const DIM = 384;               // ViT-S width
const FEAT = TOKENS * DIM;     // floats per frame per feature level

const ORT_VERSION = '1.27.0';
// Mirror (unmodified, sha256-checked) of the ONNX conversion published at
// huggingface.co/TheoreticallyTim/theoreticallypose-vda, kept on our own
// account so the feature can't break when a third-party repo moves.
const HF = 'https://huggingface.co/runes/video-depth-anything-small-webgpu/resolve/main';
const DEFAULTS = {
  // The JSEP build — the same one the roto auto engine runs on (the
  // .bundle build's native-EP wasm has corrupted its heap on big fp16 graphs).
  ortUrl: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`,
  wasmPaths: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
  encoderUrl: `${HF}/vda_vits_encoder_518x910_fp16.onnx`,
  headUrl: `${HF}/vda_vits_head_T32_37x65_fp16.onnx`,
  // Keep the window's features on the GPU (encoder outputs are copied
  // straight into the head's input buffers). false = round-trip through JS.
  gpuFeatures: true,
};
const CACHE_NAME = 'lowkey-studio-depth-v1';

function depthConfig() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('lowkey-studio.depth-config') ?? '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}

let runtimePromise = null;

/** Load onnxruntime-web + both VDA graphs (cached singleton). `onStatus(text,
 * frac)` mirrors setStatus. Rejects on failure; a retry re-attempts. */
export function loadDepth(onStatus = () => {}) {
  runtimePromise ??= (async () => {
    const cfg = depthConfig();
    try {
      onStatus('depth: loading onnxruntime…', 0);
      const ort = (await import(/* webpackIgnore: true */ cfg.ortUrl)).default ?? globalThis.ort;
      ort.env.wasm.wasmPaths = cfg.wasmPaths;
      const mb = (n) => (n / 1048576).toFixed(0);
      const encBuf = await cachedFetch(cfg.encoderUrl, (got, total) =>
        onStatus(`depth: downloading Video Depth Anything encoder — ${mb(got)}/${mb(total)} MB (one-time)`,
          0.6 * (got / total)), CACHE_NAME);
      const headBuf = await cachedFetch(cfg.headUrl, (got, total) =>
        onStatus(`depth: downloading temporal head — ${mb(got)}/${mb(total)} MB (one-time)`,
          0.6 + 0.25 * (got / total)), CACHE_NAME);
      onStatus('depth: starting model…', 0.9);
      const gpuFeatures = cfg.gpuFeatures !== false;
      let enc;
      let head;
      let ep = 'webgpu';
      try {
        enc = await ort.InferenceSession.create(encBuf, {
          executionProviders: ['webgpu'],
          ...(gpuFeatures ? { preferredOutputLocation: 'gpu-buffer' } : {}),
        });
        head = await ort.InferenceSession.create(headBuf, { executionProviders: ['webgpu'] });
      } catch (e) {
        console.warn('depth: WebGPU EP failed, falling back to wasm (slow):', e);
        enc = await ort.InferenceSession.create(encBuf, { executionProviders: ['wasm'] });
        head = await ort.InferenceSession.create(headBuf, { executionProviders: ['wasm'] });
        ep = 'wasm';
      }
      // Map encoder outputs to head inputs by name (feat1..feat4 on both
      // sides in the reference export), falling back to declaration order.
      const encOut = enc.outputNames;
      const headIn = head.inputNames.map((n, i) => (encOut.includes(n) ? n : encOut[i]));
      const rt = {
        ort, enc, head, ep,
        encIn: enc.inputNames[0],
        encOut: headIn,             // encoder output feeding head input k
        headIn: head.inputNames,
        headOut: head.outputNames[0],
        gpu: ep === 'webgpu' && gpuFeatures,
        device: null,
      };
      if (rt.gpu) {
        try { rt.device = await ort.env.webgpu.device; } catch { rt.device = null; }
        if (!rt.device) rt.gpu = false;
      }
      onStatus(`depth: ready (${ep})`, null);
      return rt;
    } catch (e) {
      runtimePromise = null;
      throw e;
    }
  })();
  return runtimePromise;
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/** ImageData-like {data, width: VDA_W, height: VDA_H} → the encoder's
 * normalized CHW float32 input. */
export function toEncoderInput(img) {
  const { data } = img;
  const px = VDA_W * VDA_H;
  const chw = new Float32Array(px * 3);
  for (let c = 0; c < 3; c++) {
    const mean = IMAGENET_MEAN[c] * 255;
    const k = 1 / (IMAGENET_STD[c] * 255);
    const base = c * px;
    for (let i = 0; i < px; i++) chw[base + i] = (data[i * 4 + c] - mean) * k;
  }
  return chw;
}

/**
 * Where a source frame goes inside the model's fixed 910×518 input.
 * Close-to-16:9 sources are squish-resized to fill it (the reference
 * resizes keeping aspect, but the static graph has one shape, and a mild
 * squish costs the ViT nothing measurable). Far-off aspects (portrait,
 * square) would be squashed 2–3× — those are fitted instead and the
 * borders padded by stretching the edge pixels outward, which a depth net
 * reads as "more of the same" rather than a wall. The same rect crops
 * the output back to the source's aspect.
 */
export function depthInputLayout(w, h) {
  const r = (w / h) / (VDA_W / VDA_H);
  if (r >= 0.72 && r <= 1.4) return { x: 0, y: 0, w: VDA_W, h: VDA_H, pad: false };
  const k = Math.min(VDA_W / w, VDA_H / h);
  const dw = Math.max(1, Math.round(w * k));
  const dh = Math.max(1, Math.round(h * k));
  return { x: Math.floor((VDA_W - dw) / 2), y: Math.floor((VDA_H - dh) / 2), w: dw, h: dh, pad: true };
}

/** Draw `src` (sw × sh) into a VDA-sized 2D context per `layout`. */
export function drawDepthInput(ctx, src, sw, sh, layout) {
  const { x, y, w, h } = layout;
  if (!layout.pad) { ctx.drawImage(src, 0, 0, sw, sh, 0, 0, VDA_W, VDA_H); return; }
  ctx.drawImage(src, 0, 0, sw, sh, x, y, w, h);
  // Edge-stretch padding: the outermost row/column smeared to the border.
  if (x > 0) {
    ctx.drawImage(ctx.canvas, x, y, 1, h, 0, y, x, h);
    ctx.drawImage(ctx.canvas, x + w - 1, y, 1, h, x + w, y, VDA_W - x - w, h);
  }
  if (y > 0) {
    ctx.drawImage(ctx.canvas, 0, y, VDA_W, 1, 0, 0, VDA_W, y);
    ctx.drawImage(ctx.canvas, 0, y + h - 1, VDA_W, 1, 0, y + h, VDA_W, VDA_H - y - h);
  }
}

/** Least-squares scale/shift so s·x + b ≈ y (the reference's
 * compute_scale_and_shift over every pixel). */
function scaleShift(xs, ys) {
  let a00 = 0, a01 = 0, a11 = 0, b0 = 0, b1 = 0;
  for (let k = 0; k < xs.length; k++) {
    const x = xs[k], y = ys[k];
    for (let i = 0; i < x.length; i++) {
      const xv = x[i];
      a00 += xv * xv;
      a01 += xv;
      b0 += xv * y[i];
      b1 += y[i];
    }
    a11 += x.length;
  }
  const det = a00 * a11 - a01 * a01;
  if (!(Math.abs(det) > 1e-12)) return [1, 0];
  return [(a11 * b0 - a01 * b1) / det, (-a01 * b0 + a00 * b1) / det];
}

function affine(d, s, b) {
  const out = new Float32Array(d.length);
  for (let i = 0; i < d.length; i++) {
    const v = d[i] * s + b;
    out[i] = v > 0 ? v : 0;
  }
  return out;
}

/**
 * The windowed, aligned VDA walk (see the header). Holds one window of
 * encoder features — on the GPU when the runtime allows (4 × 118 MB head
 * inputs, filled by buffer copies), else in JS arrays.
 */
export class DepthStream {
  constructor(rt) {
    this.rt = rt;
    this.refMax = 0;   // max disparity of the first window: the clip's scale
    this.gpu = rt.gpu;
  }

  _alloc() {
    const bytes = T * FEAT * 4;
    if (this.gpu) {
      const dev = this.rt.device;
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      this.stack = [0, 1, 2, 3].map(() => dev.createBuffer({ label: 'vda window features', size: bytes, usage }));
      this.carry = [0, 1, 2, 3].map(() => dev.createBuffer({ label: 'vda keyframe carry', size: OVERLAP * FEAT * 4, usage }));
    } else {
      this.stack = [0, 1, 2, 3].map(() => new Float32Array(T * FEAT));
    }
  }

  dispose() {
    if (this.gpu) for (const b of [...(this.stack ?? []), ...(this.carry ?? [])]) b.destroy();
    this.stack = this.carry = null;
  }

  /** Encode one frame into window slot `s` (or copy slot `from` when the
   * frame is the same one — padding past the end, or a still). */
  async _encodeSlot(s, input, from = -1) {
    const { rt } = this;
    const FB = FEAT * 4;
    if (from >= 0) {
      if (this.gpu) {
        const e2 = rt.device.createCommandEncoder();
        for (let k = 0; k < 4; k++) {
          // Same-buffer copies are illegal; hop through the carry buffer
          // (its slot 0 is free outside the carry step).
          e2.copyBufferToBuffer(this.stack[k], from * FB, this.carry[k], 0, FB);
          e2.copyBufferToBuffer(this.carry[k], 0, this.stack[k], s * FB, FB);
        }
        rt.device.queue.submit([e2.finish()]);
      } else {
        for (const a of this.stack) a.copyWithin(s * FEAT, from * FEAT, (from + 1) * FEAT);
      }
      return;
    }
    const tensor = new rt.ort.Tensor('float32', input, [1, 3, VDA_H, VDA_W]);
    const out = await rt.enc.run({ [rt.encIn]: tensor });
    if (this.gpu) {
      const enc = rt.device.createCommandEncoder();
      for (let k = 0; k < 4; k++)
        enc.copyBufferToBuffer(out[rt.encOut[k]].gpuBuffer, 0, this.stack[k], s * FB, FB);
      rt.device.queue.submit([enc.finish()]);
      for (const name of rt.encOut) out[name].dispose();
    } else {
      for (let k = 0; k < 4; k++) this.stack[k].set(out[rt.encOut[k]].data.subarray(0, FEAT), s * FEAT);
    }
  }

  /** Move the previous window's keyframe slots into slots 0..9. */
  _carryKeyframes() {
    const FB = FEAT * 4;
    if (this.gpu) {
      const enc = this.rt.device.createCommandEncoder();
      for (let k = 0; k < 4; k++) {
        KEYFRAMES.forEach((src, dst) =>
          enc.copyBufferToBuffer(this.stack[k], src * FB, this.carry[k], dst * FB, FB));
        enc.copyBufferToBuffer(this.carry[k], 0, this.stack[k], 0, OVERLAP * FB);
      }
      this.rt.device.queue.submit([enc.finish()]);
    } else {
      for (const a of this.stack) {
        const tmp = KEYFRAMES.map((src) => a.slice(src * FEAT, (src + 1) * FEAT));
        tmp.forEach((f, dst) => a.set(f, dst * FEAT));
      }
    }
  }

  async _head() {
    const { rt } = this;
    const dims = [T, TOKENS, DIM];
    const feeds = {};
    rt.headIn.forEach((name, k) => {
      feeds[name] = this.gpu
        ? rt.ort.Tensor.fromGpuBuffer(this.stack[k], { dataType: 'float32', dims })
        : new rt.ort.Tensor('float32', this.stack[k], dims);
    });
    const out = await rt.head.run(feeds);
    const data = out[rt.headOut].data;
    const px = VDA_W * VDA_H;
    // Copies: the output buffer belongs to the session.
    return Array.from({ length: T }, (_, i) => data.slice(i * px, (i + 1) * px));
  }

  /**
   * Walk `n` frames. getInput(i) → (Promise of) encoder input for frame i.
   * onDepth(i, Float32Array VDA_W×VDA_H disparity) fires once per frame in
   * order. gate(fn) serializes every model call (onnxruntime-web must never
   * run two sessions concurrently). Stops early when cancelled() is true.
   * Returns the number of frames emitted.
   */
  async run(n, getInput, onDepth, { gate = (fn) => fn(), cancelled = () => false, onProgress } = {}) {
    this._alloc();
    let emitted = 0;
    const emit = async (i, d) => {
      if (i < n && i === emitted) { await onDepth(i, d); emitted++; }
    };
    try {
      const windows = Math.max(1, Math.ceil(n / STEP));
      let pending = [];      // aligned frames of the last window's tail
      let ref = null;        // [frame-0 anchor, previous window's slot 12]
      let encoded = 0;
      for (let w = 0; w < windows; w++) {
        if (cancelled()) break;
        const fid = w * STEP;
        if (w > 0) this._carryKeyframes();
        let lastIdx = -1;
        let lastSlot = -1;
        for (let s = w === 0 ? 0 : OVERLAP; s < T; s++) {
          if (cancelled()) break;
          const idx = Math.min(fid + s, n - 1);
          if (idx === lastIdx) {
            await this._encodeSlot(s, null, lastSlot);
          } else {
            const input = await getInput(idx);
            await gate(() => this._encodeSlot(s, input));
            encoded++;
            onProgress?.(Math.min(1, encoded / n), idx);
          }
          lastIdx = idx;
          lastSlot = s;
        }
        if (cancelled()) break;
        const raw = await gate(() => this._head());
        if (w === 0) {
          for (const d of raw) for (let i = 0; i < d.length; i++) if (d[i] > this.refMax) this.refMax = d[i];
          ref = [raw[0], raw[12]];
          for (let s = 0; s < T - INTERP; s++) await emit(s, raw[s]);
          pending = raw.slice(T - INTERP).map((d, k) => ({ i: T - INTERP + k, d }));
        } else {
          const ALIGN = OVERLAP - INTERP;   // keyframe slots used for alignment
          const [sc, sh] = scaleShift(raw.slice(0, ALIGN), ref);
          // Slots 2..9 re-cover the previous window's last 8 frames.
          for (let k = 0; k < INTERP; k++) {
            const post = affine(raw[ALIGN + k], sc, sh);
            const pre = pending[k];
            const wt = k / (INTERP - 1);
            const out = new Float32Array(post.length);
            for (let i = 0; i < out.length; i++) out[i] = pre.d[i] * (1 - wt) + post[i] * wt;
            await emit(pre.i, out);
          }
          for (let s = OVERLAP; s < T - INTERP; s++) await emit(fid + s, affine(raw[s], sc, sh));
          pending = [];
          for (let s = T - INTERP; s < T; s++) pending.push({ i: fid + s, d: affine(raw[s], sc, sh) });
          ref = [ref[0], affine(raw[12], sc, sh)];
        }
      }
      if (!cancelled()) for (const p of pending) await emit(p.i, p.d);
    } finally {
      this.dispose();
    }
    return emitted;
  }
}

/* ---- storage: 16-bit frames, delta + deflate ---------------------------
 * 8 bits of depth is not enough for a matte: a tight near/far band stretches
 * a few dozen grey levels across the whole ramp, and a slice edge then
 * moves in visible steps from frame to frame — flicker by quantization.
 * Frames are stored as 16-bit, horizontally delta-coded (depth is smooth,
 * so deltas are tiny) and deflated: ~5–8× smaller than raw. */

async function streamBytes(stream) {
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function encodeDepthFrame(u16, w, h) {
  const d = new Uint16Array(u16.length);
  for (let y = 0; y < h; y++) {
    let prev = 0;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const v = u16[row + x];
      d[row + x] = (v - prev) & 0xffff;
      prev = v;
    }
  }
  const cs = new Blob([d.buffer]).stream().pipeThrough(new CompressionStream('deflate-raw'));
  return new Blob([await streamBytes(cs)], { type: 'application/octet-stream' });
}

export async function decodeDepthFrame(blob, w, h) {
  const ds = blob.stream().pipeThrough(new DecompressionStream('deflate-raw'));
  const bytes = await streamBytes(ds);
  const u16 = new Uint16Array(bytes.buffer, bytes.byteOffset, Math.min(w * h, bytes.byteLength >> 1));
  for (let y = 0; y < h; y++) {
    let prev = 0;
    const row = y * w;
    for (let x = 0; x < w; x++) {
      prev = (prev + u16[row + x]) & 0xffff;
      u16[row + x] = prev;
    }
  }
  return u16;
}

/* ---- GPU grade ---------------------------------------------------------
 * One pass turns a stored 16-bit frame into the layer's picture, in source
 * space: decode + normalize, edge-aware upsample against the source frame
 * (joint bilateral — depth is 518 px tall, the mask edge should follow the
 * picture's edges), then the mode curve, gamma, invert and output levels. */

const GRADE_WGSL = /* wgsl */ `
struct U {
  dsize : vec2<f32>,   // stored depth dims
  lohi : vec2<f32>,    // normalization: d = (raw - lo) / (hi - lo)
  mode : f32, nearV : f32, farV : f32, soft : f32,
  focus : f32, range : f32, band : f32, gamma : f32,
  outLo : f32, outHi : f32, invert : f32, refine : f32,
};
@group(0) @binding(0) var depthTex : texture_2d<u32>;
@group(0) @binding(1) var guide : texture_2d<f32>;
@group(0) @binding(2) var smp : sampler;
@group(0) @binding(3) var<uniform> u : U;

struct VSOut { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(3.0, 1.0), vec2<f32>(-1.0, 1.0));
  var o : VSOut;
  o.pos = vec4<f32>(p[i], 0.0, 1.0);
  o.uv = vec2<f32>((p[i].x + 1.0) * 0.5, (1.0 - p[i].y) * 0.5);
  return o;
}

fn dAt(p : vec2<i32>) -> f32 {
  let mx = vec2<i32>(u.dsize) - vec2<i32>(1, 1);
  let q = textureLoad(depthTex, clamp(p, vec2<i32>(0, 0), mx), 0).r;
  return (f32(q) / 65535.0 - u.lohi.x) / max(u.lohi.y - u.lohi.x, 1e-6);
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let pd = in.uv * u.dsize - vec2<f32>(0.5, 0.5);
  let b = floor(pd);
  let f = pd - b;
  let bi = vec2<i32>(b);
  var d : f32;
  if (u.refine < 0.001) {
    // Manual bilinear (integer textures don't filter).
    let d00 = dAt(bi);
    let d10 = dAt(bi + vec2<i32>(1, 0));
    let d01 = dAt(bi + vec2<i32>(0, 1));
    let d11 = dAt(bi + vec2<i32>(1, 1));
    d = mix(mix(d00, d10, f.x), mix(d01, d11, f.x), f.y);
  } else {
    // Joint bilateral upsample: 4×4 depth texels around the pixel,
    // weighted by distance AND by how alike the source picture is at
    // that texel and here — depth snaps to the picture's own edges.
    let gc = textureSampleLevel(guide, smp, in.uv, 0.0).rgb;
    let sr = mix(0.6, 0.035, u.refine);
    var acc = 0.0;
    var wsum = 0.0;
    for (var j = -1; j <= 2; j++) {
      for (var i = -1; i <= 2; i++) {
        let p = bi + vec2<i32>(i, j);
        let off = vec2<f32>(f32(i), f32(j)) - f;
        let ws = exp(-dot(off, off) / (2.0 * 0.85 * 0.85));
        let tuv = (vec2<f32>(p) + vec2<f32>(0.5, 0.5)) / u.dsize;
        let g = textureSampleLevel(guide, smp, tuv, 0.0).rgb - gc;
        let wr = exp(-dot(g, g) / (2.0 * sr * sr));
        let w = ws * wr + 1e-5 * ws;
        acc += dAt(p) * w;
        wsum += w;
      }
    }
    d = acc / wsum;
  }
  d = clamp(d, 0.0, 1.0);

  var v : f32;
  if (u.mode < 0.5) {
    // Ramp: far → black, near → white (swapping them inverts the ramp).
    let span = u.nearV - u.farV;
    v = select(step(u.farV, d), (d - u.farV) / span, abs(span) > 1e-5);
  } else if (u.mode < 1.5) {
    // Slice: white between the two cuts, soft edges.
    let lo = min(u.nearV, u.farV);
    let hi = max(u.nearV, u.farV);
    let s = max(u.soft, 5e-4);
    v = smoothstep(lo - s, lo + s, d) * (1.0 - smoothstep(hi - s, hi + s, d));
  } else {
    // Focus: black on the focal plane (± half the in-focus band), rising
    // to white over the falloff — a circle-of-confusion map.
    v = (abs(d - u.focus) - u.band * 0.5) / max(u.range, 1e-4);
  }
  v = clamp(v, 0.0, 1.0);
  v = pow(v, 1.0 / max(u.gamma, 0.01));
  v = select(v, 1.0 - v, u.invert > 0.5);
  v = mix(u.outLo, u.outHi, v);
  return vec4<f32>(v, v, v, 1.0);
}
`;

/** Shared pipeline for every depth layer on one device. */
export class DepthGrader {
  constructor(device) {
    this.device = device;
    const module = device.createShaderModule({ label: 'depth grade', code: GRADE_WGSL });
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'uint' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
      ],
    });
    this.pipeline = device.createRenderPipeline({
      label: 'depth grade',
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
  }
}

/** Per-layer GPU state: the stored frame (r16uint), the graded output
 * (source space) and its uniforms. */
export class DepthLayerGpu {
  constructor(grader) {
    this.grader = grader;
    this.device = grader.device;
    this.ubo = this.device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.depthTex = null;
    this.outTex = null;
    this.frameKey = null;   // which stored frame is uploaded
    this._bgKey = null;
  }

  /** Upload a decoded frame (Uint16Array w×h). */
  setFrame(u16, w, h, key) {
    if (!this.depthTex || this.depthTex.width !== w || this.depthTex.height !== h) {
      this.depthTex?.destroy();
      this.depthTex = this.device.createTexture({
        label: 'depth frame', size: [w, h], format: 'r16uint',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      this.depthView = this.depthTex.createView();
      this._bgKey = null;
    }
    this.device.queue.writeTexture({ texture: this.depthTex }, u16, { bytesPerRow: w * 2 }, [w, h]);
    this.frameKey = key;
  }

  _ensureOut(w, h) {
    if (this.outTex && this.outTex.width === w && this.outTex.height === h) return;
    this.outTex?.destroy();
    this.outTex = this.device.createTexture({
      label: 'depth graded', size: [w, h], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    this.outView = this.outTex.createView();
  }

  /** Encode the grade pass into `outView` (w × h, source space). `guide`
   * is the source frame's view (any size; sampled by uv). */
  encode(encoder, guideView, w, h, p) {
    if (!this.depthTex) return null;
    this._ensureOut(w, h);
    if (this._bgKey !== guideView || !this.bindGroup) {
      this.bindGroup = this.device.createBindGroup({
        layout: this.grader.layout,
        entries: [
          { binding: 0, resource: this.depthView },
          { binding: 1, resource: guideView },
          { binding: 2, resource: this.grader.sampler },
          { binding: 3, resource: { buffer: this.ubo } },
        ],
      });
      this._bgKey = guideView;
    }
    const modeIdx = p.mode === 'slice' ? 1 : p.mode === 'focus' ? 2 : 0;
    this.device.queue.writeBuffer(this.ubo, 0, new Float32Array([
      this.depthTex.width, this.depthTex.height, p.lo, p.hi,
      modeIdx, p.near, p.far, p.soft,
      p.focus, p.range, p.band, p.gamma,
      p.outLo, p.outHi, p.invert ? 1 : 0, p.refine,
    ]));
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.outView, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
    });
    pass.setPipeline(this.grader.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.draw(3);
    pass.end();
    return this.outView;
  }

  destroy() {
    this.depthTex?.destroy();
    this.outTex?.destroy();
    this.ubo.destroy();
    this.depthTex = this.outTex = null;
  }
}
