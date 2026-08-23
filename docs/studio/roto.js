/*
 * slangfx studio — AI rotoscoping backend (model layer only).
 *
 * Wraps BiRefNet (https://github.com/zhengpeng7/birefnet), a dichotomous
 * image segmentation / matting model, running in the browser through
 * onnxruntime-web. Unlike the SAM-family backend this replaced, BiRefNet
 * takes NO prompts: feed it a frame (or a crop around the subject) and it
 * returns a soft alpha matte of the most salient thing in view. Verified
 * against the onnx-community export:
 *
 *   input:   [1, 3, 1024, 1024] float32 RGB, squish-resized (aspect NOT
 *            preserved — that matches training), ImageNet mean/std
 *   output:  [1, 1, 1024, 1024] logits → sigmoid → alpha matte
 *
 * The temporal story (walking frames, subject-box tracking) lives in
 * app.js — this module is stateless per call apart from the cached
 * runtime. Everything (runtime + model) loads on demand from CDN URLs and
 * is cached in the Cache API, so the studio itself ships no ML weights;
 * override the URLs via localStorage['lowkey-studio.roto-config'] to
 * self-host or swap variants (BiRefNet general / HR / matting exports all
 * share this I/O contract).
 */

export const ROTO_SIZE = 1024;   // BiRefNet's fixed square input

const ORT_VERSION = '1.27.0';
const DEFAULTS = {
  // ort.webgpu.min.mjs is the JSEP build — the one that actually runs this
  // model. The .bundle variant uses the native-EP asyncify wasm, which
  // (measured 2026-08-23, ort 1.27) still trips an off-by-one in its
  // Concat batching and corrupts its heap near the 4 GB wasm cap.
  ortUrl: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`,
  wasmPaths: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
  // NOT the raw onnx-community export — that one cannot run on WebGPU
  // (1024-input Concats, 32-output Splits, CPU-only Sum ops, and fp16
  // constant-fold islands that blow the 4 GB wasm heap). This artifact is
  // the same weights after offline surgery (fold + cascade + Sum→Add,
  // bit-verified); studio/tools/export-birefnet.py rebuilds it.
  modelUrl: 'https://huggingface.co/runes/birefnet-lite-webgpu/resolve/main/birefnet_lite_webgpu_fp16.onnx',
  inputName: null,    // default: the session's first input
  outputName: null,   // default: the session's first output
};
const CACHE_NAME = 'lowkey-studio-roto-v2';

function rotoConfig() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('lowkey-studio.roto-config') ?? '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}

/** Inside the Electron app (studio:// origin) big downloads go through the
 * main process's disk cache (studio-window.ts /dep-cache): renderer storage
 * on custom-scheme origins is best-effort — Chromium evicts it, which is why
 * the model kept re-downloading in the app. Same origin, so no CORS. */
function appCached(url) {
  return globalThis.location?.protocol === 'studio:'
    ? `/dep-cache?url=${encodeURIComponent(url)}`
    : url;
}

/** Fetch `url` as an ArrayBuffer through the Cache API, reporting download
 * progress (the model is >100 MB — the first use must not look hung). */
async function cachedFetch(url, onProgress) {
  let cache = null;
  try {
    // Best-effort eviction protection for the browser case; ignored where
    // unsupported. Chrome grants it silently for installed/engaged sites.
    navigator.storage?.persist?.().catch(() => {});
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return await hit.arrayBuffer();
  } catch { /* Cache API unavailable (private mode) — plain fetch below */ }
  const res = await fetch(appCached(url));
  if (!res.ok) throw new Error(`roto: ${res.status} fetching ${url}`);
  const total = parseInt(res.headers.get('Content-Length') ?? '0', 10);
  if (!res.body || !total) {
    const buf = await res.arrayBuffer();
    try { await cache?.put(url, new Response(buf)); } catch {}
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  try { await cache?.put(url, new Response(buf.buffer.slice(0))); } catch {}
  return buf.buffer;
}

let runtimePromise = null;

/**
 * Load onnxruntime-web + BiRefNet (cached singleton). `onStatus(text,
 * frac)` mirrors setStatus. Resolves to the runtime handle used by
 * segmentFrame; rejects on network/WebGPU-and-wasm failure (a retry
 * re-attempts the load).
 */
export function loadRoto(onStatus = () => {}) {
  runtimePromise ??= (async () => {
    const cfg = rotoConfig();
    try {
      // The SAM-era cache held ~70 MB of models this backend never reads.
      try { caches.delete('lowkey-studio-roto-v1'); } catch {}
      onStatus('roto: loading onnxruntime…', 0);
      const ort = (await import(/* webpackIgnore: true */ cfg.ortUrl)).default
        ?? globalThis.ort;
      ort.env.wasm.wasmPaths = cfg.wasmPaths;

      const mb = (n) => (n / 1048576).toFixed(0);
      const buf = await cachedFetch(cfg.modelUrl, (got, total) =>
        onStatus(`roto: downloading BiRefNet — ${mb(got)}/${mb(total)} MB (one-time)`,
          0.85 * (got / total)));

      onStatus('roto: starting model…', 0.9);
      // WebGPU first (the whole app requires WebGPU, but ort's EP can still
      // fail independently — e.g. an fp16 limit); wasm is the fallback.
      let sess;
      let ep = 'webgpu';
      try {
        sess = await ort.InferenceSession.create(buf, { executionProviders: ['webgpu'] });
      } catch (e) {
        console.warn('roto: WebGPU EP failed, falling back to wasm:', e);
        sess = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
        ep = 'wasm';
      }
      const inName = cfg.inputName ?? sess.inputNames[0];
      const outName = cfg.outputName ?? sess.outputNames[0];
      // Exports converted with keep_io_types keep float32 I/O; a stricter
      // conversion demands float16 tensors — detect it where metadata is
      // available (segmentFrame also self-corrects on a type-mismatch).
      const meta = sess.inputMetadata?.find?.((m) => m.name === inName)
        ?? sess.inputMetadata?.[0];
      const rt = { ort, sess, ep, inName, outName, inF16: meta?.type === 'float16' };
      onStatus(`roto: ready (${ep})`, null);
      return rt;
    } catch (e) {
      runtimePromise = null;   // allow retry
      throw e;
    }
  })();
  return runtimePromise;
}

export function rotoReady() {
  return !!runtimePromise;
}

/* ---- float16 shims — only exercised by exports without float32 I/O ---- */

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

function toF16(v) {
  _f32[0] = v;
  const x = _u32[0];
  const s = (x >> 16) & 0x8000;
  let e = (x >> 23) & 0xff;
  const m = x & 0x7fffff;
  if (e === 255) return s | 0x7c00 | (m ? 1 : 0);
  e = e - 127 + 15;
  if (e >= 31) return s | 0x7c00;
  if (e <= 0) return e < -10 ? s : s | (((m | 0x800000) >> (1 - e)) >> 13);
  return s | (e << 10) | (m >> 13);
}

function fromF16(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/**
 * Run BiRefNet on one frame/crop. `imageData` must be ImageData-like
 * {data, width, height}, normally ROTO_SIZE × ROTO_SIZE (the caller
 * squish-resizes — matching training, aspect is NOT preserved). Returns a
 * Float32Array of foreground probabilities (0..1), width*height long —
 * BiRefNet's matte is genuinely soft, so treat these as alpha, never
 * threshold them for display.
 */
export async function segmentFrame(rt, imageData) {
  const { data, width, height } = imageData;
  const px = width * height;
  const chw = new Float32Array(px * 3);
  for (let c = 0; c < 3; c++) {
    const mean = IMAGENET_MEAN[c];
    const std = IMAGENET_STD[c];
    const base = c * px;
    for (let i = 0; i < px; i++)
      chw[base + i] = (data[i * 4 + c] / 255 - mean) / std;
  }
  const dims = [1, 3, height, width];
  const T = rt.ort.Tensor;
  const makeInput = () => (rt.inF16
    ? new T('float16', Uint16Array.from(chw, toF16), dims)
    : new T('float32', chw, dims));
  let out;
  try {
    out = await rt.sess.run({ [rt.inName]: makeInput() });
  } catch (e) {
    // Metadata said float32 but the graph wants halves (or vice versa) —
    // flip once and retry; the corrected flag sticks for the session.
    if (!rt.inF16 && /float16|f16|fp16/i.test(String(e))) {
      rt.inF16 = true;
      out = await rt.sess.run({ [rt.inName]: makeInput() });
    } else {
      throw e;
    }
  }
  const raw = out[rt.outName].data;
  const probs = new Float32Array(px);
  const n = Math.min(px, raw.length);
  const half = raw instanceof Uint16Array;
  // Most exports emit logits (README applies sigmoid); a variant that
  // already emits probabilities never leaves [0, 1] — detect, don't guess.
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = half ? fromF16(raw[i]) : raw[i];
    probs[i] = v;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo < -0.001 || hi > 1.001)
    for (let i = 0; i < n; i++) probs[i] = 1 / (1 + Math.exp(-probs[i]));
  return probs;
}

/** White-on-transparent canvas from a probability matte (the "layer
 * matte" convention: coverage lives in alpha, colour is premult white). */
export function matteToCanvas(probs, w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < probs.length; i++) {
    const v = Math.max(0, Math.min(255, Math.round(probs[i] * 255)));
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = v;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

/**
 * Bounding box + area of the confident matte, in matte pixel space —
 * feeds the subject-box tracker (the box follows the matte from frame to
 * frame). Null box when the matte is empty at `thresh`.
 */
export function matteBbox(probs, w, h, thresh = 0.5) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  let area = 0;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (probs[y * w + x] >= thresh) {
        area++;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
  if (x1 < x0) return { box: null, area: 0 };
  return { box: [x0, y0, x1 + 1, y1 + 1], area };
}
