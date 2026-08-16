/*
 * slangfx studio — AI rotoscoping backend (model layer only).
 *
 * Wraps a SAM-family segmentation model (MobileSAM by default) running in
 * the browser through onnxruntime-web. The split follows samexporter's
 * export format, verified against the shipped default models:
 *
 *   encoder:  input_image  f32 [h, w, 3]  RGB 0..255, LONG SIDE == 1024
 *             → image_embeddings f32 [1, 256, 64, 64]
 *   decoder:  image_embeddings + point_coords [1,n,2] + point_labels [1,n]
 *             + mask_input [1,1,256,256] + has_mask_input [1]
 *             + orig_im_size [2] (h, w — any size; masks come back at it)
 *             → masks (logits, threshold at 0), iou_predictions,
 *               low_res_masks [1,1,256,256] (feed back as next mask_input)
 *
 * Point coords are in the RESIZED image's pixel space (long side 1024).
 * The temporal story lives in app.js — this module is stateless per call
 * apart from the cached runtime. Everything (runtime + models) loads on
 * demand from CDN URLs and is cached in the Cache API, so the studio
 * itself ships no ML weights; override the URLs via
 * localStorage['lowkey-studio.roto-config'] to self-host.
 */

export const ROTO_INPUT_LONG = 1024;   // SAM's fixed long-side input size
const LOW_RES = 256;                   // decoder's mask grid

const ORT_VERSION = '1.27.0';
const DEFAULTS = {
  ortUrl: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.bundle.min.mjs`,
  wasmPaths: `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`,
  encoderUrl: 'https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx',
  // The MULTI decoder returns 4 candidate masks (part / sub-object / whole
  // …), letting tracking pick the one consistent with the previous frame —
  // the single-mask head flips between interpretations frame to frame.
  decoderUrl: 'https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_multi.onnx',
  // Text-guided selection: an open-vocabulary detector turns "the girl in
  // the red dress" into a box prompt, re-detected on every analyzed frame.
  // Measured 2026-08-13: transformers.js 4.2.0 can't create a session for
  // ANY owl model (Where-node EP assignment bug, webgpu and wasm alike),
  // and OWLv2-base-patch16 runs ~22 s/frame even warm on a 4090 — while
  // OWL-ViT patch32 is 150 ms warm with cleaner boxes (OWLv2's square
  // padding also yields score-1.0 garbage boxes past the image edge).
  transformersUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.6/dist/transformers.min.js',
  detectorModel: 'Xenova/owlvit-base-patch32',
  detectorDtype: 'fp16',   // fp16 keeps the whole graph on WebGPU (int8 ops fall back to CPU)
};
const CACHE_NAME = 'lowkey-studio-roto-v1';

function rotoConfig() {
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('lowkey-studio.roto-config') ?? '{}') };
  } catch {
    return { ...DEFAULTS };
  }
}

/** Fetch `url` as an ArrayBuffer through the Cache API, reporting download
 * progress (models are tens of MB — the first use must not look hung). */
async function cachedFetch(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) return await hit.arrayBuffer();
  } catch { /* Cache API unavailable (private mode) — plain fetch below */ }
  const res = await fetch(url);
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
 * Load onnxruntime-web + the segmentation models (cached singleton).
 * `onStatus(text, frac)` mirrors setStatus. Resolves to the runtime
 * handle used by encodeFrame/decodeMask; rejects on network/WebGPU-and-
 * wasm failure (a retry re-attempts the load).
 */
export function loadRoto(onStatus = () => {}) {
  runtimePromise ??= (async () => {
    const cfg = rotoConfig();
    try {
      onStatus('roto: loading onnxruntime…', 0);
      const ort = (await import(/* webpackIgnore: true */ cfg.ortUrl)).default
        ?? globalThis.ort;
      ort.env.wasm.wasmPaths = cfg.wasmPaths;

      const mb = (n) => (n / 1048576).toFixed(0);
      const dl = (label, base, span) => (got, total) =>
        onStatus(`roto: downloading ${label} — ${mb(got)}/${mb(total)} MB (one-time)`,
          base + span * (got / total));
      const [encBuf, decBuf] = await Promise.all([
        cachedFetch(cfg.encoderUrl, dl('encoder', 0.0, 0.55)),
        cachedFetch(cfg.decoderUrl, dl('decoder', 0.55, 0.3)),
      ]);

      onStatus('roto: starting model…', 0.9);
      // WebGPU first (the whole app requires WebGPU, but ort's EP can still
      // fail independently — e.g. an fp16 limit); wasm is the fallback.
      const make = async (eps) => ({
        enc: await ort.InferenceSession.create(encBuf, { executionProviders: eps }),
        dec: await ort.InferenceSession.create(decBuf, { executionProviders: eps }),
        ep: eps[0],
      });
      let sessions;
      try {
        sessions = await make(['webgpu']);
      } catch (e) {
        console.warn('roto: WebGPU EP failed, falling back to wasm:', e);
        sessions = await make(['wasm']);
      }
      onStatus(`roto: ready (${sessions.ep})`, null);
      return { ort, ...sessions };
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

let detectorPromise = null;

/**
 * Load the open-vocabulary detector (cached singleton). transformers.js
 * handles the tokenizer + preprocessing; weights stream from the HF hub
 * into its own cache. WebGPU fp16 first, wasm int8 as the fallback.
 */
export function loadDetector(onStatus = () => {}) {
  detectorPromise ??= (async () => {
    const cfg = rotoConfig();
    try {
      onStatus('roto: loading text-search model…', 0);
      const tjs = await import(/* webpackIgnore: true */ cfg.transformersUrl);
      tjs.env.allowLocalModels = false;
      const mb = (n) => (n / 1048576).toFixed(0);
      const progress = (p) => {
        if (p.status === 'progress' && p.total)
          onStatus(`roto: downloading text-search model — ${mb(p.loaded)}/${mb(p.total)} MB (one-time)`,
            p.loaded / p.total);
      };
      let det;
      try {
        det = await tjs.pipeline('zero-shot-object-detection', cfg.detectorModel,
          { device: 'webgpu', dtype: cfg.detectorDtype, progress_callback: progress });
      } catch (e) {
        console.warn('roto: detector WebGPU load failed, falling back to wasm:', e);
        det = await tjs.pipeline('zero-shot-object-detection', cfg.detectorModel,
          { device: 'wasm', dtype: 'q8', progress_callback: progress });
      }
      onStatus('roto: text search ready', null);
      return { det, RawImage: tjs.RawImage };
    } catch (e) {
      detectorPromise = null;   // allow retry
      throw e;
    }
  })();
  return detectorPromise;
}

/**
 * Find `query` in a frame. `imageData` is the same resized frame the
 * encoder sees, so returned boxes are already in prompt space.
 * Returns [{score, box: [x0, y0, x1, y1]}], best first.
 */
export async function detectQuery(dt, imageData, query, { threshold = 0.08, topk = 8 } = {}) {
  const { width: w, height: h } = imageData;
  const img = new dt.RawImage(imageData.data, w, h, 4);
  const out = await dt.det(img, [query], { threshold, top_k: topk });
  return out
    .map((d) => {
      const raw = [d.box.xmin, d.box.ymin, d.box.xmax, d.box.ymax];
      const box = [
        Math.max(0, raw[0]), Math.max(0, raw[1]),
        Math.min(w, raw[2]), Math.min(h, raw[3]),
      ];
      const area = Math.max(0, raw[2] - raw[0]) * Math.max(0, raw[3] - raw[1]);
      const inArea = Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
      // A box mostly outside the frame is a padding/edge artifact (OWLv2's
      // square padding produces confident ones), never a usable prompt.
      return inArea > 0 && inArea >= area * 0.5 ? { score: d.score, box } : null;
    })
    .filter(Boolean)
    .sort((a, b) => b.score - a.score);
}

/** Bounding box of the positive low-res logits, in resized-image (1024)
 * space — the previous frame's footprint, used to keep a re-detected box
 * on the SAME object among lookalikes. Null when the mask is empty. */
export function lowResBbox(lowRes, validW, validH) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let y = 0; y < validH; y++)
    for (let x = 0; x < validW; x++)
      if (lowRes[y * LOW_RES + x] > 0) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
  if (x1 < x0) return null;
  const s = ROTO_INPUT_LONG / LOW_RES;
  return [x0 * s, y0 * s, (x1 + 1) * s, (y1 + 1) * s];
}

/* ---- temporal helpers -------------------------------------------------
 * The tracker's prompts (points, prior, boxes) all describe where the
 * object WAS. estimateShift measures how far it moved between two frames
 * so they can be carried along instead of landing on its trail. */

const SHIFT_Q = 4;   // motion search runs at quarter resolution
let _grayA = null;
let _grayB = null;

function _gray(img, buf, w4, h4) {
  const { data, width } = img;
  for (let y = 0; y < h4; y++)
    for (let x = 0; x < w4; x++) {
      const i = (y * SHIFT_Q * width + x * SHIFT_Q) * 4;
      buf[y * w4 + x] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
}

/**
 * Translation of the region `bbox` (image px) between two same-sized
 * frames, by SAD template matching at quarter res. Returns [dx, dy] in
 * image px — [0, 0] when the region is degenerate or matching is flat.
 */
export function estimateShift(prevImg, curImg, bbox, maxShift = 48) {
  const w4 = (prevImg.width / SHIFT_Q) | 0;
  const h4 = (prevImg.height / SHIFT_Q) | 0;
  if (w4 < 16 || h4 < 16) return [0, 0];
  if (_grayA?.length !== w4 * h4) {
    _grayA = new Float32Array(w4 * h4);
    _grayB = new Float32Array(w4 * h4);
  }
  _gray(prevImg, _grayA, w4, h4);
  _gray(curImg, _grayB, w4, h4);

  const clamp4 = (v, hi) => Math.max(0, Math.min(hi, v | 0));
  const tx0 = clamp4(bbox[0] / SHIFT_Q, w4 - 1);
  const ty0 = clamp4(bbox[1] / SHIFT_Q, h4 - 1);
  const tx1 = clamp4(bbox[2] / SHIFT_Q, w4);
  const ty1 = clamp4(bbox[3] / SHIFT_Q, h4);
  const tw = tx1 - tx0;
  const th = ty1 - ty0;
  if (tw < 4 || th < 4) return [0, 0];
  // Subsample big templates — ~2k samples is plenty for translation.
  const stride = Math.max(1, Math.round(Math.sqrt((tw * th) / 2048)));

  const sad = (dx, dy) => {
    let sum = 0;
    let cnt = 0;
    for (let y = ty0; y < ty1; y += stride) {
      const cy = y + dy;
      if (cy < 0 || cy >= h4) continue;
      for (let x = tx0; x < tx1; x += stride) {
        const cx = x + dx;
        if (cx < 0 || cx >= w4) continue;
        sum += Math.abs(_grayA[y * w4 + x] - _grayB[cy * w4 + cx]);
        cnt++;
      }
    }
    return cnt ? sum / cnt : Infinity;
  };

  const r = Math.max(1, Math.round(maxShift / SHIFT_Q));
  let bx = 0;
  let by = 0;
  let bs = Infinity;
  for (let dy = -r; dy <= r; dy += 2)          // coarse pass
    for (let dx = -r; dx <= r; dx += 2) {
      const s = sad(dx, dy);
      if (s < bs) { bs = s; bx = dx; by = dy; }
    }
  for (let dy = by - 1; dy <= by + 1; dy++)    // refine
    for (let dx = bx - 1; dx <= bx + 1; dx++) {
      const s = sad(dx, dy);
      if (s < bs) { bs = s; bx = dx; by = dy; }
    }
  return [bx * SHIFT_Q, by * SHIFT_Q];
}

/** Translate a 256×256 low-res logit grid by whole cells; exposed borders
 * fill with a confident-background logit. */
export function shiftGrid(lowRes, dxCells, dyCells) {
  if (!dxCells && !dyCells) return lowRes;
  const out = new Float32Array(LOW_RES * LOW_RES).fill(-8);
  for (let y = 0; y < LOW_RES; y++) {
    const sy = y - dyCells;
    if (sy < 0 || sy >= LOW_RES) continue;
    for (let x = 0; x < LOW_RES; x++) {
      const sx = x - dxCells;
      if (sx >= 0 && sx < LOW_RES) out[y * LOW_RES + x] = lowRes[sy * LOW_RES + sx];
    }
  }
  return out;
}

/**
 * Soft mask with temporal smoothing: in the uncertain band around the
 * boundary, blend this frame's logits with the previous frame's (motion-
 * shifted by dx/dy px), then ramp to alpha. Confident interior/exterior
 * pixels take the current value outright, so real motion never smears —
 * only the flickering boundary is steadied.
 */
export function temporalSoftMask(logits, prevLogits, w, h, dx, dy, { blend = 0.35, band = 6 } = {}) {
  if (!prevLogits) return softMask(logits, w * h);
  const mask = new Uint8ClampedArray(w * h);
  let area = 0;
  for (let y = 0; y < h; y++) {
    const py = y - dy;
    const rowOk = py >= 0 && py < h;
    for (let x = 0; x < w; x++) {
      let v = logits[y * w + x];
      if (rowOk && v > -band && v < band) {
        const px = x - dx;
        if (px >= 0 && px < w) v = v * (1 - blend) + prevLogits[py * w + px] * blend;
      }
      if (v >= MASK_RAMP) {
        mask[y * w + x] = 255;
        area++;
      } else if (v > -MASK_RAMP) {
        mask[y * w + x] = Math.round((v / MASK_RAMP * 0.5 + 0.5) * 255);
        if (v > 0) area++;
      }
    }
  }
  return { mask, area };
}

export function boxIoU(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy;
  const area = (r) => Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
  const union = area(a) + area(b) - inter;
  return union > 0 ? inter / union : 0;
}

/**
 * Encode one frame. `imageData` must be ImageData-like {data, width,
 * height} whose LONG side is ROTO_INPUT_LONG. Returns the embedding
 * tensor (opaque — feed to decodeMask).
 */
export async function encodeFrame(rt, imageData) {
  const { data, width, height } = imageData;
  const px = width * height;
  const hwc = new Float32Array(px * 3);
  for (let i = 0; i < px; i++) {
    hwc[i * 3] = data[i * 4];
    hwc[i * 3 + 1] = data[i * 4 + 1];
    hwc[i * 3 + 2] = data[i * 4 + 2];
  }
  const input = new rt.ort.Tensor('float32', hwc, [height, width, 3]);
  const out = await rt.enc.run({ input_image: input });
  return out.image_embeddings;
}

/**
 * Run the mask decoder against one frame's embedding.
 *  - points: [[x, y], …] in resized-image space (long side 1024)
 *  - labels: [1|0, …] matching points (1 = object, 0 = background)
 *  - box:    [x0, y0, x1, y1] in the same space (SAM box prompt), or null
 *  - prior:  Float32Array(256*256) low-res logits from a previous call, or null
 *  - prevLow: previous frame's low-res logits (motion-shifted) — when given,
 *    the candidate most consistent with it wins instead of the decoder's
 *    own favourite, which stops part/whole flip-flopping between frames
 *  - outW/outH: resolution the mask is returned at
 * Returns { mask, logits, lowRes, score, area } for the chosen candidate.
 */
export async function decodeMask(rt, emb, { points = null, labels = null, box = null, prior = null, prevLow = null, outW, outH }) {
  const pts = points ? points.map((p, i) => [p[0], p[1], labels[i]]) : [];
  // SAM's box prompt is two corner points labeled 2 (top-left) / 3 (bottom-right).
  if (box) pts.push([box[0], box[1], 2], [box[2], box[3], 3]);
  const n = pts.length;
  if (!n) throw new Error('decodeMask needs points or a box');
  const coords = new Float32Array(n * 2);
  const labs = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    coords[i * 2] = pts[i][0];
    coords[i * 2 + 1] = pts[i][1];
    labs[i] = pts[i][2];
  }
  const T = rt.ort.Tensor;
  const feeds = {
    image_embeddings: emb,
    point_coords: new T('float32', coords, [1, n, 2]),
    point_labels: new T('float32', labs, [1, n]),
    mask_input: new T('float32', prior ?? new Float32Array(LOW_RES * LOW_RES), [1, 1, LOW_RES, LOW_RES]),
    has_mask_input: new T('float32', new Float32Array([prior ? 1 : 0]), [1]),
    orig_im_size: new T('float32', new Float32Array([outH, outW]), [2]),
  };
  const out = await rt.dec.run(feeds);
  const nCand = out.iou_predictions.dims.at(-1);   // 1 (single head) or 4 (multi)
  const grid = LOW_RES * LOW_RES;
  const page = outW * outH;

  let best = 0;
  let bestScore = -Infinity;
  for (let c = 0; c < nCand; c++) {
    let s = out.iou_predictions.data[c];
    if (prevLow && nCand > 1) {
      // IoU of this candidate vs the previous frame's mask, on the grid.
      let inter = 0;
      let union = 0;
      const lo = out.low_res_masks.data;
      for (let i = 0; i < grid; i++) {
        const a = lo[c * grid + i] > 0;
        const b = prevLow[i] > 0;
        if (a && b) inter++;
        if (a || b) union++;
      }
      s *= 0.35 + 0.65 * (union ? inter / union : 0);
    }
    if (s > bestScore) { bestScore = s; best = c; }
  }

  const logits = new Float32Array(out.masks.data.slice(best * page, (best + 1) * page));
  const soft = softMask(logits, page);
  const lowRes = new Float32Array(out.low_res_masks.data.slice(best * grid, (best + 1) * grid));
  return { mask: soft.mask, logits, lowRes, score: out.iou_predictions.data[best], area: soft.area };
}

// SAM's decoder output is SOFT — bilinearly upsampled logits whose natural
// edge is a feathered ramp. Hard-thresholding at 0 turns sub-pixel boundary
// wobble into a popping 1-px contour, which reads as edge flicker. Mapping
// the logits through a ramp keeps the model's own anti-aliased edge.
const MASK_RAMP = 3;   // logits -RAMP..+RAMP → alpha 0..255

export function softMask(logits, n) {
  const mask = new Uint8ClampedArray(n);
  let area = 0;
  for (let i = 0; i < n; i++) {
    const v = logits[i];
    if (v >= MASK_RAMP) {
      mask[i] = 255;
      area++;
    } else if (v > -MASK_RAMP) {
      const a = Math.round((v / MASK_RAMP * 0.5 + 0.5) * 255);
      mask[i] = a;
      if (v > 0) area++;
    }
  }
  return { mask, area };
}

/**
 * Derive tracking prompts for the NEXT frame from this frame's low-res
 * logits: up to `max` confident interior points, spread apart (greedy
 * max-min-distance), returned in resized-image (1024) space. Empty array
 * = object lost this frame.
 * validW/validH: the portion of the 256 grid the image actually occupies
 * (the rest is SAM's padding — never sample there).
 */
export function trackPoints(lowRes, validW, validH, max = 3) {
  const cells = [];
  let best = -Infinity;
  for (let y = 0; y < validH; y++)
    for (let x = 0; x < validW; x++) {
      const v = lowRes[y * LOW_RES + x];
      if (v > 0) {
        cells.push([x, y, v]);
        if (v > best) best = v;
      }
    }
  if (!cells.length) return [];
  // Confident core only — border logits hover near 0 and drift off-object.
  const core = cells.filter(([, , v]) => v >= best * 0.5);
  const pool = core.length ? core : cells;
  const chosen = [pool.reduce((a, b) => (b[2] > a[2] ? b : a))];
  while (chosen.length < max && pool.length > chosen.length) {
    let cand = null;
    let candDist = -1;
    for (const c of pool) {
      let d = Infinity;
      for (const s of chosen) {
        const dd = (c[0] - s[0]) ** 2 + (c[1] - s[1]) ** 2;
        if (dd < d) d = dd;
      }
      if (d > candDist) { candDist = d; cand = c; }
    }
    if (!cand || candDist < 9) break;   // < 3 grid cells apart adds nothing
    chosen.push(cand);
  }
  const scale = ROTO_INPUT_LONG / LOW_RES;
  return chosen.map(([x, y]) => [(x + 0.5) * scale, (y + 0.5) * scale]);
}

/** White-on-transparent canvas from a decodeMask result (the "layer
 * matte" convention: coverage lives in alpha, colour is premult white). */
export function maskToCanvas(mask, w, h) {
  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < mask.length; i++) {
    const v = mask[i];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = v;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}
