/*
 * Lowkey Studio — Gaussian splat layers: file parsing + a WebGPU renderer.
 *
 * A splat scene is a cloud of anisotropic 3D gaussians (position, 3×3
 * covariance from scale + rotation, opacity, and a colour that may vary with
 * view direction through spherical harmonics). Studio treats one as a MEDIA
 * SOURCE whose picture depends on a camera: each splat clip renders the
 * scene through its own animated camera into its own texture, and from there
 * on the clip is an ordinary 2D layer (transform, masks, effects, blend).
 *
 * Formats (parseSplatFile):
 *   .ply    the 3DGS training output (x y z, f_dc_*, f_rest_*, opacity,
 *           scale_*, rot_*) — any property types, any order. A plain
 *           coloured point cloud (x y z red green blue) also loads, as
 *           small round splats.
 *   .splat  antimatter15's 32-byte packed format (no SH)
 *   .spz    Niantic's gzipped quantized format, versions 1–3
 *
 * Frame (SplatRenderer.render, all on the GPU, one command encoder):
 *   1. project  — one thread per splat: view transform, frustum/near cull,
 *                 2D covariance (EWA, +0.3 px² low-pass), its eigen axes,
 *                 SH colour for this view direction; writes a 32-byte
 *                 screen record + a depth sort key, and atomically counts
 *                 the survivors into the indirect draw's instance count
 *   2. sort     — bitonic sort of (key, index), keys = ~depth bits so the
 *                 far splats come first; culled splats carry 0xFFFFFFFF and
 *                 sink to the tail, past the instance count
 *   3. draw     — instanced quads, back to front, premultiplied "over" into
 *                 an rgba16float accumulator
 *   4. resolve  — un-premultiply into the clip's rgba8 texture (straight
 *                 alpha, which is what the compositor expects), optionally
 *                 over a solid background
 *
 * Coordinates: object space is the file's own. `upAxis` picks which object
 * axis points up in the world (COLMAP-trained .ply scenes are y-DOWN, so
 * their default is '-y'; .spz is RUB, '+y'). The camera is world-space,
 * y-up: yaw about +Y (0 looks down -Z), pitch about the camera's right
 * axis, roll about its forward axis. Camera space follows the 3DGS / OpenCV
 * convention (x right, y down, z forward).
 */

export const SPLAT_EXTS = /\.(ply|splat|spz)$/i;

const SH_C0 = 0.28209479177387814;
const SH_DIM = [0, 3, 8, 15];   // coefficients beyond DC per degree

/* ---- f32 → f16 --------------------------------------------------------- */

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);
function toHalf(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  let e = ((x >>> 23) & 0xff) - 127 + 15;
  let m = x & 0x7fffff;
  if (e >= 31) return sign | 0x7c00;                       // overflow → inf
  if (e <= 0) {                                            // subnormal / zero
    if (e < -10) return sign;
    m = (m | 0x800000) >> (1 - e);
    return sign | ((m + 0x1000) >> 13);
  }
  return sign | (e << 10) | ((m + 0x1000) >> 13);
}

/* ---- the in-memory scene ----------------------------------------------
 * {count, pos Float32Array(3N), cov Float32Array(6N) [xx xy xz yy yz zz],
 *  rgba Uint8Array(4N) (base colour incl. the SH DC term + opacity),
 *  shDegree, sh Uint16Array (f16, per splat SH_DIM[deg]*3 halves, rgb
 *  inner, padded to an even count), antialiased, upAxis} */

function makeScene(count, shDegree) {
  const shHalves = SH_DIM[shDegree] * 3;
  const shStride = shHalves + (shHalves & 1);
  return {
    count,
    pos: new Float32Array(count * 3),
    cov: new Float32Array(count * 6),
    rgba: new Uint8Array(count * 4),
    shDegree,
    shStride,
    sh: shStride ? new Uint16Array(count * shStride) : null,
    antialiased: false,
    upAxis: '-y',
  };
}

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

/** Covariance Σ = R S Sᵀ Rᵀ from linear scales + quaternion (w, x, y, z). */
function writeCov(out, i, sx, sy, sz, qw, qx, qy, qz) {
  const n = Math.hypot(qw, qx, qy, qz) || 1;
  qw /= n; qx /= n; qy /= n; qz /= n;
  const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
  const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
  const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
  // M = R·S (columns scaled), Σ = M Mᵀ
  const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
  const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
  const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;
  const o = i * 6;
  out[o] = m00 * m00 + m01 * m01 + m02 * m02;
  out[o + 1] = m00 * m10 + m01 * m11 + m02 * m12;
  out[o + 2] = m00 * m20 + m01 * m21 + m02 * m22;
  out[o + 3] = m10 * m10 + m11 * m11 + m12 * m12;
  out[o + 4] = m10 * m20 + m11 * m21 + m12 * m22;
  out[o + 5] = m20 * m20 + m21 * m21 + m22 * m22;
}

/* ---- .ply -------------------------------------------------------------- */

const PLY_TYPES = {
  char: [1, 'getInt8'], int8: [1, 'getInt8'],
  uchar: [1, 'getUint8'], uint8: [1, 'getUint8'],
  short: [2, 'getInt16'], int16: [2, 'getInt16'],
  ushort: [2, 'getUint16'], uint16: [2, 'getUint16'],
  int: [4, 'getInt32'], int32: [4, 'getInt32'],
  uint: [4, 'getUint32'], uint32: [4, 'getUint32'],
  float: [4, 'getFloat32'], float32: [4, 'getFloat32'],
  double: [8, 'getFloat64'], float64: [8, 'getFloat64'],
};

function parsePly(buf) {
  const bytes = new Uint8Array(buf);
  // Header is ASCII, terminated by "end_header\n".
  const probe = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 64 * 1024)));
  const endIdx = probe.indexOf('end_header');
  if (!probe.startsWith('ply') || endIdx < 0) throw new Error('not a PLY file');
  const nl = probe.indexOf('\n', endIdx);
  const bodyStart = new TextEncoder().encode(probe.slice(0, nl + 1)).length;
  const lines = probe.slice(0, endIdx).split(/\r?\n/);
  let format = '';
  const elements = [];
  let capture = null, captureUp = null;
  for (const line of lines) {
    const w = line.trim().split(/\s+/);
    // Written by the media server's splat task: the first capture frame's
    // camera and the capture's mean up vector (object space).
    if (w[0] === 'comment' && w[1] === 'lowkey_capture_camera' && w.length >= 12) {
      const n = w.slice(2, 12).map(Number);
      if (n.every(Number.isFinite)) capture = { pos: n.slice(0, 3), forward: n.slice(3, 6), up: n.slice(6, 9), fov: n[9] };
    } else if (w[0] === 'comment' && w[1] === 'lowkey_capture_up' && w.length >= 5) {
      const n = w.slice(2, 5).map(Number);
      if (n.every(Number.isFinite) && Math.hypot(...n) > 1e-6) captureUp = n;
    }
    if (w[0] === 'format') format = w[1];
    else if (w[0] === 'element') elements.push({ name: w[1], count: +w[2], props: [] });
    else if (w[0] === 'property') {
      const el = elements[elements.length - 1];
      if (w[1] === 'list') throw new Error(`PLY list properties aren't supported (${w[4]})`);
      const t = PLY_TYPES[w[1]];
      if (!t) throw new Error(`unknown PLY property type ${w[1]}`);
      el.props.push({ name: w[2], size: t[0], get: t[1] });
    }
  }
  if (format !== 'binary_little_endian') throw new Error(`PLY format ${format} isn't supported — export binary little-endian`);
  if (elements.some((e) => e.name === 'chunk'))
    throw new Error('compressed PLY (SuperSplat) isn\'t supported yet — export an uncompressed .ply or .spz');
  // Skip any elements before "vertex".
  let off = bodyStart;
  let vert = null;
  for (const el of elements) {
    let stride = 0;
    for (const p of el.props) { p.offset = stride; stride += p.size; }
    el.stride = stride;
    if (el.name === 'vertex') { vert = el; break; }
    off += el.count * stride;
  }
  if (!vert) throw new Error('PLY has no vertex element');
  const N = vert.count;
  const idx = Object.fromEntries(vert.props.map((p, i) => [p.name, i]));
  const has = (n) => n in idx;
  if (!has('x') || !has('y') || !has('z')) throw new Error('PLY vertices have no x/y/z');
  const isSplat = has('scale_0') && has('rot_0') && has('opacity');
  let rest = 0;
  while (has(`f_rest_${rest}`)) rest++;
  const deg = rest >= 45 ? 3 : rest >= 24 ? 2 : rest >= 9 ? 1 : 0;
  const scene = makeScene(N, isSplat ? deg : 0);

  // Read every vertex property as float. Fast path: all-float32 rows read
  // through one aligned Float32Array; otherwise DataView per property.
  const P = vert.props.length;
  const allFloat = vert.props.every((p) => p.get === 'getFloat32');
  const row = new Float32Array(P);
  let readRow;
  if (allFloat) {
    const body = new Float32Array(buf.slice(off, off + N * vert.stride));
    readRow = (i) => { const b = i * P; for (let k = 0; k < P; k++) row[k] = body[b + k]; };
  } else {
    const dv = new DataView(buf, off, N * vert.stride);
    readRow = (i) => {
      const b = i * vert.stride;
      for (let k = 0; k < P; k++) row[k] = dv[vert.props[k].get](b + vert.props[k].offset, true);
    };
  }
  const ix = idx.x, iy = idx.y, iz = idx.z;
  if (isSplat) {
    const dc = [idx.f_dc_0, idx.f_dc_1, idx.f_dc_2];
    const sc = [idx.scale_0, idx.scale_1, idx.scale_2];
    const rq = [idx.rot_0, idx.rot_1, idx.rot_2, idx.rot_3];
    const io = idx.opacity;
    const dim = SH_DIM[deg];
    const restIdx = [];
    // f_rest is channel-major: all R coefficients, then G, then B.
    for (let k = 0; k < dim; k++)
      for (let c = 0; c < 3; c++) restIdx.push(idx[`f_rest_${c * (rest / 3) + k}`]);
    for (let i = 0; i < N; i++) {
      readRow(i);
      scene.pos[i * 3] = row[ix]; scene.pos[i * 3 + 1] = row[iy]; scene.pos[i * 3 + 2] = row[iz];
      writeCov(scene.cov, i, Math.exp(row[sc[0]]), Math.exp(row[sc[1]]), Math.exp(row[sc[2]]),
        row[rq[0]], row[rq[1]], row[rq[2]], row[rq[3]]);
      const o = i * 4;
      scene.rgba[o] = clamp01(0.5 + SH_C0 * row[dc[0]]) * 255;
      scene.rgba[o + 1] = clamp01(0.5 + SH_C0 * row[dc[1]]) * 255;
      scene.rgba[o + 2] = clamp01(0.5 + SH_C0 * row[dc[2]]) * 255;
      scene.rgba[o + 3] = sigmoid(row[io]) * 255;
      if (scene.sh) {
        const so = i * scene.shStride;
        for (let k = 0; k < restIdx.length; k++) scene.sh[so + k] = toHalf(row[restIdx[k]]);
      }
    }
  } else {
    // A plain point cloud: round splats sized to the average point spacing.
    const cr = idx.red ?? idx.r, cg = idx.green ?? idx.g, cb = idx.blue ?? idx.b;
    const colorScale = cr != null && vert.props[cr].get.includes('Float') ? 255 : 1;
    let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < N; i++) {
      readRow(i);
      const x = row[ix], y = row[iy], z = row[iz];
      scene.pos[i * 3] = x; scene.pos[i * 3 + 1] = y; scene.pos[i * 3 + 2] = z;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      const o = i * 4;
      scene.rgba[o] = cr != null ? row[cr] * colorScale : 200;
      scene.rgba[o + 1] = cg != null ? row[cg] * colorScale : 200;
      scene.rgba[o + 2] = cb != null ? row[cb] * colorScale : 200;
      scene.rgba[o + 3] = 255;
    }
    const vol = Math.max(1e-9, (maxX - minX) * (maxY - minY) * (maxZ - minZ));
    const s = 0.5 * Math.cbrt(vol / Math.max(1, N));
    for (let i = 0; i < N; i++) writeCov(scene.cov, i, s, s, s, 1, 0, 0, 0);
  }
  if (capture) scene.capture = capture;
  if (captureUp) { scene.captureUp = captureUp; scene.upAxis = 'capture'; }
  return scene;
}

/* ---- .splat (antimatter15) --------------------------------------------- */

function parseDotSplat(buf) {
  const ROW = 32;
  const N = Math.floor(buf.byteLength / ROW);
  if (!N) throw new Error('empty .splat file');
  const f = new Float32Array(buf, 0, N * 8);
  const u = new Uint8Array(buf);
  const scene = makeScene(N, 0);
  for (let i = 0; i < N; i++) {
    const b = i * 8;
    scene.pos[i * 3] = f[b]; scene.pos[i * 3 + 1] = f[b + 1]; scene.pos[i * 3 + 2] = f[b + 2];
    const q = i * ROW + 28;
    writeCov(scene.cov, i, f[b + 3], f[b + 4], f[b + 5],
      (u[q] - 128) / 128, (u[q + 1] - 128) / 128, (u[q + 2] - 128) / 128, (u[q + 3] - 128) / 128);
    const c = i * ROW + 24;
    scene.rgba[i * 4] = u[c]; scene.rgba[i * 4 + 1] = u[c + 1];
    scene.rgba[i * 4 + 2] = u[c + 2]; scene.rgba[i * 4 + 3] = u[c + 3];
  }
  return scene;
}

/* ---- .spz (Niantic) ---------------------------------------------------- */

async function gunzip(buf) {
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

async function parseSpz(raw) {
  const buf = await gunzip(raw);
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x5053474e) throw new Error('not an SPZ file');
  const version = dv.getUint32(4, true);
  if (version < 1 || version > 3) throw new Error(`SPZ version ${version} isn't supported`);
  const N = dv.getUint32(8, true);
  const deg = dv.getUint8(12);
  const fracBits = dv.getUint8(13);
  const flags = dv.getUint8(14);
  if (deg > 3) throw new Error(`SPZ SH degree ${deg} isn't supported`);
  const u = new Uint8Array(buf);
  const scene = makeScene(N, deg);
  scene.antialiased = (flags & 1) === 1;
  scene.upAxis = '+y';   // SPZ is RUB (y up)
  let o = 16;
  // positions
  if (version === 1) {
    const h = new Uint16Array(buf.slice(o, o + N * 6));
    const halfToFloat = (x) => {
      const s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, m = x & 0x3ff;
      if (e === 0) return s * m * 2 ** -24;
      if (e === 31) return m ? NaN : s * Infinity;
      return s * (1 + m / 1024) * 2 ** (e - 15);
    };
    for (let i = 0; i < N * 3; i++) scene.pos[i] = halfToFloat(h[i]);
    o += N * 6;
  } else {
    const scale = 1 / (1 << fracBits);
    for (let i = 0; i < N * 3; i++) {
      let v = u[o] | (u[o + 1] << 8) | (u[o + 2] << 16);
      if (v & 0x800000) v |= 0xff000000;   // sign-extend 24 → 32
      scene.pos[i] = v * scale;
      o += 3;
    }
  }
  const aOff = o; o += N;
  const cOff = o; o += N * 3;
  const sOff = o; o += N * 3;
  const rOff = o; o += N * (version >= 3 ? 4 : 3);
  const shOff = o;
  const COLOR_SCALE = 0.15;
  const dim = SH_DIM[deg];
  for (let i = 0; i < N; i++) {
    scene.rgba[i * 4 + 3] = u[aOff + i];
    for (let c = 0; c < 3; c++) {
      const dc = (u[cOff + i * 3 + c] / 255 - 0.5) / COLOR_SCALE;
      scene.rgba[i * 4 + c] = clamp01(0.5 + SH_C0 * dc) * 255;
    }
    const sx = Math.exp(u[sOff + i * 3] / 16 - 10);
    const sy = Math.exp(u[sOff + i * 3 + 1] / 16 - 10);
    const sz = Math.exp(u[sOff + i * 3 + 2] / 16 - 10);
    let qx, qy, qz, qw;
    if (version >= 3) {
      // "smallest three": 2 bits pick the largest component, three 10-bit
      // signed magnitudes (scaled by 1/√2) carry the rest, highest index first.
      const b = rOff + i * 4;
      let comp = (u[b] | (u[b + 1] << 8) | (u[b + 2] << 16) | (u[b + 3] << 24)) >>> 0;
      const q = [0, 0, 0, 0];
      const iLargest = comp >>> 30;
      let sum = 0;
      for (let k = 3; k >= 0; k--) {
        if (k === iLargest) continue;
        const mag = comp & 511;
        const neg = (comp >>> 9) & 1;
        comp >>>= 10;
        q[k] = Math.SQRT1_2 * (mag / 511) * (neg ? -1 : 1);
        sum += q[k] * q[k];
      }
      q[iLargest] = Math.sqrt(Math.max(0, 1 - sum));
      [qx, qy, qz, qw] = q;
    } else {
      const b = rOff + i * 3;
      qx = u[b] / 127.5 - 1; qy = u[b + 1] / 127.5 - 1; qz = u[b + 2] / 127.5 - 1;
      qw = Math.sqrt(Math.max(0, 1 - qx * qx - qy * qy - qz * qz));
    }
    writeCov(scene.cov, i, sx, sy, sz, qw, qx, qy, qz);
    if (scene.sh) {
      // coefficient-major, rgb inner — already our layout
      const src = shOff + i * dim * 3, dst = i * scene.shStride;
      for (let k = 0; k < dim * 3; k++) scene.sh[dst + k] = toHalf((u[src + k] - 128) / 128);
    }
  }
  return scene;
}

/** Parse a splat file into the in-memory scene. */
export async function parseSplatFile(file) {
  const buf = await file.arrayBuffer();
  const name = file.name ?? '';
  if (/\.spz$/i.test(name)) return parseSpz(buf);
  if (/\.splat$/i.test(name)) return parseDotSplat(buf);
  return parsePly(buf);
}

/* ---- scene framing ------------------------------------------------------ */

/** An up-axis setting as the renderer takes it: 'capture' becomes the
 * scene's recorded up vector (falling back to COLMAP's −Y). */
export function resolveUp(upAxis, scene) {
  if (upAxis === 'capture') return scene?.captureUp ?? '-y';
  return upAxis;
}

/** Up-axis → world rotation (3×3, row-major) applied to object space.
 * `up` is a preset name, or an object-space vector that becomes world +Y. */
export function upAxisMatrix(up) {
  if (Array.isArray(up)) {
    // Rodrigues: the shortest rotation taking unit v onto +Y.
    const l = Math.hypot(up[0], up[1], up[2]) || 1;
    const v = [up[0] / l, up[1] / l, up[2] / l];
    const c = v[1];                       // v · Y
    if (c < -1 + 1e-9) return [1, 0, 0, 0, -1, 0, 0, 0, -1];
    const k = [-v[2], 0, v[0]];           // v × Y
    const f = 1 / (1 + c);
    return [
      1 - f * (k[1] * k[1] + k[2] * k[2]), -k[2] + f * k[0] * k[1], k[1] + f * k[0] * k[2],
      k[2] + f * k[0] * k[1], 1 - f * (k[0] * k[0] + k[2] * k[2]), -k[0] + f * k[1] * k[2],
      -k[1] + f * k[0] * k[2], k[0] + f * k[1] * k[2], 1 - f * (k[0] * k[0] + k[1] * k[1]),
    ];
  }
  switch (up) {
    case '+y': return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    case '+z': return [1, 0, 0, 0, 0, 1, 0, -1, 0];     // rotX(-90°): +Z → +Y
    case '-z': return [1, 0, 0, 0, 0, -1, 0, 1, 0];     // rotX(+90°): -Z → +Y
    case '-y':
    default: return [1, 0, 0, 0, -1, 0, 0, 0, -1];      // rotX(180°): -Y → +Y
  }
}

const mul3 = (m, x, y, z) => [
  m[0] * x + m[1] * y + m[2] * z,
  m[3] * x + m[4] * y + m[5] * z,
  m[6] * x + m[7] * y + m[8] * z,
];

/** The scene's subject, robustly: {center, radius} in world space. Center
 * is the per-axis median, radius the median distance from it — a capture's
 * background (sky, far walls, floaters) is most of its extent but not most
 * of its splats, so the medians land on what was actually photographed. */
export function sceneBounds(scene, upAxis = scene.upAxis) {
  const m = upAxisMatrix(resolveUp(upAxis, scene));
  const step = Math.max(1, Math.floor(scene.count / 50000));
  const xs = [], ys = [], zs = [];
  for (let i = 0; i < scene.count; i += step) {
    const p = mul3(m, scene.pos[i * 3], scene.pos[i * 3 + 1], scene.pos[i * 3 + 2]);
    if (!Number.isFinite(p[0] + p[1] + p[2])) continue;
    xs.push(p[0]); ys.push(p[1]); zs.push(p[2]);
  }
  if (!xs.length) return { center: [0, 0, 0], radius: 1 };
  const median = (a) => [...a].sort((p, r) => p - r)[a.length >> 1];
  const center = [median(xs), median(ys), median(zs)];
  const d = xs.map((x, i) => Math.hypot(x - center[0], ys[i] - center[1], zs[i] - center[2]));
  return { center, radius: Math.max(1e-3, median(d)) };
}

/* ---- camera ------------------------------------------------------------- */

const DEG = Math.PI / 180;

/** Camera basis from yaw/pitch/roll (degrees): {r, u, f} world vectors. */
export function cameraBasis(yaw, pitch, roll) {
  const cy = Math.cos(yaw * DEG), sy = Math.sin(yaw * DEG);
  const cp = Math.cos(pitch * DEG), sp = Math.sin(pitch * DEG);
  const f = [sy * cp, sp, -cy * cp];
  const r0 = [cy, 0, sy];
  // u0 = r0 × f
  const u0 = [
    r0[1] * f[2] - r0[2] * f[1],
    r0[2] * f[0] - r0[0] * f[2],
    r0[0] * f[1] - r0[1] * f[0],
  ];
  const cr = Math.cos(roll * DEG), sr = Math.sin(roll * DEG);
  const r = [0, 1, 2].map((k) => r0[k] * cr + u0[k] * sr);
  const u = [0, 1, 2].map((k) => u0[k] * cr - r0[k] * sr);
  return { r, u, f };
}

/** Yaw/pitch (degrees) that look along world direction d. */
export function lookAngles(d) {
  const len = Math.hypot(d[0], d[1], d[2]) || 1;
  const x = d[0] / len, y = d[1] / len, z = d[2] / len;
  return { yaw: Math.atan2(x, -z) / DEG, pitch: Math.asin(Math.max(-1, Math.min(1, y))) / DEG };
}

/** The capture's first-frame camera in world space (the view the video
 * started on), or null when the file doesn't carry one. */
export function captureCamera(scene, upAxis) {
  const c = scene.capture;
  if (!c) return null;
  const m = upAxisMatrix(resolveUp(upAxis, scene));
  const pos = mul3(m, ...c.pos), f = mul3(m, ...c.forward), u = mul3(m, ...c.up);
  const { yaw, pitch } = lookAngles(f);
  const b = cameraBasis(yaw, pitch, 0);
  const dot = (a, q) => a[0] * q[0] + a[1] * q[1] + a[2] * q[2];
  const roll = Math.atan2(-dot(u, b.r), dot(u, b.u)) / DEG;
  const { center, radius } = sceneBounds(scene, upAxis);
  const fl = Math.hypot(...f) || 1;
  const ahead = dot([center[0] - pos[0], center[1] - pos[1], center[2] - pos[2]], f) / fl;
  return {
    x: pos[0], y: pos[1], z: pos[2], yaw, pitch, roll,
    fov: Number.isFinite(c.fov) && c.fov > 1 ? c.fov : 50,
    dist: Math.max(radius * 0.25, ahead),
  };
}

/** A camera that frames the subject from slightly above, looking at its
 * centre: {x,y,z,yaw,pitch,roll,fov,dist} (dist = orbit distance). */
export function frameCamera(scene, upAxis, fov = 50) {
  const { center, radius } = sceneBounds(scene, upAxis);
  const dist = radius * 2;
  const pitch = -15;
  const { f } = cameraBasis(0, pitch, 0);
  return {
    x: center[0] - f[0] * dist, y: center[1] - f[1] * dist, z: center[2] - f[2] * dist,
    yaw: 0, pitch, roll: 0, fov, dist,
  };
}

/* ---- thumbnails --------------------------------------------------------- */

/** Paint a quick CPU preview (sampled points, painter-sorted) onto a 2D
 * canvas — for the media bin; the real picture comes from the GPU. */
export function paintSplatThumb(scene, upAxis, w, h) {
  const cv = document.createElement('canvas');
  cv.width = w; cv.height = h;
  const ctx = cv.getContext('2d');
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, w, h);
  const cam = captureCamera(scene, upAxis) ?? frameCamera(scene, upAxis);
  const m = upAxisMatrix(resolveUp(upAxis, scene));
  const { r, u, f } = cameraBasis(cam.yaw, cam.pitch, cam.roll);
  const fy = (h / 2) / Math.tan((cam.fov * DEG) / 2);
  const step = Math.max(1, Math.floor(scene.count / 30000));
  const pts = [];
  for (let i = 0; i < scene.count; i += step) {
    if (scene.rgba[i * 4 + 3] < 40) continue;
    const p = mul3(m, scene.pos[i * 3], scene.pos[i * 3 + 1], scene.pos[i * 3 + 2]);
    const dx = p[0] - cam.x, dy = p[1] - cam.y, dz = p[2] - cam.z;
    const z = f[0] * dx + f[1] * dy + f[2] * dz;
    if (z < 1e-3) continue;
    const x = (r[0] * dx + r[1] * dy + r[2] * dz) / z * fy + w / 2;
    const y = -(u[0] * dx + u[1] * dy + u[2] * dz) / z * fy + h / 2;
    if (x < 0 || y < 0 || x >= w || y >= h) continue;
    pts.push([z, x, y, i]);
  }
  pts.sort((a, b) => b[0] - a[0]);
  for (const [, x, y, i] of pts) {
    const o = i * 4;
    ctx.fillStyle = `rgb(${scene.rgba[o]},${scene.rgba[o + 1]},${scene.rgba[o + 2]})`;
    ctx.fillRect(x - 0.75, y - 0.75, 1.5, 1.5);
  }
  return cv;
}

/* ---- WGSL --------------------------------------------------------------- */

const COMMON_WGSL = /* wgsl */`
struct U {
  view0: vec4<f32>,   // object → camera rows (x right, y down, z forward)
  view1: vec4<f32>,
  view2: vec4<f32>,
  camObj: vec4<f32>,  // camera position in object space
  focal: vec4<f32>,   // fx, fy, width, height (px)
  params: vec4<f32>,  // size² multiplier, tan limit x, tan limit y, near
  counts: vec4<u32>,  // N, Npad, sh degree, flags (1 = antialiased)
};
`;

const PROJECT_WGSL = COMMON_WGSL + /* wgsl */`
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> splats: array<vec4<f32>>;   // 3 per splat
@group(0) @binding(2) var<storage, read> sh: array<u32>;
@group(0) @binding(3) var<storage, read_write> proj: array<vec4<u32>>; // 2 per splat
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals: array<u32>;
@group(0) @binding(6) var<storage, read_write> indirect: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> shStride: vec4<u32>;

const C1 = 0.4886025119029199;
const C2 = array<f32, 5>(1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396);
const C3 = array<f32, 7>(-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435);

fn shCoef(base: u32, k: u32) -> vec3<f32> {
  // halves k*3 .. k*3+2 of this splat's run
  let h0 = k * 3u;
  let a = unpack2x16float(sh[base + (h0 >> 1u)]);
  let b = unpack2x16float(sh[base + ((h0 + 2u) >> 1u)]);
  if ((h0 & 1u) == 0u) { return vec3<f32>(a.x, a.y, b.x); }
  return vec3<f32>(a.y, b.x, b.y);
}

fn cull(i: u32) {
  keys[i] = 0xffffffffu;
  vals[i] = i;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  let i = (wg.y * nwg.x + wg.x) * 256u + li;
  if (i >= u.counts.y) { return; }
  if (i >= u.counts.x) { cull(i); return; }
  let s0 = splats[i * 3u];
  let s1 = splats[i * 3u + 1u];
  let s2 = splats[i * 3u + 2u];
  let p = vec4<f32>(s0.xyz, 1.0);
  let t = vec3<f32>(dot(u.view0, p), dot(u.view1, p), dot(u.view2, p));
  if (t.z < u.params.w) { cull(i); return; }
  let color = unpack4x8unorm(bitcast<u32>(s0.w));
  var alpha = color.a;
  if (alpha < 1.0 / 255.0) { cull(i); return; }

  let fx = u.focal.x; let fy = u.focal.y;
  let W = u.focal.z; let H = u.focal.w;
  let px = fx * t.x / t.z + 0.5 * W;
  let py = fy * t.y / t.z + 0.5 * H;
  if (px < -0.3 * W || px > 1.3 * W || py < -0.3 * H || py > 1.3 * H) { cull(i); return; }

  // EWA: Σ' = J W Σ Wᵀ Jᵀ
  let sz = u.params.x;
  let cov = mat3x3<f32>(
    vec3<f32>(s1.x, s1.y, s1.z),
    vec3<f32>(s1.y, s1.w, s2.x),
    vec3<f32>(s1.z, s2.x, s2.y)) * sz;
  let txz = clamp(t.x / t.z, -u.params.y, u.params.y) * t.z;
  let tyz = clamp(t.y / t.z, -u.params.z, u.params.z) * t.z;
  let J = mat3x3<f32>(
    vec3<f32>(fx / t.z, 0.0, 0.0),
    vec3<f32>(0.0, fy / t.z, 0.0),
    vec3<f32>(-fx * txz / (t.z * t.z), -fy * tyz / (t.z * t.z), 0.0));
  // Wm columns = object axes in camera space (view rows transposed)
  let Wm = transpose(mat3x3<f32>(u.view0.xyz, u.view1.xyz, u.view2.xyz));
  let T = J * Wm;
  let c2 = T * cov * transpose(T);
  var a = c2[0][0]; let b = c2[0][1]; var c = c2[1][1];
  let detOrig = a * c - b * b;
  a += 0.3; c += 0.3;
  let det = a * c - b * b;
  if (det <= 0.0) { cull(i); return; }
  if ((u.counts.w & 1u) == 1u) { alpha *= sqrt(max(detOrig, 0.0) / det); }

  let mid = 0.5 * (a + c);
  let rad = length(vec2<f32>(0.5 * (a - c), b));
  let l1 = mid + rad;
  let l2 = max(mid - rad, 0.1);
  var dv = vec2<f32>(b, l1 - a);
  if (dot(dv, dv) < 1e-12) { dv = select(vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), a >= c); }
  dv = normalize(dv);
  let major = min(sqrt(2.0 * l1), 2048.0) * dv;
  let minor = min(sqrt(2.0 * l2), 2048.0) * vec2<f32>(dv.y, -dv.x);

  // View-dependent colour
  var rgb = color.rgb;
  let deg = u.counts.z;
  if (deg > 0u) {
    let d = normalize(s0.xyz - u.camObj.xyz);
    let x = d.x; let y = d.y; let z = d.z;
    let base = i * shStride.x;
    var r = -C1 * y * shCoef(base, 0u) + C1 * z * shCoef(base, 1u) - C1 * x * shCoef(base, 2u);
    if (deg > 1u) {
      let xx = x * x; let yy = y * y; let zz = z * z;
      r += C2[0] * x * y * shCoef(base, 3u)
         + C2[1] * y * z * shCoef(base, 4u)
         + C2[2] * (2.0 * zz - xx - yy) * shCoef(base, 5u)
         + C2[3] * x * z * shCoef(base, 6u)
         + C2[4] * (xx - yy) * shCoef(base, 7u);
      if (deg > 2u) {
        r += C3[0] * y * (3.0 * xx - yy) * shCoef(base, 8u)
           + C3[1] * x * y * z * shCoef(base, 9u)
           + C3[2] * y * (4.0 * zz - xx - yy) * shCoef(base, 10u)
           + C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shCoef(base, 11u)
           + C3[4] * x * (4.0 * zz - xx - yy) * shCoef(base, 12u)
           + C3[5] * z * (xx - yy) * shCoef(base, 13u)
           + C3[6] * x * (xx - 3.0 * yy) * shCoef(base, 14u);
      }
    }
    rgb = max(rgb + r, vec3<f32>(0.0));
  }

  // NDC (y up)
  let ndc = vec2<f32>(px / W * 2.0 - 1.0, 1.0 - py / H * 2.0);
  let toNdc = vec2<f32>(2.0 / W, -2.0 / H);
  // centre stays f32 (f16 is ~2 px off at 4K); the axes are relative
  proj[i * 2u] = vec4<u32>(bitcast<u32>(ndc.x), bitcast<u32>(ndc.y),
                           pack2x16float(major * toNdc), pack2x16float(minor * toNdc));
  proj[i * 2u + 1u] = vec4<u32>(pack2x16float(rgb.rg), pack2x16float(vec2<f32>(rgb.b, alpha)), 0u, 0u);
  keys[i] = 0xffffffffu - bitcast<u32>(t.z);
  vals[i] = i;
  atomicAdd(&indirect[1], 1u);
}
`;

/* Bitonic sort over (keys, vals), ascending by key. One workgroup owns a
 * 512-element block in shared memory for every j < 512; larger j strides
 * run as global compare-swap passes. */
const SORT_WGSL = /* wgsl */`
struct S { k: u32, j: u32, mode: u32, pad: u32 };
@group(0) @binding(0) var<storage, read_write> keys: array<u32>;
@group(0) @binding(1) var<storage, read_write> vals: array<u32>;
@group(0) @binding(2) var<uniform> s: S;

var<workgroup> lk: array<u32, 512>;
var<workgroup> lv: array<u32, 512>;

fn wgIndex(wg: vec3<u32>, nwg: vec3<u32>) -> u32 { return wg.y * nwg.x + wg.x; }

@compute @workgroup_size(256)
fn global_pass(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
               @builtin(local_invocation_index) li: u32) {
  let t = wgIndex(wg, nwg) * 256u + li;
  let j = s.j; let k = s.k;
  let i = 2u * j * (t / j) + (t % j);
  let l = i + j;
  let up = (i & k) == 0u;
  let ki = keys[i]; let kl = keys[l];
  if ((ki > kl) == up) {
    keys[i] = kl; keys[l] = ki;
    let vi = vals[i]; vals[i] = vals[l]; vals[l] = vi;
  }
}

// mode 0: full local sort (k = 2 … 512); mode 1: finish stage s.k for j < 512
@compute @workgroup_size(256)
fn local_pass(@builtin(workgroup_id) wg: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>,
              @builtin(local_invocation_index) li: u32) {
  let base = wgIndex(wg, nwg) * 512u;
  lk[li] = keys[base + li]; lk[li + 256u] = keys[base + li + 256u];
  lv[li] = vals[base + li]; lv[li + 256u] = vals[base + li + 256u];
  workgroupBarrier();
  var k = select(2u, s.k, s.mode == 1u);
  let kEnd = select(512u, s.k, s.mode == 1u);
  loop {
    var j = select(k >> 1u, 256u, s.mode == 1u);
    loop {
      let i = 2u * j * (li / j) + (li % j);
      let l = i + j;
      let up = ((base + i) & k) == 0u;
      let ki = lk[i]; let kl = lk[l];
      if ((ki > kl) == up) {
        lk[i] = kl; lk[l] = ki;
        let vi = lv[i]; lv[i] = lv[l]; lv[l] = vi;
      }
      workgroupBarrier();
      if (j == 1u) { break; }
      j = j >> 1u;
    }
    if (k >= kEnd) { break; }
    k = k << 1u;
  }
  keys[base + li] = lk[li]; keys[base + li + 256u] = lk[li + 256u];
  vals[base + li] = lv[li]; vals[base + li + 256u] = lv[li + 256u];
}
`;

const DRAW_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage, read> proj: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> vals: array<u32>;

struct V {
  @builtin(position) pos: vec4<f32>,
  @location(0) local: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> V {
  let idx = vals[ii];
  let a = proj[idx * 2u];
  let b = proj[idx * 2u + 1u];
  let center = vec2<f32>(bitcast<f32>(a.x), bitcast<f32>(a.y));
  let major = unpack2x16float(a.z);
  let minor = unpack2x16float(a.w);
  let corner = vec2<f32>(select(-2.0, 2.0, (vi & 1u) == 1u), select(-2.0, 2.0, (vi & 2u) == 2u));
  var o: V;
  o.pos = vec4<f32>(center + corner.x * major + corner.y * minor, 0.0, 1.0);
  o.local = corner;
  let rg = unpack2x16float(b.x);
  let ba = unpack2x16float(b.y);
  o.color = vec4<f32>(rg, ba);
  return o;
}

@fragment
fn fs(i: V) -> @location(0) vec4<f32> {
  let A = -dot(i.local, i.local);
  if (A < -4.0) { discard; }
  let alpha = min(0.99, exp(A) * i.color.a);
  if (alpha < 1.0 / 255.0) { discard; }
  return vec4<f32>(i.color.rgb * alpha, alpha);
}
`;

const RESOLVE_WGSL = /* wgsl */`
@group(0) @binding(0) var accum: texture_2d<f32>;
@group(0) @binding(1) var<uniform> bg: vec4<f32>;   // rgb, a = 1 → fill

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  let c = textureLoad(accum, vec2<i32>(fc.xy), 0);
  if (bg.a > 0.5) { return vec4<f32>(c.rgb + bg.rgb * (1.0 - c.a), 1.0); }
  if (c.a <= 1e-4) { return vec4<f32>(0.0); }
  return vec4<f32>(clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0)), c.a);
}
`;

/* ---- GPU ---------------------------------------------------------------- */

const SU = { STORAGE: 128, COPY_DST: 8, COPY_SRC: 4, UNIFORM: 64, INDIRECT: 256 };
const TU = { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, RENDER_ATTACHMENT: 16 };

/** Upload a parsed scene once per device: returns the GPU half of an asset
 * (shared by every clip that shows this splat). Throws when the scene
 * exceeds the device's buffer limits. */
export function uploadScene(device, scene) {
  const N = scene.count;
  const maxBind = device.limits.maxStorageBufferBindingSize;
  const splatBytes = N * 48;
  if (splatBytes > maxBind) throw new Error(`${N.toLocaleString()} splats is more than this GPU can bind (${Math.floor(maxBind / 48).toLocaleString()} max)`);
  const data = new Float32Array(N * 12);
  const u32 = new Uint32Array(data.buffer);
  for (let i = 0; i < N; i++) {
    const o = i * 12;
    data[o] = scene.pos[i * 3]; data[o + 1] = scene.pos[i * 3 + 1]; data[o + 2] = scene.pos[i * 3 + 2];
    u32[o + 3] = scene.rgba[i * 4] | (scene.rgba[i * 4 + 1] << 8)
      | (scene.rgba[i * 4 + 2] << 16) | (scene.rgba[i * 4 + 3] << 24);
    for (let k = 0; k < 6; k++) data[o + 4 + k] = scene.cov[i * 6 + k];
  }
  const splats = device.createBuffer({ label: 'splat data', size: data.byteLength, usage: SU.STORAGE | SU.COPY_DST });
  device.queue.writeBuffer(splats, 0, data);
  // SH: drop degrees until it fits the binding limit.
  let deg = scene.shDegree;
  let shWords = 0;
  while (deg > 0) {
    const halves = SH_DIM[deg] * 3;
    shWords = (halves + (halves & 1)) / 2;
    if (N * shWords * 4 <= maxBind) break;
    deg--;
  }
  let sh;
  if (deg > 0) {
    const words = new Uint32Array(N * shWords);
    const srcStride = scene.shStride;
    const halves = SH_DIM[deg] * 3;
    const w16 = new Uint16Array(words.buffer);
    for (let i = 0; i < N; i++)
      for (let k = 0; k < halves; k++) w16[i * shWords * 2 + k] = scene.sh[i * srcStride + k];
    sh = device.createBuffer({ label: 'splat sh', size: words.byteLength, usage: SU.STORAGE | SU.COPY_DST });
    device.queue.writeBuffer(sh, 0, words);
  } else {
    sh = device.createBuffer({ label: 'splat sh (none)', size: 16, usage: SU.STORAGE });
    shWords = 0;
  }
  return { count: N, splats, sh, shDegree: deg, shWords, antialiased: scene.antialiased,
    destroy() { splats.destroy(); sh.destroy(); } };
}

/** Device-wide pipelines, built once. */
class SplatPipelines {
  constructor(device) {
    this.device = device;
    const pm = device.createShaderModule({ label: 'splat project', code: PROJECT_WGSL });
    this.project = device.createComputePipeline({ label: 'splat project', layout: 'auto', compute: { module: pm, entryPoint: 'main' } });
    const sm = device.createShaderModule({ label: 'splat sort', code: SORT_WGSL });
    this.sortLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
      ],
    });
    const sl = device.createPipelineLayout({ bindGroupLayouts: [this.sortLayout] });
    this.sortGlobal = device.createComputePipeline({ label: 'splat sort global', layout: sl, compute: { module: sm, entryPoint: 'global_pass' } });
    this.sortLocal = device.createComputePipeline({ label: 'splat sort local', layout: sl, compute: { module: sm, entryPoint: 'local_pass' } });
    const dm = device.createShaderModule({ label: 'splat draw', code: DRAW_WGSL });
    this.draw = device.createRenderPipeline({
      label: 'splat draw', layout: 'auto',
      vertex: { module: dm, entryPoint: 'vs' },
      fragment: {
        module: dm, entryPoint: 'fs',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip' },
    });
    const rm = device.createShaderModule({ label: 'splat resolve', code: RESOLVE_WGSL });
    this.resolve = device.createRenderPipeline({
      label: 'splat resolve', layout: 'auto',
      vertex: { module: rm, entryPoint: 'vs' },
      fragment: { module: rm, entryPoint: 'fs', targets: [{ format: 'rgba8unorm' }] },
      primitive: { topology: 'triangle-list' },
    });
    // Sort pass parameters, one 256-byte slot per pass (dynamic offsets),
    // grown on demand to the largest padded count seen.
    this.sortParams = null;
    this.sortSlots = new Map();   // `${k}|${j}|${mode}` -> slot
  }

  /** Sort schedule for a padded count: [{pipe, k, j, mode}] */
  schedule(Npad) {
    const passes = [{ local: true, k: 512, j: 0, mode: 0 }];
    for (let k = 1024; k <= Npad; k <<= 1) {
      for (let j = k >> 1; j >= 512; j >>= 1) passes.push({ local: false, k, j, mode: 0 });
      passes.push({ local: true, k, j: 0, mode: 1 });
    }
    const need = passes.length;
    if (!this.sortParams || this.sortParams.slots < need) {
      this.sortParams?.buffer.destroy();
      const slots = Math.max(need, 64);
      const data = new Uint32Array(slots * 64);
      const buffer = this.device.createBuffer({ label: 'splat sort params', size: data.byteLength, usage: SU.UNIFORM | SU.COPY_DST });
      this.sortParams = { buffer, slots, data };
      this.sortSlots.clear();
    }
    let dirty = false;
    for (const p of passes) {
      const key = `${p.k}|${p.j}|${p.mode}`;
      let slot = this.sortSlots.get(key);
      if (slot == null) {
        slot = this.sortSlots.size;
        this.sortSlots.set(key, slot);
        this.sortParams.data.set([p.k, p.j, p.mode, 0], slot * 64);
        dirty = true;
      }
      p.offset = slot * 256;
    }
    if (dirty) this.device.queue.writeBuffer(this.sortParams.buffer, 0, this.sortParams.data);
    return passes;
  }
}

const pipelineCache = new WeakMap();
function pipelinesFor(device) {
  let p = pipelineCache.get(device);
  if (!p) { p = new SplatPipelines(device); pipelineCache.set(device, p); }
  return p;
}

function dispatch2D(pass, groups) {
  const x = Math.min(groups, 32768);
  pass.dispatchWorkgroups(x, Math.ceil(groups / x));
}

/**
 * One clip's view of a splat scene: its own sort buffers + output texture.
 * render() re-runs only when the camera or settings change.
 */
export class SplatRenderer {
  constructor(device, gpuScene) {
    this.device = device;
    this.pipes = pipelinesFor(device);
    this.scene = gpuScene;
    const N = gpuScene.count;
    this.Npad = Math.max(512, 2 ** Math.ceil(Math.log2(N)));
    this.uniform = device.createBuffer({ label: 'splat uniforms', size: 112, usage: SU.UNIFORM | SU.COPY_DST });
    this.shUniform = device.createBuffer({ label: 'splat sh stride', size: 16, usage: SU.UNIFORM | SU.COPY_DST });
    device.queue.writeBuffer(this.shUniform, 0, new Uint32Array([gpuScene.shWords, 0, 0, 0]));
    this.proj = device.createBuffer({ label: 'splat proj', size: N * 32, usage: SU.STORAGE });
    this.keys = device.createBuffer({ label: 'splat keys', size: this.Npad * 4, usage: SU.STORAGE });
    this.vals = device.createBuffer({ label: 'splat vals', size: this.Npad * 4, usage: SU.STORAGE });
    this.indirect = device.createBuffer({ label: 'splat indirect', size: 16, usage: SU.STORAGE | SU.INDIRECT | SU.COPY_DST });
    device.queue.writeBuffer(this.indirect, 0, new Uint32Array([4, 0, 0, 0]));
    this.bgUniform = device.createBuffer({ label: 'splat bg', size: 16, usage: SU.UNIFORM | SU.COPY_DST });
    this.projectBG = device.createBindGroup({
      layout: this.pipes.project.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: gpuScene.splats } },
        { binding: 2, resource: { buffer: gpuScene.sh } },
        { binding: 3, resource: { buffer: this.proj } },
        { binding: 4, resource: { buffer: this.keys } },
        { binding: 5, resource: { buffer: this.vals } },
        { binding: 6, resource: { buffer: this.indirect } },
        { binding: 7, resource: { buffer: this.shUniform } },
      ],
    });
    this.sortBG = null;   // built lazily (needs the params buffer)
    this.drawBG = device.createBindGroup({
      layout: this.pipes.draw.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.proj } },
        { binding: 1, resource: { buffer: this.vals } },
      ],
    });
    this.w = 0; this.h = 0;
    this.key = '';
  }

  _ensureTargets(w, h) {
    if (w === this.w && h === this.h) return;
    this.accum?.destroy();
    this.texture?.destroy();
    this.w = w; this.h = h;
    this.accum = this.device.createTexture({
      label: 'splat accum', size: [w, h], format: 'rgba16float',
      usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING,
    });
    this.texture = this.device.createTexture({
      label: 'splat out', size: [w, h], format: 'rgba8unorm',
      usage: TU.RENDER_ATTACHMENT | TU.TEXTURE_BINDING | TU.COPY_SRC,
    });
    this.view = this.texture.createView();
    this.resolveBG = this.device.createBindGroup({
      layout: this.pipes.resolve.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.accum.createView() },
        { binding: 1, resource: { buffer: this.bgUniform } },
      ],
    });
    this.key = '';
  }

  /**
   * Render the scene through `cam` into this.view.
   * @param cam {x,y,z,yaw,pitch,roll,fov,size} — world-space camera, fov
   *            vertical degrees, size = splat size multiplier (1 = as trained)
   * @param opts {upAxis, background: null | [r,g,b] 0..1, w, h}
   * @returns true when it rendered, false when the cached frame still holds
   */
  render(encoder, cam, { upAxis = '-y', background = null, w, h }) {
    this._ensureTargets(w, h);
    const key = JSON.stringify([cam, upAxis, background, w, h]);
    if (key === this.key) return false;
    this.key = key;
    const dev = this.device;
    const M = upAxisMatrix(upAxis);
    const { r, u: up, f } = cameraBasis(cam.yaw, cam.pitch, cam.roll);
    const dn = [-up[0], -up[1], -up[2]];   // camera y points down
    const C = [cam.x, cam.y, cam.z];
    // row' = Mᵀ row (object-space row), translation = -row·C
    const rowObj = (v) => [
      M[0] * v[0] + M[3] * v[1] + M[6] * v[2],
      M[1] * v[0] + M[4] * v[1] + M[7] * v[2],
      M[2] * v[0] + M[5] * v[1] + M[8] * v[2],
    ];
    const dotC = (v) => v[0] * C[0] + v[1] * C[1] + v[2] * C[2];
    const camObj = rowObj(C);   // Mᵀ C
    const fovRad = Math.max(1, Math.min(170, cam.fov)) * DEG;
    const fy = (h / 2) / Math.tan(fovRad / 2);
    const fx = fy;
    const tanY = Math.tan(fovRad / 2), tanX = tanY * (w / h);
    const ub = new ArrayBuffer(112);
    const fv = new Float32Array(ub);
    const uv = new Uint32Array(ub);
    fv.set([...rowObj(r), -dotC(r)], 0);
    fv.set([...rowObj(dn), -dotC(dn)], 4);
    fv.set([...rowObj(f), -dotC(f)], 8);
    fv.set([...camObj, 0], 12);
    fv.set([fx, fy, w, h], 16);
    const size = Math.max(0, cam.size ?? 1);
    fv.set([size * size, 1.3 * tanX, 1.3 * tanY, 0.01], 20);
    uv.set([this.scene.count, this.Npad, this.scene.shDegree, this.scene.antialiased ? 1 : 0], 24);
    dev.queue.writeBuffer(this.uniform, 0, ub);
    dev.queue.writeBuffer(this.bgUniform, 0, new Float32Array(background
      ? [background[0], background[1], background[2], 1] : [0, 0, 0, 0]));

    // 1. project
    encoder.clearBuffer(this.indirect, 4, 4);
    let pass = encoder.beginComputePass({ label: 'splat project' });
    pass.setPipeline(this.pipes.project);
    pass.setBindGroup(0, this.projectBG);
    dispatch2D(pass, this.Npad / 256);
    pass.end();

    // 2. sort
    const passes = this.pipes.schedule(this.Npad);
    if (!this.sortBG || this.sortBG.params !== this.pipes.sortParams.buffer) {
      this.sortBG = dev.createBindGroup({
        layout: this.pipes.sortLayout,
        entries: [
          { binding: 0, resource: { buffer: this.keys } },
          { binding: 1, resource: { buffer: this.vals } },
          { binding: 2, resource: { buffer: this.pipes.sortParams.buffer, size: 16 } },
        ],
      });
      this.sortBG.params = this.pipes.sortParams.buffer;
    }
    pass = encoder.beginComputePass({ label: 'splat sort' });
    const groups = this.Npad / 512;
    for (const p of passes) {
      pass.setPipeline(p.local ? this.pipes.sortLocal : this.pipes.sortGlobal);
      pass.setBindGroup(0, this.sortBG, [p.offset]);
      dispatch2D(pass, groups);
    }
    pass.end();

    // 3. draw
    let rp = encoder.beginRenderPass({
      label: 'splat draw',
      colorAttachments: [{ view: this.accum.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    rp.setPipeline(this.pipes.draw);
    rp.setBindGroup(0, this.drawBG);
    rp.drawIndirect(this.indirect, 0);
    rp.end();

    // 4. resolve to straight alpha
    rp = encoder.beginRenderPass({
      label: 'splat resolve',
      colorAttachments: [{ view: this.view, loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
    });
    rp.setPipeline(this.pipes.resolve);
    rp.setBindGroup(0, this.resolveBG);
    rp.draw(3);
    rp.end();
    return true;
  }

  destroy() {
    for (const b of [this.uniform, this.shUniform, this.proj, this.keys, this.vals, this.indirect, this.bgUniform]) b.destroy();
    this.accum?.destroy();
    this.texture?.destroy();
  }
}
