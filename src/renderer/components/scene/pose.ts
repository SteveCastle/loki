// The mannequin: a bipedal human rig on a paint layer, kept as the 18 COCO keypoints OpenPose uses and drawn the way
// pose-conditioned image models were trained to read a pose (OpenPose's own picture: every limb and joint in its fixed
// colour, on black when it is a guide; see resolveLayerRefs in media-server/tasks/scene.go). Posing is forward
// kinematics: dragging a joint swings it and everything below it round its parent, so limbs keep their length.

/** A posed rig. kp: x, y of each joint (POSE_JOINTS order) in frame px; w: limb width, frame px. */
export interface PoseOp {
  t: 'pose';
  kp: number[];
  w: number;
}

/** OpenPose's COCO-18 joints, in its order. The figure's right is on the picture's left when it faces the viewer. */
export const POSE_JOINTS = [
  'nose',
  'neck',
  'right shoulder',
  'right elbow',
  'right wrist',
  'left shoulder',
  'left elbow',
  'left wrist',
  'right hip',
  'right knee',
  'right ankle',
  'left hip',
  'left knee',
  'left ankle',
  'right eye',
  'left eye',
  'right ear',
  'left ear',
];

/** OpenPose's limbs (joint pairs); limb i is drawn in POSE_COLORS[i]. */
export const POSE_LIMBS: Array<[number, number]> = [
  [1, 2],
  [1, 5],
  [2, 3],
  [3, 4],
  [5, 6],
  [6, 7],
  [1, 8],
  [8, 9],
  [9, 10],
  [1, 11],
  [11, 12],
  [12, 13],
  [1, 0],
  [0, 14],
  [14, 16],
  [0, 15],
  [15, 17],
];

/** OpenPose's colours: joint i and limb i. */
export const POSE_COLORS = [
  [255, 0, 0],
  [255, 85, 0],
  [255, 170, 0],
  [255, 255, 0],
  [170, 255, 0],
  [85, 255, 0],
  [0, 255, 0],
  [0, 255, 85],
  [0, 255, 170],
  [0, 255, 255],
  [0, 170, 255],
  [0, 85, 255],
  [0, 0, 255],
  [85, 0, 255],
  [170, 0, 255],
  [255, 0, 255],
  [255, 0, 170],
  [255, 0, 85],
].map(([r, g, b]) => `rgb(${r}, ${g}, ${b})`);

/** Each joint's parent in the rig; the neck is the root. */
export const POSE_PARENT = [1, -1, 1, 2, 3, 1, 5, 6, 1, 8, 9, 1, 11, 12, 0, 0, 14, 15];

export const POSE_ROOT = 1;

// Standing, facing the viewer, arms a little out: each joint in figure heights from the neck. The head's top is
// REST_TOP above the neck and the ankles REST_FEET below it.
const REST = [
  0, -0.1, 0, 0, -0.11, 0.01, -0.15, 0.17, -0.17, 0.32, 0.11, 0.01, 0.15, 0.17, 0.17, 0.32, -0.07, 0.33, -0.08, 0.58, -0.08, 0.83, 0.07, 0.33, 0.08,
  0.58, 0.08, 0.83, -0.025, -0.125, 0.025, -0.125, -0.06, -0.11, 0.06, -0.11,
];
const REST_TOP = -0.17;
const REST_FEET = 0.83;

const round = (v: number) => Math.round(v * 10) / 10;

/** A standing rig whose head's top is at (cx, top) and whose feet are `height` below it. */
export function restPose(cx: number, top: number, height: number): PoseOp {
  const h = height / (REST_FEET - REST_TOP);
  const neckY = top - REST_TOP * h;
  return { t: 'pose', kp: REST.map((v, i) => round(i % 2 ? neckY + v * h : cx + v * h)), w: Math.max(2, Math.round(height * 0.012)) };
}

// The head is the same in every preset: nose, then (after the 14 body joints) eyes and ears.
const HEAD = { nose: [0, -0.1], face: [-0.025, -0.125, 0.025, -0.125, -0.06, -0.11, 0.06, -0.11] };

/** A preset's body as [shoulders..ankles] pairs (joints 2-13), in REST's units; the neck is (0, 0). */
function body(...pts: number[]): number[] {
  return [...HEAD.nose, 0, 0, ...pts, ...HEAD.face];
}

/**
 * Ready poses, facing the viewer (the figure's right on the picture's left), in REST's units. Applying one keeps the
 * figure where it is and its size.
 */
export const POSE_PRESETS: Array<{ id: string; label: string; kp: number[] }> = [
  { id: 'stand', label: 'Standing', kp: REST },
  {
    id: 'tpose',
    label: 'T-pose',
    kp: body(-0.11, 0.01, -0.27, 0.01, -0.43, 0.01, 0.11, 0.01, 0.27, 0.01, 0.43, 0.01, -0.07, 0.33, -0.07, 0.58, -0.07, 0.83, 0.07, 0.33, 0.07, 0.58, 0.07, 0.83),
  },
  {
    id: 'cheer',
    label: 'Arms up',
    kp: body(-0.11, 0.01, -0.2, -0.12, -0.24, -0.28, 0.11, 0.01, 0.2, -0.12, 0.24, -0.28, -0.07, 0.33, -0.08, 0.58, -0.08, 0.83, 0.07, 0.33, 0.08, 0.58, 0.08, 0.83),
  },
  {
    id: 'wave',
    label: 'Waving',
    kp: body(-0.11, 0.01, -0.22, 0.05, -0.25, -0.12, 0.11, 0.01, 0.15, 0.17, 0.17, 0.32, -0.07, 0.33, -0.08, 0.58, -0.08, 0.83, 0.07, 0.33, 0.08, 0.58, 0.08, 0.83),
  },
  {
    id: 'walk',
    label: 'Walking',
    kp: body(-0.11, 0.01, -0.13, 0.17, -0.08, 0.31, 0.11, 0.01, 0.15, 0.16, 0.2, 0.29, -0.07, 0.33, -0.1, 0.58, -0.14, 0.82, 0.07, 0.33, 0.09, 0.57, 0.13, 0.82),
  },
  {
    id: 'run',
    label: 'Running',
    kp: body(-0.09, 0.01, -0.18, 0.12, -0.1, 0.22, 0.09, 0.01, 0.17, 0.16, 0.25, 0.08, -0.06, 0.33, -0.15, 0.5, -0.1, 0.7, 0.06, 0.33, 0.12, 0.55, 0.24, 0.74),
  },
  {
    id: 'sit',
    label: 'Sitting',
    kp: body(-0.11, 0.01, -0.14, 0.2, -0.1, 0.36, 0.11, 0.01, 0.14, 0.2, 0.1, 0.36, -0.07, 0.33, -0.09, 0.43, -0.09, 0.66, 0.07, 0.33, 0.09, 0.43, 0.09, 0.66),
  },
];

/** How many frame px one REST unit is in op: from its torso (neck to mid-hip is 0.33), else its shoulders (0.22). */
function unitOf(kp: number[], fallback: number): number {
  const at = (j: number) => [kp[j * 2], kp[j * 2 + 1]];
  const [nx, ny] = at(1);
  const hx = (kp[16] + kp[22]) / 2;
  const hy = (kp[17] + kp[23]) / 2;
  const torso = Math.hypot(hx - nx, hy - ny) / 0.33;
  if (torso > 1) return torso;
  const sh = Math.hypot(kp[4] - kp[10], kp[5] - kp[11]) / 0.22;
  return sh > 1 ? sh : fallback;
}

/** op in preset `id`, keeping where its neck is and its size. */
export function applyPreset(op: PoseOp, id: string): PoseOp {
  const p = POSE_PRESETS.find((x) => x.id === id);
  if (!p) return op;
  const u = unitOf(op.kp, 100);
  const nx = op.kp[POSE_ROOT * 2];
  const ny = op.kp[POSE_ROOT * 2 + 1];
  return { ...op, kp: p.kp.map((v, i) => round(i % 2 ? ny + v * u : nx + v * u)) };
}

/** A new figure in preset `id`: head's top at (cx, top), `height` to the feet of a standing one. */
export function presetPose(id: string, cx: number, top: number, height: number): PoseOp {
  return applyPreset(restPose(cx, top, height), id);
}

/**
 * A rig from detected joints (OpenPose order, frame px, with confidence): confident joints where they were found, the
 * rest hung off their parent the way a standing figure would have them, at the detected figure's size. box: the
 * character's box, for the size and place when too little was found.
 */
export function fromDetection(joints: Array<{ x: number; y: number; score: number }>, box: { x: number; y: number; w: number; h: number }, minScore = 0.3): PoseOp {
  const ok = joints.map((j) => j.score >= minScore);
  const found = joints.flatMap((j) => [j.x, j.y]);
  // Size: from the confident torso or shoulders; else the box is about one standing figure tall.
  const torsoOk = ok[1] && ok[8] && ok[11];
  const shouldersOk = ok[2] && ok[5];
  let u = box.h / (REST_FEET - REST_TOP);
  if (torsoOk || shouldersOk) u = unitOf(found, u);
  const kp = new Array<number>(36).fill(0);
  const neck = ok[1] ? [found[2], found[3]] : [box.x + box.w / 2, box.y + (0 - REST_TOP) * u];
  const order = poseSubtree(POSE_ROOT);
  order.forEach((j) => {
    if (ok[j] || j === POSE_ROOT) {
      kp[j * 2] = round(j === POSE_ROOT ? neck[0] : found[j * 2]);
      kp[j * 2 + 1] = round(j === POSE_ROOT ? neck[1] : found[j * 2 + 1]);
      return;
    }
    const p = POSE_PARENT[j];
    kp[j * 2] = round(kp[p * 2] + (REST[j * 2] - REST[p * 2]) * u);
    kp[j * 2 + 1] = round(kp[p * 2 + 1] + (REST[j * 2 + 1] - REST[p * 2 + 1]) * u);
  });
  return { t: 'pose', kp, w: Math.max(2, Math.round(u * 0.012)) };
}

/** The joint of op nearest (x, y) within tol (frame px), or -1. */
export function poseHit(op: PoseOp, x: number, y: number, tol: number): number {
  let best = -1;
  let bestD = tol;
  for (let j = 0; j < POSE_JOINTS.length; j++) {
    const d = Math.hypot(op.kp[j * 2] - x, op.kp[j * 2 + 1] - y);
    if (d <= bestD) {
      best = j;
      bestD = d;
    }
  }
  return best;
}

/** Joint j and every joint below it in the rig. */
export function poseSubtree(j: number): number[] {
  const out = [j];
  for (let k = 0; k < out.length; k++) POSE_PARENT.forEach((p, c) => p === out[k] && out.push(c));
  return out;
}

/**
 * How dragging a joint changes the rig: swing it (and what hangs from it) round its parent, keeping limb lengths;
 * move it alone (to change proportions); move the whole figure; or scale it about the neck.
 */
export type PoseDrag = 'swing' | 'free' | 'move' | 'scale';

/** op (as the drag started) with joint j dragged to (x, y) in the given way. */
export function posed(op: PoseOp, j: number, x: number, y: number, mode: PoseDrag): PoseOp {
  const kp = op.kp.slice();
  const jx = op.kp[j * 2];
  const jy = op.kp[j * 2 + 1];
  const p = POSE_PARENT[j];
  if (mode === 'move' || (mode === 'swing' && p < 0)) {
    for (let i = 0; i < kp.length; i += 2) {
      kp[i] = round(kp[i] + x - jx);
      kp[i + 1] = round(kp[i + 1] + y - jy);
    }
    return { ...op, kp };
  }
  if (mode === 'scale') {
    const cx = op.kp[POSE_ROOT * 2];
    const cy = op.kp[POSE_ROOT * 2 + 1];
    const from = Math.hypot(jx - cx, jy - cy);
    const f = from > 1 ? Math.max(0.05, Math.hypot(x - cx, y - cy) / from) : 1;
    for (let i = 0; i < kp.length; i += 2) {
      kp[i] = round(cx + (kp[i] - cx) * f);
      kp[i + 1] = round(cy + (kp[i + 1] - cy) * f);
    }
    return { ...op, kp, w: Math.max(1, Math.round(op.w * f)) };
  }
  if (mode === 'free') {
    kp[j * 2] = round(x);
    kp[j * 2 + 1] = round(y);
    return { ...op, kp };
  }
  const px = op.kp[p * 2];
  const py = op.kp[p * 2 + 1];
  const a = Math.atan2(y - py, x - px) - Math.atan2(jy - py, jx - px);
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  poseSubtree(j).forEach((k) => {
    const dx = op.kp[k * 2] - px;
    const dy = op.kp[k * 2 + 1] - py;
    kp[k * 2] = round(px + dx * cos - dy * sin);
    kp[k * 2 + 1] = round(py + dx * sin + dy * cos);
  });
  return { ...op, kp };
}

/** Mirror the figure left to right about its neck (it turns to face the other way round). */
export function mirrorPose(op: PoseOp): PoseOp {
  const cx = op.kp[POSE_ROOT * 2];
  return { ...op, kp: op.kp.map((v, i) => (i % 2 ? v : round(2 * cx - v))) };
}

/** Bounds of the rig's joints, padded by its limb width. */
export function poseBox(op: PoseOp): { x: number; y: number; w: number; h: number } {
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (let i = 0; i + 1 < op.kp.length; i += 2) {
    x0 = Math.min(x0, op.kp[i]);
    y0 = Math.min(y0, op.kp[i + 1]);
    x1 = Math.max(x1, op.kp[i]);
    y1 = Math.max(y1, op.kp[i + 1]);
  }
  return { x: x0 - op.w, y: y0 - op.w, w: x1 - x0 + 2 * op.w, h: y1 - y0 + 2 * op.w };
}

/**
 * Draw the rig at scale. colored: OpenPose's picture (joints in their colours, then every limb in its colour at 60%,
 * as OpenPose draws them); else white coverage, for a mask.
 */
export function drawPose(ctx: CanvasRenderingContext2D, op: PoseOp, scale: number, colored: boolean): void {
  const P = (j: number) => [op.kp[j * 2] * scale, op.kp[j * 2 + 1] * scale];
  const w = Math.max(1, op.w * scale);
  ctx.save();
  ctx.lineCap = 'round';
  ctx.globalAlpha = 1;
  POSE_JOINTS.forEach((_, j) => {
    const [x, y] = P(j);
    ctx.fillStyle = colored ? POSE_COLORS[j] : '#fff';
    ctx.beginPath();
    ctx.arc(x, y, w, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.lineWidth = w * 2;
  if (colored) ctx.globalAlpha = 0.6;
  POSE_LIMBS.forEach(([a, b], i) => {
    const [ax, ay] = P(a);
    const [bx, by] = P(b);
    ctx.strokeStyle = colored ? POSE_COLORS[i] : '#fff';
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  });
  ctx.restore();
}
