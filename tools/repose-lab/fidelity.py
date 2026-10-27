"""How much of the person that should not change did not: masked SSIM of the face and of the torso/hips
(arms, where they were and where they go, are left out), best over small shifts."""
import numpy as np
from PIL import Image, ImageDraw
from common import torso

ARMS = [(2, 3), (3, 4), (5, 6), (6, 7)]

def _box(img, r):
    c = np.pad(img, ((r + 1, r), (r + 1, r)), mode='edge').cumsum(0).cumsum(1)
    n = 2 * r + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)

def ssim_map(a, b, r=4):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = _box(a, r), _box(b, r)
    va, vb = _box(a * a, r) - ma * ma, _box(b * b, r) - mb * mb
    cov = _box(a * b, r) - ma * mb
    return ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2))

def region(kp, joints, pad, size, extra_down=0):
    xs = [kp[j][0] for j in joints]; ys = [kp[j][1] for j in joints]
    x0, y0, x1, y1 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad + extra_down
    W, H = size
    return [int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))]

def fidelity(src_path, out_path, c):
    src = Image.open(src_path).convert('L')
    out = Image.open(out_path).convert('L')
    s = out.width / src.width
    src = src.resize(out.size, Image.BILINEAR)
    S = [(x * s, y * s) for x, y in c['source']]
    Tg = [(x * s, y * s) for x, y in c['target']]
    T = torso(S)
    mask = Image.new('L', out.size, 255)
    d = ImageDraw.Draw(mask)
    for kp in (S, Tg):
        for a, b in ARMS:
            d.line([kp[a], kp[b]], fill=0, width=int(0.22 * T))
            d.ellipse([kp[b][0] - 0.12 * T, kp[b][1] - 0.12 * T, kp[b][0] + 0.12 * T, kp[b][1] + 0.12 * T], fill=0)
    m = np.asarray(mask) > 0
    A, B = np.asarray(src, float), np.asarray(out, float)
    res = {}
    for name, box in (('face', region(S, [0, 14, 15, 16, 17], 0.12 * T, out.size)),
                      ('body', region(S, [2, 5, 8, 11], 0.18 * T, out.size, 0.3 * T))):
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue
        best = -1
        sh = max(1, int(0.04 * T))
        for dy in range(-sh, sh + 1, max(1, sh // 3)):
            for dx in range(-sh, sh + 1, max(1, sh // 3)):
                ya, yb = max(0, y0 + dy), min(B.shape[0], y1 + dy)
                xa, xb = max(0, x0 + dx), min(B.shape[1], x1 + dx)
                a = A[ya - dy:yb - dy, xa - dx:xb - dx]
                b = B[ya:yb, xa:xb]
                mm = m[ya - dy:yb - dy, xa - dx:xb - dx] & m[ya:yb, xa:xb]
                if mm.sum() < 50:
                    continue
                best = max(best, float(ssim_map(a, b)[mm].mean()))
        res[name] = best
    return res
