"""Shape flicker of a mask sequence: consecutive-frame IoU after the best integer translation (+-R px, coarse
to fine), i.e. how much the mask's SHAPE changes from one frame to the next independent of motion.
flicker.py DIR [DIR ...]"""
import glob
import os
import sys

import numpy as np
from PIL import Image


def best_shift_iou(a, b, r=48):
    best = 0.0
    ys, xs = np.nonzero(a)
    yb, xb = np.nonzero(b)
    if len(ys) == 0 or len(yb) == 0:
        return 1.0 if len(ys) == len(yb) else 0.0
    dy0, dx0 = int(round(yb.mean() - ys.mean())), int(round(xb.mean() - xs.mean()))
    for dy in range(dy0 - 4, dy0 + 5):
        for dx in range(dx0 - 4, dx0 + 5):
            s = np.roll(np.roll(a, dy, 0), dx, 1)
            u = (s | b).sum()
            best = max(best, (s & b).sum() / u if u else 1.0)
    return best


for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    ms = [np.asarray(Image.open(f).convert("L")) > 127 for f in files]
    v = np.array([best_shift_iou(ms[i], ms[i + 1]) for i in range(len(ms) - 1)])
    print(f"{d}: motion-compensated consecutive IoU mean {v.mean():.4f} p5 {np.percentile(v, 5):.4f} min {v.min():.4f}")
