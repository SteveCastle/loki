"""Temporal consistency of a mask sequence: temporal.py DIR [DIR ...]  (DIR = <out>/<id> with 00000.png ...)
Reports mean IoU between consecutive frames, the number of "jumps" (consecutive IoU < 0.85) and mean |soft diff|."""
import glob
import os
import sys

import numpy as np
from PIL import Image

for d in sys.argv[1:]:
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    prev = None
    ious, diffs = [], []
    for f in files:
        m = np.asarray(Image.open(f).convert("L")).astype(np.float32) / 255.0
        if prev is not None:
            a, b = prev > 0.5, m > 0.5
            u = (a | b).sum()
            ious.append((a & b).sum() / u if u else 1.0)
            diffs.append(np.abs(m - prev).mean())
        prev = m
    ious = np.array(ious)
    print(f"{d}: {len(files)} frames, consecutive IoU mean {ious.mean():.4f} min {ious.min():.4f}, jumps(<0.85) {(ious < 0.85).sum()}, mean|diff| {np.mean(diffs):.5f}")
