"""Overlay soft masks on an image for eyeballing: overlay.py IMAGE OUT.png MASK.png [MASK.png ...]"""
import sys
import numpy as np
from PIL import Image

img = np.asarray(Image.open(sys.argv[1]).convert("RGB")).astype(np.float32)
cols = [(255, 60, 60), (60, 200, 255), (80, 255, 80), (255, 220, 40), (220, 80, 255), (255, 140, 0), (0, 255, 200), (255, 0, 160)]
out = img.copy()
for i, mp in enumerate(sys.argv[3:]):
    m = np.asarray(Image.open(mp).convert("L")).astype(np.float32)[..., None] / 255.0
    c = np.array(cols[i % len(cols)], dtype=np.float32)
    out = out * (1 - 0.55 * m) + c * 0.55 * m
    edge = (np.abs(np.diff(m[..., 0] > 0.5, axis=0, prepend=0)) + np.abs(np.diff(m[..., 0] > 0.5, axis=1, prepend=0))) > 0
    out[edge] = c
im = Image.fromarray(out.clip(0, 255).astype(np.uint8))
im.thumbnail((1400, 1400))
im.save(sys.argv[2])
