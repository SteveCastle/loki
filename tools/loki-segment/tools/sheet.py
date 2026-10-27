"""Contact sheet of a video's mask overlays: sheet.py FRAMES_DIR MASKS_DIR OUT.png [frame ...]"""
import os
import sys

import numpy as np
from PIL import Image

fdir, mdir, out = sys.argv[1:4]
frames = [int(x) for x in sys.argv[4:]] or [0, 40, 80, 120, 160, 199]
ids = sorted(d for d in os.listdir(mdir) if os.path.isdir(os.path.join(mdir, d)))
cols = [(255, 60, 60), (60, 200, 255), (80, 255, 80), (255, 220, 40)]
tiles = []
for f in frames:
    img = np.asarray(Image.open(os.path.join(fdir, f"{f:05d}.jpg")).convert("RGB")).astype(np.float32)
    for i, oid in enumerate(ids):
        m = np.asarray(Image.open(os.path.join(mdir, oid, f"{f:05d}.png")).convert("L")).astype(np.float32)[..., None] / 255
        c = np.array(cols[i % 4], dtype=np.float32)
        img = img * (1 - 0.5 * m) + c * 0.5 * m
    t = Image.fromarray(img.clip(0, 255).astype(np.uint8))
    t.thumbnail((480, 480))
    tiles.append(t)
w, h = tiles[0].size
sheet = Image.new("RGB", (w * 3, h * ((len(tiles) + 2) // 3)))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 3) * w, (i // 3) * h))
sheet.save(out)
