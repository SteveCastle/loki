"""Ping-pong consistency: frame m and frame N-1-m are the same image; report mask IoU over those pairs."""
import sys
import numpy as np
from PIL import Image

d, n = sys.argv[1], int(sys.argv[2])
def m(i):
    return np.asarray(Image.open(f"{d}/{i:05d}.png").convert("L")) > 127
ious = []
for k in range(0, n // 2):
    a, b = m(k), m(n - 1 - k)
    u = (a | b).sum()
    ious.append((a & b).sum() / u if u else 1.0)
ious = np.array(ious)
print(f"{d}: mirror-pair IoU mean {ious.mean():.4f} min {ious.min():.4f}  wrap pair (0 vs {n-1}) {ious[0]:.4f}")
