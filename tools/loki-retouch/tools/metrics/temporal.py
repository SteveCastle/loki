"""Temporal consistency between two consecutive outputs: mean |out2 - out1| over the source region (960x540,
lightly blurred) divided by the same for the two sources placed at the matched zoom/centre, so 1.0 means
the output moves as much as the source does and more means extra flicker.
usage: temporal.py <src1> <src2> <out1> <out2>
"""
import sys, numpy as np, cv2
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from framing import analyse

def place(src_path, out_path):
    src = cv2.imread(src_path); out = cv2.imread(out_path)
    o = cv2.resize(out, (960, 540), interpolation=cv2.INTER_AREA)
    mv, z, cx, cy = analyse(src_path, out_path)
    sh, sw = src.shape[:2]; s = 540 / sh * z
    t = cv2.resize(src, (int(sw * s), int(sh * s)), interpolation=cv2.INTER_CUBIC)
    c = np.zeros_like(o); m = np.zeros(o.shape[:2], bool)
    x0 = int(cx * 960 - t.shape[1] / 2); y0 = int(cy * 540 - t.shape[0] / 2)
    sx, sy = max(0, -x0), max(0, -y0); dx, dy = max(0, x0), max(0, y0)
    w = min(t.shape[1] - sx, 960 - dx); h = min(t.shape[0] - sy, 540 - dy)
    c[dy:dy + h, dx:dx + w] = t[sy:sy + h, sx:sx + w]; m[dy:dy + h, dx:dx + w] = True
    return c, o, m, z

g = lambda im: cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 2)
if __name__ == '__main__':
    c1, o1, m1, z1 = place(sys.argv[1], sys.argv[3])
    c2, o2, m2, z2 = place(sys.argv[2], sys.argv[4])
    m = m1 & m2
    ds = np.abs(g(c1) - g(c2))[m].mean(); do = np.abs(g(o1) - g(o2))[m].mean()
    print(f"{sys.argv[4]:45s} source motion {ds:5.2f}  output motion {do:5.2f}  ratio {do / max(ds, 1e-6):4.2f}  zoom {z1:.2f}->{z2:.2f}")
