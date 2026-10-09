"""Eye and mouth crops across source 1, source 2, output 1, output 2 (each source placed at its output's
matched zoom/centre). usage: crops.py <src1> <src2> <out1> <out2> <panel.png> [label]"""
import sys, numpy as np, cv2
sys.path.insert(0, __file__.replace("\\", "/").rsplit("/", 1)[0])
from temporal import place
c1, o1, _, _ = place(sys.argv[1], sys.argv[3]); c2, o2, _, _ = place(sys.argv[2], sys.argv[4])
boxes = {"eye": (120, 200, 400, 400), "mouth": (380, 120, 700, 340)}
rows = []
for name, (x0, y0, x1, y1) in boxes.items():
    cr = lambda im: cv2.resize(im[y0:y1, x0:x1], (480, int(480 * (y1 - y0) / (x1 - x0))), interpolation=cv2.INTER_CUBIC)
    rows.append(np.concatenate([cr(c1), cr(c2), cr(o1), cr(o2)], axis=1))
panel = np.concatenate(rows, axis=0)
label = sys.argv[6] if len(sys.argv) > 6 else ""
cv2.putText(panel, f"src1 | src2 | out1 | out2  {label}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite(sys.argv[5], panel)
