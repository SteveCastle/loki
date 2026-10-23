"""Compare two reference dump dirs (or a dump dir and a native dump dir): cosine / max-abs per tensor."""
import json
import sys

import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
only = sys.argv[3:]
ia = json.load(open(f"{a_dir}/index.json"))
ib = json.load(open(f"{b_dir}/index.json"))
for k, s in ia.items():
    if k not in ib or (only and not any(o in k for o in only)):
        continue
    a = np.fromfile(f"{a_dir}/{k}.bin", dtype=np.float32).astype(np.float64)
    b = np.fromfile(f"{b_dir}/{k}.bin", dtype=np.float32).astype(np.float64)
    if a.size != b.size:
        print(f"{k:28s} size mismatch {s} vs {ib[k]}")
        continue
    cos = (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
    rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)
    print(f"{k:28s} {str(s):20s} cos {cos:.6f} rel {rel:.2e} maxabs {np.abs(a-b).max():.4g} (|ref|max {np.abs(a).max():.4g})")
