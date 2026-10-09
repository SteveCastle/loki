"""Estimate where/at what scale each source frame landed in its 4K output: multi-scale template matching
of the source's central crop against the output (both downscaled 4x). Prints zoom (1.0 = source height fills
the output height), the centre of the source region in output coordinates (as a fraction), and the match score.
usage: framing.py <src_dir> <out1> [<out2> ...]   (source frame = src_dir/frame_NNN.png matched by name, or --src FILE)
"""
import sys, re, numpy as np, cv2
def analyse(src_path, out_path):
    src=cv2.imread(src_path); out=cv2.imread(out_path)
    out=cv2.resize(out,(960,540),interpolation=cv2.INTER_AREA)
    sh,sw=src.shape[:2]
    best=None
    for z in np.arange(0.70,1.80,0.02):
        s=540.0/sh*z
        tw,th=int(sw*s),int(sh*s)
        t=cv2.resize(src,(tw,th),interpolation=cv2.INTER_AREA)
        # central 50% crop of the source as the template (the borders may be cropped away by a zoom)
        cx0,cy0=int(tw*0.25),int(th*0.25); crop=t[cy0:cy0+th//2, cx0:cx0+tw//2]
        if crop.shape[0]>=540 or crop.shape[1]>=960: continue
        r=cv2.matchTemplate(out,crop,cv2.TM_CCOEFF_NORMED)
        _,mv,_,ml=cv2.minMaxLoc(r)
        if best is None or mv>best[0]:
            cx=(ml[0]-cx0+tw/2)/960; cy=(ml[1]-cy0+th/2)/540
            best=(mv,z,cx,cy)
    return best
if __name__=='__main__':
    args=sys.argv[1:]
    src_file=None
    if args[0]=="--src": src_file=args[1]; args=args[2:]
    src_dir=None if src_file else args.pop(0)
    rows=[]
    for o in args:
        if src_file: s=src_file
        else:
            m=re.search(r"frame_(\d+)",o); s=f"{src_dir}/frame_{m.group(1)}.png"
        mv,z,cx,cy=analyse(s,o)
        rows.append((z,cx,cy)); print(f"{o:45s} zoom {z:4.2f}  centre ({cx:5.3f},{cy:5.3f})  score {mv:.3f}")
    if len(rows)>1:
        a=np.array(rows); print(f"spread: zoom std {a[:,0].std():.3f} (min {a[:,0].min():.2f} max {a[:,0].max():.2f}); centre std x {a[:,1].std():.3f} y {a[:,2].std():.3f}")
