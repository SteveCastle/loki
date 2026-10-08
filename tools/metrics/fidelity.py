"""Geometric fidelity of a 4K output to its source frame: the source is placed on the output canvas at the
template-matched zoom/centre, both at 960x540, then compared (SSIM on grayscale, mean abs diff) over the
source region. Optionally writes a comparison panel: source | output | diff heat, plus a 2x crop of the middle.
usage: fidelity.py <src.png> <out.png> [panel.png]
"""
import sys, numpy as np, cv2
sys.path.insert(0, __file__.rsplit("/",1)[0])
from framing import analyse
src=cv2.imread(sys.argv[1]); out=cv2.imread(sys.argv[2])
mv,z,cx,cy=analyse(sys.argv[1],sys.argv[2])
o=cv2.resize(out,(960,540),interpolation=cv2.INTER_AREA)
sh,sw=src.shape[:2]; s=540.0/sh*z; tw,th=int(round(sw*s)),int(round(sh*s))
t=cv2.resize(src,(tw,th),interpolation=cv2.INTER_CUBIC)
x0,y0=int(round(cx*960-tw/2)),int(round(cy*540-th/2))
canvas=np.zeros_like(o); mask=np.zeros(o.shape[:2],bool)
sx0,sy0=max(0,-x0),max(0,-y0); dx0,dy0=max(0,x0),max(0,y0)
w=min(tw-sx0,960-dx0); h=min(th-sy0,540-dy0)
canvas[dy0:dy0+h,dx0:dx0+w]=t[sy0:sy0+h,sx0:sx0+w]; mask[dy0:dy0+h,dx0:dx0+w]=True
# compare at the source's own detail level: blur both to kill the restoration's added detail
ga=cv2.GaussianBlur(cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),2)
gb=cv2.GaussianBlur(cv2.cvtColor(o,cv2.COLOR_BGR2GRAY).astype(np.float32),(0,0),2)
def ssim(a,b):
    C1,C2=(0.01*255)**2,(0.03*255)**2
    mu_a=cv2.GaussianBlur(a,(0,0),1.5); mu_b=cv2.GaussianBlur(b,(0,0),1.5)
    va=cv2.GaussianBlur(a*a,(0,0),1.5)-mu_a**2; vb=cv2.GaussianBlur(b*b,(0,0),1.5)-mu_b**2; cov=cv2.GaussianBlur(a*b,(0,0),1.5)-mu_a*mu_b
    return ((2*mu_a*mu_b+C1)*(2*cov+C2))/((mu_a**2+mu_b**2+C1)*(va+vb+C2))
sm=ssim(ga,gb); d=np.abs(ga-gb)
print(f"{sys.argv[2]:45s} zoom {z:4.2f} match {mv:.3f}  SSIM(region) {sm[mask].mean():.3f}  mean|diff| {d[mask].mean():5.2f}")
if len(sys.argv)>3:
    heat=cv2.applyColorMap(np.clip(d*3,0,255).astype(np.uint8),cv2.COLORMAP_INFERNO); heat[~mask]=0
    top=np.concatenate([canvas,o,heat],axis=1)
    # 2x crops of the central region
    cxp,cyp=int(cx*960),int(cy*540); cw,ch=240,135
    crop=lambda im: cv2.resize(im[cyp-ch:cyp+ch, cxp-cw:cxp+cw],(960,540),interpolation=cv2.INTER_CUBIC)
    bot=np.concatenate([crop(canvas),crop(o),crop(heat)],axis=1)
    cv2.imwrite(sys.argv[3],np.concatenate([top,bot],axis=0))
