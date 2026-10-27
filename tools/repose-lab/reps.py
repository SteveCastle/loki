"""The same target pose rendered several ways, all at the generation size (long edge EDGE)."""
import json, math, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from common import ROOT, render, torso, LIMBS, COLORS as COLORS_

EDGE = int(sys.argv[1]) if len(sys.argv) > 1 else 640
ARMS = {(2, 3), (3, 4), (5, 6), (6, 7)}
LEGS = {(8, 9), (9, 10), (11, 12), (12, 13)}

def scale(kp, s):
    return [(x * s, y * s) for x, y in kp]

def capsule(kp, size, bg, fg):
    T = torso(kp)
    img = Image.new('RGB', size, bg); d = ImageDraw.Draw(img)
    def cap(a, b, w):
        d.line([a, b], fill=fg, width=int(w))
        for p in (a, b):
            d.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=fg)
    d.polygon([kp[2], kp[5], kp[11], kp[8]], fill=fg)
    cap(kp[1], ((kp[8][0] + kp[11][0]) / 2, (kp[8][1] + kp[11][1]) / 2), 0.32 * T)
    for a, b in LEGS:
        cap(kp[a], kp[b], 0.2 * T)
    for a, b in ARMS:
        cap(kp[a], kp[b], 0.12 * T)
    cap(kp[1], kp[0], 0.12 * T)
    r = 0.2 * T
    d.ellipse([kp[0][0] - r, kp[0][1] - r * 1.2, kp[0][0] + r, kp[0][1] + r * 1.1], fill=fg)
    # a soft edge shade so it reads as a solid figure
    return img.filter(ImageFilter.GaussianBlur(1))

def stick(kp, size):
    T = torso(kp)
    img = Image.new('RGB', size, 'white'); d = ImageDraw.Draw(img)
    for a, b in LIMBS[:12] + [(1, 0)]:
        d.line([kp[a], kp[b]], fill='black', width=max(2, int(0.025 * T)))
    r = 0.15 * T
    d.ellipse([kp[0][0] - r, kp[0][1] - r, kp[0][0] + r, kp[0][1] + r], outline='black', width=max(2, int(0.025 * T)))
    return img

def warp(src, S, Tg, size):
    """Her own limbs, cut out along each bone and turned to the new pose; the old limbs blurred out."""
    T = torso(S)
    base = src.copy()
    # cover the old limbs that move with a blur of their surroundings
    hole = Image.new('L', size, 0); dh = ImageDraw.Draw(hole)
    moving = []
    for a, b in sorted(ARMS | LEGS):
        if math.dist(S[a], Tg[a]) + math.dist(S[b], Tg[b]) > 0.1 * T:
            moving.append((a, b))
            w = 0.24 * T if (a, b) in ARMS else 0.34 * T
            dh.line([S[a], S[b]], fill=255, width=int(w))
            for p in (S[a], S[b]):
                dh.ellipse([p[0] - w / 2, p[1] - w / 2, p[0] + w / 2, p[1] + w / 2], fill=255)
    blurred = src.filter(ImageFilter.GaussianBlur(0.25 * T))
    PAD = int(T)  # limbs that run off the frame are cut from its edge, extended, not from black
    padded = Image.fromarray(np.pad(np.asarray(src), ((PAD, PAD), (PAD, PAD), (0, 0)), mode='edge'))
    base = Image.composite(blurred, base, hole.filter(ImageFilter.GaussianBlur(4)))
    for a, b in moving:
        w = 0.2 * T if (a, b) in ARMS else 0.3 * T
        L = math.dist(S[a], S[b]); L2 = math.dist(Tg[a], Tg[b])
        if L < 2:
            continue
        # a strip along the source bone, straightened, then placed along the target bone
        ang = math.degrees(math.atan2(S[b][1] - S[a][1], S[b][0] - S[a][0]))
        cx, cy = (S[a][0] + S[b][0]) / 2 + PAD, (S[a][1] + S[b][1]) / 2 + PAD
        r = padded.rotate(ang, center=(cx, cy), resample=Image.BILINEAR)
        strip = r.crop((int(cx - L / 2 - w / 2), int(cy - w / 2), int(cx + L / 2 + w / 2), int(cy + w / 2)))
        strip = strip.resize((max(1, int(L2 + w)), strip.height))
        m = Image.new('L', strip.size, 0)
        ImageDraw.Draw(m).rounded_rectangle([0, 0, strip.width - 1, strip.height - 1], radius=int(w / 2), fill=255)
        m = m.filter(ImageFilter.GaussianBlur(3))
        ang2 = math.degrees(math.atan2(Tg[b][1] - Tg[a][1], Tg[b][0] - Tg[a][0]))
        big = Image.new('RGBA', strip.size); big.paste(strip, (0, 0)); big.putalpha(m)
        rot = big.rotate(-ang2, expand=True, resample=Image.BILINEAR)
        tx, ty = (Tg[a][0] + Tg[b][0]) / 2, (Tg[a][1] + Tg[b][1]) / 2
        base.paste(rot, (int(tx - rot.width / 2), int(ty - rot.height / 2)), rot)
    return base

cases = json.load(open(f'{ROOT}/cases.json'))
for c in cases:
    src = Image.open(f"{ROOT}/cases/{c['subject']}.png").convert('RGB')
    s = EDGE / max(src.size)
    size = (round(src.width * s), round(src.height * s))
    src = src.resize(size, Image.LANCZOS)
    sp = f"{ROOT}/cases/{c['subject']}-{EDGE}.png"
    if not os.path.exists(sp):  # never rewrite: a running engine may be reading it
        src.save(sp)
    Tg, S = scale(c['target'], s), scale(c['source'], s)
    w = max(3, round(c['w'] * s))
    p = f"{ROOT}/cases/{c['id']}-{EDGE}"
    render(Tg, w, size).save(p + '-black.png')
    render(Tg, 2 * w, size).save(p + '-thick.png')
    im = render(Tg, w, size); a = np.asarray(im); bgm = (a.sum(2) == 0)
    a = a.copy(); a[bgm] = 255; Image.fromarray(a).save(p + '-white.png')
    capsule(Tg, size, 'white', (150, 150, 150)).save(p + '-capsule.png')
    # capsule mass in OpenPose's colours on black, then the joints/limbs over it
    oc = Image.new('RGB', size, 'black'); od = ImageDraw.Draw(oc); Tt = torso(Tg)
    od.polygon([Tg[2], Tg[5], Tg[11], Tg[8]], fill=(60, 60, 60))
    for i, (a, b) in enumerate(LIMBS[:12]):
        wd = 0.12 * Tt if (a, b) in ARMS else 0.2 * Tt if (a, b) in LEGS else 0
        if wd:
            col = tuple(int(v * 0.6) for v in COLORS_[i])
            od.line([Tg[a], Tg[b]], fill=col, width=int(wd))
            for q in (Tg[a], Tg[b]):
                od.ellipse([q[0] - wd / 2, q[1] - wd / 2, q[0] + wd / 2, q[1] + wd / 2], fill=col)
    fig = render(Tg, w, size)
    oc.paste(fig, (0, 0), Image.fromarray(((np.asarray(fig).max(2) > 0) * 255).astype('uint8')))
    oc.save(p + '-ocapsule.png')
    stick(Tg, size).save(p + '-stick.png')
    pair = Image.new('RGB', (size[0] * 2, size[1]))
    pair.paste(render(S, w, size), (0, 0)); pair.paste(render(Tg, w, size), (size[0], 0))
    pair.save(p + '-pair.png')
    warp(src, S, Tg, size).save(p + '-warp.png')
print('ok', len(cases))
