"""The pose in words, from the 18 joints alone (what the product could compute from a mannequin)."""
import math
from common import torso

def arm(kp, s, e, w, hip, side, T):
    sx, sy = kp[s]; ex, ey = kp[e]; wx, wy = kp[w]
    head_y = kp[0][1]
    up, fore = math.dist(kp[s], kp[e]), math.dist(kp[e], kp[w])
    straight = math.dist(kp[s], kp[w]) > 0.9 * (up + fore)
    if straight and wy > sy + 0.8 * (up + fore) * 0.9:
        hand = 'hanging relaxed at the side'
    elif wy < head_y - 0.15 * T:
        hand = 'raised above the head'
    elif math.dist(kp[w], kp[0]) < 0.4 * T:
        hand = 'up at the face'
    elif math.dist(kp[w], kp[hip]) < 0.5 * T and ey < wy + 0.2 * T:
        hand = 'resting on the hip, the elbow pointing out to the side'
    elif abs(wy - sy) < 0.3 * T and abs(wx - sx) > 0.6 * T:
        hand = 'stretched straight out to the side at shoulder height'
    elif wy < sy + 0.1 * T:
        hand = 'raised with the elbow out to the side and the hand up'
    elif wy > kp[hip][1] - 0.1 * T and abs(wx - sx) < 0.35 * T:
        hand = 'hanging relaxed at the side'
    elif (wx - kp[1][0]) * (sx - kp[1][0]) < 0:
        hand = 'across the front of the body'
    else:
        hand = 'held down and away from the body'
    return f'the arm on the {side} of the picture {hand}'

def leg(kp, h, k, a, side, T):
    hx, hy = kp[h]; kx, ky = kp[k]; ax, ay = kp[a]
    thigh = math.dist(kp[h], kp[k])
    if ky < hy + 0.45 * thigh:
        out = (' toward the left of the picture' if kx < hx else ' toward the right of the picture') if abs(kx - hx) > 0.5 * thigh else ''
        return f'the leg on the {side} of the picture lifted, the knee raised high{out} and the foot off the ground'
    if ay < ky + 0.2 * thigh:
        return f'the leg on the {side} of the picture bent at the knee with the foot lifted up behind or beside it'
    if abs(ax - hx) > 0.5 * thigh:
        return f'the leg on the {side} of the picture stepping out to the side'
    return f'the leg on the {side} of the picture standing straight'

def describe(kp, size=None, only=None):
    """only: joint ids that move; limbs with none of them are left out (a delta from the current pose)."""
    T = torso(kp)
    W, H = size or (1e9, 1e9)
    seen = lambda *js: all(0 <= kp[j][0] < W and 0 <= kp[j][1] < H for j in js)
    cx = kp[1][0]
    def sides(a, b):
        """Where two limbs are (their far joints against the body's middle): left/right, or one/the other."""
        xa, xb = a - cx, b - cx
        if xa * xb < 0 or abs(xa - xb) > 0.3 * T:
            return ('the {} on the left of the picture', 'the {} on the right of the picture') if xa < xb else ('the {} on the right of the picture', 'the {} on the left of the picture')
        return ('one {}', 'the other {}')
    out = []
    folded = math.dist(kp[4], kp[7]) < 0.35 * T and min(kp[2][1], kp[5][1]) - 0.1 * T < (kp[4][1] + kp[7][1]) / 2 < (kp[8][1] + kp[11][1]) / 2
    if folded and seen(2, 5) and (only is None or {3, 4, 6, 7} & set(only)):
        out.append((0, 'both arms folded across the chest'))
    na, nb = sides((kp[3][0] + kp[4][0]) / 2, (kp[6][0] + kp[7][0]) / 2)
    for (s, e, w, hip), n in (((2, 3, 4, 8), na), ((5, 6, 7, 11), nb)):
        if seen(s) and not folded and (only is None or {e, w} & set(only)):
            out.append((kp[w][0], arm(kp, s, e, w, hip, '', T).replace('the arm on the  of the picture', n.format('arm'))))
    legs = []
    ma, mb = sides((kp[9][0] + kp[10][0]) / 2, (kp[12][0] + kp[13][0]) / 2)
    for (h, k, a), n in (((8, 9, 10), ma), ((11, 12, 13), mb)):
        if seen(h, k) and (only is None or {k, a} & set(only)):
            legs.append((kp[k][0], leg(kp, h, k, a, '', T).replace('the leg on the  of the picture', n.format('leg'))))
    key = lambda p: (p[1].startswith('the other'), p[0])
    parts = sorted(out, key=key) + sorted(legs, key=key)
    return '; '.join(p for _, p in parts)

if __name__ == '__main__':
    import json
    from common import ROOT
    for c in json.load(open(f'{ROOT}/cases.json')):
        print(c['id'], '->', describe(c['target'], c['size']))
