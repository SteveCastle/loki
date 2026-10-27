import json, math, os, shutil
from PIL import Image
from common import *

SUBJECTS = {
    'window': ('ba93750b-d5c9-43b9-96ae-4286f3edc3da', (250, 200, 520, 1150)),
    'pool': ('05db132b-c532-4292-a6a9-634c855ab53c', (300, 480, 680, 1220)),
    'suit': ('110784cb-089d-4d4b-90c4-00b6a985556e', (150, 50, 1100, 1990)),
}
# Arm A = joints 2,3,4; arm B = 5,6,7; leg A = 8,9,10. Angles in degrees, image space (0 = right, 90 = down).
POSES = {
    'armsup': ({'A': (250, 262), 'B': (290, 278)}, 'both arms raised high above the head'),
    'wave': ({'A': (95, 92), 'B': (345, 275)}, 'one arm hanging relaxed at the side, the other raised with the elbow out to the side and the hand up, waving'),
    'hips': ({'A': (125, 45), 'B': (55, 135)}, 'both hands resting on the hips, elbows pointing out to the sides'),
    'tpose': ({'A': (180, 180), 'B': (0, 0)}, 'both arms stretched straight out to the sides at shoulder height'),
    'knee': ({'L': (160, 95)}, 'standing on one leg with the other knee lifted up and out to the side, the foot hanging below it'),
    'behindhead': ({'A': (200, 330), 'B': (340, 210)}, 'both hands behind the head, elbows pointing out to the sides'),
    'reach': ({'A': (95, 92), 'B': (275, 272)}, 'one arm hanging at the side, the other reaching straight up above the head'),
    'point': ({'A': (100, 95), 'B': (330, 330)}, 'one arm hanging at the side, the other stretched out and up, pointing into the distance'),
    'folded': ({'A': (70, 0), 'B': (110, 180)}, 'both arms folded across the chest'),
    'step': ({'L': (120, 100)}, 'standing with the feet wide apart, one leg stepping out to the side'),
    'headhand': ({'A': (95, 92), 'B': (300, 200)}, 'one arm hanging at the side, the other raised with the hand resting on top of the head'),
    'combo1': ({'A': (240, 255), 'B': (55, 135), 'L': (160, 95)}, 'one arm raised above the head, the other hand on the hip, and one knee lifted up and out to the side'),
    'combo2': ({'A': (180, 180), 'B': (95, 92), 'L': (120, 100)}, 'one arm stretched straight out to the side, the other hanging down, and the feet wide apart'),
}
CHAINS = {'A': (2, 3, 4), 'B': (5, 6, 7), 'L': (8, 9, 10)}
USER = {
    'window': ('lmvms3begnx90', 'arms folded across the chest, standing on one leg with the other knee raised forward'),
    'pool': ('lmvmqooqsr4zc', 'standing upright with the arms hanging relaxed at the sides, one knee raised out to the side with the foot lifted'),
}

def fk(kp, chain, angles):
    a, b, c = chain
    l1, l2 = math.dist(kp[a], kp[b]), math.dist(kp[b], kp[c])
    if c == 10:  # an ankle out of frame reads short: a shin is about a thigh long
        l2 = max(l2, 0.9 * l1)
    t1, t2 = map(math.radians, angles)
    kp[b] = (kp[a][0] + l1 * math.cos(t1), kp[a][1] + l1 * math.sin(t1))
    kp[c] = (kp[b][0] + l2 * math.cos(t2), kp[b][1] + l2 * math.sin(t2))

os.makedirs(f'{ROOT}/cases', exist_ok=True)
cases = []
for name, (sid, box) in SUBJECTS.items():
    src = f'{SCENES}/{sid}/base.png'
    size = Image.open(src).size
    shutil.copy(src, f'{ROOT}/cases/{name}.png')
    det = detect(src, box)
    kp0 = [(x, y) for x, y, _ in det]
    w = max(6, round(0.0125 * size[1]))
    render(kp0, w, size, src, 0.3).save(f'{ROOT}/cases/{name}-detected.jpg')
    for pid, (spec, words) in POSES.items():
        kp = list(kp0)
        for k, ang in spec.items():
            fk(kp, CHAINS[k], ang)
        cases.append(dict(id=f'{name}-{pid}', subject=name, src=src, size=size, box=box, target=kp, w=w, words=words))
    if name in USER:
        lid, words = USER[name]
        scene = json.load(open(f'{SCENES}/{sid}/scene.json'))
        op = next(o for l in scene['layers'] if l['id'] == lid for o in l['shapes'] if o.get('t') == 'pose')
        kp = [(op['kp'][2 * i], op['kp'][2 * i + 1]) for i in range(18)]
        cases.append(dict(id=f'{name}-user', subject=name, src=src, size=size, box=box, target=kp, w=op['w'], words=words))
for c in cases:
    render(c['target'], c['w'], tuple(c['size'])).save(f"{ROOT}/cases/{c['id']}-pose.png")
    render(c['target'], c['w'], tuple(c['size']), c['src'], 0.55).save(f"{ROOT}/cases/{c['id']}-over.png")
    c['source'] = [(x, y) for x, y, _ in detect(c['src'], c['box'])]
    c['moved'] = moved(c['source'], c['target'])
    c['before'] = score(c['source'], c['target'], c['size'], c['moved'])
json.dump(cases, open(f'{ROOT}/cases.json', 'w'), indent=1)
for c in cases:
    print(c['id'], 'moved', c['moved'], 'source scores', round(c['before'], 2))
