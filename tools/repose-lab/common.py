import json, math, os, subprocess
from PIL import Image, ImageDraw
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SCENES = os.path.join(os.environ['APPDATA'], 'Lowkey Media Viewer', 'scenes')
MODEL = os.path.join(os.environ['APPDATA'], 'Lowkey Media Viewer', 'models', 'rtmpose-body', 'model.onnx')
EMBED = r'C:\Users\steph\dev\loki\media-server\bin\embed.exe'
LIMBS = [(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13),(1,0),(0,14),(14,16),(0,15),(15,17)]
COLORS = [(255,0,0),(255,85,0),(255,170,0),(255,255,0),(170,255,0),(85,255,0),(0,255,0),(0,255,85),(0,255,170),(0,255,255),(0,170,255),(0,85,255),(0,0,255),(85,0,255),(170,0,255),(255,0,255),(255,0,170),(255,0,85)]

def detect(path, box):
    out = subprocess.run([EMBED, '--pose', '--model=' + MODEL, '--image=' + path, '--box=%d,%d,%d,%d' % tuple(box)], capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr)
    return [(j['x'], j['y'], j['score']) for j in json.loads(out.stdout)['joints']]

def render(kp, w, size, base=None, dim=0.0):
    """kp: 18 (x,y). As pose.ts drawPose: joints, then each limb at 60% over them."""
    if base is None:
        img = Image.new('RGBA', size, (0, 0, 0, 255))
    else:
        img = Image.open(base).convert('RGBA')
        if dim:
            img = Image.blend(img, Image.new('RGBA', size, (0, 0, 0, 255)), dim)
    d = ImageDraw.Draw(img)
    for j, (x, y) in enumerate(kp):
        d.ellipse([x - w, y - w, x + w, y + w], fill=COLORS[j] + (255,))
    for i, (a, b) in enumerate(LIMBS):
        layer = Image.new('RGBA', size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).line([kp[a], kp[b]], fill=COLORS[i] + (153,), width=2 * w)
        for p in (kp[a], kp[b]):
            ImageDraw.Draw(layer).ellipse([p[0] - w, p[1] - w, p[0] + w, p[1] + w], fill=COLORS[i] + (153,))
        img = Image.alpha_composite(img, layer)
    return img.convert('RGB')

def torso(kp):
    return math.dist(kp[1], ((kp[8][0] + kp[11][0]) / 2, (kp[8][1] + kp[11][1]) / 2))

def moved(src, target):
    """Elbows, wrists, knees, ankles the target puts more than 0.3 torso from where they are."""
    t = torso(target)
    return [j for j in (3, 4, 6, 7, 9, 10, 12, 13) if math.dist(src[j][:2], target[j]) > 0.3 * t]

def score(det, target, size, joints):
    """Share of joints within 0.25 torso of target (in frame only); best of L/R as given or swapped."""
    W, H = size
    torso_ = torso(target)
    swap = {2:5,3:6,4:7,5:2,6:3,7:4,8:11,9:12,10:13,11:8,12:9,13:10}
    best = 0
    for sw in (False, True):
        ok = n = 0
        for j in joints:
            tx, ty = target[j]
            if not (0 <= tx < W and 0 <= ty < H):
                continue
            dj = det[swap[j] if sw else j]
            n += 1
            ok += math.dist((dj[0], dj[1]), (tx, ty)) < 0.25 * torso_
        best = max(best, ok / max(1, n))
    return best

BONES = [(2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13)]

def angle_score(det, target, joints):
    """Share of the bones the target moves (into a moved joint) that point within 30 degrees of the target's,
    whatever their length or whether they leave the frame; best of as-is / left-right swapped."""
    swap = {2:5,3:6,4:7,5:2,6:3,7:4,8:11,9:12,10:13,11:8,12:9,13:10}
    bones = [(a, b) for a, b in BONES if b in joints]
    if not bones:
        return 0.0
    best = 0.0
    for sw in (False, True):
        ok = 0
        for a, b in bones:
            da, db = (det[swap[a]], det[swap[b]]) if sw else (det[a], det[b])
            vx, vy = db[0] - da[0], db[1] - da[1]
            tx, ty = target[b][0] - target[a][0], target[b][1] - target[a][1]
            n = math.hypot(vx, vy) * math.hypot(tx, ty)
            ok += n > 0 and (vx * tx + vy * ty) / n > math.cos(math.radians(30))
        best = max(best, ok / len(bones))
    return best
