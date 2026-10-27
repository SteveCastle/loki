"""Repose lab: evolve the Repose finish (prompt + pose reference) against a fixed case set.

  python lab.py run            run every queued generation (generations/gen*.json) that isn't finished; resumable
  python lab.py board          rewrite LEADERBOARD.md from results/
  python lab.py rescore        re-score results that predate a metric change (--all: every one)
  touch PAUSE                  stop after the current chunk (delete it and `run` again to go on)

Each strategy runs all cases x seeds in batches through one engine process (loki-retouch --jobs), so the models load once
per chunk. Finished images are never redone. Scores: pose = share of the joints the target moves that RTMPose finds within
0.25 torso of the target in the output; face / body = masked SSIM of the face and torso+hips against the source (arms left
out), i.e. how much of the person that should not change did not. See SEARCH_SPACE.md for what has been tried.
"""
import glob, json, os, subprocess, sys, time
from PIL import Image
from common import ROOT, detect, score, angle_score
from fidelity import fidelity

LAB = os.path.dirname(os.path.abspath(__file__))
_built = [os.path.join(LAB, '..', 'loki-retouch', d, 'release', 'loki-retouch.exe') for d in ('target-lab', 'target')]
ENGINE = os.environ.get('REPOSE_ENGINE', next((p for p in _built if os.path.exists(p)), _built[-1]))
MODELS = os.environ.get('REPOSE_MODELS', os.path.expanduser('~/bin/models'))
MODEL_ARGS = ['--dit', f'{MODELS}/qwen_image_2.1_int8_convrot.safetensors', '--text-encoder', f'{MODELS}/qwen3vl_8b_int8_convrot.safetensors',
              '--vae', f'{MODELS}/qwen_image_2.1_vae_bf16.safetensors']
EDGE = 640
CHUNK = 40

def paused():
    return os.path.exists(os.path.join(LAB, 'PAUSE'))

def fitness(r):
    return 0.25 * r['pose'] + 0.25 * r.get('angle', r['pose']) + 0.25 * r['face'] + 0.25 * r['body']

def run_strategy(gen, name, strat, cases, steps, seeds, tag='', extra=(), seed0=1000):
    from prompts import STRATS
    prompt_fn, rep, inp = strat[:3]
    sizes = strat[3] if len(strat) > 3 else None
    out = f'{ROOT}/evo/{name}{tag}-{steps}'
    os.makedirs(out, exist_ok=True)
    jobs = []
    for c in cases:
        src = f"{ROOT}/cases/{c['subject']}-{EDGE}.png"
        first = f"{ROOT}/cases/{c['id']}-{EDGE}-warp.png" if inp == 'warp' else src
        # rep: one or more of the case's renders joined by '+', 'src' = the picture itself, 'none' = no reference
        refs = [] if rep == 'none' else [src if r == 'src' else f"{ROOT}/cases/{c['subject']}-crop.png" if r == 'crop' else f"{ROOT}/cases/{c['id']}-{EDGE}-{r}.png" for r in rep.split('+')]
        if inp == 'warp':
            refs = [src]
        prompt = prompt_fn(c)
        open(f'{out}/{c["id"]}.txt', 'w').write(prompt)
        for k in range(seeds):
            o = f'{out}/{c["id"]}-s{k}_edit.png'
            if not os.path.exists(o):
                jobs.append(dict(input=first, prompt=prompt, refs=refs, seed=seed0 + k, out=o, **({'sizes': sizes} if sizes else {})))
    for i in range(0, len(jobs), CHUNK):
        if paused():
            print('  paused'); return None
        jf = f'{out}/jobs.jsonl'
        open(jf, 'w').write('\n'.join(json.dumps(j) for j in jobs[i:i + CHUNK]))
        r = subprocess.run([ENGINE, '-q', '--jobs', jf, '--steps', str(steps)] + list(extra) + MODEL_ARGS, capture_output=True, text=True)
        if r.returncode:
            print(f'  engine failed: {r.stderr[-600:]}')
    res = []
    for c in cases:
        W, H = c['size']
        for k in range(seeds):
            e = f'{out}/{c["id"]}-s{k}_edit.png'
            if not os.path.exists(e):
                continue
            im = Image.open(e); sx = im.width / W
            xs = [p[0] for p in c['target']] + [c['box'][0], c['box'][0] + c['box'][2]]
            ys = [p[1] for p in c['target']] + [c['box'][1], c['box'][1] + c['box'][3]]
            x0, y0 = max(0, min(xs)), max(0, min(ys))
            box = [x0 * sx, y0 * sx, (min(W, max(xs)) - x0) * sx, (min(H, max(ys)) - y0) * sx]
            det = [(x / sx, y / sx, s) for x, y, s in detect(e, box)]
            f = fidelity(f"{ROOT}/cases/{c['subject']}.png", e, c)
            res.append(dict(case=c['id'], seed=k, pose=score(det, c['target'], c['size'], c['moved']), angle=angle_score(det, c['target'], c['moved']), face=f.get('face', 0), body=f.get('body', 0)))
    json.dump(res, open(f'{out}/scores.json', 'w'), indent=1)
    m = lambda key: sum(r[key] for r in res) / max(1, len(res))
    summary = dict(gen=gen, name=name + tag, strategy=name, extra=list(extra), seed0=seed0, steps=steps, seeds=seeds, n=len(res), pose=m('pose'), angle=m('angle'), face=m('face'), body=m('body'),
                   zero=sum(1 for r in res if r['pose'] == 0), per_case=res)
    summary['fitness'] = fitness(summary)
    os.makedirs(f'{LAB}/results', exist_ok=True)
    json.dump(summary, open(f'{LAB}/results/{gen}--{name}{tag}-{steps}.json', 'w'), indent=1)
    return summary

def rescore():
    """Re-score every finished result from its images (after a metric change); runs on CPU."""
    cases = json.load(open(f'{ROOT}/cases.json'))
    for p in sorted(glob.glob(f'{LAB}/results/*.json')):
        r = json.load(open(p))
        if 'angle' in r and '--all' not in sys.argv:
            continue
        out = f"{ROOT}/evo/{r['name']}-{r['steps']}"  # name includes the tag
        res = []
        for c in cases:
            W, H = c['size']
            for k in range(r['seeds']):
                e = f'{out}/{c["id"]}-s{k}_edit.png'
                if not os.path.exists(e):
                    continue
                im = Image.open(e); sx = im.width / W
                xs = [q[0] for q in c['target']] + [c['box'][0], c['box'][0] + c['box'][2]]
                ys = [q[1] for q in c['target']] + [c['box'][1], c['box'][1] + c['box'][3]]
                x0, y0 = max(0, min(xs)), max(0, min(ys))
                box = [x0 * sx, y0 * sx, (min(W, max(xs)) - x0) * sx, (min(H, max(ys)) - y0) * sx]
                det = [(x / sx, y / sx, s) for x, y, s in detect(e, box)]
                f = fidelity(f"{ROOT}/cases/{c['subject']}.png", e, c)
                res.append(dict(case=c['id'], seed=k, pose=score(det, c['target'], c['size'], c['moved']), angle=angle_score(det, c['target'], c['moved']), face=f.get('face', 0), body=f.get('body', 0)))
        m = lambda key: sum(x[key] for x in res) / max(1, len(res))
        r.update(n=len(res), pose=m('pose'), angle=m('angle'), face=m('face'), body=m('body'), zero=sum(1 for x in res if x['pose'] == 0), per_case=res)
        r['fitness'] = fitness(r)
        json.dump(r, open(p, 'w'), indent=1)
        json.dump(res, open(f'{out}/scores.json', 'w'), indent=1)
        print('rescored', r['name'], f"angle {r['angle']:.2f}")
    board()

def board():
    rows = [json.load(open(p)) for p in glob.glob(f'{LAB}/results/*.json')]
    rows.sort(key=lambda r: -r['fitness'])
    lines = ['# Repose lab leaderboard', '', f'All cases x seeds at {EDGE} px. fitness = (pose + angle + face + body) / 4. angle: moved bones pointing within 30 deg of target (length-free). '
             'pose: moved joints landing on target (0 = source). face/body: SSIM vs the source where nothing should change.', '',
             '| # | strategy | gen | steps | fitness | pose | angle | face | body | no move | n |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows):
        lines.append(f"| {i + 1} | {r['name']} | {r['gen']} | {r['steps']} | {r['fitness']:.3f} | {r['pose']:.2f} | {r.get('angle', float('nan')):.2f} | {r['face']:.3f} | {r['body']:.3f} | {r['zero']} | {r['n']} |")
    open(f'{LAB}/LEADERBOARD.md', 'w').write('\n'.join(lines) + '\n')
    return rows

def run():
    from prompts import STRATS
    cases = json.load(open(f'{ROOT}/cases.json'))
    for gf in sorted(glob.glob(f'{LAB}/generations/gen*.json')):
        g = json.load(open(gf))
        # a strategy entry is a name, or {"name", "tag", "extra": [engine flags], "seed0", "steps"} for a sampling variant
        for entry in g['strategies']:
            e = entry if isinstance(entry, dict) else {'name': entry}
            name, tag, steps = e['name'], e.get('tag', ''), e.get('steps', g['steps'])
            if os.path.exists(f"{LAB}/results/{g['gen']}--{name}{tag}-{steps}.json"):
                continue
            if paused():
                print('paused'); return
            t = time.time()
            r = run_strategy(g['gen'], name, STRATS[name], cases, steps, g['seeds'], tag, e.get('extra', []), e.get('seed0', 1000))
            if r is None:
                return
            print(f"{g['gen']} {name:22s} fit {r['fitness']:.3f} pose {r['pose']:.2f} face {r['face']:.3f} body {r['body']:.3f} no-move {r['zero']}/{r['n']} {time.time() - t:.0f}s", flush=True)
            board()

if __name__ == '__main__':
    {'run': run, 'board': board, 'rescore': rescore}[sys.argv[1]]()
