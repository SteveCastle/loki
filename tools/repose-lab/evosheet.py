"""Rows = cases, columns = target then each strategy's seed K. evosheet.py STEPS seed names case,case,..."""
import json, sys
from PIL import Image, ImageDraw
from common import ROOT
steps, seed, names, ids = sys.argv[1], int(sys.argv[2]), sys.argv[3].split(','), sys.argv[4].split(',')
ch = 300
cells = []
for cid in ids:
    row = [(f'{ROOT}/cases/{cid}-over.png', cid)]
    for n in names:
        sc = {(r['case'], r['seed']): r for r in json.load(open(f'{ROOT}/evo/{n}-{steps}/scores.json'))}
        r = sc.get((cid, seed), {})
        row.append((f'{ROOT}/evo/{n}-{steps}/{cid}-s{seed}_edit.png', f"{n} p{r.get('pose', 0):.2f} f{r.get('face', 0):.2f} b{r.get('body', 0):.2f}"))
    cells.append(row)
im0 = Image.open(cells[0][0][0]); cw = int(im0.width * ch / im0.height)
s = Image.new('RGB', (cw * len(cells[0]), ch * len(cells)), 'black'); d = ImageDraw.Draw(s)
for r, row in enumerate(cells):
    for k, (p, label) in enumerate(row):
        try:
            s.paste(Image.open(p).convert('RGB').resize((cw, ch)), (k * cw, r * ch))
        except FileNotFoundError:
            pass
        d.rectangle([k * cw, r * ch, k * cw + cw, r * ch + 12], fill='black')
        d.text((k * cw + 2, r * ch), label, fill='yellow')
out = f"{ROOT}/evo/sheet-{'-'.join(n[:6] for n in names)}-{ids[0]}.jpg"
s.save(out, quality=85); print(out)
