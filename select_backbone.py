"""Select the segmentation backbone on validation patients by pooled lesion-level F1."""
import numpy as np, json, os, sys
from common import load_split
from track import instances, match_to_gt

CANDS = ['unet', 'unetr', 'paired', 'full', 'nomem', 'gate', 'unetssl']
G = {sp: load_split(sp) for sp in ('val', 'val2')}
res = {}
for m in CANDS:
    tp = fp = gt = 0; dice = []
    ok = True
    for sp in ('val', 'val2'):
        f = f'./res/pred/{m}_s0_{sp}.npz'
        if not os.path.exists(f):
            ok = False; break
        P = np.load(f); p, lb = P['p'], P['lobe']; inst = G[sp]['inst']
        for i in range(len(p)):
            lab, objs = instances(p[i].astype(np.float32), lb[i]); mp, gts = match_to_gt(lab, objs, inst[i])
            tp += len(mp); fp += len(objs) - len(mp); gt += len(gts)
            g = inst[i] > 0; q = p[i] > 0.5
            if g.any():
                dice.append(2 * (g & q).sum() / (g.sum() + q.sum()))
    if not ok:
        continue
    sens = tp / gt; prec = tp / max(tp + fp, 1)
    res[m] = dict(f1=2 * sens * prec / (sens + prec), sens=sens, prec=prec, dice=float(np.mean(dice)))
    print(m, {k: round(v, 4) for k, v in res[m].items()}, flush=True)
sel = max(res, key=lambda k: res[k]['f1'])
ex = json.load(open('./res/extra.json')) if os.path.exists('./res/extra.json') else {}
ex.update(selected=sel, validation=res)
json.dump(ex, open('./res/extra.json', 'w'), indent=1)
print('selected', sel)
