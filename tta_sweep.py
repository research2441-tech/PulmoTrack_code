"""Segmentation metrics of TTA strength settings on the first patients of site D (seed 0, selected backbone)."""
import numpy as np, json, glob, os, sys
sys.path.insert(0, '.')
from common import load_split, patients
from track import instances, match_to_gt
D = load_split('extD'); meta, inst = D['meta'], D['inst']
out = {}
for f in sorted(glob.glob('./res/pred/unetssl_s0_extD*_p30.npz')) + ['./res/pred/unetssl_s0_extD%s.npz' % t for t in ('', '_anat', '_tent')]:
    P = np.load(f); p, lb = P['p'], P['lobe']
    idx = [i for pat in patients(meta)[:30] for i in pat]
    tp = fp = gt = 0; dice = []
    for i in idx:
        lab, objs = instances(p[i].astype(np.float32), lb[i]); mp, gts = match_to_gt(lab, objs, inst[i])
        tp += len(mp); fp += len(objs) - len(mp); gt += len(gts)
        g = inst[i] > 0; q = p[i] > 0.5
        if g.any():
            dice.append(2 * (g & q).sum() / (g.sum() + q.sum()))
    out[os.path.basename(f)[:-4]] = dict(dice=float(np.mean(dice)), sens=tp / gt, fp=fp / len(idx))
    print(os.path.basename(f), out[os.path.basename(f)[:-4]])
json.dump(out, open('./res/tta_sweep.json', 'w'), indent=1)
