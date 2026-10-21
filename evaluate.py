"""Per-scan and per-interval evaluation records for one prediction file.

usage: python evaluate.py <tag> <split>   (reads res/pred/<tag>.npz)
"""
import numpy as np, json, sys, pickle, time
from scipy import ndimage as ndi
from common import load_split, patients, warp
from track import *

METHODS = ['iou', 'centroid', 'prob', 'prob_novol', 'prob_nolobe', 'prob_constbd']


def load_params(model='full_s0'):
    import os
    p = f'./res/assoc/{model}.json'
    if not os.path.exists(p):
        p = './res/assoc/full_s0.json'
    d = json.load(open(p))
    prm = AssocParams(sig=d['sig'], s0=d['s0'], s1=d['s1'], mu=d['mu'], kappa=d['kappa'], bd=d['bd'], dd=d['dd'],
                      eta=d['eta'], gate=d['gate'])
    prm.cgate = d['cgate']; prm.const_b = d['const_b']; prm.const_d = d['const_d']; prm.iou_thr = d.get('iou_thr', 0.05)
    prm.nu = d.get('nu', 3.0)
    prm.event_matcher = d.get('event_matcher', 'prob')
    return prm


def link_eval(links, P, Q, mpP, mpQ):
    """Identity scoring: a link is correct when both ends map to the same reference
    lesion, an identity error when the ends differ and at least one is a real lesion,
    and neutral when both ends are false positives."""
    gP = [mpP.get(o['k']) for o in P]; gQ = [mpQ.get(o['k']) for o in Q]
    linkable = sum(1 for g in gP if g is not None and g in gQ)
    correct = sum(1 for i, j in links if gP[i] is not None and gP[i] == gQ[j])
    wrong = sum(1 for i, j in links if (gP[i] is not None or gQ[j] is not None) and gP[i] != gQ[j])
    return dict(n_links=len(links), correct=correct, wrong=wrong, linkable=linkable)


def evaluate(tag, split, prm, oracle=False, methods=METHODS):
    D = load_split(split, grids=True)
    meta, img, lobe_gt, inst = D['meta'], D['img'], D['lobe'], D['inst']
    if oracle:
        P_ = (inst > 0).astype(np.float32); PL = (lobe_gt > 0).astype(np.float32); LB = lobe_gt
    else:
        f = np.load(f'./res/pred/{tag}.npz')
        P_, PL, LB = f['p'], f['plung'], f['lobe']
    scans, ints = [], []
    for pat in patients(meta):
        cache = {}
        for i in pat:
            p = P_[i].astype(np.float32)
            pl = PL[i].astype(np.float32) / (1.0 if oracle else 255.0)
            lb = LB[i]
            lab, objs = instances(p, lb)
            mp, gts = match_to_gt(lab, objs, inst[i])
            gmask = inst[i] > 0; pmask = p > 0.5
            rec = dict(i=i, pid=meta[i]['pid'], tp=meta[i]['tp'], site=meta[i]['site'],
                       n_gt=len(gts), tp_det=len(mp), n_pred=len(objs),
                       les_dice=dice(pmask, gmask) if gmask.any() else np.nan,
                       les_hd95=hd95(pmask, gmask),
                       lobe_dice=float(np.mean([dice(lb == k, lobe_gt[i] == k) for k in range(1, 6)])),
                       B_hat=float(p.sum() / max(pl.sum(), 1)), B_gt=float(gmask.sum() / max((lobe_gt[i] > 0).sum(), 1)),
                       V_hat=float(p.sum()), V_gt=float(gmask.sum()), U=float((p * (1 - p)).sum()),
                       inst_conf=[o['meanp'] for o in objs], inst_tp=[o['k'] in mp for o in objs],
                       gt_sizes=[int((inst[i] == g).sum()) for g in gts], gt_hit=[g in mp.values() for g in gts])
            # small lesions (< 20 voxels, about 3 mL)
            scans.append(rec)
            cache[i] = (p, lab, objs, mp, rec)
        for a, b in zip(pat[:-1], pat[1:]):
            pa, laba, Pa, mpa, ra = cache[a]; pb, labb, Qb, mpb, rb = cache[b]
            g = D['grid'][b].astype(np.float32)
            dt = meta[b]['t'] - meta[a]['t']
            gt_a = {x for x in np.unique(inst[a]) if x and (inst[a] == x).sum() >= MINVOX}
            gt_b = {x for x in np.unique(inst[b]) if x and (inst[b] == x).sum() >= MINVOX}
            new_gt, res_gt = gt_b - gt_a, gt_a - gt_b
            # false persistence: a reference lesion that resolved is still segmented at follow-up
            fpers = 0
            for gidr in res_gt:
                wm = warp((inst[a] == gidr).astype(np.float32), g, 1) > 0.3
                wm = ndi.binary_dilation(wm, STRUCT) & (inst[b] == 0)
                if wm.any() and (pb[wm] > 0.5).sum() >= MINVOX:
                    fpers += 1
            rec = dict(pid=meta[b]['pid'], a=a, b=b, site=meta[b]['site'], dt=dt, n_new_gt=len(new_gt), n_res_gt=len(res_gt),
                       false_persist=fpers,
                       n_new_gt_detected=sum(1 for x in new_gt if x in mpb.values()),
                       n_res_gt_detected_prior=sum(1 for x in res_gt if x in mpa.values()))
            W = None
            for mth in methods:
                links, W = associate(mth, Pa, Qb, laba, labb, g, dt, prm, W)
                rec['link_' + mth] = link_eval(links, Pa, Qb, mpa, mpb)
                if mth == prm.event_matcher:
                    main_links = links
            # change-event candidates under the matcher selected on validation patients
            mi = {i for i, _ in main_links}; mj = {j for _, j in main_links}
            pa_w = warp(pa, g, 1)
            cands = []
            for j, q in enumerate(Qb):
                if j in mj:
                    continue
                gid = mpb.get(q['k'])
                truth = gid is not None and gid in new_gt
                cands.append(dict(kind='new', x=list(event_features('new', q, labb, pa_w, dt, pb)), y=bool(truth),
                                  gid=gid))
            for i, o in enumerate(Pa):
                if i in mi:
                    continue
                w = W[i][2] if W is not None else warp((laba == o['k']).astype(np.float32), g, 1)
                m = w > 0.3
                if not m.any():
                    m = w >= w.max() * 0.5 if w.max() > 0 else np.zeros_like(w, bool)
                nb = ndi.binary_dilation(m, STRUCT, iterations=2)
                ev = float(pb[nb].max()) if nb.any() else 0.0
                evm = float(pb[m].mean()) if m.any() else 0.0
                gid = mpa.get(o['k'])
                truth = gid is not None and gid in res_gt
                cands.append(dict(kind='res', x=[o['meanp'], float(np.log(o['nvox'])), ev, evm, o['unc'] * 10,
                                                 float(np.log(max(dt, 1) / 100))], y=bool(truth), gid=gid))
            rec['cands'] = cands
            rec['Ba'], rec['Bb'], rec['Ba_gt'], rec['Bb_gt'] = ra['B_hat'], rb['B_hat'], ra['B_gt'], rb['B_gt']
            rec['Va'], rec['Vb'], rec['Ua'], rec['Ub'] = ra['V_hat'], rb['V_hat'], ra['U'], rb['U']
            ints.append(rec)
    return scans, ints


if __name__ == '__main__':
    tag, split = sys.argv[1], sys.argv[2]
    oracle = tag == 'oracle'
    t0 = time.time()
    sel = json.load(open('./res/extra.json'))['selected'] if oracle else None
    prm = load_params('_'.join(tag.split('_')[:2]) if not oracle else f'{sel}_s0')
    scans, ints = evaluate(tag, split, prm, oracle)
    pickle.dump(dict(scans=scans, ints=ints), open(f'./res/eval/{tag}.pkl' if not oracle
                                                    else f'./res/eval/oracle_{split}.pkl', 'wb'))
    print(tag, split, round(time.time() - t0), flush=True)
