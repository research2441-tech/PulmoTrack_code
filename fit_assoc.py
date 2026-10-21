"""Fit the probabilistic association model on validation patients, using the
instances predicted by the segmentation model (paired with reference identities):
residual scales, growth variance, lobe-consistency penalty and size-dependent
birth/death priors.  The shared offset eta and the parameters of the comparison
methods (centroid gate, IoU threshold, constant birth/death cost) are tuned on the
same patients by identity F1."""
import numpy as np, json, sys, copy
from sklearn.linear_model import LogisticRegression
from common import load_split, patients
from track import *

tag = sys.argv[1]
D = load_split('val', grids=True)
meta, inst = D['meta'], D['inst']
f = np.load(f'./res/pred/{tag}.npz'); P_, LB = f['p'], f['lobe']
items, res, lrs, dts, lobechg, Xb, yb, Xd, yd = [], [], [], [], [], [], [], [], []
for pat in patients(meta):
    c = {}
    for i in pat:
        lab, objs = instances(P_[i].astype(np.float32), LB[i]); mp, _ = match_to_gt(lab, objs, inst[i]); c[i] = (lab, objs, mp)
    for a, b in zip(pat[:-1], pat[1:]):
        g = D['grid'][b].astype(np.float32); dt = meta[b]['t'] - meta[a]['t']
        (la, A, ma), (lb_, B, mb) = c[a], c[b]
        W = warped_centroids(la, A, g) if A else None
        items.append((c[a], c[b], g, dt, W))
        gA = [ma.get(o['k']) for o in A]; gB = [mb.get(o['k']) for o in B]
        for n, o in enumerate(A):
            if gA[n] is not None and gA[n] in gB:
                q = B[gB.index(gA[n])]
                res.append(q['c'] - W[n][0]); lrs.append(np.log(q['nvox'] / o['nvox'])); dts.append(dt)
                lobechg.append(o['lobe'] != q['lobe'])
        for n, o in enumerate(B):
            Xb.append(np.log(o['nvox'])); yb.append(not (gB[n] is not None and gB[n] in gA))
        for n, o in enumerate(A):
            Xd.append(np.log(o['nvox'])); yd.append(not (gA[n] is not None and gA[n] in gB))
res = np.array(res); lrs = np.array(lrs); dts = np.array(dts)
sig = 1.4826 * np.median(np.abs(res - np.median(res, 0)), 0) + 1.0
mu = float(np.median(lrs))
A_ = np.stack([np.ones_like(dts), dts], 1)
coef = np.linalg.lstsq(A_, (lrs - mu) ** 2, rcond=None)[0]
s0 = float(np.sqrt(max(coef[0], 0.01))); s1 = float(np.sqrt(max(coef[1], 1e-6)))
q = max(np.mean(lobechg), 0.01); kappa = float(np.clip(-np.log(q), 0.5, 5))
lb = LogisticRegression(C=1e3).fit(np.array(Xb)[:, None], yb); ld = LogisticRegression(C=1e3).fit(np.array(Xd)[:, None], yd)
prm = AssocParams(sig=sig, s0=s0, s1=s1, mu=mu, kappa=kappa, bd=(float(lb.intercept_[0]), float(lb.coef_[0, 0])),
                  dd=(float(ld.intercept_[0]), float(ld.coef_[0, 0])), eta=0.0, gate=100.0, nu=3.0)
print('pairs', len(res), 'sig', sig.round(2), 'mu', round(mu, 3), 's0', round(s0, 3), 's1', round(s1, 4), 'kappa', round(kappa, 2),
      'lobechg', round(float(np.mean(lobechg)), 3), 'bd', np.round(prm.bd, 2), 'dd', np.round(prm.dd, 2))


def _set(p, **kw):
    p = copy.copy(p)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def f1(mth, p):
    cor = wr = lk = 0
    for (la, Pa, ma), (lb_, Qb, mb), g, dt, W in items:
        links, _ = associate(mth, Pa, Qb, la, lb_, g, dt, p, W)
        gP = [ma.get(o['k']) for o in Pa]; gQ = [mb.get(o['k']) for o in Qb]
        lk += sum(1 for x in gP if x is not None and x in gQ)
        cor += sum(1 for i, j in links if gP[i] is not None and gP[i] == gQ[j])
        wr += sum(1 for i, j in links if (gP[i] is not None or gQ[j] is not None) and gP[i] != gQ[j])
    pr = cor / max(cor + wr, 1); rc = cor / max(lk, 1)
    return 2 * pr * rc / max(pr + rc, 1e-9)


prm.cgate, prm.const_b, prm.const_d, prm.iou_thr = 15.0, 3.0, 3.0, 0.05
grid_eta = np.arange(-3, 6.01, 0.5); sc = [f1('prob', _set(prm, eta=e)) for e in grid_eta]
prm.eta = float(grid_eta[int(np.argmax(sc))])
grid_g = [5, 7.5, 10, 12.5, 15, 20, 25, 30]; sg = [f1('centroid', _set(prm, cgate=g)) for g in grid_g]
prm.cgate = float(grid_g[int(np.argmax(sg))])
grid_i = [0.01, 0.05, 0.1, 0.2, 0.3]; si = [f1('iou', _set(prm, iou_thr=t)) for t in grid_i]
prm.iou_thr = float(grid_i[int(np.argmax(si))])
grid_c = np.arange(-2, 8.01, 0.5); scc = [f1('prob_constbd', _set(prm, const_b=c_, const_d=c_)) for c_ in grid_c]
prm.const_b = prm.const_d = float(grid_c[int(np.argmax(scc))])
scores = {'centroid': max(sg), 'iou': max(si), 'prob': max(sc), 'prob_constbd': max(scc)}
order = ['centroid', 'iou', 'prob_constbd', 'prob']        # simplest first on ties
best = max(scores.values())
event_matcher = [m for m in order if scores[m] >= best - 1e-9][0]
print('validation identity F1', {k: round(v, 4) for k, v in scores.items()}, '-> event matcher', event_matcher)
print('eta', prm.eta, round(max(sc), 3), '| cgate', prm.cgate, round(max(sg), 3), '| iou', prm.iou_thr, round(max(si), 3),
      '| const', prm.const_b, round(max(scc), 3))
json.dump(dict(sig=list(map(float, sig)), s0=s0, s1=s1, mu=mu, kappa=kappa, bd=prm.bd, dd=prm.dd, eta=prm.eta, gate=100.0, nu=3.0,
               cgate=prm.cgate, iou_thr=prm.iou_thr, const_b=prm.const_b, const_d=prm.const_d, npairs=len(res),
               lobechg=float(np.mean(lobechg)), fitted_on=tag, event_matcher=event_matcher, val_f1=scores,
               tune=dict(eta=[list(map(float, grid_eta)), list(map(float, sc))], cgate=[grid_g, list(map(float, sg))],
                         iou=[grid_i, list(map(float, si))])),
          open(f'./res/assoc/{tag.rsplit("_val", 1)[0]}.json', 'w'), indent=1)
