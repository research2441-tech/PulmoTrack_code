"""Lesion instances, longitudinal association, change events, burden and metrics."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from common import warp

SP = np.array([6.0, 5.0, 5.0])
VOX_ML = 0.15
STRUCT = np.ones((3, 3, 3), bool)
MINVOX = 3


# ------------------------------------------------------------------ instances
def instances(p, lobe, tau=0.5, minvox=MINVOX):
    lab, n = ndi.label(p > tau, STRUCT)
    out = []
    if n == 0:
        return lab, out
    idx = np.arange(1, n + 1)
    cnt = ndi.sum(np.ones_like(p), lab, idx)
    for k, c in zip(idx, cnt):
        if c < minvox:
            lab[lab == k] = 0
            continue
        m = lab == k
        pv = p[m]
        lb = np.bincount(lobe[m], minlength=6)[1:]
        out.append(dict(k=int(k), nvox=int(c), soft=float(pv.sum()), meanp=float(pv.mean()), maxp=float(pv.max()),
                        unc=float((pv * (1 - pv)).mean()), c=(np.argwhere(m).mean(0) * SP),
                        lobe=int(lb.argmax() + 1) if lb.sum() else 0))
    return lab, out


def gt_instances(inst):
    out = []
    for g in np.unique(inst):
        if g == 0:
            continue
        m = inst == g
        if m.sum() < MINVOX:
            continue
        out.append(dict(k=int(g), nvox=int(m.sum()), c=np.argwhere(m).mean(0) * SP))
    return out


def match_to_gt(lab, objs, inst, thr=0.1):
    """One-to-one assignment of predicted components to GT lesion ids (IoU)."""
    gts = [g for g in np.unique(inst) if g > 0 and (inst == g).sum() >= MINVOX]
    if not objs or not gts:
        return {}, gts
    I = np.zeros((len(objs), len(gts)))
    for a, o in enumerate(objs):
        m = lab == o['k']
        for b, g in enumerate(gts):
            gm = inst == g
            inter = (m & gm).sum()
            if inter:
                I[a, b] = inter / (m.sum() + gm.sum() - inter)
    r, c = linear_sum_assignment(-I)
    mp = {objs[a]['k']: int(gts[b]) for a, b in zip(r, c) if I[a, b] >= thr}
    return mp, gts


# ---------------------------------------------------------------- association
def warped_centroids(lab_prev, objs_prev, grid):
    """Warp each prior instance into the follow-up frame and return centroid/volume."""
    out = []
    for o in objs_prev:
        w = warp((lab_prev == o['k']).astype(np.float32), grid, 1)
        if w.sum() < 0.3:
            out.append((o['c'], o['nvox'], w)); continue
        out.append(((np.argwhere(w > 0.3).mean(0) * SP) if (w > 0.3).any() else o['c'], o['nvox'], w))
    return out


class AssocParams:
    """Parameters of the probabilistic association model, fitted on validation
    patients with reference instances."""
    def __init__(self, sig=(6.0, 5.0, 5.0), s0=0.25, s1=0.004, mu=0.0, kappa=2.0,
                 bd=(0.0, 0.0), dd=(0.0, 0.0), eta=0.0, gate=100.0, nu=3.0):
        self.sig, self.s0, self.s1, self.mu, self.kappa = np.array(sig), s0, s1, mu, kappa
        self.bd, self.dd, self.eta, self.gate, self.nu = bd, dd, eta, gate, nu


def assoc_cost(P, Q, W, dt, prm, use_vol=True, use_lobe=True):
    C = np.full((len(P), len(Q)), np.inf)
    for i, (ci, vi, _) in enumerate(W):
        for j, q in enumerate(Q):
            d = (q['c'] - ci) / prm.sig
            dd = float(d @ d)
            if dd > prm.gate:
                continue
            c = 0.5 * (prm.nu + 3) * np.log1p(dd / prm.nu)      # multivariate Student-t residual
            if use_vol:
                lr = np.log(q['nvox'] / max(vi, 1))
                var = prm.s0 ** 2 + prm.s1 ** 2 * dt
                c += 0.5 * (lr - prm.mu) ** 2 / var + 0.5 * np.log(var)
            if use_lobe and P[i]['lobe'] and q['lobe'] and P[i]['lobe'] != q['lobe']:
                c += prm.kappa
            C[i, j] = c
    return C


def solve_with_birth_death(C, b, d):
    """Augmented assignment: rows = prior lesions + birth slots, cols = current + death slots."""
    n, m = C.shape
    BIG = 1e6
    A = np.full((n + m, m + n), BIG)
    A[:n, :m] = np.where(np.isfinite(C), C, BIG)
    for i in range(n):
        A[i, m + i] = d[i]
    for j in range(m):
        A[n + j, j] = b[j]
    A[n:, m:] = 0.0
    r, c = linear_sum_assignment(A)
    return [(i, j) for i, j in zip(r, c) if i < n and j < m and A[i, j] < BIG / 2]


def logit_size_prior(coef, nvox):
    a, k = coef
    z = a + k * np.log(nvox)
    return 1 / (1 + np.exp(-z))


def associate(method, P, Q, lab_prev, lab_cur, grid, dt, prm, W=None):
    if not P or not Q:
        return [], W
    if W is None:
        W = warped_centroids(lab_prev, P, grid)
    if method == 'iou':
        I = np.zeros((len(P), len(Q)))
        for i, (_, _, w) in enumerate(W):
            wm = w > 0.3
            for j, q in enumerate(Q):
                m = lab_cur == q['k']
                inter = (wm & m).sum()
                if inter:
                    I[i, j] = inter / (wm.sum() + m.sum() - inter)
        r, c = linear_sum_assignment(-I)
        return [(i, j) for i, j in zip(r, c) if I[i, j] > getattr(prm, 'iou_thr', 0.05)], W
    if method.startswith('centroid'):
        g = prm.cgate
        D = np.array([[np.linalg.norm(q['c'] - ci) for q in Q] for (ci, _, _) in W])
        r, c = linear_sum_assignment(D)
        return [(i, j) for i, j in zip(r, c) if D[i, j] <= g], W
    # probabilistic association (proposed) and its ablations
    use_vol = 'novol' not in method
    use_lobe = 'nolobe' not in method
    C = assoc_cost(P, Q, W, dt, prm, use_vol, use_lobe)
    if 'constbd' in method:
        b = np.full(len(Q), prm.const_b); d = np.full(len(P), prm.const_d)
    else:
        pb = logit_size_prior(prm.bd, np.array([q['nvox'] for q in Q]))
        pd = logit_size_prior(prm.dd, np.array([p['nvox'] for p in P]))
        b = -np.log(pb / (1 - pb)) + prm.eta
        d = -np.log(pd / (1 - pd)) + prm.eta
    return solve_with_birth_death(C, b, d), W


# -------------------------------------------------------------- event evidence
def event_features(kind, o, lab, p_other_warped, dt, own_p):
    """Features of a candidate change event.
    kind 'new': o is an unmatched follow-up lesion; p_other_warped is the prior
    probability map resampled to the follow-up frame.  kind 'res': o is an
    unmatched prior lesion; p_other_warped is the follow-up map resampled to the
    prior frame."""
    m = lab == o['k']
    nb = ndi.binary_dilation(m, STRUCT, iterations=2)
    ev = float(p_other_warped[nb].max()) if nb.any() else 0.0
    ev_mean = float(p_other_warped[m].mean())
    return np.array([o['meanp'], np.log(o['nvox']), ev, ev_mean, o['unc'] * 10, np.log(max(dt, 1) / 100)])


# ------------------------------------------------------------------ metrics
def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def hd95(a, b):
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    sa = a & ~ndi.binary_erosion(a, STRUCT); sb = b & ~ndi.binary_erosion(b, STRUCT)
    da = ndi.distance_transform_edt(~sb, sampling=SP)[sa]
    db = ndi.distance_transform_edt(~sa, sampling=SP)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def response_category(r, new):
    if new:
        return 2
    if r >= 1.728:
        return 2
    if r <= 0.343:
        return 0
    return 1
