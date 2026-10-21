"""Event control, burden intervals, response decisions and statistics."""
import numpy as np, pickle, json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from track import response_category

EV = './res/eval'
EPS = 1e-4
LOG_PD, LOG_PR = np.log(1.728), np.log(0.343)


def load(stem):
    return pickle.load(open(f'{EV}/{stem}.pkl', 'rb'))


# ----------------------------------------------------------------- events
def cand_arrays(ints, kind):
    X, y, pid = [], [], []
    for r in ints:
        for c in r['cands']:
            if c['kind'] == kind:
                X.append(c['x']); y.append(c['y']); pid.append(r['pid'])
    return np.array(X).reshape(-1, 6), np.array(y, bool), np.array(pid)


def fit_scorer(val_ints, kind):
    X, y, _ = cand_arrays(val_ints, kind)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(X, y)
    return clf


def conformal_pvalues(s_test, s_cal, y_cal):
    """Conformal p-values for selection (Jin & Candes): calibration scores of
    true events are set to -inf so only false candidates enter the count."""
    n = len(s_cal)
    nulls = np.sort(s_cal[~y_cal])
    # number of null calibration scores >= s
    ge = len(nulls) - np.searchsorted(nulls, s_test, side='left')
    return (1.0 + ge) / (n + 1.0)


def bh(p, alpha):
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    thr = alpha * np.arange(1, m + 1) / m
    ok = p[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    sel = np.zeros(m, bool); sel[o[:k]] = True
    return sel


def crc_threshold(cal_ints, clf, kind, alpha, B=3.0):
    """Conformal risk control at patient level: smallest score threshold whose
    inflated empirical risk of false change events per patient is at most alpha."""
    X, y, pid = cand_arrays(cal_ints, kind)
    pats = sorted({r['pid'] for r in cal_ints}); n = len(pats)
    s = clf.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    idx = {p: k for k, p in enumerate(pats)}
    grid = np.unique(np.concatenate([s, [1.0 + 1e-9]]))
    for lam in grid:                      # risk is non-increasing in lam
        cnt = np.zeros(n)
        for k in np.where((s >= lam) & ~y)[0]:
            cnt[idx[pid[k]]] += 1
        R = np.minimum(cnt, B).mean()
        if n / (n + 1) * R + B / (n + 1) <= alpha:
            return float(lam)
    return float(grid[-1])


def event_decisions(ints, clf, cal_ints, kind, alpha, mode='conformal', thr=0.5):
    X, y, pid = cand_arrays(ints, kind)
    if len(X) == 0:
        return np.zeros(0, bool), y, pid
    s = clf.predict_proba(X)[:, 1]
    if mode == 'naive':
        sel = np.ones(len(X), bool)
    elif mode == 'thresh':
        sel = s >= thr
    elif mode == 'crc':
        sel = s >= crc_threshold(cal_ints, clf, kind, alpha)
    else:
        Xc, yc, _ = cand_arrays(cal_ints, kind)
        sc = clf.predict_proba(Xc)[:, 1]
        pv = conformal_pvalues(s, sc, yc)
        sel = bh(pv, alpha)
    return sel, y, pid


def event_metrics(sel, y, ints, kind):
    n_gt = sum(r['n_new_gt'] if kind == 'new' else r['n_res_gt'] for r in ints)
    d = int(sel.sum()); fd = int((sel & ~y).sum()); td = int((sel & y).sum())
    npat = len({r['pid'] for r in ints})
    return dict(declared=d, false=fd, true=td, fdr=fd / max(d, 1), power=td / max(n_gt, 1), n_gt=n_gt,
                false_per_patient=fd / max(npat, 1), n_patients=npat,
                n_cand=len(y), n_true_cand=int(y.sum()))


def best_f1_threshold(val_ints, clf, kind):
    X, y, _ = cand_arrays(val_ints, kind)
    s = clf.predict_proba(X)[:, 1]
    best, bt = -1, 0.5
    for t in np.linspace(0.05, 0.95, 19):
        sel = s >= t
        tp = (sel & y).sum(); f1 = 2 * tp / max(sel.sum() + y.sum(), 1)
        if f1 > best:
            best, bt = f1, t
    return bt


# ----------------------------------------------------------------- burden
def burden_terms(r):
    lr_hat = np.log((r['Bb'] + EPS) / (r['Ba'] + EPS))
    lr_gt = np.log((r['Bb_gt'] + EPS) / (r['Ba_gt'] + EPS))
    sd = np.sqrt(r['Ub'] / (r['Vb'] + 3) ** 2 + r['Ua'] / (r['Va'] + 3) ** 2) + 0.02
    return lr_hat, lr_gt, sd


def conformal_q(cal_ints, alpha):
    s = np.array([abs(a - b) / c for a, b, c in map(burden_terms, cal_ints)])
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1])


def gt_category(r):
    _, lr, _ = burden_terms(r)
    return response_category(np.exp(lr), r['n_new_gt'] > 0)


def decide(r, q, new_flag):
    lr, _, sd = burden_terms(r)
    lo, hi = lr - q * sd, lr + q * sd
    if new_flag or lo >= LOG_PD:
        return 2
    if hi <= LOG_PR:
        return 0
    if lo > LOG_PR and hi < LOG_PD:
        return 1
    return -1   # indeterminate: referred to the reader


# ----------------------------------------------------------------- statistics
def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps)
    adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def wilcoxon(a, b):
    a, b = np.asarray(a), np.asarray(b); ok = ~(np.isnan(a) | np.isnan(b)); d = a[ok] - b[ok]
    if np.allclose(d, 0):
        return 1.0, 0.0
    p = stats.wilcoxon(d, zero_method='wilcox').pvalue
    r = d[d != 0]; rk = stats.rankdata(np.abs(r))
    rrb = (rk[r > 0].sum() - rk[r < 0].sum()) / rk.sum()
    return float(p), float(rrb)


def boot_ci(x, n=2000, seed=0, fn=np.nanmean):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    v = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))


def per_patient(scans, key, fn=np.nanmean, cond=None):
    d = {}
    for s in scans:
        if cond is None or cond(s):
            d.setdefault(s['pid'], []).append(s[key])
    return {k: fn(v) if not np.all(np.isnan(np.array(v, float))) else np.nan for k, v in d.items()}


def det_counts(scans, small=None):
    """Lesion sensitivity, FP/scan and lesion F1 (pooled)."""
    tp = fn = 0; fp = 0
    for s in scans:
        hits = [h for h, z in zip(s['gt_hit'], s['gt_sizes']) if small is None or (z < small)]
        tp += sum(hits); fn += len(hits) - sum(hits)
        fp += s['n_pred'] - s['tp_det']
    sens = tp / max(tp + fn, 1); prec = sum(s['tp_det'] for s in scans) / max(sum(s['n_pred'] for s in scans), 1)
    return dict(sens=sens, fp_scan=fp / len(scans), prec=prec, f1=2 * sens * prec / max(sens + prec, 1e-9))


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok, float)
    if len(conf) == 0:
        return np.nan
    e = 0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return float(e)


def crc_from_arrays(s, y, pid, pats, alpha, B=3.0):
    """Vectorised conformal risk control threshold from candidate arrays."""
    n = len(pats)
    idx = {p: k for k, p in enumerate(pats)}
    grid = np.unique(np.concatenate([s, [1.0 + 1e-9]]))
    cnt = np.zeros((n, len(grid)))
    fs = ~y
    for k in np.where(fs)[0]:
        cnt[idx[pid[k]], :np.searchsorted(grid, s[k], side='right')] += 1
    R = np.minimum(cnt, B).mean(0)
    ok = n / (n + 1) * R + B / (n + 1) <= alpha
    return float(grid[np.argmax(ok)]) if ok.any() else float(grid[-1])


def crc_resample(pool_ints, clf, kind, alphas, n_test=60, reps=500, seed=0):
    """Repeated random calibration/test partitions of exchangeable patients:
    mean realised false calls per test patient and power for each alpha."""
    X, y, pid = cand_arrays(pool_ints, kind)
    s = clf.predict_proba(X)[:, 1] if len(X) else np.zeros(0)
    pats = np.array(sorted({r['pid'] for r in pool_ints}))
    ngt = {}
    for r in pool_ints:
        ngt[r['pid']] = ngt.get(r['pid'], 0) + (r['n_new_gt'] if kind == 'new' else r['n_res_gt'])
    rng = np.random.default_rng(seed)
    out = {a: [] for a in alphas}
    for _ in range(reps):
        te = set(rng.choice(pats, n_test, replace=False)); ca = [p for p in pats if p not in te]
        mc = np.isin(pid, ca); mt = ~mc
        for a in alphas:
            lam = crc_from_arrays(s[mc], y[mc], pid[mc], ca, a)
            sel = mt & (s >= lam)
            fp = (sel & ~y).sum() / n_test
            pw = (sel & y).sum() / max(sum(ngt[p] for p in te), 1)
            out[a].append((fp, pw))
    return {a: dict(mean_false=float(np.mean([v[0] for v in out[a]])), q90_false=float(np.percentile([v[0] for v in out[a]], 90)),
                    frac_exceed=float(np.mean([v[0] > a for v in out[a]])), mean_power=float(np.mean([v[1] for v in out[a]])))
            for a in alphas}


def burden_resample(pool_ints, alphas, n_test=60, reps=500, seed=0):
    pats = np.array(sorted({r['pid'] for r in pool_ints})); rng = np.random.default_rng(seed)
    L = {}
    for r in pool_ints:
        L.setdefault(r['pid'], []).append(burden_terms(r))
    out = {a: [] for a in alphas}
    for _ in range(reps):
        te = set(rng.choice(pats, n_test, replace=False))
        cal = np.array([abs(a - b) / c for p in pats if p not in te for a, b, c in L[p]])
        tst = [t for p in te for t in L[p]]
        for a in alphas:
            k = int(np.ceil((len(cal) + 1) * (1 - a))); q = np.sort(cal)[min(k, len(cal)) - 1]
            out[a].append(np.mean([abs(x - g) <= q * sd for x, g, sd in tst]))
    return {a: dict(mean_cov=float(np.mean(out[a])), q10_cov=float(np.percentile(out[a], 10))) for a in alphas}
