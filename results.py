"""Aggregate every experiment into res/summary.json (all manuscript numbers come from here)."""
import numpy as np, pickle, json, os, glob
from analysis import *
from track import response_category

EV = './res/eval'
MAIN = ['unet', 'unetr', 'paired', 'full']
ABL = ['nomem', 'gate', 'unetssl']
SPLITS = ['test', 'extD', 'extE']
SEEDS = [0, 1, 2]
OUT = {}


def have(stem):
    return os.path.exists(f'{EV}/{stem}.pkl')


def seg_metrics(scans, ints=None):
    s = scans
    det = det_counts(s); sm = det_counts(s, 20); fu = det_counts([x for x in s if x['tp'] > 0])
    conf = [c for x in s for c in x['inst_conf']]; ok = [c for x in s for c in x['inst_tp']]
    out = dict(les_dice=float(np.nanmean([x['les_dice'] for x in s])),
               hd95=float(np.nanmedian([x['les_hd95'] for x in s])),
               lobe_dice=float(np.mean([x['lobe_dice'] for x in s])),
               sens=det['sens'], fp_scan=det['fp_scan'], prec=det['prec'], f1=det['f1'],
               sens_small=sm['sens'], sens_fu=fu['sens'], fp_fu=fu['fp_scan'],
               ece_inst=ece(conf, ok), brier_inst=float(np.mean((np.array(conf) - np.array(ok, float)) ** 2)) if conf else np.nan)
    if ints is not None:
        nres = sum(r['n_res_gt'] for r in ints)
        out['false_persist'] = sum(r['false_persist'] for r in ints) / max(nres, 1)
        out['n_res_gt'] = nres
        out['new_det'] = sum(r['n_new_gt_detected'] for r in ints) / max(sum(r['n_new_gt'] for r in ints), 1)
    return out


def per_patient_seg(scans):
    pp = {}
    for x in scans:
        d = pp.setdefault(x['pid'], dict(dice=[], tp=0, gt=0, fp=0, n=0))
        if not np.isnan(x['les_dice']):
            d['dice'].append(x['les_dice'])
        d['tp'] += sum(x['gt_hit']); d['gt'] += len(x['gt_hit']); d['fp'] += x['n_pred'] - x['tp_det']; d['n'] += 1
    return {k: dict(dice=np.mean(v['dice']) if v['dice'] else np.nan, sens=v['tp'] / v['gt'] if v['gt'] else np.nan,
                    fp=v['fp'] / v['n']) for k, v in pp.items()}


def link_metrics(ints, m):
    c = sum(r['link_' + m]['correct'] for r in ints); w = sum(r['link_' + m]['wrong'] for r in ints)
    l = sum(r['link_' + m]['linkable'] for r in ints)
    p = c / max(c + w, 1); rr = c / max(l, 1)
    return dict(prec=p, rec=rr, f1=2 * p * rr / max(p + rr, 1e-9), wrong=w, correct=c, linkable=l)


def link_pp(ints, m):
    pp = {}
    for r in ints:
        d = pp.setdefault(r['pid'], [0, 0, 0]); x = r['link_' + m]
        d[0] += x['correct']; d[1] += x['wrong']; d[2] += x['linkable']
    out = {}
    for k, (c, w, l) in pp.items():
        if l == 0 and c + w == 0:
            continue
        p = c / max(c + w, 1); rr = c / max(l, 1) if l else 1.0
        out[k] = 2 * p * rr / max(p + rr, 1e-9)
    return out


def event_block(model, split, val, cal, test):
    res = {}
    for kind in ('new', 'res'):
        clf = fit_scorer(val, kind)
        bt = best_f1_threshold(val, clf, kind)
        r = {}
        for name, mode, a, thr in [('naive', 'naive', 0, 0), ('thresh', 'thresh', 0, 0.5), ('threshF1', 'thresh', 0, bt),
                                   ('bh10', 'conformal', 0.10, 0), ('bh20', 'conformal', 0.20, 0),
                                   ('crc05', 'crc', 0.05, 0), ('crc10', 'crc', 0.10, 0), ('crc20', 'crc', 0.20, 0)]:
            sel, y, pid = event_decisions(test, clf, cal, kind, a, mode, thr)
            r[name] = event_metrics(sel, y, test, kind)
        # alpha sweep for the calibration figure
        r['sweep'] = []; r['sweep_bh'] = []
        for a in [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]:
            sel, y, pid = event_decisions(test, clf, cal, kind, a, 'crc')
            m = event_metrics(sel, y, test, kind); r['sweep'].append([a, m['false_per_patient'], m['power'], m['declared'], m['fdr']])
        for a in [0.05, 0.1, 0.2, 0.3, 0.4]:
            sel, y, pid = event_decisions(test, clf, cal, kind, a, 'conformal')
            m = event_metrics(sel, y, test, kind); r['sweep_bh'].append([a, m['fdr'], m['power'], m['declared']])
        r['coef'] = clf.coef_[0].tolist(); r['thrF1'] = bt
        res[kind] = r
    return res


def burden_block(cal, test, val):
    out = {}
    for aB in (0.1, 0.2):
        q = conformal_q(cal, aB)
        cov, wid, err = [], [], []
        for r in test:
            lr, lg, sd = burden_terms(r)
            cov.append(abs(lr - lg) <= q * sd); wid.append(2 * q * sd); err.append(abs(lr - lg))
        small = np.array([min(r['Ba_gt'], r['Bb_gt']) for r in test]); cov_a = np.array(cov, bool)
        out[f'a{int(aB * 100)}'] = dict(q=q, coverage=float(np.mean(cov)), width_med=float(np.median(wid)),
                                       abs_err_med=float(np.median(err)),
                                       miss_burden=float(np.median(small[~cov_a])) if (~cov_a).any() else None,
                                       hit_burden=float(np.median(small[cov_a])) if cov_a.any() else None)
    out['B_mae'] = float(np.mean([abs(r['Bb'] - r['Bb_gt']) for r in test] + [abs(r['Ba'] - r['Ba_gt']) for r in test])) * 100
    bh = np.array([r['Bb'] for r in test] + [r['Ba'] for r in test]); bg = np.array([r['Bb_gt'] for r in test] + [r['Ba_gt'] for r in test])
    out['B_r'] = float(np.corrcoef(bh, bg)[0, 1])
    return out


def response_block(val, cal, test, scorer_new):
    def flags(mode, a=0.1):
        sel, y, pid = event_decisions(test, scorer_new, cal, 'new', a, mode)
        f = np.zeros(len(test), bool); k = 0
        for n, r in enumerate(test):
            for c in r['cands']:
                if c['kind'] == 'new':
                    if sel[k]:
                        f[n] = True
                    k += 1
        return f
    out = {}
    L = np.array([burden_terms(r) for r in test])
    # (1) volumetric burden category from the lung-normalised ratio only
    gtb = np.array([response_category(np.exp(l), False) for l in L[:, 1]])
    ptb = np.array([response_category(np.exp(l), False) for l in L[:, 0]])
    out['burden_point'] = dict(acc=float((gtb == ptb).mean()), macro_f1=macro_f1(gtb, ptb))
    rc = []
    for aB in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
        q = conformal_q(cal, aB)
        pr = np.array([decide(r, q, False) for r in test]); d = pr >= 0
        rc.append([aB, float(d.mean()), float((pr[d] == gtb[d]).mean()) if d.any() else np.nan,
                   macro_f1(gtb[d], pr[d]) if d.any() else np.nan])
        if aB in (0.1, 0.2):
            out[f'burden_interval_a{int(aB * 100)}'] = dict(coverage=float(d.mean()),
                                                           acc=float((pr[d] == gtb[d]).mean()) if d.any() else np.nan,
                                                           macro_f1=macro_f1(gtb[d], pr[d]) if d.any() else np.nan,
                                                           point_acc_on_referred=float((ptb[~d] == gtb[~d]).mean()) if (~d).any() else np.nan)
    margin = np.abs(np.minimum(np.abs(L[:, 0] - LOG_PD), np.abs(L[:, 0] - LOG_PR)))
    hc = []
    for cov in np.linspace(0.1, 1.0, 10):
        k = max(1, int(round(cov * len(test)))); idx = np.argsort(-margin)[:k]
        hc.append([float(cov), float((ptb[idx] == gtb[idx]).mean())])
    out['rc_conformal'] = rc; out['rc_heuristic'] = hc
    out['gt_burden_counts'] = np.bincount(gtb, minlength=3).tolist()
    # (2) new-lesion progression flag at interval level
    gtn = np.array([r['n_new_gt'] > 0 for r in test])
    for name, f in [('naive', flags('naive')), ('crc10', flags('crc', 0.1)), ('bh20', flags('conformal', 0.2))]:
        tp = int((f & gtn).sum()); fp = int((f & ~gtn).sum())
        out[f'newflag_{name}'] = dict(sens=tp / max(gtn.sum(), 1), spec=float((~f & ~gtn).sum() / max((~gtn).sum(), 1)),
                                      ppv=tp / max(tp + fp, 1), n_flag=int(f.sum()))
    out['n_intervals_with_new'] = int(gtn.sum())
    # (3) combined RECIST-style category
    gtc = np.array([gt_category(r) for r in test])
    for name, f in [('naive', flags('naive')), ('crc10', flags('crc', 0.1))]:
        pc = np.array([response_category(np.exp(L[n, 0]), f[n]) for n in range(len(test))])
        out[f'combined_point_{name}'] = dict(acc=float((pc == gtc).mean()), macro_f1=macro_f1(gtc, pc))
    q = conformal_q(cal, 0.1); fc = flags('crc', 0.1)
    pr = np.array([decide(r, q, fc[n]) for n, r in enumerate(test)]); d = pr >= 0
    out['combined_interval_a10'] = dict(coverage=float(d.mean()), acc=float((pr[d] == gtc[d]).mean()) if d.any() else np.nan)
    out['gt_counts'] = np.bincount(gtc, minlength=3).tolist()
    return out


def macro_f1(y, p):
    if len(y) == 0:
        return np.nan
    f = []
    for c in range(3):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        if tp + fp + fn:
            f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


SEL = json.load(open('./res/extra.json'))['selected'] if os.path.exists('./res/extra.json') else 'unet'


def main():
    # ---- segmentation, all models and seeds
    seg = {}
    for m in MAIN + ABL + [SEL + '25', 'unet25', 'unetssl25']:
        for s in SEEDS:
            for sp in SPLITS + ['dense']:
                st = f'{m}_s{s}_{sp}'
                if have(st):
                    d = load(st); seg[st] = seg_metrics(d['scans'], d['ints'])
    for sp in ('extD', 'extE'):
        for tta in ('tent', 'anat'):
            for s in SEEDS:
                st = f'{SEL}_s{s}_{sp}_{tta}'
                if have(st):
                    d = load(st); seg[st] = seg_metrics(d['scans'], d['ints'])
    OUT['seg'] = seg
    # mean and sd over seeds
    agg = {}
    for m in MAIN + ABL:
        for sp in SPLITS:
            vals = [seg[f'{m}_s{s}_{sp}'] for s in SEEDS if f'{m}_s{s}_{sp}' in seg]
            if vals:
                agg[f'{m}_{sp}'] = {k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals])), len(vals)]
                                    for k in vals[0]}
    OUT['seg_agg'] = agg
    # ---- statistics: full vs each baseline, per patient, seed 0, Holm within split
    stats_out = {}
    for sp in SPLITS:
        if not have(f'{SEL}_s0_{sp}'):
            continue
        F = per_patient_seg(load(f'{SEL}_s0_{sp}')['scans'])
        rows = []
        for m in [x for x in MAIN + ABL if x != SEL]:
            if not have(f'{m}_s0_{sp}'):
                continue
            Bm = per_patient_seg(load(f'{m}_s0_{sp}')['scans'])
            ks = sorted(set(F) & set(Bm))
            for met in ('dice', 'sens', 'fp'):
                a = [F[k][met] for k in ks]; b = [Bm[k][met] for k in ks]
                p, rrb = wilcoxon(a, b)
                diff = np.array(a, float) - np.array(b, float)
                rows.append(dict(model=m, metric=met, p=p, rrb=rrb, diff=float(np.nanmean(diff)),
                                 ci=boot_ci(diff[~np.isnan(diff)])))
        adj = holm([r['p'] for r in rows])
        for r, a in zip(rows, adj):
            r['p_holm'] = float(a)
        stats_out[sp] = rows
    OUT['stats'] = stats_out
    # ---- association
    assoc = {}
    for sp in SPLITS + ['dense']:
        for src in ('oracle', f'{SEL}_s0'):
            st = f'oracle_{sp}' if src == 'oracle' else f'{SEL}_s0_{sp}'
            if have(st):
                ints = load(st)['ints']
                assoc[f'{src}_{sp}'] = {m: link_metrics(ints, m) for m in
                                        ['iou', 'centroid', 'prob', 'prob_novol', 'prob_nolobe', 'prob_constbd']}
                if src != 'oracle':
                    P = link_pp(ints, 'prob')
                    for m in ['iou', 'centroid']:
                        Q = link_pp(ints, m); ks = sorted(set(P) & set(Q))
                        p, rrb = wilcoxon([P[k] for k in ks], [Q[k] for k in ks])
                        assoc[f'{src}_{sp}'][m]['p_vs_prob'] = p; assoc[f'{src}_{sp}'][m]['rrb'] = rrb
    OUT['assoc'] = assoc
    # ---- events, burden, response for every model/seed with val+cal available
    ev, bu, rp = {}, {}, {}
    for m in MAIN + ABL:
        for s in SEEDS:
            key = f'{m}_s{s}'
            if not (have(f'{key}_val') and have(f'{key}_cal')):
                continue
            val = load(f'{key}_val')['ints'] + (load(f'{key}_val2')['ints'] if have(f'{key}_val2') else [])
            cal = load(f'{key}_cal')['ints'] + (load(f'{key}_cal2')['ints'] if have(f'{key}_cal2') else [])
            clf_new = fit_scorer(val, 'new')
            for sp in SPLITS + ['dense'] + [f'{x}_{t}' for x in ('extD', 'extE') for t in ('tent', 'anat')]:
                if not have(f'{key}_{sp}'):
                    continue
                test = load(f'{key}_{sp}')['ints']
                ev[f'{key}_{sp}'] = event_block(key, sp, val, cal, test)
                bu[f'{key}_{sp}'] = burden_block(cal, test, val)
                rp[f'{key}_{sp}'] = response_block(val, cal, test, clf_new)
    OUT['events'] = ev; OUT['burden'] = bu; OUT['response'] = rp
    # resampling validation of the guarantees on exchangeable development-site patients (calibration + internal test)
    rs = {}
    for m in MAIN + ABL:
        key = f'{m}_s0'
        if not (have(f'{key}_val') and have(f'{key}_cal') and have(f'{key}_test')):
            continue
        val = load(f'{key}_val')['ints'] + (load(f'{key}_val2')['ints'] if have(f'{key}_val2') else [])
        pool = load(f'{key}_cal')['ints'] + (load(f'{key}_cal2')['ints'] if have(f'{key}_cal2') else []) + load(f'{key}_test')['ints']
        rs[key] = dict(new=crc_resample(pool, fit_scorer(val, 'new'), 'new', [0.05, 0.1, 0.2], reps=500),
                       res=crc_resample(pool, fit_scorer(val, 'res'), 'res', [0.05, 0.1, 0.2], reps=500),
                       burden=burden_resample(pool, [0.1, 0.2], reps=500))
    OUT['resample'] = rs
    # local recalibration at unseen sites: part of the site's patients serve as a local calibration set
    lr = {}
    key = f'{SEL}_s0'
    val = load(f'{key}_val')['ints'] + load(f'{key}_val2')['ints']
    for sp in ('extD', 'extE', 'dense'):
        if not have(f'{key}_{sp}'):
            continue
        pool = load(f'{key}_{sp}')['ints']; npat = len({x['pid'] for x in pool}); nt = npat // 2
        lr[sp] = dict(new=crc_resample(pool, fit_scorer(val, 'new'), 'new', [0.1, 0.2], n_test=nt, reps=300),
                      res=crc_resample(pool, fit_scorer(val, 'res'), 'res', [0.1, 0.2], n_test=nt, reps=300),
                      burden=burden_resample(pool, [0.1, 0.2], n_test=nt, reps=300), n_local=npat - nt)
    OUT['local_recal'] = lr
    # scorer feature ablation: without cross-examination evidence (columns 2, 3) and without boundary uncertainty (4)
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    sa = {}
    pool = load(f'{key}_cal')['ints'] + load(f'{key}_cal2')['ints'] + load(f'{key}_test')['ints']
    for kind in ('new', 'res'):
        Xv, yv, _ = cand_arrays(val, kind); Xp, yp, pp = cand_arrays(pool, kind)
        for name, cols in (('all', [0, 1, 2, 3, 4, 5]), ('no_evidence', [0, 1, 4, 5]), ('no_uncertainty', [0, 1, 2, 3, 5])):
            clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xv[:, cols], yv)
            auc = roc_auc_score(yp, clf.predict_proba(Xp[:, cols])[:, 1])
            class Wrap:
                def __init__(s, c, cols): s.c, s.cols = c, cols
                def predict_proba(s, X): return s.c.predict_proba(X[:, s.cols])
            rsm = crc_resample(pool, Wrap(clf, cols), kind, [0.1], reps=300)
            sa[f'{kind}_{name}'] = dict(auc=float(auc), power=rsm[0.1]['mean_power'], false=rsm[0.1]['mean_false'])
    OUT['scorer_ablation'] = sa
    json.dump(OUT, open('./res/summary.json', 'w'), indent=1, default=float)
    print('saved', len(seg), len(ev))


if __name__ == '__main__':
    main()
