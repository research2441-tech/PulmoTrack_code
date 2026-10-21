"""Summarise every split of PulmoTrack-Sim (Table of cohort characteristics) -> res/cohort.json."""
import json, os, numpy as np
from collections import Counter
VOX_ML = 6 * 5 * 5 / 1000.0

def summarise(meta):
    pids = sorted({m['pid'] for m in meta})
    inst = [(m['pid'], m['tp'], l) for m in meta for l in m['lesions'] if l['nvox'] >= 3]   # reference instances
    present = {(p, t, l['id']) for p, t, l in inst}
    obs = [l['nvox'] * VOX_ML for _, _, l in inst]
    by_p = {}
    for m in meta:
        by_p.setdefault(m['pid'], []).append(m)
    ntp = {p: max(m['tp'] for m in v) for p, v in by_p.items()}
    new = sum(1 for p, t, l in inst if t > 0 and (p, t - 1, l['id']) not in present)
    res = sum(1 for p, t, l in inst if t < ntp[p] and (p, t + 1, l['id']) not in present)
    dts = [m['t'] - prev['t'] for v in by_p.values() for prev, m in zip(sorted(v, key=lambda e: e['tp']), sorted(v, key=lambda e: e['tp'])[1:])]
    sw = sum(1 for v in by_p.values() for a, b in zip(sorted(v, key=lambda e: e['tp']), sorted(v, key=lambda e: e['tp'])[1:]) if a['site'] != b['site'])
    return dict(patients=len(pids), exams=len(meta), lesions=len(inst), new=new, resolved=res,
                sites=dict(Counter(m['site'] for m in meta)), kinds=dict(Counter(l['kind'] for _, _, l in inst)),
                vol_med=float(np.median(obs)), vol_iqr=[float(np.percentile(obs, 25)), float(np.percentile(obs, 75))],
                dt_med=float(np.median(dts)) if dts else 0, process=dict(Counter(by_p[p][0]['process'] for p in pids)),
                contrast=float(np.mean([m['contrast'] for m in meta])), site_switch=sw,
                small=sum(1 for x in obs if x < 3.0), les_per_exam=len(inst) / len(meta))

if __name__ == '__main__':
    out = {}
    for sp in ['train', 'val', 'cal', 'test', 'extD', 'extE', 'ssl', 'val2', 'cal2', 'dense']:
        if os.path.exists(f'data/{sp}.json'):
            out[sp] = summarise(json.load(open(f'data/{sp}.json')))
    json.dump(out, open('res/cohort.json', 'w'), indent=1)
