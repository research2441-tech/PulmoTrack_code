"""Generate the PulmoTrack-Sim cohort (patient-level splits)."""
import numpy as np, json, os, sys
from multiprocessing import Pool
from sim import simulate_patient, site_sequence
OUT = './data'

def job(a):
    pid, base, seed, ntp = a[:4]
    rng = np.random.default_rng(seed + 7)
    seq = site_sequence(rng, base, n_tp=ntp)
    return simulate_patient(pid, seq, seed, n_tp=ntp, process=a[4] if len(a) > 4 else None)

def build(name, specs):
    with Pool(2) as p:
        res = p.map(job, specs, chunksize=4)
    img = np.stack([e['img'] for r in res for e in r]); lobe = np.stack([e['lobe'] for r in res for e in r])
    inst = np.stack([e['inst'] for r in res for e in r]); meta = [e['meta'] for r in res for e in r]
    np.savez(f'{OUT}/{name}.npz', img=img, lobe=lobe, inst=inst)
    json.dump(meta, open(f'{OUT}/{name}.json', 'w'))
    print(name, img.shape, flush=True)

if __name__ == '__main__':
    specs = {k: [] for k in ['train', 'val', 'cal', 'test', 'extD', 'extE', 'ssl']}
    pid = 0
    for si, s in enumerate('ABC'):
        for k in range(120):
            split = 'train' if k < 60 else 'val' if k < 70 else 'cal' if k < 100 else 'test'
            specs[split].append((pid, s, 1000 + pid, 3)); pid += 1
    for s in 'DE':
        for k in range(80):
            specs['ext' + s].append((pid, s, 1000 + pid, 3)); pid += 1
    for k in range(400):
        specs['ssl'].append((pid, 'ABC'[k % 3], 1000 + pid, 1)); pid += 1
    for k in range(45):     # evaluation-only stress cohort with crowded multifocal disease
        specs.setdefault('dense', []).append((pid, 'ABC'[k % 3], 1000 + pid, 3, 'dense')); pid += 1
    for s in 'ABC':        # additional development-site patients: scorer fitting (val2) and calibration (cal2)
        for k in range(20):
            specs.setdefault('val2', []).append((pid, s, 1000 + pid, 3)); pid += 1
        for k in range(50):
            specs.setdefault('cal2', []).append((pid, s, 1000 + pid, 3)); pid += 1
    for name in (sys.argv[1:] or specs):
        build(name, specs[name])
