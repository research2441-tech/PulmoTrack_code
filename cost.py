"""Measure parameters and wall-clock costs on the experiment hardware (two CPU cores, idle)."""
import time, json, numpy as np, torch, sys
sys.path.insert(0, '.')
from train import build
from common import load_split, norm_hu, warp
from sliding import slide
from demons import demons_grid
from track import instances, warped_centroids, associate, event_features
from evaluate import load_params
torch.set_num_threads(2)
X = json.load(open('./res/extra.json')); SEL = X['selected']
D = load_split('test', grids=True); img = D['img']
out = {}
for cfg in (SEL, 'unetr'):
    net, _ = build(cfg); net.load_state_dict(torch.load(f'./ckpt/{cfg}_s0.pt')); net.eval()
    npar = sum(p.numel() for p in net.parameters())
    x = torch.from_numpy(norm_hu(img[1]))[None, None]
    with torch.no_grad():
        slide(lambda a, b: net(a), x)
        t = time.time()
        for _ in range(5):
            slide(lambda a, b: net(a), x)
    out['backbone' if cfg == SEL else 'unetr'] = dict(params=f'{npar / 1e6:.2f} M', time=f'{(time.time() - t) / 5:.2f} s per exam')
t = time.time()
for k in range(3):
    demons_grid(img[2 * k], img[2 * k + 1])
out['registration'] = dict(params='–', time=f'{(time.time() - t) / 3:.2f} s per pair')
prm = load_params(f'{SEL}_s0')
P = np.load(f'./res/pred/{SEL}_s0_test.npz'); p, lb = P['p'], P['lobe']
t = time.time(); n = 0
for b in range(1, 30, 3):
    a = b - 1
    la, A = instances(p[a].astype(np.float32), lb[a]); lb_, B = instances(p[b].astype(np.float32), lb[b])
    g = D['grid'][b].astype(np.float32)
    links, W = associate('prob', A, B, la, lb_, g, 90.0, prm)
    pw = warp(p[a].astype(np.float32), g, 1)
    for q in B:
        event_features('new', q, lb_, pw, 90.0, p[b])
    n += 1
out['association'] = dict(params='–', time=f'{(time.time() - t) / n:.2f} s per pair')
tr = json.load(open(f'./res/trainlog_{SEL}_s0.json'))
out['training'] = dict(params='–', time=f'{tr["sec"] / 60:.0f} min')
log = open('./res/infer.log').read().split('\n')
tt = [l for l in log if l.startswith(f'{SEL}_s0_extD_anat ')]
if tt:
    out['tta'] = dict(params='–', time=f'{float(tt[-1].split()[-1]) / 240:.2f} s per exam')
X['cost'] = out
json.dump(X, open('./res/extra.json', 'w'), indent=1)
print(out)
