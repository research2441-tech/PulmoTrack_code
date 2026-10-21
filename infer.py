"""Sequential longitudinal inference with optional test-time adaptation.

usage: python infer.py <cfg> <seed> <split> [tta: none|tent|anat] [steps]
Saves lesion probability (float16), lung probability (uint8) and lobe labels.
"""
import numpy as np, torch, sys, time, json, copy
import torch.nn as nn
from model import PulmoTrackNet, UNet, UNETRLite
from train import build
from common import load_split, norm_hu, patients, warp
from sliding import slide, PATCH

torch.set_num_threads(2)
cfg, seed, split = sys.argv[1], int(sys.argv[2]), sys.argv[3]
tta = sys.argv[4] if len(sys.argv) > 4 else 'none'
K = int(sys.argv[5]) if len(sys.argv) > 5 else 3
LR = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-4
MAXP = int(sys.argv[7]) if len(sys.argv) > 7 else 10 ** 6
base_cfg = cfg[:-2] if cfg.endswith('25') else cfg
uses_mem = base_cfg in ('full', 'noanat', 'nossl')
PRIOR = json.load(open('./res/train_priors.json'))


def norm_params(net):
    ps = []
    for m in net.modules():
        if isinstance(m, (nn.InstanceNorm3d, nn.LayerNorm)):
            ps += [m.weight, m.bias]
    return ps


def tta_loss(out, kind):
    p = out['p']; pl = torch.softmax(out['lobe'], 1)
    H_les = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    H_lobe = -(pl * torch.log(pl.clamp(min=1e-6))).sum(1, keepdim=True)
    if kind == 'tent':
        return H_les.mean() + H_lobe.mean()
    # anatomy-constrained, uncertainty-filtered objective
    plung = 1 - pl[:, :1]
    rel = (H_les < 0.4).float() * (plung > 0.5).float()          # reliable lung voxels only
    ent = (H_les * rel).sum() / rel.sum().clamp(min=1) + (H_lobe * (H_lobe < 0.5).float()).mean()
    frac = pl[:, 1:].sum((2, 3, 4)) / pl[:, 1:].sum((1, 2, 3, 4)).clamp(min=1)
    pi = torch.tensor(PRIOR['lobe_frac']).view(1, 5)
    kl = (pi * (torch.log(pi) - torch.log(frac.clamp(min=1e-6)))).sum()
    lv = torch.log(plung.sum())
    zv = ((lv - PRIOR['loglung_mu']) / PRIOR['loglung_sd']) ** 2
    return ent + 1.0 * kl + 0.1 * zv


def fwd(net, xc, Mc, dt):
    if uses_mem:
        return net(xc, Mc, dt.expand(xc.shape[0]) if dt is not None else None)
    return net(xc)


def run(net, x, M=None, dt=None):
    with torch.no_grad():
        return slide(lambda a, b: fwd(net, a, b, dt), x, M)


def adapt_and_run(net0, x, M, dt, rng):
    if tta == 'none':
        return run(net0, x, M, dt)
    net = copy.deepcopy(net0)      # episodic adaptation: reset for every examination
    for p in net.parameters():
        p.requires_grad_(False)
    ps = norm_params(net)
    for p in ps:
        p.requires_grad_(True)
    opt = torch.optim.Adam(ps, LR)
    for _ in range(K):
        loss = 0
        for _c in range(2):        # two random training-size crops per step
            z, y, xx = rng.integers(0, 17), rng.integers(0, 17), rng.integers(0, 17)
            sl = (slice(None), slice(None), slice(z, z + PATCH[0]), slice(y, y + PATCH[1]), slice(xx, xx + PATCH[2]))
            loss = loss + tta_loss(fwd(net, x[sl], M[sl], dt), tta)
        opt.zero_grad(); loss.backward(); opt.step()
    return run(net, x, M, dt)


if __name__ == '__main__':
    net, _ = build(base_cfg)
    net.load_state_dict(torch.load(f'./ckpt/{cfg}_s{seed}.pt'))
    net.eval()
    D = load_split(split, grids=True)
    meta, img = D['meta'], D['img']
    N = len(img)
    P = np.zeros((N, 48, 64, 64), np.float16); PL = np.zeros((N, 48, 64, 64), np.uint8)
    LB = np.zeros((N, 48, 64, 64), np.uint8)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for pat in patients(meta)[:MAXP]:
        prev = None
        for i in pat:
            x = torch.from_numpy(norm_hu(img[i]))[None, None]
            M = torch.zeros_like(x); dt = torch.zeros(1)
            if meta[i]['tp'] > 0 and prev is not None:
                g = D['grid'][i]
                Mw = warp(prev, g, 1)
                M = torch.from_numpy(Mw)[None, None]; dt = torch.tensor([meta[i]['t'] - meta[i - 1]['t']], dtype=torch.float32)
                if base_cfg == 'paired':
                    pi = norm_hu(warp(img[i - 1].astype(np.float32), g, 1))
                    x = torch.from_numpy(np.stack([norm_hu(img[i]), pi, Mw]))[None]
            elif base_cfg == 'paired':
                x = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
            out = adapt_and_run(net, x, M, dt, rng)
            p = out['p'][0, 0].numpy(); pl = out['lobe_prob'][0].numpy()
            P[i] = p; PL[i] = np.round((1 - pl[0]) * 255); LB[i] = pl.argmax(0)
            prev = p.astype(np.float32)
    tag = f'{cfg}_s{seed}_{split}' + ('' if tta == 'none' else f'_{tta}')
    if len(sys.argv) > 6:
        tag += f'_lr{LR:g}_k{K}_p{MAXP}'
    np.savez(f'./res/pred/{tag}.npz', p=P, plung=PL, lobe=LB)
    print(tag, round(time.time() - t0), flush=True)
