"""Supervised training of PulmoTrack variants and comparison models.

usage: python train.py <config> <seed> [epochs]
configs: full, noanat, nomem, nossl, unet, unetr, paired
"""
import numpy as np, torch, time, sys, json
from scipy import ndimage as ndi
from model import PulmoTrackNet, UNet, UNETRLite, seg_loss
from common import load_split, norm_hu, set_seed, warp

torch.set_num_threads(2)


def build(cfg):
    if cfg == 'full':   return PulmoTrackNet(True, True), True
    if cfg == 'noanat': return PulmoTrackNet(False, True), True
    if cfg == 'nomem':  return PulmoTrackNet(True, False), True
    if cfg == 'nossl':  return PulmoTrackNet(True, True), False
    if cfg == 'unet':   return UNet(1), False
    if cfg == 'unetr':  return UNETRLite(grid=(4, 6, 6)), False
    if cfg == 'paired': return UNet(3), False
    if cfg == 'gate':   return PulmoTrackNet(True, False), False
    if cfg == 'unetssl': return UNet(1), True


def corrupt_prior(mask, lung, rng):
    """Simulate imperfect prior-scan predictions: dropped lesions, spurious
    blobs, blur and under-confidence."""
    lab, n = ndi.label(mask > 0.5)
    m = mask.astype(np.float32).copy()
    for c in range(1, n + 1):
        if rng.random() < 0.15:
            m[lab == c] = 0
    if rng.random() < 0.25:
        pts = np.argwhere(lung)
        if len(pts):
            c = pts[rng.integers(len(pts))]
            r = rng.uniform(1.0, 2.0)
            zz, yy, xx = np.ogrid[:48, :64, :64]
            m[((zz - c[0]) * 1.2) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= r * r] = 1
    s = rng.uniform(0, 1.0)
    if s > 0.2:
        m = ndi.gaussian_filter(m, s)
    return np.clip(m * rng.uniform(0.5, 1.0), 0, 1)


def augment(h, rng):
    h = h.astype(np.float32) * rng.uniform(0.97, 1.03) + rng.uniform(-25, 25)
    if rng.random() < 0.5:
        h = h + rng.uniform(0, 20) * rng.standard_normal(h.shape).astype(np.float32)
    if rng.random() < 0.3:
        h = ndi.gaussian_filter(h, rng.uniform(0.2, 0.6))
    return h


if __name__ == '__main__':
    cfg, seed = sys.argv[1], int(sys.argv[2])
    frac = 1.0
    if cfg.endswith('25'):
        cfg, frac = cfg[:-2], 0.25
    tag = cfg + ('25' if frac < 1 else '')
    EP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    set_seed(seed)
    net, use_ssl = build(cfg)
    if use_ssl:
        net.enc.load_state_dict(torch.load('./ckpt/ssl_encoder.pt'))
    D = load_split('train', grids=True)
    meta, img, lobe, inst = D['meta'], D['img'], D['lobe'], D['inst']
    keep = np.arange(len(img))
    if frac < 1:      # label-efficiency runs: first fraction of training patients per site
        bysite = {}
        for k, e in enumerate(meta):
            if e['tp'] == 0:
                bysite.setdefault(e['site'], []).append(e['pid'])
        pids = {p for v in bysite.values() for p in v[:int(round(len(v) * frac))]}
        keep = np.array([k for k, e in enumerate(meta) if e['pid'] in pids])
    rng = np.random.default_rng(seed)
    B = 2
    steps = EP * (len(keep) // B)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    uses_mem = cfg in ('full', 'noanat', 'nossl')
    t0, log = time.time(), []
    for ep in range(EP):
        perm = rng.permutation(keep); tot = 0
        for b in range(0, len(perm) - B + 1, B):
            X, L, G, M, T = [], [], [], [], []
            for i in perm[b:b + B]:
                # nnU-Net-style foreground oversampling: half of the crops contain a lesion voxel
                fg = np.argwhere(inst[i] > 0) if rng.random() < 0.5 else []
                if len(fg):
                    c = fg[rng.integers(len(fg))]
                    z = int(np.clip(c[0] - rng.integers(4, 28), 0, 16)); y = int(np.clip(c[1] - rng.integers(6, 42), 0, 16))
                    x = int(np.clip(c[2] - rng.integers(6, 42), 0, 16))
                else:
                    z, y, x = rng.integers(0, 17), rng.integers(0, 17), rng.integers(0, 17)
                sl = (slice(z, z + 32), slice(y, y + 48), slice(x, x + 48))
                cur = norm_hu(augment(img[i], rng))
                tp = meta[i]['tp']
                prior_m = np.zeros_like(cur); prior_i = np.zeros_like(cur); dt = 0.0
                if tp > 0 and (uses_mem or cfg == 'paired'):
                    g = D['grid'][i]
                    pm = warp((inst[i - 1] > 0).astype(np.float32), g, 1)
                    prior_m = corrupt_prior(pm, lobe[i] > 0, rng)
                    dt = meta[i]['t'] - meta[i - 1]['t']
                    if cfg == 'paired':
                        prior_i = norm_hu(warp(augment(img[i - 1], rng), g, 1))
                if cfg == 'paired':
                    X.append(np.stack([cur, prior_i, prior_m])[(slice(None),) + sl])
                else:
                    X.append(cur[sl][None])
                L.append(lobe[i][sl]); G.append(inst[i][sl] > 0); M.append(prior_m[sl][None]); T.append(dt)
            X = torch.from_numpy(np.stack(X)); Lt = torch.from_numpy(np.stack(L)).long()
            Gt = torch.from_numpy(np.stack(G)); Mt = torch.from_numpy(np.stack(M)); Tt = torch.tensor(T, dtype=torch.float32)
            out = net(X, Mt, Tt) if uses_mem else net(X)
            loss = seg_loss(out, Lt, Gt)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item()
        log.append(tot / (len(perm) // B))
        print(tag, seed, ep, round(log[-1], 4), round(time.time() - t0), flush=True)
    if cfg == 'unetr':   # store grid for full-volume inference by interpolating positional embedding
        pass
    torch.save(net.state_dict(), f'./ckpt/{tag}_s{seed}.pt')
    json.dump(dict(loss=log, sec=time.time() - t0, epochs=EP, n_scans=int(len(keep))), open(f'./res/trainlog_{tag}_s{seed}.json', 'w'))
