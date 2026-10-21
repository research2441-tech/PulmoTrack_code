"""Sliding-window inference with the training patch size (32x48x48) and Gaussian weighting."""
import numpy as np, torch
PATCH = (32, 48, 48)
STARTS = [(z, y, x) for z in (0, 16) for y in (0, 16) for x in (0, 16)]
_w = None
def weight():
    global _w
    if _w is None:
        g = [np.exp(-0.5 * ((np.arange(n) - (n - 1) / 2) / (n / 4)) ** 2) for n in PATCH]
        _w = torch.tensor(np.einsum('i,j,k->ijk', *g), dtype=torch.float32)
    return _w
def slide(fn, x, M=None):
    """fn(xcrops, Mcrops) -> dict(p=(B,1,...), lobe=(B,6,...)); all eight windows are run as one batch."""
    W = weight(); D, H, Wd = x.shape[2:]
    sls = [(slice(None), slice(None), slice(z, z + PATCH[0]), slice(y, y + PATCH[1]), slice(xx, xx + PATCH[2]))
           for z, y, xx in STARTS]
    xb = torch.cat([x[s] for s in sls], 0)
    Mb = None if M is None else torch.cat([M[s] for s in sls], 0)
    o = fn(xb, Mb)
    P = torch.zeros(1, 1, D, H, Wd); L = torch.zeros(1, 6, D, H, Wd); S = torch.zeros(1, 1, D, H, Wd)
    lp = torch.softmax(o['lobe'], 1)
    for k, s in enumerate(sls):
        P[s] += o['p'][k:k + 1] * W; L[s] += lp[k:k + 1] * W; S[s] += W
    return dict(p=P / S, lobe_prob=L / S)
