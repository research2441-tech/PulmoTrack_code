"""Lung-normalised anatomical coordinates, prior-to-current warps and
moment-based affine alignment."""
import numpy as np
from scipy import ndimage as ndi

Z, Y, X = 48, 64, 64
SIDES = {'R': (1, 2, 3), 'L': (4, 5)}


def _smooth(a, k=3):
    if len(a) < k:
        return a
    p = np.pad(a, (k // 2, k // 2), mode='edge')
    return np.convolve(p, np.ones(k) / k, mode='valid')


def lung_profile(lobe):
    """Per lung: craniocaudal extent and slice-wise in-plane bounding boxes."""
    prof = {}
    for s, ids in SIDES.items():
        m = np.isin(lobe, ids)
        if m.sum() < 50:
            prof[s] = None; continue
        zs = np.where(m.any((1, 2)))[0]
        z0, z1 = float(zs.min()), float(zs.max() + 1)
        bb = np.zeros((Z, 4))
        valid = np.zeros(Z, bool)
        for z in zs:
            yx = np.argwhere(m[z])
            if len(yx) < 4:
                continue
            bb[z] = [yx[:, 0].min(), yx[:, 0].max() + 1, yx[:, 1].min(), yx[:, 1].max() + 1]
            valid[z] = True
        vz = np.where(valid)[0]
        for j in range(4):   # fill invalid slices by nearest valid and smooth
            bb[:, j] = np.interp(np.arange(Z), vz, bb[vz, j])
            bb[:, j] = _smooth(bb[:, j], 3)
        cx = np.argwhere(m)[:, 2].mean()
        prof[s] = dict(z0=z0, z1=z1, bb=bb, cx=cx)
    return prof


def _side_map(prof):
    """Assign each x column to the nearer lung (midline split)."""
    if prof['R'] is None or prof['L'] is None:
        return np.where(np.arange(X) < X / 2, 0, 1)
    mid = 0.5 * (prof['R']['cx'] + prof['L']['cx'])
    return np.where(np.arange(X) < mid, 0, 1)


def lungnorm_grid(lobe_src, lobe_dst, prof_src=None, prof_dst=None):
    """Sampling grid (in source voxel coordinates) for every destination voxel,
    obtained by matching lung-normalised coordinates (u, v, w)."""
    ps = prof_src or lung_profile(lobe_src)
    pd = prof_dst or lung_profile(lobe_dst)
    zz, yy, xx = np.meshgrid(np.arange(Z, dtype=np.float32), np.arange(Y, dtype=np.float32),
                             np.arange(X, dtype=np.float32), indexing='ij')
    grid = np.stack([zz.copy(), yy.copy(), xx.copy()])
    side = _side_map(pd)
    for k, s in enumerate(('R', 'L')):
        a, b = ps[s], pd[s]
        if a is None or b is None:
            continue
        cols = side == k
        u = (np.arange(Z) - b['z0']) / max(b['z1'] - b['z0'], 1)
        zs = a['z0'] + u * (a['z1'] - a['z0'])                      # source slice per dest slice
        zi = np.clip(zs, 0, Z - 1)
        bbs = np.stack([np.interp(zi, np.arange(Z), a['bb'][:, j]) for j in range(4)], 1)
        bbd = b['bb']
        for z in range(Z):
            y0d, y1d, x0d, x1d = bbd[z]; y0s, y1s, x0s, x1s = bbs[z]
            v = (np.arange(Y) - y0d) / max(y1d - y0d, 1)
            w = (np.arange(X) - x0d) / max(x1d - x0d, 1)
            ys = y0s + v * (y1s - y0s); xs = x0s + w * (x1s - x0s)
            grid[0, z][:, cols] = zs[z]
            grid[1, z][:, cols] = ys[:, None].repeat(cols.sum(), 1)
            grid[2, z][:, cols] = xs[None, cols].repeat(Y, 0)
    return grid


def norm_coords(prof, pts):
    """Lung-normalised coordinates (side, u, v, w) for points (N,3)."""
    side = _side_map(prof)
    out = []
    for p in pts:
        k = int(side[int(np.clip(round(p[2]), 0, X - 1))])
        s = ('R', 'L')[k]; a = prof[s]
        if a is None:
            out.append((k, p[0] / Z, p[1] / Y, p[2] / X)); continue
        u = (p[0] - a['z0']) / max(a['z1'] - a['z0'], 1)
        z = int(np.clip(round(p[0]), 0, Z - 1))
        y0, y1, x0, x1 = a['bb'][z]
        out.append((k, u, (p[1] - y0) / max(y1 - y0, 1), (p[2] - x0) / max(x1 - x0, 1)))
    return np.array(out, dtype=np.float64)


def affine_grid(lobe_src, lobe_dst):
    """Moment-matching affine alignment of the lung masks (axis-aligned)."""
    sp = np.array([6.0, 5.0, 5.0])
    ps = np.argwhere(lobe_src > 0) * sp; pd = np.argwhere(lobe_dst > 0) * sp
    cs, cd = ps.mean(0), pd.mean(0)
    Cs, Cd = np.cov((ps - cs).T), np.cov((pd - cd).T)
    ws, Vs = np.linalg.eigh(Cs); wd, Vd = np.linalg.eigh(Cd)
    Ss = Vs @ np.diag(np.sqrt(ws)) @ Vs.T
    Sd_inv = Vd @ np.diag(1 / np.sqrt(wd)) @ Vd.T
    A = Ss @ Sd_inv
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing='ij')
    P = np.stack([zz, yy, xx], -1).reshape(-1, 3) * sp
    Q = (P - cd) @ A.T + cs
    return (Q / sp).T.reshape(3, Z, Y, X).astype(np.float32)


def apply(arr, grid, order=1):
    return ndi.map_coordinates(arr.astype(np.float32), grid, order=order, mode='constant', cval=0.0)
