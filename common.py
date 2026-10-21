import numpy as np, json, torch
from scipy import ndimage as ndi

DATA = './data'


def norm_hu(a):
    return ((np.clip(a.astype(np.float32), -1024, 600) + 300.0) / 500.0).astype(np.float32)


def load_split(sp, grids=False):
    d = np.load(f'{DATA}/{sp}.npz')
    out = dict(img=d['img'], lobe=d['lobe'], inst=d['inst'], meta=json.load(open(f'{DATA}/{sp}.json')))
    if grids:
        G = np.load(f'{DATA}/grid_{sp}.npy')
        idx = json.load(open(f'{DATA}/grid_{sp}_idx.json'))
        out['grid'] = {i: G[k] for k, i in enumerate(idx)}
    return out


def patients(meta):
    P = {}
    for i, e in enumerate(meta):
        P.setdefault(e['pid'], []).append(i)
    return [sorted(v, key=lambda i: meta[i]['tp']) for v in P.values()]


def warp(arr, grid, order=1):
    return ndi.map_coordinates(arr.astype(np.float32), grid.astype(np.float32), order=order, mode='constant', cval=0.0)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
