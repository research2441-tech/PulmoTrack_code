"""PulmoTrack-Sim: in-silico longitudinal multicentre chest CT cohort.

Each patient has a canonical thorax (body wall, ribs, spine, mediastinum, heart,
airways, two lungs split into five lobes with incomplete fissures, and a
branching vascular tree).  Lesions live in canonical coordinates and evolve
between examinations (growth, regression, resolution, emergence).  Every
examination applies a breathing/positioning deformation and a site-specific
acquisition model (point-spread function, kernel, correlated noise, HU
calibration, contrast phase).
"""
import json, sys, os
import numpy as np
from scipy import ndimage as ndi

Z, Y, X = 48, 64, 64
SPACING = (6.0, 5.0, 5.0)            # mm (z, y, x)
VOXEL_ML = np.prod(SPACING) / 1000.0  # 0.15 mL

# tissue codes in the canonical label volume
AIR, FAT, SOFT, BONE, LUNG, BLOOD, AIRWAY = 0, 1, 2, 3, 4, 5, 6

SITES = {
    # name: noise sd (HU), noise corr (vox), psf sd xy, psf sd z, sharpen gain,
    #       HU offset, HU scale, p(contrast)
    'A': dict(noise=18., ncorr=0.7, psf=0.60, psfz=0.70, sharp=0.0, off=0., scale=1.00, pc=0.0),
    'B': dict(noise=12., ncorr=1.0, psf=0.90, psfz=0.80, sharp=0.0, off=15., scale=1.00, pc=1.0),
    'C': dict(noise=34., ncorr=0.7, psf=0.65, psfz=0.75, sharp=0.0, off=-10., scale=1.00, pc=0.0),
    'D': dict(noise=22., ncorr=0.8, psf=0.70, psfz=1.40, sharp=0.0, off=25., scale=1.00, pc=0.5),
    'E': dict(noise=52., ncorr=0.45, psf=0.45, psfz=0.70, sharp=0.8, off=-30., scale=1.04, pc=0.0),
}

zz, yy, xx = np.meshgrid(np.arange(Z, dtype=np.float32), np.arange(Y, dtype=np.float32),
                         np.arange(X, dtype=np.float32), indexing='ij')


def smooth_noise(rng, sigma, shape=(Z, Y, X)):
    n = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return n / (n.std() + 1e-6)


def splat_tube(vol, pts, rad, mask=None):
    """Max-splat soft spheres of given radii at points."""
    for (z, y, x), r in zip(pts, rad):
        R = int(np.ceil(r + 1.5))
        z0, z1 = max(0, int(z) - R), min(Z, int(z) + R + 2)
        y0, y1 = max(0, int(y) - R), min(Y, int(y) + R + 2)
        x0, x1 = max(0, int(x) - R), min(X, int(x) + R + 2)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue
        d = np.sqrt((zz[z0:z1, y0:y1, x0:x1] * 1.2 - z * 1.2) ** 2 + (yy[z0:z1, y0:y1, x0:x1] - y) ** 2 +
                    (xx[z0:z1, y0:y1, x0:x1] - x) ** 2)
        f = np.clip((r - d) / 0.6 + 0.5, 0, 1)
        np.maximum(vol[z0:z1, y0:y1, x0:x1], f, out=vol[z0:z1, y0:y1, x0:x1])


def make_anatomy(rng):
    cx, cy = 32 + rng.normal(0, 0.6), 33 + rng.normal(0, 0.6)
    ax, ay = rng.uniform(27, 29.5), rng.uniform(21, 24)
    taper = 1 - 0.12 * np.clip((8 - zz) / 8, 0, 1)        # narrower near apex
    rr = np.sqrt(((xx - cx) / (ax * taper)) ** 2 + ((yy - cy) / (ay * taper)) ** 2)
    tissue = np.full((Z, Y, X), AIR, np.uint8)
    tissue[rr <= 1.0] = FAT
    tissue[rr <= 0.93] = SOFT
    phase = rng.uniform(0, 2 * np.pi)
    rib = (rr > 0.85) & (rr <= 0.90) & (np.sin(2 * np.pi * zz / 5.5 + phase) > 0.45) & (yy > cy - ay * 0.75)
    tissue[rib] = BONE
    spine = ((yy - (cy + ay * 0.72)) ** 2 + (xx - cx) ** 2) < 4.2 ** 2
    tissue[spine & (rr <= 0.9)] = BONE
    cavity = rr <= 0.85
    # lungs
    lung = np.zeros((Z, Y, X), bool)
    side_info = {}
    zc = rng.uniform(24, 26.5)
    rz = rng.uniform(21, 23)
    for side, sgn in (('R', -1), ('L', 1)):
        xl = cx + sgn * ax * rng.uniform(0.42, 0.47)
        yl = cy + rng.normal(0, 0.8)
        rx = ax * rng.uniform(0.36, 0.40) * (0.94 if side == 'L' else 1.0)
        ry = ay * rng.uniform(0.78, 0.84)
        e = ((xx - xl) / rx) ** 2 + ((yy - yl) / ry) ** 2 + ((zz - zc) / rz) ** 2
        # flattened diaphragmatic dome
        dome = zz > zc + rz * 0.62 - 5.0 * (((xx - xl) / rx) ** 2 + ((yy - yl) / ry) ** 2)
        m = (e <= 1) & ~dome & cavity
        side_info[side] = (xl, yl, rx, ry)
        lung |= m
    heart = (((zz - rng.uniform(32, 35)) / 9.5) ** 2 + ((yy - (cy - ay * 0.25)) / 9.5) ** 2 +
             ((xx - (cx + rng.uniform(2, 4))) / 10.5) ** 2) <= 1
    medi = (np.abs(xx - cx) < 4.5 + 2.0 * np.clip((cy - yy) / ay, 0, 1) * 2)
    lung &= ~heart & ~medi & ~spine
    tissue[lung] = LUNG
    tissue[heart & cavity] = BLOOD
    aorta = (((yy - (cy + 4)) ** 2 + (xx - (cx + 5)) ** 2) < 3.2 ** 2) & (zz > 6) & cavity
    tissue[aorta] = BLOOD
    # airways: trachea and main bronchi
    air = np.zeros((Z, Y, X), np.float32)
    tz = np.arange(0, 17, 0.5)
    splat_tube(air, [(z, cy - 3, cx) for z in tz], [1.6] * len(tz))
    for sgn in (-1, 1):
        s = np.linspace(0, 1, 16)
        splat_tube(air, [(16 + 6 * t, cy - 3 + 2 * t, cx + sgn * 9 * t) for t in s], [1.2] * len(s))
    tissue[(air > 0.5) & cavity] = AIRWAY
    # lobes
    lobe = np.zeros((Z, Y, X), np.uint8)
    fiss = np.zeros((Z, Y, X), np.float32)
    zlo, zhi = zz[lung].min(), zz[lung].max()
    for side in ('R', 'L'):
        xl = side_info[side][0]
        m = lung & ((xx < cx) if side == 'R' else (xx >= cx))
        # oblique fissure: posterior-superior to anterior-inferior
        za, ya = zlo + rng.uniform(3, 7), cy + ay * rng.uniform(0.35, 0.55)
        zb, yb = zhi - rng.uniform(0, 3), cy - ay * rng.uniform(0.25, 0.45)
        curv = rng.uniform(-2, 2)
        yf = ya + (yb - ya) * (zz - za) / (zb - za) + curv * np.sin(np.pi * (xx - xl) / 24)
        lower = m & (yy > yf)
        dist_obl = np.abs(yy - yf)
        if side == 'R':
            zh = zlo + (zhi - zlo) * rng.uniform(0.45, 0.55) + rng.uniform(-0.06, 0.06) * (yy - cy)
            middle = m & ~lower & (zz > zh)
            upper = m & ~lower & ~middle
            lobe[upper], lobe[middle], lobe[lower] = 1, 2, 3
            dist_h = np.where(~lower, np.abs(zz - zh), 99)
            dfis = np.minimum(dist_obl, dist_h)
        else:
            lobe[m & ~lower], lobe[lower] = 4, 5
            dfis = dist_obl
        f = np.clip(1 - dfis / 0.7, 0, 1) * m
        hole = smooth_noise(rng, 3.0) > rng.uniform(-0.2, 0.6)   # incomplete fissures
        fiss = np.maximum(fiss, f * hole)
    # vascular tree
    vessel = np.zeros((Z, Y, X), np.float32)
    for side in ('R', 'L'):
        xl, yl, rx, ry = side_info[side]
        m = lung & (lobe <= 3 if side == 'R' else lobe >= 4)
        cand = np.argwhere(m)
        hil = np.array([rng.uniform(20, 24), yl + rng.normal(0, 1), xl - np.sign(xl - cx) * rx * 0.75])
        pts, rad = [], []

        def branch(p0, gen, r):
            tgt = cand[rng.integers(len(cand))].astype(np.float32)
            d = tgt - p0
            L = np.linalg.norm(d) * rng.uniform(0.3, 0.5)
            d = d / (np.linalg.norm(d) + 1e-6)
            n = max(2, int(L / 0.5))
            wob = rng.normal(0, 0.03, (n, 3)).cumsum(0)
            for k in range(n):
                q = p0 + d * (k * 0.5) + wob[k]
                pts.append(q); rad.append(r)
            p1 = p0 + d * L
            if gen < 5:
                for _ in range(rng.integers(2, 4)):
                    branch(p1, gen + 1, r * 0.72)
        for _ in range(3):
            branch(hil, 1, rng.uniform(1.15, 1.4))
        splat_tube(vessel, pts, rad)
    vessel *= (lung | (tissue == SOFT)).astype(np.float32) * (lung.astype(np.float32) * 1.0 + 0.0)
    texture = smooth_noise(rng, 1.5)
    return dict(tissue=tissue, lobe=lobe, fiss=fiss, vessel=vessel, texture=texture,
                lung=lung, cx=cx, cy=cy)


def lung_side_R_or(m):
    return m


# ---------------------------------------------------------------- lesions
TRAJ = ['grow', 'stable', 'regress', 'resolve']


def sample_lesions(rng, anat, process):
    lung, lobe = anat['lung'], anat['lobe']
    edt = ndi.distance_transform_edt(lung)
    lesions = []
    dense = process == 'dense'
    n0 = 0 if rng.random() < 0.05 else 1 + min(rng.poisson(2.3), 7)
    if dense:          # stress-test cohort: crowded multifocal disease
        n0 = 6 + min(rng.poisson(4), 8)
    probs = {'progressive': [0.70, 0.20, 0.10, 0.0],
             'responding': [0.05, 0.15, 0.55, 0.25],
             'stable': [0.20, 0.50, 0.25, 0.05],
             'dense': [0.40, 0.35, 0.20, 0.05]}[process]
    new_rate = {'progressive': 0.9, 'responding': 0.12, 'stable': 0.3, 'dense': 1.2}[process]
    margin = 1.0 if dense else 2.5

    def place(kind, r):
        for _ in range(200):
            if kind == 'ggo':
                pool = np.argwhere(lung & (edt > 1.5) & (edt < 6) & (lobe > 0))
            else:
                pool = np.argwhere(lung & (edt > min(r, 3.0)) & (lobe > 0))
            c = pool[rng.integers(len(pool))].astype(np.float32)
            ok = all(np.linalg.norm((c - l['c']) * [1.2, 1, 1]) > r + l['rmax'] + margin for l in lesions)
            if ok:
                return c
        return None

    def new_lesion(t_on, small=False):
        kind = rng.choice(['solid', 'ggo', 'partsolid'], p=[0.5, 0.3, 0.2])
        if small:
            r0 = rng.uniform(1.1, 2.2)
        elif dense:
            r0 = rng.uniform(1.2, 3.0)
        else:
            r0 = rng.uniform(1.3, 4.5) if kind == 'solid' else rng.uniform(2.0, 5.2)
        traj = 'grow' if small else rng.choice(TRAJ, p=probs)
        if traj == 'grow':
            dt = rng.uniform(60, 260) if process in ('progressive', 'dense') else rng.uniform(250, 700)
            rate = np.log(2) / dt
        elif traj == 'stable':
            rate = rng.normal(0, 0.0005)
        elif traj == 'regress':
            rate = -np.log(2) / rng.uniform(30, 130)
        else:
            rate = -np.log(2) / rng.uniform(10, 22)
        rmax = r0 * np.exp(rate * 400 / 3) if rate > 0 else r0
        rmax = min(rmax, 7.5)
        c = place(kind, min(rmax, 6))
        if c is None:
            return None
        A = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        axes = rng.uniform(0.78, 1.25, 3)
        harm = rng.normal(0, 1, (4, 3)) ; amp = rng.uniform(0.05, 0.12) if kind == 'solid' else rng.uniform(0.12, 0.25)
        hu = {'solid': rng.uniform(5, 45), 'ggo': rng.uniform(-640, -460), 'partsolid': rng.uniform(-420, -250)}[kind]
        return dict(c=c, r0=float(r0), rate=float(rate), t_on=float(t_on), kind=str(kind), traj=str(traj),
                    A=A, axes=axes, harm=harm, amp=float(amp), hu=float(hu), rmax=float(rmax),
                    seed=int(rng.integers(1 << 30)))

    for _ in range(n0):
        l = new_lesion(0.0)
        if l is not None:
            lesions.append(l)
    return lesions, new_rate, new_lesion


def lesion_radius(l, t):
    if t < l['t_on']:
        return 0.0
    r = l['r0'] * np.exp(l['rate'] * (t - l['t_on']) / 3.0)   # volume rate/3 for radius
    return float(min(r, 7.5))


def render_lesions(lesions, t, lung):
    frac = np.zeros((Z, Y, X), np.float32)
    hu = np.zeros((Z, Y, X), np.float32)
    lid = np.zeros((Z, Y, X), np.uint8)
    for i, l in enumerate(lesions):
        r = lesion_radius(l, t)
        if r < 0.9:
            continue
        R = int(np.ceil(r * 1.6 + 2))
        c = l['c']
        z0, z1 = max(0, int(c[0]) - R), min(Z, int(c[0]) + R + 2)
        y0, y1 = max(0, int(c[1]) - R), min(Y, int(c[1]) + R + 2)
        x0, x1 = max(0, int(c[2]) - R), min(X, int(c[2]) + R + 2)
        P = np.stack([(zz[z0:z1, y0:y1, x0:x1] - c[0]) * 1.2, yy[z0:z1, y0:y1, x0:x1] - c[1],
                      xx[z0:z1, y0:y1, x0:x1] - c[2]], -1)
        Q = P @ l['A'] / l['axes']
        d = np.linalg.norm(Q, axis=-1) + 1e-6
        u = Q / d[..., None]
        lob = 1 + l['amp'] * np.tanh((u @ l['harm'].T).sum(-1) / 2)
        f = np.clip((r * lob - d) / 0.7 + 0.5, 0, 1)
        f *= lung[z0:z1, y0:y1, x0:x1]
        sub = frac[z0:z1, y0:y1, x0:x1]
        upd = f > sub
        sub[upd] = f[upd]
        hsub = hu[z0:z1, y0:y1, x0:x1]
        if l['kind'] == 'partsolid':
            core = np.clip((r * 0.45 - d) / 0.7 + 0.5, 0, 1)
            h = l['hu'] * (1 - core) + 20 * core
        elif l['kind'] == 'ggo':
            h = l['hu'] + 60 * np.sin(d * 2.1 + l['seed'] % 7)
        else:
            h = np.full_like(f, l['hu'])
        hsub[upd] = h[upd]
        lsub = lid[z0:z1, y0:y1, x0:x1]
        lsub[upd & (f > 0.5)] = i + 1
    return frac, hu, lid


# ------------------------------------------------------------ deformation
def deformation(rng, lungmask, baseline=False):
    """Image->canonical sampling grid: inspiration (non-uniform craniocaudal
    scaling), AP expansion, small shift and in-plane rotation."""
    zl = np.argwhere(lungmask)[:, 0]
    za = float(zl.min())
    if baseline:
        sz, bz, sy, sx, sh, th = 1.0, 0.0, 1.0, 1.0, np.zeros(3), 0.0
    else:
        sz = rng.uniform(0.90, 1.08); bz = rng.uniform(-0.10, 0.10)
        sy = rng.uniform(0.95, 1.05); sx = rng.uniform(0.97, 1.03)
        sh = rng.uniform(-1.8, 1.8, 3); th = np.deg2rad(rng.uniform(-3, 3))
    h = (zz - za) / 40.0
    zc = za + (zz - za) / (sz + bz * h) + sh[0]
    yc0 = 52 + (yy - 52) / sy
    xc0 = 32 + (xx - 32) / sx
    yc = 33 + np.cos(th) * (yc0 - 33) - np.sin(th) * (xc0 - 32) + sh[1]
    xc = 32 + np.sin(th) * (yc0 - 33) + np.cos(th) * (xc0 - 32) + sh[2]
    return np.stack([zc, yc, xc]), dict(sz=sz, bz=bz, sy=sy, sx=sx, sh=[float(s) for s in sh], th=float(th))


def acquire(rng, anat, frac, lhu, site, grid):
    S = SITES[site]
    contrast = rng.random() < S['pc']
    t = anat['tissue']
    hu = np.zeros((Z, Y, X), np.float32)
    hu[t == AIR] = -1000; hu[t == FAT] = -100; hu[t == SOFT] = 45; hu[t == BONE] = 650
    hu[t == AIRWAY] = -990
    hu[t == BLOOD] = 170 if contrast else 42
    lung = t == LUNG
    hu[lung] = (-865 + 22 * anat['texture'])[lung]
    hu += anat['fiss'] * 160
    vhu = (175 if contrast else 40)
    v = anat['vessel']
    hu = hu * (1 - v) + vhu * v
    hu = hu * (1 - frac) + lhu * frac
    # warp to the examination frame
    W = lambda a, o: ndi.map_coordinates(a, grid, order=o, mode='nearest')
    img = W(hu, 1)
    img = ndi.gaussian_filter(img, (S['psfz'] * rng.uniform(0.9, 1.1), S['psf'], S['psf']))
    if S['sharp'] > 0:
        img = img + S['sharp'] * (img - ndi.gaussian_filter(img, (0.6, 1.0, 1.0)))
    noise = ndi.gaussian_filter(rng.standard_normal(img.shape).astype(np.float32), S['ncorr'])
    noise /= noise.std()
    sd = S['noise'] * rng.uniform(0.8, 1.25)
    img = img + sd * noise
    img = img * S['scale'] + S['off'] + rng.normal(0, 5)
    img = np.clip(img, -1024, 1500)
    return img.astype(np.float16), contrast


def simulate_patient(pid, site_seq, seed, n_tp=3, process=None):
    rng = np.random.default_rng(seed)
    anat = make_anatomy(rng)
    if process is None:
        process = str(rng.choice(['progressive', 'responding', 'stable'], p=[0.35, 0.35, 0.30]))
    lesions, new_rate, new_lesion = sample_lesions(rng, anat, process)
    times = [0.0]
    for k in range(1, n_tp):
        times.append(times[-1] + float(rng.uniform(56, 150)))
    # new lesions emerging in each interval
    for k in range(1, n_tp):
        for _ in range(rng.poisson(new_rate)):
            l = new_lesion(rng.uniform(times[k - 1] + 5, times[k] - 5), small=True)
            if l is not None:
                lesions.append(l)
    out = []
    for k, t in enumerate(times):
        frac, lhu, lid = render_lesions(lesions, t, anat['lung'])
        grid, dpar = deformation(rng, anat['lung'], baseline=(k == 0))
        img, contrast = acquire(rng, anat, frac, lhu, site_seq[k], grid)
        W0 = lambda a: ndi.map_coordinates(a, grid, order=0, mode='nearest')
        lobe = W0(anat['lobe']).astype(np.uint8)
        fr = ndi.map_coordinates(frac, grid, order=1, mode='nearest')
        ids = W0(lid).astype(np.uint8)
        # ids must cover the warped lesion support
        ids_d = ndi.grey_dilation(ids, size=(2, 2, 2))
        les = ((fr > 0.5) & (lobe > 0)).astype(np.uint8)
        inst = np.where(les > 0, np.where(ids > 0, ids, ids_d), 0).astype(np.uint8)
        info = []
        for i, l in enumerate(lesions):
            nv = int((inst == i + 1).sum())
            if nv == 0:
                continue
            cz = np.argwhere(inst == i + 1).mean(0)
            info.append(dict(id=i + 1, nvox=nv, kind=l['kind'], traj=l['traj'], centroid=[float(v) for v in cz],
                             new_after_baseline=bool(l['t_on'] > 0)))
        out.append(dict(img=img, lobe=lobe, inst=inst,
                        meta=dict(pid=pid, tp=k, t=t, site=site_seq[k], contrast=bool(contrast), process=process,
                                  deform=dpar, lesions=info)))
    return out


def site_sequence(rng, base, dev_sites=('A', 'B', 'C'), n_tp=3, p_switch=0.15):
    seq = [base]
    for _ in range(1, n_tp):
        if base in dev_sites and rng.random() < p_switch:
            seq.append(str(rng.choice([s for s in dev_sites if s != base])))
        else:
            seq.append(base)
    return seq


if __name__ == '__main__':
    import time
    t0 = time.time()
    o = simulate_patient(0, ['A', 'A', 'B'], 1)
    print('sec', time.time() - t0)
    for e in o:
        m = e['meta']
        print(m['tp'], m['t'], m['site'], m['process'], [(l['id'], l['nvox'], l['traj']) for l in m['lesions']],
              (e['lobe'] > 0).sum(), np.bincount(e['lobe'].ravel()))
    np.savez_compressed('./data/demo.npz', img=np.stack([e['img'] for e in o]),
                        lobe=np.stack([e['lobe'] for e in o]), inst=np.stack([e['inst'] for e in o]))
