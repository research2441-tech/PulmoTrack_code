"""Fig. 3: examples of the five simulated sites and a longitudinal case."""
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7})

D = './data'


def pick(split, site, tp=0):
    m = json.load(open(f'{D}/{split}.json'))
    for i, e in enumerate(m):
        if e['site'] == site and e['tp'] == tp and len(e['lesions']) >= 3:
            return i
    return 0


def win(a):
    return np.clip((a.astype(np.float32) + 1000) / 1400, 0, 1)


W, H = 3.45, 1.86
fig = plt.figure(figsize=(W, H))
pw = 0.64; gap = (W - 5 * pw) / 6
sites = [('test', 'A'), ('test', 'B'), ('test', 'C'), ('extD', 'D'), ('extE', 'E')]
cache = {}
for k, (sp, s) in enumerate(sites):
    if sp not in cache:
        cache[sp] = np.load(f'{D}/{sp}.npz')
    d = cache[sp]; i = pick(sp, s)
    inst = d['inst'][i]; lung = (d['lobe'][i] > 0).sum((1, 2))
    z = int(np.argmax((inst > 0).sum((1, 2)) + 0.02 * lung))
    ax = fig.add_axes([(gap + k * (pw + gap)) / W, (H - 0.16 - pw) / H, pw / W, pw / H])
    ax.imshow(win(d['img'][i][z]), cmap='gray', vmin=0, vmax=1)
    ax.contour(inst[z] > 0, [0.5], colors='white', linewidths=0.5)
    ax.set_title(f'Site {s}', fontsize=7, pad=1.5); ax.axis('off')
d = cache['test']; m = json.load(open(f'{D}/test.json'))
pid = None
for e in m:
    if e['tp'] == 0 and e['process'] == 'progressive' and 3 <= len(e['lesions']) <= 5:
        pid = e['pid']; break
idx = [i for i, e in enumerate(m) if e['pid'] == pid]
yc = int(np.argmax((d['inst'][idx[-1]] > 0).sum((0, 2))))
cw = 0.80; ch = cw * 48 * 1.2 / 64; x0 = 0.95
for k, i in enumerate(idx):
    ax = fig.add_axes([(x0 + k * (cw + 0.04)) / W, 0.04 / H, cw / W, ch / H])
    ax.imshow(win(d['img'][i][:, yc, :]), cmap='gray', vmin=0, vmax=1, aspect=1.2)
    ax.contour(d['lobe'][i][:, yc, :], levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors='#bbbbbb', linewidths=0.35)
    ax.contour(d['inst'][i][:, yc, :] > 0, [0.5], colors='white', linewidths=0.6)
    ax.set_title(f'Visit {k + 1} (day {m[i]["t"]:.0f})', fontsize=6.3, pad=1.5); ax.axis('off')
fig.text(0.05 / W, (0.04 + ch / 2) / H, 'Coronal view,\none patient\n(progressive\ncourse, site ' + m[idx[0]]['site'] +
         ')\n\nWhite: lesions\nGrey: lobes', fontsize=6.2, va='center', ha='left')
fig.savefig('./figs/fig3_cohort.png', dpi=600)
print('ok', pid, [m[i]['site'] for i in idx], ch)
