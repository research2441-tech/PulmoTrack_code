"""Result figures (grayscale), all values read from res/summary.json and evaluation records."""
import json, pickle, numpy as np, matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
                     'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'legend.frameon': False,
                     'hatch.linewidth': 0.5})
S = json.load(open('./res/summary.json'))
OUT = './figs'
NAMES = {'unet': 'U-Net', 'unetr': 'UNETR-lite', 'paired': 'Paired U-Net', 'full': 'PulmoTrack'}
FILL = {'unet': '#f2f2f2', 'unetr': '#bdbdbd', 'paired': '#7f7f7f', 'full': '#262626'}
HATCH = {'unet': '////', 'unetr': '', 'paired': '....', 'full': ''}
SPL = [('test', 'Internal'), ('extD', 'Site D'), ('extE', 'Site E')]


def fig4():
    SEL = json.load(open('./res/extra.json'))['selected']
    nm = {'unetssl': 'SSL U-Net', 'unet': 'U-Net', 'gate': 'Gate', 'nomem': 'SSL + gate', 'full': 'SSL + gate + memory',
          'paired': 'Paired U-Net'}
    fig, axs = plt.subplots(2, 1, figsize=(3.45, 3.75), gridspec_kw=dict(height_ratios=[1, 1]))
    fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.10, hspace=0.70)
    a = axs[0]
    labs = ['<1', '1-2', '2-3', '3-5', '5-10', '>10']
    mk = {'unetssl': ('o', '-'), 'unet': ('s', ':'), 'paired': ('^', '--'), 'full': ('D', '-.')}
    for m in ['unetssl', 'unet', 'paired', 'full']:
        c = size_curve(f'{m}_s0_test')
        a.plot(range(6), [v for v, _ in c], marker=mk[m][0], ls=mk[m][1], color='black', ms=3, lw=0.8, mfc='white',
               label=nm[m])
    a.set_xticks(range(6)); a.set_xticklabels(labs, fontsize=6.3)
    a.set_xlabel('Reference lesion volume (mL), internal test'); a.set_ylabel('Detection sensitivity'); a.set_ylim(0, 1.04)
    a.legend(loc='upper center', ncol=2, fontsize=6.1, bbox_to_anchor=(0.5, 1.42), handlelength=2.4, columnspacing=1.6)
    a.text(-0.12, 1.05, '(a)', transform=a.transAxes, fontsize=7, fontweight='bold')
    b = axs[1]
    cfg = ['unetssl', 'unet', 'gate', 'nomem', 'full', 'paired']
    sps = [('test', 'Internal', 'white', '////'), ('extD', 'Site D', '#9e9e9e', ''), ('extE', 'Site E', '#262626', '')]
    w = 0.26
    for j, (sp, lab, fc, ht) in enumerate(sps):
        vals = [S['seg'][f'{m}_s0_{sp}']['false_persist'] for m in cfg]
        b.bar(np.arange(len(cfg)) + (j - 1) * w, vals, w * 0.92, color=fc, edgecolor='black', linewidth=0.5, hatch=ht, label=lab)
    b.set_xticks(range(len(cfg))); b.set_xticklabels(['SSL\nU-Net', 'U-Net', 'Gate', 'SSL +\ngate', 'SSL + gate\n+ memory', 'Paired\nU-Net'], fontsize=6.0)
    b.set_ylabel('False persistence'); b.set_ylim(0, 0.62)
    b.legend(loc='upper left', ncol=3, fontsize=6.1, handlelength=1.6, columnspacing=0.9)
    b.text(-0.12, 1.05, '(b)', transform=b.transAxes, fontsize=7, fontweight='bold')
    for x in axs:
        x.spines['top'].set_visible(False); x.spines['right'].set_visible(False)
    fig.savefig(f'{OUT}/fig4_backbones.png', dpi=600); plt.close(fig)


def fig5(key=None):
    key = key or json.load(open('./res/extra.json'))['selected'] + '_s0'
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.80))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.21, wspace=0.40)
    mk = {'test': ('o', '-'), 'extD': ('s', '--'), 'extE': ('^', ':'), 'dense': ('D', '-.')}
    nm = {'test': 'Internal', 'extD': 'Site D', 'extE': 'Site E', 'dense': 'Crowded'}
    top = 0.0
    for sp in ['test', 'extD', 'extE', 'dense']:
        a = np.array(S['events'][f'{key}_{sp}']['new']['sweep'], float)
        axs[0].plot(a[:, 0], a[:, 1], marker=mk[sp][0], ls=mk[sp][1], color='black', ms=2.8, lw=0.8, mfc='white', label=nm[sp])
        axs[1].plot(a[:, 0], a[:, 2], marker=mk[sp][0], ls=mk[sp][1], color='black', ms=2.8, lw=0.8, mfc='white')
        top = max(top, a[:, 1].max())
    axs[0].plot([0, 0.32], [0, 0.32], color='#888888', lw=0.6)
    axs[0].set_xlabel('Target $\\alpha$'); axs[0].set_ylabel('False new calls / patient')
    axs[1].set_xlabel('Target $\\alpha$'); axs[1].set_ylabel('Power, new lesions')
    for a in axs:
        a.spines['top'].set_visible(False); a.spines['right'].set_visible(False); a.set_xlim(0, 0.32)
    axs[0].set_ylim(0, top * 1.08); axs[1].set_ylim(0, None)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, fontsize=6.4, bbox_to_anchor=(0.55, 1.0), handlelength=2.2,
               columnspacing=1.0)
    fig.savefig(f'{OUT}/fig5_events.png', dpi=600); plt.close(fig)


def fig6(key=None):
    key = key or json.load(open('./res/extra.json'))['selected'] + '_s0'
    from analysis import load, burden_terms, conformal_q, LOG_PD, LOG_PR
    cal = load(f'{key}_cal')['ints']; test = load(f'{key}_test')['ints']
    q = conformal_q(cal, 0.1)
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.20, wspace=0.55)
    a = axs[0]
    L = np.array([burden_terms(r) for r in test])
    o = np.argsort(L[:, 1])
    for k in range(len(L)):
        lr, lg, sd = L[k]
        cover = abs(lr - lg) <= q * sd
        a.plot([lg, lg], [lr - q * sd, lr + q * sd], color='#999999' if cover else 'black', lw=0.5, zorder=1)
    a.scatter(L[:, 1], L[:, 0], s=4, facecolor='white', edgecolor='black', linewidth=0.5, zorder=2)
    lim = [min(L[:, 1].min(), L[:, 0].min()) - 0.3, max(L[:, 1].max(), L[:, 0].max()) + 0.3]
    a.plot(lim, lim, color='#888888', lw=0.6)
    for t in (LOG_PD, LOG_PR):
        a.axvline(t, color='#bbbbbb', lw=0.5, ls='--'); a.axhline(t, color='#bbbbbb', lw=0.5, ls='--')
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_xlabel('Reference $\\log r_t$'); a.set_ylabel('Predicted $\\log r_t$ and $C_t$')
    b = axs[1]
    for sp, nm in SPL:
        rc = np.array(S['response'][f'{key}_{sp}']['rc_conformal'], float)
        mk = {'test': 'o', 'extD': 's', 'extE': '^'}[sp]; ls = {'test': '-', 'extD': '--', 'extE': ':'}[sp]
        b.plot(rc[:, 1], rc[:, 2], marker=mk, ls=ls, color='black', ms=3, lw=0.8, mfc='white', label=nm)
    hc = np.array(S['response'][f'{key}_test']['rc_heuristic'], float)
    b.plot(hc[:, 0], hc[:, 1], color='#888888', lw=1.0, ls='-', label='Margin heuristic')
    b.set_xlabel('Coverage (not referred)'); b.set_ylabel('Selective accuracy')
    for x in axs:
        x.spines['top'].set_visible(False); x.spines['right'].set_visible(False)
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, fontsize=6.3, bbox_to_anchor=(0.62, 1.0), handlelength=2.2,
               columnspacing=0.9)
    fig.savefig(f'{OUT}/fig6_burden.png', dpi=600); plt.close(fig)


def size_curve(stem):
    d = pickle.load(open(f'./res/eval/{stem}.pkl', 'rb'))['scans']
    z = np.array([v for s in d for v in s['gt_sizes']]) * 0.15
    h = np.array([v for s in d for v in s['gt_hit']], float)
    bins = [0, 1, 2, 3, 5, 10, 1e9]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (z >= lo) & (z < hi)
        out.append((h[m].mean() if m.any() else np.nan, int(m.sum())))
    return out


def fig7():
    sw = json.load(open('./res/tta_sweep.json'))
    K = json.load(open('./res/extra.json'))['selected'] + '_s0'
    fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.95), gridspec_kw=dict(width_ratios=[1, 1.25]))
    fig.subplots_adjust(left=0.13, right=0.99, top=0.72, bottom=0.25, wspace=0.55)
    a = axs[0]
    lrs = [5e-4, 5e-3, 2e-2]
    for t, mk, ls, nm in (('tent', 's', '--', 'TENT'), ('anat', 'o', '-', 'Anatomy TTA')):
        ys = [sw.get(f'{K}_extD_{t}', {}).get('dice')] + [sw.get(f'{K}_extD_{t}_lr{lr:g}_k3_p30', {}).get('dice') for lr in lrs[1:]]
        a.plot(range(3), ys, marker=mk, ls=ls, color='black', ms=3, lw=0.8, mfc='white', label=nm)
    a.axhline(sw[f'{K}_extD']['dice'], color='#888888', lw=0.7, ls=':', label='No adaptation')
    a.set_xticks(range(3)); a.set_xticklabels(['5e-4', '5e-3', '2e-2'], fontsize=6.3)
    a.set_xlabel('Learning rate'); a.set_ylabel('Lesion Dice, site D')
    vals = [v['dice'] for v in sw.values()]
    a.set_ylim(min(vals) - 0.03, max(vals) + 0.03)
    a.text(-0.32, 1.06, '(a)', transform=a.transAxes, fontsize=7, fontweight='bold')
    b = axs[1]
    g = lambda k, m: S['seg'][k][m]
    groups = [('Dice', 'les_dice', '25'), ('Dice', 'les_dice', ''), ('Sens. <3 mL', 'sens_small', '25'), ('Sens. <3 mL', 'sens_small', '')]
    w = 0.36
    for j, (m, fc, ht, nm) in enumerate((('unet', 'white', '////', 'U-Net'), ('unetssl', '#404040', '', 'SSL U-Net'))):
        ys = [g(f'{m}{fr}_s0_test', key) for _, key, fr in groups]
        b.bar(np.arange(4) + (j - 0.5) * w, ys, w * 0.92, color=fc, edgecolor='black', linewidth=0.5, hatch=ht, label=nm)
    b.set_xticks(range(4)); b.set_xticklabels(['Dice\n25%', 'Dice\n100%', 'Small\n25%', 'Small\n100%'], fontsize=6.2)
    b.set_ylim(0, 1.0); b.set_ylabel('Dice / sensitivity')
    b.text(-0.32, 1.07, '(b)', transform=b.transAxes, fontsize=7, fontweight='bold')
    for x in axs:
        x.spines['top'].set_visible(False); x.spines['right'].set_visible(False)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', ncol=3, fontsize=6.1, bbox_to_anchor=(0.54, 1.0), handlelength=2.0,
               columnspacing=0.9)
    fig.savefig(f'{OUT}/fig7_robust.png', dpi=600); plt.close(fig)


if __name__ == '__main__':
    for f in sys.argv[1:] or ['fig4', 'fig5', 'fig6', 'fig7']:
        globals()[f]()
        print('ok', f)
