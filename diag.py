"""Grayscale schematic engine: boxes on an inch canvas, edge-anchored
orthogonal arrows, and an overlap checker that refuses to save a figure
whose boxes, labels or arrow labels collide."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7})

FILL = {'data': 'white', 'net': '#d9d9d9', 'op': '#efefef', 'out': '#bdbdbd', 'stat': '#f7f7f7'}


class Canvas:
    def __init__(self, w, h):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1]); self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off')
        self.boxes = {}; self.texts = []; self.frames = []

    def box(self, name, x, y, w, h, text, kind='net', fs=6.6, bold_first=False, ls='-'):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.04', fc=FILL[kind], ec='black',
                           lw=0.7, ls=ls, zorder=2)
        self.ax.add_patch(p)
        t = self.ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, zorder=3, linespacing=1.15,
                         fontweight='normal')
        self.boxes[name] = (x, y, w, h); self.texts.append((name, t, (x, y, w, h)))
        return name

    def frame(self, x, y, w, h, label, fs=7):
        self.ax.add_patch(Rectangle((x, y), w, h, fc='none', ec='#555555', lw=0.6, ls=(0, (4, 2)), zorder=1))
        t = self.ax.text(x + 0.06, y + h - 0.05, label, ha='left', va='top', fontsize=fs, fontstyle='italic', zorder=3)
        self.frames.append((x, y, w, h)); self.texts.append(('frame:' + label, t, None))

    def anchor(self, name, side, frac=0.5):
        x, y, w, h = self.boxes[name]
        return {'l': (x, y + h * frac), 'r': (x + w, y + h * frac), 't': (x + w * frac, y + h),
                'b': (x + w * frac, y)}[side]

    def arrow(self, pts, dashed=False, label=None, lpos=None, lfs=6.2, head=True, lw=0.7):
        """pts: list of (x,y) or (name, side[, frac]) anchors; polyline with arrow head at end."""
        P = [self.anchor(*p) if isinstance(p[0], str) else p for p in pts]
        for a, b in zip(P[:-1], P[1:]):
            if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:
                self.diag_segments = getattr(self, 'diag_segments', []) + [(a, b)]
        ls = (0, (3, 2)) if dashed else '-'
        for a, b in zip(P[:-2], P[1:-1]):
            self.ax.plot([a[0], b[0]], [a[1], b[1]], color='black', lw=lw, ls=ls, zorder=1.5, solid_capstyle='butt')
        a, b = P[-2], P[-1]
        self.ax.annotate('', xy=b, xytext=a, zorder=1.6,
                         arrowprops=dict(arrowstyle='-|>' if head else '-', color='black', lw=lw, ls=ls,
                                         shrinkA=0, shrinkB=0.5, mutation_scale=6))
        if label:
            x, y = lpos
            t = self.ax.text(x, y, label, fontsize=lfs, ha='center', va='center', zorder=4,
                             bbox=dict(fc='white', ec='none', pad=0.4))
            self.texts.append(('label:' + label, t, None))

    def note(self, x, y, s, fs=6.2, ha='center', style='normal'):
        t = self.ax.text(x, y, s, fontsize=fs, ha=ha, va='center', zorder=4, fontstyle=style)
        self.texts.append(('note:' + s, t, None))

    def check(self, pad=0.03):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        probs = []
        names = list(self.boxes)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                x1, y1, w1, h1 = self.boxes[names[i]]; x2, y2, w2, h2 = self.boxes[names[j]]
                if x1 < x2 + w2 + pad and x2 < x1 + w1 + pad and y1 < y2 + h2 + pad and y2 < y1 + h1 + pad:
                    probs.append(('box overlap', names[i], names[j]))
        bbs = []
        for name, t, bx in self.texts:
            bb = t.get_window_extent(r).transformed(inv)
            if bx is not None:
                x, y, w, h = bx
                if bb.x0 < x + 0.015 or bb.x1 > x + w - 0.015 or bb.y0 < y + 0.01 or bb.y1 > y + h - 0.01:
                    probs.append(('text exceeds box', name, (round(bb.width, 2), round(bb.height, 2), w, h)))
            else:
                for bn, (x, y, w, h) in self.boxes.items():
                    if bb.x0 < x + w and x < bb.x1 and bb.y0 < y + h and y < bb.y1:
                        probs.append(('label over box', name, bn))
                if bb.x0 < 0 or bb.x1 > self.W or bb.y0 < 0 or bb.y1 > self.H:
                    probs.append(('label outside canvas', name))
                for n2, b2 in bbs:
                    if bb.overlaps(b2):
                        probs.append(('labels overlap', name, n2))
                bbs.append((name, bb))
        return probs

    def save(self, path, strict=True):
        probs = self.check()
        if probs and strict:
            raise RuntimeError('\n'.join(map(str, probs)))
        self.fig.savefig(path, dpi=600)
        plt.close(self.fig)
        return probs
