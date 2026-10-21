"""Networks: PulmoTrack encoder/decoder, SSL masked-volume pretraining head,
and the comparison models (U-Net, UNETR-lite, paired-scan U-Net)."""
import math
import torch, torch.nn as nn, torch.nn.functional as F

CH = (8, 16, 32, 64)


def norm(c):
    return nn.InstanceNorm3d(c, affine=True)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False); self.n1 = norm(cout)
        self.c2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False); self.n2 = norm(cout)
        self.sk = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), norm(cout))

    def forward(self, x):
        h = F.leaky_relu(self.n1(self.c1(x)), 0.01)
        h = self.n2(self.c2(h))
        return F.leaky_relu(h + (x if self.sk is None else self.sk(x)), 0.01)


class Encoder(nn.Module):
    def __init__(self, cin=1):
        super().__init__()
        self.s = nn.ModuleList([Block(cin, CH[0]), Block(CH[0], CH[1], 2), Block(CH[1], CH[2], 2), Block(CH[2], CH[3], 2)])

    def forward(self, x):
        f = []
        for b in self.s:
            x = b(x); f.append(x)
        return f


class Decoder(nn.Module):
    def __init__(self, mem=False):
        super().__init__()
        self.u = nn.ModuleList([Block(CH[3] + CH[2], CH[2]), Block(CH[2] + CH[1], CH[1]), Block(CH[1] + CH[0], CH[0])])
        self.mem = mem
        if mem:
            # memory encoders for the warped prior lesion map at each decoder scale
            self.m = nn.ModuleList([nn.Sequential(nn.Conv3d(1, c, 3, 1, 1), nn.LeakyReLU(0.01), nn.Conv3d(c, c, 3, 1, 1))
                                    for c in (CH[2], CH[1], CH[0])])
            # interval-conditioned channel gates g(dt)
            self.g = nn.Sequential(nn.Linear(2, 32), nn.LeakyReLU(0.01), nn.Linear(32, CH[2] + CH[1] + CH[0]))

    def forward(self, f, M=None, dt=None):
        x = f[3]
        if self.mem:
            e = torch.stack([torch.log(dt.clamp(min=1.0) / 100.0), (dt > 0).float()], 1)
            gates = torch.sigmoid(self.g(e)).split([CH[2], CH[1], CH[0]], 1)
        for k, (blk, skip) in enumerate(zip(self.u, (f[2], f[1], f[0]))):
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = blk(torch.cat([x, skip], 1))
            if self.mem:
                Mk = F.adaptive_avg_pool3d(M, skip.shape[2:])
                has = (dt > 0).float().view(-1, 1, 1, 1, 1)
                x = x + has * gates[k].view(x.shape[0], -1, 1, 1, 1) * self.m[k](Mk)
        return x


class PulmoTrackNet(nn.Module):
    """anatomy=True: lobe-conditioned lesion head with hierarchical gating
    p(lesion) = p(lesion | lung) p(lung).  mem=True: interval-gated lesion memory."""
    def __init__(self, anatomy=True, mem=True):
        super().__init__()
        self.anatomy, self.memf = anatomy, mem
        self.enc = Encoder(1)
        self.dec = Decoder(mem)
        self.lobe = nn.Conv3d(CH[0], 6, 1)
        cin = CH[0] + (6 if anatomy else 0)
        self.les = nn.Sequential(nn.Conv3d(cin, CH[0], 3, 1, 1), nn.LeakyReLU(0.01), nn.Conv3d(CH[0], 1, 1))

    def forward(self, x, M=None, dt=None):
        f = self.enc(x)
        if self.memf:
            if M is None:
                M = torch.zeros_like(x); dt = torch.zeros(x.shape[0], device=x.device)
            h = self.dec(f, M, dt)
        else:
            h = self.dec(f)
        lobe_logit = self.lobe(h)
        plobe = torch.softmax(lobe_logit, 1)
        z = self.les(torch.cat([h, plobe], 1) if self.anatomy else h)
        if self.anatomy:
            plung = 1 - plobe[:, :1]
            p = torch.sigmoid(z) * plung
        else:
            p = torch.sigmoid(z)
        return dict(lobe=lobe_logit, p=p.clamp(1e-6, 1 - 1e-6), z=z)


class UNet(nn.Module):
    """Single-timepoint (cin=1) or paired-scan early-fusion (cin=3) U-Net with
    independent lobe and lesion heads (nnU-Net-style configuration)."""
    def __init__(self, cin=1):
        super().__init__()
        self.enc = Encoder(cin); self.dec = Decoder(False)
        self.lobe = nn.Conv3d(CH[0], 6, 1); self.les = nn.Conv3d(CH[0], 1, 1)

    def forward(self, x, M=None, dt=None):
        h = self.dec(self.enc(x))
        z = self.les(h)
        return dict(lobe=self.lobe(h), p=torch.sigmoid(z).clamp(1e-6, 1 - 1e-6), z=z)


class UNETRLite(nn.Module):
    """ViT encoder on 8^3 patches with a convolutional decoder (UNETR-style)."""
    def __init__(self, dim=144, depth=4, heads=4, grid=(6, 8, 8)):
        super().__init__()
        self.grid = grid
        self.pe = nn.Conv3d(1, dim, 8, 8)
        self.pos = nn.Parameter(torch.zeros(1, math.prod(grid), dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim * 2, 0.0, batch_first=True, norm_first=True)
                                     for _ in range(depth)])
        self.stem = Block(1, CH[0])
        self.p8 = nn.Conv3d(dim, CH[3], 1)
        self.d4 = nn.Sequential(nn.ConvTranspose3d(dim, CH[2], 2, 2))
        self.d2 = nn.Sequential(nn.ConvTranspose3d(dim, CH[1], 2, 2), nn.ConvTranspose3d(CH[1], CH[1], 2, 2))
        self.u4 = Block(CH[3] + CH[2], CH[2]); self.u2 = Block(CH[2] + CH[1], CH[1]); self.u1 = Block(CH[1] + CH[0], CH[0])
        self.lobe = nn.Conv3d(CH[0], 6, 1); self.les = nn.Conv3d(CH[0], 1, 1)

    def forward(self, x, M=None, dt=None):
        B = x.shape[0]
        g = tuple(s // 8 for s in x.shape[2:])
        pos = self.pos
        if g != tuple(self.grid):   # interpolate positional embedding to the inference grid
            pos = F.interpolate(pos.transpose(1, 2).reshape(1, -1, *self.grid), size=g, mode='trilinear',
                                align_corners=False).flatten(2).transpose(1, 2)
        t = self.pe(x).flatten(2).transpose(1, 2) + pos
        outs = []
        for l in self.layers:
            t = l(t); outs.append(t)
        sh = lambda a: a.transpose(1, 2).reshape(B, -1, *g)
        f8 = self.p8(sh(outs[-1]))
        h = F.interpolate(f8, scale_factor=2, mode='trilinear', align_corners=False)
        h = self.u4(torch.cat([h, self.d4(sh(outs[2]))], 1))
        h = F.interpolate(h, scale_factor=2, mode='trilinear', align_corners=False)
        h = self.u2(torch.cat([h, self.d2(sh(outs[1]))], 1))
        h = F.interpolate(h, scale_factor=2, mode='trilinear', align_corners=False)
        h = self.u1(torch.cat([h, self.stem(x)], 1))
        z = self.les(h)
        return dict(lobe=self.lobe(h), p=torch.sigmoid(z).clamp(1e-6, 1 - 1e-6), z=z)


class MVMHead(nn.Module):
    """Masked-volume-modelling decoder used only during self-supervised pretraining."""
    def __init__(self):
        super().__init__()
        self.dec = Decoder(False); self.out = nn.Conv3d(CH[0], 1, 1)

    def forward(self, f):
        return self.out(self.dec(f))


def seg_loss(out, lobe_gt, les_gt, w_lobe=1.0):
    lobe_ce = F.cross_entropy(out['lobe'], lobe_gt)
    pl = torch.softmax(out['lobe'], 1)
    oh = F.one_hot(lobe_gt, 6).permute(0, 4, 1, 2, 3).float()
    inter = (pl * oh).sum((2, 3, 4)); den = pl.sum((2, 3, 4)) + oh.sum((2, 3, 4))
    lobe_dice = 1 - ((2 * inter + 1) / (den + 1))[:, 1:].mean()
    p = out['p'][:, 0]; g = les_gt.float()
    bce = F.binary_cross_entropy(p, g)
    inter = (p * g).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + g.sum((1, 2, 3))
    dice = 1 - ((2 * inter + 1) / (den + 1)).mean()
    return w_lobe * (lobe_ce + lobe_dice) + 2.0 * bce + dice
