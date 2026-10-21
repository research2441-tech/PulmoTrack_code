"""Self-supervised masked-volume modelling of the CT encoder on unlabelled scans."""
import numpy as np, torch, time, sys, json
from model import Encoder, MVMHead
from common import load_split, norm_hu, set_seed

torch.set_num_threads(2)
set_seed(0)
d = load_split('ssl')
imgs = d['img']
enc, head = Encoder(1), MVMHead()
params = list(enc.parameters()) + list(head.parameters())
EP, B = int(sys.argv[1]) if len(sys.argv) > 1 else 10, 2
steps = EP * len(imgs) // B
opt = torch.optim.AdamW(params, 2e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
log, t0, k = [], time.time(), 0
for ep in range(EP):
    perm = np.random.permutation(len(imgs)); tot = 0
    for b in range(0, len(perm) - B + 1, B):
        xs = []
        for i in perm[b:b + B]:
            z, y, x = np.random.randint(0, 17), np.random.randint(0, 17), np.random.randint(0, 17)
            xs.append(norm_hu(imgs[i][z:z + 32, y:y + 48, x:x + 48]))
        x = torch.from_numpy(np.stack(xs))[:, None]
        m = (torch.rand(B, 1, 4, 6, 6) < 0.5).float()
        m = m.repeat_interleave(8, 2).repeat_interleave(8, 3).repeat_interleave(8, 4)
        rec = head(enc(x * (1 - m)))
        loss = ((rec - x).abs() * m).sum() / m.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); tot += loss.item(); k += 1
    log.append(tot / (len(perm) // B))
    print(ep, round(log[-1], 4), round(time.time() - t0), flush=True)
torch.save(enc.state_dict(), './ckpt/ssl_encoder.pt')
json.dump(dict(loss=log, sec=time.time() - t0), open('./res/ssl_log.json', 'w'))
