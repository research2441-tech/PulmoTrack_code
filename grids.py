"""Pre-compute demons registration grids (prior -> follow-up) for every split."""
import numpy as np, json, sys
from multiprocessing import Pool
from demons import demons_grid
def one(a):
    p, c = a
    return demons_grid(p, c).astype(np.float16)
if __name__ == '__main__':
    for sp in sys.argv[1:]:
        d = np.load(f'data/{sp}.npz'); m = json.load(open(f'data/{sp}.json'))
        img = d['img']; del d
        idx = [i for i in range(len(m)) if m[i]['tp'] > 0]
        with Pool(2) as pool:
            G = pool.imap(one, ((img[i - 1], img[i]) for i in idx), chunksize=4)
            G = np.stack(list(G))
        np.save(f'data/grid_{sp}.npy', G); del G, img; json.dump(idx, open(f'data/grid_{sp}_idx.json', 'w'))
        print(sp, len(idx), flush=True)
