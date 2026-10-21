#!/bin/bash
# Reproduces every experiment in the manuscript from scratch (PulmoTrack-Sim generation to figures).
# Runtime on two CPU cores is roughly a day; all paths are relative to this folder.
set -e
cd "$(dirname "$0")"
mkdir -p data res res/pred res/eval res/assoc ckpt figs
python3 gen.py                                          # cohort: train/val/val2/cal/cal2/test/extD/extE/dense/ssl
python3 cohort_stats.py
python3 grids.py train val val2 cal cal2 test extD extE dense
python3 pretrain.py 10                                  # masked-volume pretraining of the encoder
python3 - <<'PY'
import numpy as np, json
d = np.load('data/train.npz'); fr, lv = [], []
for l in d['lobe']:
    c = np.bincount(l.ravel(), minlength=6)[1:].astype(float); fr.append(c / c.sum()); lv.append(np.log(c.sum()))
json.dump(dict(lobe_frac=list(np.mean(fr, 0)), loglung_mu=float(np.mean(lv)), loglung_sd=float(np.std(lv))), open('res/train_priors.json', 'w'))
PY
A="val val2 cal cal2 test extD extE dense"
J=()
for m in unetssl unet gate nomem full paired; do
  J+=("train $m 0"); for sp in $A; do J+=("infer $m 0 $sp"); done; J+=("fit ${m}_s0"); for sp in $A; do J+=("eval ${m}_s0_$sp $sp"); done
done
J+=("train unetr 0"); for sp in val test extD extE dense; do J+=("infer unetr 0 $sp"); done; J+=("fit unetr_s0")
for sp in test extD extE dense; do J+=("eval unetr_s0_$sp $sp"); done
./queue.sh "${J[@]}"
python3 select_backbone.py                              # validation lesion F1 -> selected backbone
J=()
for sp in extD extE; do for t in anat tent; do J+=("infer unetssl 0 $sp $t" "eval unetssl_s0_${sp}_$t $sp"); done; done
J+=("train unet25 0" "train unetssl25 0" "infer unet25 0 test" "infer unetssl25 0 test" "eval unet25_s0_test test" "eval unetssl25_s0_test test")
for m in unetssl unet; do J+=("train $m 1"); for sp in test extD extE; do J+=("infer $m 1 $sp" "eval ${m}_s1_$sp $sp"); done; done
./queue.sh "${J[@]}"
for sp in test extD extE dense; do python3 evaluate.py oracle $sp; done
for t in anat tent; do for lr in 0.005 0.02; do python3 infer.py unetssl 0 extD $t 3 $lr 30; done; done
python3 tta_sweep.py
python3 results.py                                      # res/summary.json: every number in the paper
python3 cost.py
mkdir -p figs
python3 sim.py                                          # demo patient shown in the cohort figure
python3 fig_arch.py && python3 fig_cohort.py && python3 plots.py
