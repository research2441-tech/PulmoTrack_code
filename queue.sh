#!/bin/bash
# usage: queue.sh "train full 1" "infer full 1 test" ...
cd "$(dirname "$0")"
for job in "$@"; do
  set -- $job
  kind=$1; shift
  if [ "$kind" = train ]; then
    [ -f ckpt/$1_s$2.pt ] || python3 train.py $1 $2 ${3:-12} > res/train_$1_s$2.log 2>&1
  elif [ "$kind" = infer ]; then
    tag=$1_s$2_$3; [ -n "$4" ] && [ "$4" != none ] && tag=${tag}_$4
    [ -f res/pred/$tag.npz ] || python3 infer.py "$@" >> res/infer.log 2>&1
  elif [ "$kind" = fit ]; then
    [ -f res/assoc/$1.json ] || python3 fit_assoc.py $1_val >> res/fit.log 2>&1
  elif [ "$kind" = eval ]; then
    [ -f res/eval/$1.pkl ] || python3 evaluate.py $1 $2 >> res/eval.log 2>&1
  fi
  echo "$(date +%H:%M) done: $kind $*" >> res/queue.log
done
