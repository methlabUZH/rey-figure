#!/bin/sh

DATA_ROOT=""
RES_DIR=""

angle=0
while [ $angle -le 30 ]; do
  python eval_multilabel_semantic.py --data-root "$DATA_ROOT" \
    --results-dir "$RES_DIR" \
    --image-size 232 300 \
    --transform "rotation" \
    --angles $angle $((angle + 5))
  angle=$((angle + 5))
done
