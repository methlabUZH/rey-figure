#!/bin/sh

angle=0
while [ $angle -le 30 ]; do
  python eval_multilabel_semantic.py --image-size "232 300" --transform "rotation" --angles $angle $((angle + 5)) --augmented 0
  python eval_multilabel_semantic.py --image-size "232 300" --transform "rotation" --angles $angle $((angle + 5)) --augmented 1
  angle=$((angle + 5))
done
