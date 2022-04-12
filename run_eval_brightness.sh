#!/bin/sh

for brightness in $(seq 0.1 0.1 2.0); do
  python eval_multilabel_semantic.py --image-size "232 300" --transform "brightness" --brightness $brightness --augmented 0
  python eval_multilabel_semantic.py --image-size "232 300" --transform "brightness" --brightness $brightness --augmented 1
done
