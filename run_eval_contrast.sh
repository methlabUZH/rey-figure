#!/bin/sh

for contrast in $(seq 0.1 0.1 2.0); do
  python eval_multilabel_semantic.py --image-size "232 300" --transform "contrast" --contrast $contrast --augmented 0
  python eval_multilabel_semantic.py --image-size "232 300" --transform "contrast" --contrast $contrast --augmented 1
done
