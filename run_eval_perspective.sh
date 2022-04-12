#!/bin/sh

for distortion in $(seq 0.1 0.1 1.0); do
  python eval_multilabel_semantic.py --image-size "232 300" --transform "perspective" --distortion $distortion --augmented 0
  python eval_multilabel_semantic.py --image-size "232 300" --transform "perspective" --distortion $distortion --augmented 1
done
