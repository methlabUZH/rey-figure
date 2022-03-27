#!/bin/sh

DATA_ROOT="/mnt/ds3lab-scratch/webermau/rey-figure/data/serialized-data/data-2018-2021-232x300-pp0"
RES_DIR="/home/webermau/rey-figure/spaceml-results/data-2018-2021-232x300-pp0/final-bigsize-aug/rey-multilabel-classifier"

for distortion in $(seq 0.1 0.1 1.0); do
  python eval_multilabel_semantic.py --data-root "$DATA_ROOT" \
    --results-dir "$RES_DIR" \
    --image-size 232 300 \
    --transform "perspective" \
    --distortion $distortion
done
