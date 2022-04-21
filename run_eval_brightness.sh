#!/bin/sh

RES_DIR1="/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_232x300-seed_1/final-3_scores/rey-multilabel-classifier"
RES_DIR2="/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier"

for brightness in $(seq 0.1 0.1 2.0); do
  python eval_multilabel_semantic.py --results-dir $RES_DIR1 --transform "brightness" --brightness $brightness
  python eval_multilabel_semantic.py --results-dir $RES_DIR2 --transform "brightness" --brightness $brightness
done
