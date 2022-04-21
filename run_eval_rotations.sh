#!/bin/sh

RES_DIR1="/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_232x300-seed_1/final-3_scores/rey-multilabel-classifier"
RES_DIR2="/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier"

angle=0
while [ $angle -le 45 ]; do
  python eval_multilabel_semantic.py --results-dir $RES_DIR1 --transform "rotation" --angles $angle $((angle + 5))
  python eval_multilabel_semantic.py --results-dir $RES_DIR2 --transform "rotation" --angles $angle $((angle + 5))
  angle=$((angle + 5))
done
