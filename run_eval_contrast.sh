#!/bin/sh

RES_DIR_ML="/home/webermau/rey-figure/lukas-final/final-aug_01/rey-multilabel-classifier/"
RES_DIR_REG="/home/webermau/rey-figure/lukas-final/final-reg_01/rey-regressor-v2"

for contrast in $(seq 0.1 0.1 2.0); do
  python eval_multilabel_semantic.py --results-dir $RES_DIR_ML --transform "contrast" --contrast $contrast --tta
  #  python eval_regressor_semantic.py --results-dir $RES_DIR_REG --transform "contrast" --contrast $contrast
done
