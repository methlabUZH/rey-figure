#!/bin/sh

RES_DIR_ML="/home/webermau/rey-figure/lukas-final/final-aug_01/rey-multilabel-classifier/"
RES_DIR_REG="/home/webermau/rey-figure/lukas-final/final-reg_01/rey-regressor-v2"

angle=0
while [ $angle -le 45 ]; do
  python eval_multilabel_semantic.py --results-dir $RES_DIR_ML --transform "rotation" --angles $angle $((angle + 5)) --tta
#  python eval_regressor_semantic.py --results-dir $RES_DIR_REG --transform "rotation" --angles $angle $((angle + 5))
  angle=$((angle + 5))
done
