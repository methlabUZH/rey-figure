#!/bin/sh

python train_multilabel.py --image-size "${1} ${2}" --augment 0 --seed 1 --n_classes 3
python train_multilabel.py --image-size "${1} ${2}" --augment 1 --seed 1 --n_classes 3
python eval_multilabel.py --results-dir "../spaceml-results/data_${1}x${2}-seed_1/final-3_scores/rey-multilabel-classifier/"
python eval_multilabel.py --results-dir "../spaceml-results/data_${1}x${2}-seed_1/final-aug-3_scores/rey-multilabel-classifier/"

