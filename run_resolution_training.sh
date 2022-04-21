#!/bin/sh

#python train_multilabel.py --image-size "${1} ${2}" --augment 0 --seed 1 --n_classes 3
#python train_multilabel.py --image-size "${1} ${2}" --augment 1 --seed 1 --n_classes 3

python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_78x100-seed_1/final-3_scores/rey-multilabel-classifier/"
python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure//spaceml-results/data_78x100-seed_1/final-aug-3_scores/rey-multilabel-classifier/"

python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_116x150-seed_1/final-3_scores/rey-multilabel-classifier/"
python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure//spaceml-results/data_116x150-seed_1/final-aug-3_scores/rey-multilabel-classifier/"

python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_232x300-seed_1/final-3_scores/rey-multilabel-classifier/"
python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure//spaceml-results/data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier/"

python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/data_348x450-seed_1/final-3_scores/rey-multilabel-classifier/"
python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure//spaceml-results/data_348x450-seed_1/final-aug-3_scores/rey-multilabel-classifier/"

