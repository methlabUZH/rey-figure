#!/bin/sh

python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/1000-data_232x300-seed_1/final-3_scores/rey-multilabel-classifier/"
python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/1000-data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier/"

# data progression
for num_samples in $(seq 2000 2000 16000); do
  # without data augmentation
  #    python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n "$num_samples" --n_classes 3
  python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/${num_samples}-data_232x300-seed_1/final-3_scores/rey-multilabel-classifier/"

  # with data augmentation
  #    python train_multilabel.py --image-size "232 300" --augment 1 --seed 1 --max_n "$num_samples" --n_classes 3
  python eval_multilabel.py --results-dir "/mnt/ds3lab-scratch/webermau/rey-figure/spaceml-results/${num_samples}-data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier/"
done
