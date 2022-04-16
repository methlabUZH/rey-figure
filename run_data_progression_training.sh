#!/bin/sh

# data progression
for num_samples in $(seq $1 2000 $2); do
  # without data augmentation
  python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n "$num_samples" --n_classes 3
  python eval_multilabel.py --results-dir "../spaceml-results/${num_samples}-data_232x300-seed_1/final-3_scores/rey-multilabel-classifier/"

  # with data augmentation
  python train_multilabel.py --image-size "232 300" --augment 1 --seed 1 --max_n "$num_samples" --n_classes 3
  python eval_multilabel.py --results-dir "../spaceml-results/${num_samples}-data_232x300-seed_1/final-aug-3_scores/rey-multilabel-classifier/"
done
