#!/bin/sh
set -e

DATA_ROOT=""
TRAIN_ID="multiclass-id1"

# train models
item=1
while [ $item -le 18 ]; do
  python train_classifier.py --data-root "$DATA_ROOT" \
    --results-dir "./spaceml-results" \
    --id $TRAIN_ID \
    --item $item \
    --n_blocks 4 \
    --batch-size 64 \
    --epochs 30 \
    --eval-test

  item=$((item + 1))
done

## evaluate models
python eval_multiclass.py --data-root "$DATA_ROOT" \
  --results-dir "./results/data-2018-2021-116x150-pp0-augmented/$TRAIN_ID/4-way-item-classifier/" \
  --batch-size 128 \
  --n_blocks 4




python train_classifier.py --data-root "$DATA_ROOT" --results-dir "./spaceml-results" --id "multiclass-id1" --n_blocks 4 --batch-size 64 --epochs 30 --eval-test --item 1