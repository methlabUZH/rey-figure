#!/bin/bash
out=$'/console.out'
ckpt_dir=$'./checkpoints/'
date=$(date +%y-%m-%d-%H-%M)    

nohup \
python train_multilabel.py \
    --data-root ./data/serialized-data/data_116x150-seed_1 \
    --results-dir ./results \
    --eval-test \
    --epochs 30 \
    --batch-size 64 \
    --lr 0.01 \
    --gamma 0.95 \
    --weighted-sampling 1 \
    --image-size 116 150 \
> "./log/$date$_console.out" & 
