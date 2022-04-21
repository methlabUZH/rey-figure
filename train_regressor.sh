#!/bin/bash
out=$'/console.out'
ckpt_dir=$'./checkpoints/'
date=$(date +%y-%m-%d-%H-%M)    

nohup \
python train_regressor.py \
    --arch v2 \
    --workers 10 \
    --id final-reg \
    --data-root ./data/serialized-data/data_232x300-seed_1 \
    --results-dir ./results \
    --eval-test \
    --epochs 40 \
    --batch-size 16 \
    --lr 0.01 \
    --gamma 0.95 \
    --wd 0.0 \
    --weighted-sampling 1 \
    --image-size 232 300 \
> "./log/$date$_console.out" & 
