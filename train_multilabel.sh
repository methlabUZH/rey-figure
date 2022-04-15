#!/bin/bash
out=$'/console.out'
ckpt_dir=$'./checkpoints/'
date=$(date +%y-%m-%d-%H-%M)    

nohup \
python train_multilabel.py --image-size '232 300' --augment 1 \
> "./log/$date$_console.out" & 
