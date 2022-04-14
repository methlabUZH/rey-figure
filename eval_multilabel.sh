#!/bin/bash
out=$'/console.out'
ckpt_dir=$'./checkpoints/'
date=$(date +%y-%m-%d-%H-%M)    

nohup \
python eval_multilabel.py --image-size '232 300' --augmented 1 --tta \
> "./log/$date$_console.out" & 
