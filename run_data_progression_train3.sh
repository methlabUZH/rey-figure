#!/bin/sh


CUDA_VISIBLE_DEVICES="3" python train_multilabel.py --image-size "232 300" --augment 1 --seed 1 --max_n 10000
CUDA_VISIBLE_DEVICES="3" python eval_multilabel.py --image-size "232 300" --augment 1 --n_train 10000