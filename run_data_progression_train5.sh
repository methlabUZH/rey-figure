#!/bin/sh


CUDA_VISIBLE_DEVICES="4" python train_multilabel.py --image-size "232 300" --augment 1 --seed 1 --max_n 12000
CUDA_VISIBLE_DEVICES="4" python train_multilabel.py --image-size "232 300" --augment 1 --seed 1 --max_n 14000
CUDA_VISIBLE_DEVICES="4" python train_multilabel.py --image-size "232 300" --augment 1 --seed 1 --max_n 16000


CUDA_VISIBLE_DEVICES="4" python eval_multilabel.py --image-size "232 300" --augment 1 --n_train 12000
CUDA_VISIBLE_DEVICES="4" python eval_multilabel.py --image-size "232 300" --augment 1 --n_train 14000
CUDA_VISIBLE_DEVICES="4" python eval_multilabel.py --image-size "232 300" --augment 1 --n_train 16000
