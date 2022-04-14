#!/bin/sh


CUDA_VISIBLE_DEVICES="0" python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n 1000
CUDA_VISIBLE_DEVICES="0" python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n 2000
CUDA_VISIBLE_DEVICES="0" python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n 4000
CUDA_VISIBLE_DEVICES="0" python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n 6000

CUDA_VISIBLE_DEVICES="0" python eval_multilabel.py --image-size "232 300" --augment 0 --n_train 1000
CUDA_VISIBLE_DEVICES="0" python eval_multilabel.py --image-size "232 300" --augment 0 --n_train 2000
CUDA_VISIBLE_DEVICES="0" python eval_multilabel.py --image-size "232 300" --augment 0 --n_train 4000
CUDA_VISIBLE_DEVICES="0" python eval_multilabel.py --image-size "232 300" --augment 0 --n_train 6000