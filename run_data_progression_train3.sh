#!/bin/sh



CUDA_VISIBLE_DEVICES="2" python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n 12000
CUDA_VISIBLE_DEVICES="2" python train_multilabel.py --image-size "232 300" --augment 0 --seed 1 --max_n 14000
CUDA_VISIBLE_DEVICES="2" python eval_multilabel.py --image-size "232 300" --augment 0 --n_train 12000
CUDA_VISIBLE_DEVICES="2" python eval_multilabel.py --image-size "232 300" --augment 0 --n_train 14000