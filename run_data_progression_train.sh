#!/bin/sh


CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 1000
CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 2000
CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 4000
CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 60000
CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 10000
CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 15000
CUDA_VISIBLE_DEVICES="5" python train_multilabel.py --image-size "116 150" --augment 0 --seed 1 --max_n 20000