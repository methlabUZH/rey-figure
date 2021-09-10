#!/usr/bin/bash



learning_rates=(0.1 0.01)
weight_decays=(5e-4 5e-3)

# resnet 18 not augmented
for lr in "${learning_rates[@]}"
do
  for wd in "${weight_decays[@]}"
  do
    bsub -n 8 -R "rusage[mem=4500,ngpus_excl_p=4]" -R "select[gpu_model0==TITANRTX]" python train.py --data-root $ROCF_DATA/serialized-data/scans-2018-224x224 \
    --results_dir $ROCF_DATA/serialized_data/results/ \
    --finetune-file ./resnet18-finetune.txt \
    --arch resnet18 --wd "$wd" --lr "$lr" --optimizer adam --lr-decay exponential --gamma 0.99

    bsub -n 8 -R "rusage[mem=4500,ngpus_excl_p=4]" -R "select[gpu_model0==TITANRTX]" python train.py --data-root $ROCF_DATA/serialized-data/scans-2018-224x224 \
        --results_dir $ROCF_DATA/serialized_data/results/ \
        --finetune-file ./resnet18-finetune.txt \
        --arch resnet18 --wd "$wd" --lr "$lr" --optimizer sgd --lr-decay stepwise --gamma 0.1
  done
done


# resnext29_16x64d not augmented
for lr in "${learning_rates[@]}"
do
  for wd in "${weight_decays[@]}"
  do
    bsub -n 8 -R "rusage[mem=4500,ngpus_excl_p=4]" -R "select[gpu_model0==TITANRTX]" python train.py --data-root $ROCF_DATA/serialized-data/scans-2018-224x224 \
    --results_dir $ROCF_DATA/serialized_data/results/ \
    --finetune-file ./resnext29_16x64d-finetune.txt \
    --arch resnext29_16x64d --wd "$wd" --lr "$lr" --optimizer adam --lr-decay exponential --gamma 0.99

    bsub -n 8 -R "rusage[mem=4500,ngpus_excl_p=4]" -R "select[gpu_model0==TITANRTX]" python train.py --data-root $ROCF_DATA/serialized-data/scans-2018-224x224 \
        --results_dir $ROCF_DATA/serialized_data/results/ \
        --finetune-file ./resnext29_16x64d-finetune.txt \
        --arch resnext29_16x64d --wd "$wd" --lr "$lr" --optimizer sgd --lr-decay stepwise --gamma 0.1
  done
done