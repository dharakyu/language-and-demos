#!/bin/bash

for train_percent in 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.05;
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=1 python train.py \
            --cuda \
            --chain_length 2 \
            --perfect_teacher \
            --learn_from_demos \
            --num_examples_for_demos 10 \
            --pedagogical_sampling \
            --train_percent ${train_percent} \
            --lr 5e-06 \
            --wandb \
            --name "perfect-teacher_pedagogical-demos_k=10_train-percent=${train_percent}_${i}" \
            --group "perfect-teacher_pedagogical-demos_k=10_train-percent=${train_percent}"
    done
done