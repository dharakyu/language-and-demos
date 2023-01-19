#!/bin/bash

for num_features in 3 4 5 6 7;
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=0 python train.py \
            --cuda \
            --chain_length 2 \
            --perfect_teacher \
            --learn_from_demos \
            --num_examples_for_demos 2 \
            --num_colors ${num_features} \
            --num_shapes ${num_features} \
            --lr 5e-06 \
            --wandb \
            --name "perfect-teacher_random-sampling_${num_features}x${num_features}_k=2_${i}" \
            --group "perfect-teacher_random-sampling_${num_features}x${num_features}_k=2"
    done
done