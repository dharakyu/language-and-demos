#!/bin/bash

for num_utilities in 14 12 10 8 6 4 2
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=5 python train.py \
            --cuda \
            --chain_length 2 \
            --partial_reward_matrix \
            --chunks ${num_utilities} ${num_utilities} \
            --learn_from_demos \
            --num_examples_for_demos 2 \
            --pedagogical_sampling \
            --optimize_jointly \
            --lr 5e-06 \
            --wandb \
            --name "imperfect-teacher_num-utils=${num_utilities}_pedagogical-demos_${i}" \
            --group "imperfect-teacher_num-utils=${num_utilities}_pedagogical-demos"
    done
done