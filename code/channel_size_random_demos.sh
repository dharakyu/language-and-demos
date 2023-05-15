#!/bin/bash

for num_demos in 1 2 5 10 20;
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=4 python train.py \
            --cuda \
            --chain_length 2 \
            --perfect_teacher \
            --learn_from_demos \
            --num_examples_for_demos ${num_demos} \
            --optimize_jointly \
            --lr 5e-06 \
            --wandb \
            --name "perfect-teacher_random-demo_k=${num_demos}_${i}" \
            --group "perfect-teacher_random-demo_k=${num_demos}"
    done
done