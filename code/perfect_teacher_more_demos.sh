#!/bin/bash

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=9 python train.py \
        --cuda \
        --chain_length 2 \
        --perfect_teacher \
        --learn_from_demos \
        --num_examples_for_demos 5 \
        --pedagogical_sampling \
        --optimize_jointly \
        --lr 5e-06 \
        --epochs 50 \
        --wandb \
        --name "perfect-teacher_pedagogical-demo_k=5_${i}" \
        --group "perfect-teacher_pedagogical-demo_k=5"
done