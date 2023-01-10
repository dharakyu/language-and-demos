#!/bin/bash

for i in 1000 10000;
do
    CUDA_VISIBLE_DEVICES=7 python train.py \
        --cuda \
        --chain_length 2 \
        --perfect_teacher \
        --learn_from_demos \
        --num_examples_for_demos 1 \
        --pedagogical_sampling \
        --optimize_jointly \
        --teacher_alpha ${i} \
        --lr 5e-06 \
        --wandb \
        --name "perfect-teacher_pedagogical-demo_k=1_teacher-alpha=${i}"
done