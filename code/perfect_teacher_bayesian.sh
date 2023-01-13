#!/bin/bash

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --cuda \
        --chain_length 2 \
        --perfect_teacher \
        --learn_from_demos \
        --num_examples_for_demos 1 \
        --pedagogical_sampling \
        --use_bayesian_teacher \
        --lr 5e-06 \
        --epochs 50 \
        --wandb \
        --name "perfect-teacher_bayesian-teacher_k=1_${i}" \
        --group "perfect-teacher_bayesian-teacher_k=1"
done