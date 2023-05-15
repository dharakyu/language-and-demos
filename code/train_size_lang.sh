#!/bin/bash

for train_percent in 0.1 0.05;
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=2 python train.py \
            --cuda \
            --chain_length 2 \
            --perfect_teacher \
            --discrete_comm \
            --max_message_len 12 \
            --train_percent ${train_percent} \
            --lr 5e-06 \
            --wandb \
            --name "perfect-teacher_lang_train-percent=${train_percent}_${i}" \
            --group "perfect-teacher_lang_train-percent=${train_percent}"
    done
done