#!/bin/bash

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=4 python train.py \
        --cuda \
        --chain_length 2 \
        --perfect_teacher \
        --discrete_comm \
        --max_message_len 8 \
        --optimize_jointly \
        --lr 5e-06 \
        --epochs 50 \
        --wandb \
        --name "perfect-teacher_long-lang_${i}" \
        --group "perfect-teacher_long-lang"
done