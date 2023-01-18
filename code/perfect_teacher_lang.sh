#!/bin/bash

for num_messages in 3 4 7 12 22;
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=9 python train.py \
            --cuda \
            --chain_length 2 \
            --perfect_teacher \
            --discrete_comm \
            --max_message_len ${num_messages} \
            --optimize_jointly \
            --lr 5e-06 \
            --wandb \
            --name "perfect-teacher_lang-len=${num_messages}_${i}" \
            --group "perfect-teacher_lang-len=${num_messages}"
    done
done