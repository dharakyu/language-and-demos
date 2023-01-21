#!/bin/bash

for num_utilities in 14 12 10 8 6 4 2
do
    for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=3 python train.py \
            --cuda \
            --chain_length 2 \
            --partial_reward_matrix \
            --chunks ${num_utilities} ${num_utilities} \
            --discrete_comm \
            --max_message_len 3 \
            --optimize_jointly \
            --lr 5e-06 \
            --wandb \
            --name "imperfect-teacher_num-utils=${num_utilities}_lang_max-message-len=3_${i}" \
            --group "imperfect-teacher_num-utils=${num_utilities}_lang_max-message-len=3"
    done
done