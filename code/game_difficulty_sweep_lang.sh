#!/bin/bash

for k in 4 6;
    for num_features in 3 4 5 6 7;
    do
        for i in {1..5};
        do
            CUDA_VISIBLE_DEVICES=9 python train.py \
                --cuda \
                --chain_length 2 \
                --perfect_teacher \
                --discrete_comm \
                --max_message_len ${k} \
                --num_colors ${num_features} \
                --num_shapes ${num_features} \
                --lr 5e-06 \
                --wandb \
                --name "perfect-teacher_lang_${num_features}x${num_features}_len=${k}_${i}" \
                --group "perfect-teacher_lang_${num_features}x${num_features}_len=${k}"
        done
    done
done