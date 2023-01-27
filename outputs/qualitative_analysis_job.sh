#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python train.py \
                --cuda \
                --chain_length 2 \
                --perfect_teacher \
                --discrete_comm \
                --max_message_len 12 \
                --lr 5e-06 \
                --train_percent 0.8 \
                --name perfect-teacher_lang_max-message-len=12_train-percent=0.8_log \
                --save_outputs \
                --wandb 

CUDA_VISIBLE_DEVICES=2 python train.py \
                --cuda \
                --chain_length 2 \
                --perfect_teacher \
                --discrete_comm \
                --max_message_len 12 \
                --lr 5e-06 \
                --train_percent 0.05 \
                --name perfect-teacher_lang_max-message-len=12_train-percent=0.05_log \
                --save_outputs \
                --wandb 

CUDA_VISIBLE_DEVICES=2 python train.py \
                --cuda \
                --chain_length 2 \
                --perfect_teacher \
                --learn_from_demos \
                --num_examples_for_demos 2 \
                --pedagogical_sampling \
                --lr 5e-06 \
                --train_percent 0.8 \
                --name perfect-teacher_pedagogical-demos_k=2_train-percent=0.8_log \
                --save_outputs \
                --wandb 

CUDA_VISIBLE_DEVICES=2 python train.py \
                --cuda \
                --chain_length 2 \
                --perfect_teacher \
                --learn_from_demos \
                --num_examples_for_demos 2 \
                --pedagogical_sampling \
                --lr 5e-06 \
                --train_percent 0.05 \
                --name perfect-teacher_pedagogical-demos_k=2_train-percent=0.05_log \
                --save_outputs \
                --wandb  