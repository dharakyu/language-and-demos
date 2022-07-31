#!/bin/bash

for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=2 python train.py \
            --epochs 30
            --cuda \
            --wandb \
            --name "optimize-jointly_chain-len-4_run-${i}" \
            --group "optimize-jointly_chain-len-4" \
            --discrete_comm \
            --chain_length 4 \
            --partial_reward_matrix \
            --chunks 2 4 6 8 10 12 \
            --optimize_jointly
    done