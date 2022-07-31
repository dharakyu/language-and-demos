#!/bin/bash

for i in {1..5};
    do
        CUDA_VISIBLE_DEVICES=3 python train.py \
            --epochs 30 \
            --cuda \
            --wandb \
            --name "optimize-separately_chain-len-4_run-${i}" \
            --group "optimize-separately_chain-len-4" \
            --discrete_comm \
            --chain_length 4 \
            --partial_reward_matrix \
            --chunks 2 4 6 8 10 12
    done