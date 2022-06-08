#!/bin/bash

for chain_length in 2;
do
    for i in {4..5};
    do
        CUDA_VISIBLE_DEVICES=5 python train.py \
            --cuda \
            --wandb \
            --name "4x4-partial_chunk-2_chain-len-${chain_length}_run-${i}" \
            --group "4x4-partial_chunk-2_chain-len-${chain_length}" \
            --discrete_comm \
            --chain_length ${chain_length} \
            --partial_reward_matrix \
            --use_same_agent \
            --chunks 2 4 6 8 10 12 \
            --save_outputs
    done
done