#!/bin/bash

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --cuda \
        --wandb \
        --name "4x4-full_chain-len-2_run-${i}" \
        --group "4x4-full_chain-len-2" \
        --discrete_comm \
        --chain_length 2 \
        --use_same_agent \
        --save_outputs
done

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --cuda \
        --wandb \
        --name "4x4-partial_chunk-10_chain-len-2_run-${i}" \
        --group "4x4-partial_chunk-10_chain-len-2" \
        --discrete_comm \
        --chain_length 2 \
        --partial_reward_matrix \
        --chunks 10 15
        --use_same_agent \
        --save_outputs
done
