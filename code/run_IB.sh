#!/bin/bash

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=5 python train.py \
        --cuda \
        --wandb \
        --name "learn_from_language_IB-${i}" \
        --group "learn_from_language_IB" \
        --discrete_comm \
        --chain_length 2 \
        --partial_reward_matrix \
        --chunks 2 4 \
        --num_listener_views 1 \
        --inductive_bias
done

for i in {1..5};
do
    CUDA_VISIBLE_DEVICES=5 python train.py \
        --cuda \
        --wandb \
        --name "learn_from_demos_k=15_IB-${i}" \
        --group "learn_from_demos_k=15_IB" \
        --learn_from_demos \
        --num_examples_for_demos 15 \
        --chain_length 2 \
        --partial_reward_matrix \
        --chunks 2 4 \
        --num_listener_views 1 \
        --inductive_bias
done