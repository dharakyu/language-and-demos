#!/bin/bash

for chain_length in 2 3 4 5;
do
    for i in {1..3};
    do
        CUDA_VISIBLE_DEVICES=7 python train.py --wandb --name "run-${i}" --group "chain-length-${chain_length}" --chain_length ${chain_length}
    done
done