#!/bin/bash

for embedding_size in 64 128 256;
do
    for hidden_size in 200 300 400;
    do
        for j in {1..5};
        do
            python train.py --wandb --name "run-${j}" --group "discrete_${embedding_size}-embed_${hidden_size}-hidden" --epochs 40 --embedding_size ${embedding_size} --hidden_size ${hidden_size}
        done
    done
done