for k in 1 2 5 10 20 50 100;
do
    echo ${k}
    CUDA_VISIBLE_DEVICES=1 python train.py \
        --cuda \
        --wandb \
        --name "good-teacher_random-demo_k=${k}" \
        --learn_from_demos \
        --num_examples_for_demos ${k} \
        --chain_length 2 \
        --partial_reward_matrix \
        --chunks 6 8 \
        --optimize_jointly \
        --lr 5e-06
done