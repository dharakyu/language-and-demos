from speaker import Speaker
from listener import Listener
from game import SignalingBanditsGame
from arguments import get_args
from agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from statistics import mean
from collections import defaultdict
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def run_epoch(split, game, agents, optimizer, args):
    """
    Arguments:
    split: string
    game: np.array of size (batch_size, num_choices, object_encoding_len)
    agents: list of Agent objects
    optimizer: PyTorch optimizer object
    args: 

    Return:
    metrics: dict
    """
    training = split == 'train'

    batch_rewards_after_agent_i = [[] for _ in range(args.chain_length)]
    batch_losses_after_agent_i = [[] for _ in range(args.chain_length)]
    batch_accuracy_after_agent_i = [[] for _ in range(args.chain_length)]
    
    for batch_idx in range(args.num_batches_per_epoch):
        start = time.time()
        reward_matrices, listener_views = game.sample_batch(num_listener_views=args.num_listener_views)
        reward_matrices = torch.from_numpy(reward_matrices).float()
        listener_views = torch.from_numpy(listener_views).float()
        
        if args.cuda:
            reward_matrices = reward_matrices.cuda()
            listener_views = listener_views.cuda()

        end = time.time()
        #print('time to sample game:', end-start)
        lang_i = None
        lang_len_i = None
        losses_for_curr_batch = []
        for i in range(args.chain_length):
            agent_i = agents[i]
            start = time.time()
            lang_i, lang_len_i, scores_i = agent_i(reward_matrices=reward_matrices,
                                                    input_lang=lang_i,
                                                    input_lang_len=lang_len_i,
                                                    games=listener_views
                                                )

            if args.cuda:
                lang_i = lang_i.cuda()
                lang_len_i = lang_len_i.cuda()
            end = time.time()
            #print('time to produce message', end-start)
            if i != 0:
                # get the listener predictions
                preds = torch.argmax(scores_i, dim=-1)    # (batch_size)
                
                start = time.time()
                # get the rewards associated with the objects in each game
                game_rewards = game.compute_rewards(listener_views, reward_matrices)  # (batch_size, num_choices)
                # move to GPU if necessary
                game_rewards = game_rewards.to(reward_matrices.device)
                end = time.time()
                #print('time to compute rewards:', end-start)

                # what reward did the model actually earn
                model_rewards = game_rewards.gather(-1, preds.unsqueeze(-1))

                # what is the maximum reward and the associated index
                max_rewards, argmax_rewards = game_rewards.max(-1)

                avg_reward = model_rewards.squeeze(-1).mean().item()
                avg_max_reward = max_rewards.mean().item()

                accuracy = (argmax_rewards == preds).float().mean().item()

                # multiply by -1 bc we are maximizing the expected reward which means minimizing the negative of that
                #losses = -1 * torch.bmm(scores_i.exp().unsqueeze(1), game_rewards.unsqueeze(-1)).squeeze().sum()

                losses = -1 * torch.einsum('bvc,bvc->bv',(scores_i.exp(), game_rewards))    # size (batch_size, num_views)
                loss = losses.mean()    # take the mean over all views and each reward matrix in the batch

                batch_rewards_after_agent_i[i].append(avg_reward / avg_max_reward)
                batch_losses_after_agent_i[i].append(loss.item())
                batch_accuracy_after_agent_i[i].append(accuracy)

                losses_for_curr_batch.append(loss)

            else:
                batch_rewards_after_agent_i[i].append(None)
                batch_losses_after_agent_i[i].append(None)
                batch_accuracy_after_agent_i[i].append(None)

        if training:    # use the losses from all the agents (starting with agent 1)
            #breakpoint()
            loss_across_agents = torch.mean(torch.stack(losses_for_curr_batch))
            loss_across_agents.backward()
            optimizer.step()
    
    print(split)
    for i in range(1, args.chain_length):
        print('metrics after passing through agent', i)
        print('avg reward', mean(batch_rewards_after_agent_i[i]))
        print('train loss', mean(batch_losses_after_agent_i[i]))
        print('accuracy', mean(batch_accuracy_after_agent_i[i]))
        #print('predictions', preds.float().mean().item())
    
    metrics = {}
    for i in range(1, args.chain_length):
        metrics['reward_' + str(i)] = mean(batch_rewards_after_agent_i[i])
        metrics['loss_' + str(i)] = mean(batch_losses_after_agent_i[i])
        metrics['accuracy_' + str(i)] = mean(batch_accuracy_after_agent_i[i])

    return metrics

def main():
    args = get_args()
    
    agents = nn.ModuleList([Agent(hidden_size=args.hidden_size) for _ in range(args.chain_length)])
    if args.cuda:
        agents = nn.ModuleList([agent.cuda() for agent in agents])

    optimizer = optim.Adam(agents.parameters(),
                           lr=args.lr)
    
    metrics = defaultdict(list)
    game = SignalingBanditsGame(num_reward_matrices=args.num_reward_matrices)

    if args.wandb:
        import wandb
        if args.group is not None:
            wandb.init(args.wandb_project_name, group=args.group, config=args)
        else:
            wandb.init(args.wandb_project_name, config=args)

        if args.name is not None:
            wandb.run.name = args.name
        else:
            args.name = wandb.run.name

    for i in range(args.epochs):
        print('epoch', i)
        start = time.time()
        train_metrics = run_epoch('train', game, agents, optimizer, args)
        val_metrics = run_epoch('val', game, agents, optimizer, args)

        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)] = value

        for metric, value in val_metrics.items():
            metrics['val_{}'.format(metric)] = value

        metrics['current_epoch'] = i
        end = time.time()
        print('elapsed time:', end-start)

        if args.wandb:
            wandb.log(metrics)
        
        
if __name__ == "__main__":
    main()