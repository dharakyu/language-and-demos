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
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def run_epoch(dataset_split, game, agents, optimizer, args):
    """
    Arguments:
    dataset_split: string
    game: np.array of size (batch_size, num_choices, object_encoding_len)
    agents: list of Agent objects
    optimizer: PyTorch optimizer object
    args: 

    Return:
    metrics: dict
    """
    training = dataset_split == 'train'

    batch_rewards_after_agent_i = [[] for _ in range(args.chain_length)]
    batch_losses_after_agent_i = [[] for _ in range(args.chain_length)]
    batch_accuracy_after_agent_i = [[] for _ in range(args.chain_length)]
    
    for batch_idx in range(args.num_batches_per_epoch):
        reward_matrices, listener_views = game.sample_batch(num_listener_views=args.num_listener_views)

        reward_matrices = torch.from_numpy(reward_matrices).float()
        listener_views = torch.from_numpy(listener_views).float()

        if args.partial_reward_matrix:
            reward_matrices_views = game.generate_masked_reward_matrix_views(reward_matrices, args.chain_length)
        
        if args.cuda:
            reward_matrices = reward_matrices.cuda()
            listener_views = listener_views.cuda()
            
            if args.partial_reward_matrix:
                reward_matrices_views = reward_matrices_views.cuda()

        lang_i = None
        lang_len_i = None
        losses_for_curr_batch = []
        for i in range(args.chain_length):
            agent_i = agents[i]
            #breakpoint()
            if args.partial_reward_matrix:
                agent_view = reward_matrices_views[i]
            else:
                agent_view = reward_matrices

            lang_i, lang_len_i, scores_i = agent_i(reward_matrices=agent_view,
                                                    input_lang=lang_i,
                                                    input_lang_len=lang_len_i,
                                                    games=listener_views
                                                )

            if args.cuda:
                lang_i = lang_i.cuda()
                if args.discrete_comm: lang_len_i = lang_len_i.cuda()

            # get the listener predictions
            preds = torch.argmax(scores_i, dim=-1)    # (batch_size)
                
            # get the rewards associated with the objects in each game
            game_rewards = game.compute_rewards(listener_views, reward_matrices)  # (batch_size, num_choices)
            # move to GPU if necessary
            game_rewards = game_rewards.to(reward_matrices.device)

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

        if training:    # use the losses from all the agents
            #breakpoint()
            loss_across_agents = torch.mean(torch.stack(losses_for_curr_batch))
            loss_across_agents.backward()
            optimizer.step()
    
    print(dataset_split)
    for i in range(args.chain_length):
        print('metrics after passing through agent', i)
        print('avg reward', mean(batch_rewards_after_agent_i[i]))
        print('train loss', mean(batch_losses_after_agent_i[i]))
        print('accuracy', mean(batch_accuracy_after_agent_i[i]))
        #print('predictions', preds.float().mean().item())
    
    metrics = {}
    for i in range(args.chain_length):
        metrics['reward_' + str(i)] = mean(batch_rewards_after_agent_i[i])
        metrics['loss_' + str(i)] = mean(batch_losses_after_agent_i[i])
        metrics['accuracy_' + str(i)] = mean(batch_accuracy_after_agent_i[i])

    return metrics

def main():
    args = get_args()
    
    if args.use_same_agent:
        agent = Agent(hidden_size=args.hidden_size,
                        use_discrete_comm=args.discrete_comm)
        agents = nn.ModuleList([agent for _ in range(args.chain_length)])
    else:
        agents = nn.ModuleList([Agent(hidden_size=args.hidden_size,
                                    use_discrete_comm=args.discrete_comm) 
                                for _ in range(args.chain_length)])

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
        #breakpoint()
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