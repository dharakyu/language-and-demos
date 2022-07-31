from speaker import Speaker
from listener import Listener
from game import SignalingBanditsGame
from arguments import get_args
from agent import Agent

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd

from statistics import mean
from collections import defaultdict
import time
import os
import copy
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

    
    data_to_log = []
    for batch_idx in range(args.num_batches_per_epoch):
        messages_to_log = []
        reward_matrices_to_log = []
        reward_matrices_views_to_log = []

        reward_matrices, listener_views = game.sample_batch(num_listener_views=args.num_listener_views)
        batch_size = reward_matrices.shape[0]

        # append flattened ground truth reward matrix
        flattened_reward_matrices = reward_matrices.reshape(batch_size, -1)
        reward_matrices_to_log.append(flattened_reward_matrices)

        # convert reward matrices and listener views to tensors
        reward_matrices = torch.from_numpy(reward_matrices).float()
        listener_views = torch.from_numpy(listener_views).float()

        if args.partial_reward_matrix:
            reward_matrices_views = game.generate_masked_reward_matrix_views(reward_matrices, 
                                                                                chunks=args.chunks,
                                                                                num_views=args.chain_length)
        
        if args.cuda:   # move to GPU
            reward_matrices = reward_matrices.cuda()
            listener_views = listener_views.cuda()
            
            if args.partial_reward_matrix:
                reward_matrices_views = reward_matrices_views.cuda()

        lang_i = None
        lang_len_i = None
        losses_for_curr_batch = []
        if training and args.train_chain_length is not None:
            num_gens_to_iterate_over = args.train_chain_length
        else:
            num_gens_to_iterate_over = args.chain_length

        for i in range(num_gens_to_iterate_over):
            agent_i = agents[i]
            
            # what view is the agent seeing?
            if args.partial_reward_matrix:
                agent_view = reward_matrices_views[i]
            else:
                agent_view = reward_matrices

            lang_i, lang_len_i, scores_i = agent_i(reward_matrices=agent_view,
                                                    input_lang=lang_i,
                                                    input_lang_len=lang_len_i,
                                                    games=listener_views
                                                )
            
            # append flattened message produced by agent i
            flattened_message = lang_i.view(batch_size, -1).cpu().detach().numpy()
            messages_to_log.append(flattened_message)

            # append flattened view of reward matrices seen by agent i
            flattened_agent_view = agent_view.view(batch_size, -1).cpu().detach().numpy()
            reward_matrices_views_to_log.append(flattened_agent_view)

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

            # minimize negative expected reward
            losses = -1 * torch.einsum('bvc,bvc->bv',(scores_i.exp(), game_rewards))    # size (batch_size, num_views)
            loss = losses.mean()    # take the mean over all views and each reward matrix in the batch

            batch_rewards_after_agent_i[i].append(avg_reward / avg_max_reward)
            batch_losses_after_agent_i[i].append(loss.item())
            batch_accuracy_after_agent_i[i].append(accuracy)

            losses_for_curr_batch.append(loss)
        
        if training:    # use the losses from all the agents
            if args.optimize_jointly:
                loss_across_agents = torch.mean(torch.stack(losses_for_curr_batch))
                loss_across_agents.backward()
                optimizer.step()
            else:
                for i in range(num_gens_to_iterate_over - 1):
                    # loss for agent i = loss for agent i+1 + ... + loss for agent n-1
                    loss = torch.sum(torch.stack(losses_for_curr_batch)[i:])
                    loss.backward(retain_graph=True)    # need to retain computation graph bc of the way we compute the loss
                    optimizer[i].step()


        batch_data_to_log = reward_matrices_to_log + reward_matrices_views_to_log + messages_to_log
        data_to_log.append(batch_data_to_log)
        
    
    print(dataset_split)
    #breakpoint()
    metrics = {}
    for i in range(num_gens_to_iterate_over):
        metrics['reward_' + str(i)] = mean(batch_rewards_after_agent_i[i])
        metrics['loss_' + str(i)] = mean(batch_losses_after_agent_i[i])
        metrics['accuracy_' + str(i)] = mean(batch_accuracy_after_agent_i[i])

        print('metrics after passing through agent', i)
        print('reward:', metrics['reward_' + str(i)])
        print('loss:', metrics['loss_' + str(i)])
        print('accuracy:', metrics['accuracy_' + str(i)])

    # initialize a DataFrame for logging reward matrices and messages
    reward_matrix_col_names = ['reward_matrix_' + str(i) for i in range(num_gens_to_iterate_over)]
    message_col_names = ['message_' + str(i) for i in range(num_gens_to_iterate_over)]
    col_names = ['reward_matrix'] + reward_matrix_col_names + message_col_names
    df = pd.DataFrame(data_to_log, columns=col_names)
    
    return metrics, df

def main():
    args = get_args()
    
    if args.use_same_agent:
        agent = Agent(object_encoding_len = args.num_colors + args.num_shapes,
                        num_objects = args.num_colors * args.num_shapes,
                        hidden_size=args.hidden_size,
                        use_discrete_comm=args.discrete_comm,
                        max_message_len=args.max_message_len,
                        vocab_size=args.vocab_size)
        agents = nn.ModuleList([agent for _ in range(args.chain_length)])
    else:
        agents = nn.ModuleList([Agent(object_encoding_len = args.num_colors + args.num_shapes,
                                    num_objects = args.num_colors * args.num_shapes,
                                    hidden_size=args.hidden_size,
                                    use_discrete_comm=args.discrete_comm,
                                    max_message_len=args.max_message_len,
                                    vocab_size=args.vocab_size) 
                                for _ in range(args.chain_length)])

    if args.cuda:
        agents = nn.ModuleList([agent.cuda() for agent in agents])

    if args.optimize_jointly:
        optimizer = optim.Adam(agents.parameters(),
                           lr=args.lr)
    else:
        optimizer = [optim.Adam(agent.parameters(), lr=args.lr)
                        for agent in agents]

    metrics = defaultdict(list)
    game = SignalingBanditsGame(num_colors=args.num_colors, num_shapes=args.num_shapes)

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

        train_metrics, train_df = run_epoch('train', game, agents, optimizer, args)
        val_metrics, val_df = run_epoch('val', game, agents, optimizer, args)

        # just save the validation set, and rewrite it after each iter
        if args.save_outputs:
            pkl_full_save_path = os.path.join(args.save_dir, args.name + '_val.pkl')
            val_df.to_pickle(pkl_full_save_path)

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