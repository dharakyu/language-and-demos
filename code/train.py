from speaker import Speaker
from listener import Listener
from game import SignalingBanditsGame
from arguments import get_args

import torch
import torch.optim as optim

import numpy as np

from statistics import mean
from collections import defaultdict

def run_epoch(split, game, speaker, listener, optimizer, args):
    training = split == 'train'
    batch_rewards = []
    batch_losses = []
    batch_accuracy = []

    for _ in range(args.num_batches_per_epoch):
        reward_matrices, listener_views = game.sample_batch()
        listener_views = torch.from_numpy(listener_views).float()
        reward_matrices = torch.from_numpy(reward_matrices).float()

        messages, message_lens = speaker(reward_matrices)
        
        scores = listener(listener_views, messages, message_lens)
        
        # get the listener predictions
        preds = torch.argmax(scores, dim=-1)    # (batch_size)

        # get the rewards associated with the objects in each game
        game_rewards = game.compute_rewards(listener_views, reward_matrices)  # (batch_size, num_choices)

        # what reward did the model actually earn
        model_rewards = game_rewards.gather(-1, preds.unsqueeze(-1))

        # what is the maximum reward and the associated index
        max_rewards, argmax_rewards = game_rewards.max(-1)

        avg_reward = model_rewards.squeeze(-1).mean().item()
        avg_max_reward = max_rewards.mean().item()

        accuracy = (argmax_rewards == preds).float().mean().item()

        # multiply by -1 bc we are maximizing the expected reward which means minimizing the negative of that
        losses = -1 * torch.bmm(scores.exp().unsqueeze(1), game_rewards.unsqueeze(-1)).squeeze().sum()
        loss = losses.mean()

        if training:
            loss.backward()
            optimizer.step()

        batch_rewards.append(avg_reward / avg_max_reward)
        batch_losses.append(loss.item())
        batch_accuracy.append(accuracy)
    
    print(split)
    print('avg reward', mean(batch_rewards))
    print('train loss', mean(batch_losses))
    print('accuracy', mean(batch_accuracy))
    print('predictions', preds.float().mean().item())
    
    metrics = {
        'reward': mean(batch_rewards),
        'loss': mean(batch_losses),
        'accuracy': mean(batch_accuracy),
        'prediction_index': preds.float().mean().item()
    }

    return metrics

def main():
    args = get_args()
    s = Speaker(hidden_size=args.hidden_size)
    l = Listener(hidden_size=args.hidden_size)

    optimizer = optim.Adam(list(s.parameters()) +
                           list(l.parameters()),
                           lr=args.lr)
    
    metrics = defaultdict(list)
    game = SignalingBanditsGame(num_reward_matrices=args.num_reward_matrices)

    if args.wandb:
        import wandb
        wandb.init(args.wandb_project_name, group=args.group, config=args)

        if args.name is not None:
            wandb.run.name = args.name
        else:
            args.name = wandb.run.name

    for i in range(args.epochs):
        print('epoch', i)
        
        train_metrics = run_epoch('train', game, s, l, optimizer, args)
        val_metrics = run_epoch('val', game, s, l, optimizer, args)

        for metric, value in train_metrics.items():
            metrics['train_{}'.format(metric)] = value

        for metric, value in val_metrics.items():
            metrics['val_{}'.format(metric)] = value

        metrics['current_epoch'] = i

        if args.wandb:
            wandb.log(metrics)
        
        
main()