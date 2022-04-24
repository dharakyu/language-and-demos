from speaker import Speaker
from listener import Listener
from game import SimpleGame, SignalingBanditsGame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from statistics import mean

lr = 1e-4
num_epochs = 1000
num_batches_per_epoch = 100

def run_epoch(split, speaker, listener, optimizer, loss_func):
    training = split == 'train'
    batch_rewards = []
    batch_losses = []
    batch_accuracy = []
    for _ in range(num_batches_per_epoch):
        game = SignalingBanditsGame()
        
        games = game.sample_batch()
        games = torch.from_numpy(games)
        reward_matrix = game.reward_matrix
        reward_matrix = torch.from_numpy(reward_matrix)

        #breakpoint()
        messages, message_lens = speaker(games, reward_matrix)
        scores = listener(games, messages, message_lens)

        # get the listener predictions
        preds = torch.argmax(scores, dim=-1)    # (batch_size)

        # get the rewards associated with the objects in each game
        game_rewards = game.compute_rewards(games)  # (batch_size, num_choices)

        # what reward did the model actually earn
        model_rewards = game_rewards.gather(-1, preds.unsqueeze(-1))

        # what is the maximum reward and the associated index
        max_rewards, argmax_rewards = game_rewards.max(-1)

        avg_reward = model_rewards.squeeze(-1).mean().item()
        avg_max_reward = max_rewards.mean().item()

        accuracy = (argmax_rewards == preds).float().mean().item()

        # multiply by -1 bc we are maximizing the expected reward which means minimizing the negative of that
        loss = -1 * torch.bmm(scores.unsqueeze(1), game_rewards.unsqueeze(-1)).squeeze().sum()

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

def main():
    s = Speaker()
    l = Listener()
    lr = 1e-5
    loss_func = nn.CrossEntropyLoss()

    optimizer = optim.Adam(list(s.parameters()) +
                           list(l.parameters()),
                           lr=lr)

    for i in range(num_epochs):
        print('epoch', i)
        run_epoch('train', s, l, optimizer, loss_func)
        run_epoch('val', s, l, optimizer, loss_func)

main()