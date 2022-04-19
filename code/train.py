from speaker import Speaker
from listener import Listener
from game import SignalingBanditsGame

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from statistics import mean

lr = 1e-4
num_epochs = 1000
num_batches_per_epoch = 20

def compute_stats(utilities, choices, rewards):
    #breakpoint()
    # avg_reward = (rewards.squeeze(-1) / utilities.sum(1)).mean()
    avg_reward = (rewards.squeeze(-1)).mean()
    return avg_reward

def run_epoch(split, speaker, listener, optimizer, loss_func):
    training = split == 'train'
    batch_rewards = []
    batch_losses = []
    batch_accuracy = []
    for _ in range(num_batches_per_epoch):
        game = SignalingBanditsGame()
        utilities = game.generate_batch()
        utilities = torch.from_numpy(utilities)

        messages, message_lens = speaker(utilities)
        #breakpoint()
        logits = listener(messages, message_lens)
        #choices = torch.tensor(np.random.randint(0, 3, size=(32,)))
       
        #breakpoint()

        #rf_loss = -(log_probs * rewards).mean()
        #loss = rf_loss
        model_choices = torch.argmax(logits, dim=-1)
        ground_truth = torch.argmax(utilities, dim=-1)

        rewards = game.compute_rewards(utilities, model_choices.unsqueeze(-1))
        avg_reward = compute_stats(utilities, model_choices, rewards)
        accuracy = (ground_truth == model_choices).float().mean().item()

        #breakpoint()
        #choices_one_hot = F.one_hot(choices).float()
        #ground_truth_one_hot = F.one_hot(ground_truth).float()
        loss = loss_func(logits, ground_truth)
        #breakpoint()

        if training:
            loss.backward()
            optimizer.step()

        batch_rewards.append(avg_reward.item())
        batch_losses.append(loss.item())
        batch_accuracy.append(accuracy)
    
    print(split)
    print('avg reward', mean(batch_rewards))
    print('train loss', mean(batch_losses))
    print('accuracy', mean(batch_accuracy))
    print('choices', model_choices.float().mean().item())

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
        if i > 10: breakpoint()
        run_epoch('train', s, l, optimizer, loss_func)
        run_epoch('val', s, l, optimizer, loss_func)

main()