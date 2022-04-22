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
        game = SimpleGame()
        
        utilities = game.generate_batch()
        utilities = torch.from_numpy(utilities)

        messages, message_lens = speaker(utilities)
        #breakpoint()
        logits = listener(messages, message_lens)
        #choices = torch.tensor(np.random.randint(0, 3, size=(32,)))
       
        #breakpoint()

        #rf_loss = -(log_probs * rewards).mean()
        #loss = rf_loss
        preds = torch.argmax(logits, dim=-1)
        ground_truth = torch.argmax(utilities, dim=-1)

        rewards = game.compute_rewards(utilities, preds.unsqueeze(-1))
        avg_reward = compute_stats(utilities, preds, rewards)
        accuracy = (ground_truth == preds).float().mean().item()

        # multiply by -1 bc we are maximizing the expected reward which means minimizing the negative of that
        loss = -1 * torch.bmm(logits.unsqueeze(1), utilities.unsqueeze(-1)).squeeze().sum()
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