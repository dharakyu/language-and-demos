import numpy as np
import torch

class SignalingBanditsGame():
    def __init__(self, num_choices=3):
        self.num_choices = num_choices

    def sample_utilities(self):
        utilities = np.array([1, 9, 10], dtype=np.float32)
        #utilities = np.random.choice(10, size=self.num_choices, replace=False) + 1
        #utilities = np.zeros(shape=(3,))
        #idx = np.random.choice(3)
        #utilities[idx] = 1
        return utilities.astype(np.float32) / utilities.max()

        return utilities.astype(np.float32)

    def generate_batch(self, batch_size=32):
        batch = []
        for _ in range(batch_size):
            batch.append(self.sample_utilities())

        return np.array(batch)

    def compute_rewards(
        self, utilities, choices
    ):
        """
        Given a batch of utilities and agent choices, compute the rewards for each agent.
        """
        rewards = utilities.gather(-1, choices)

        return rewards