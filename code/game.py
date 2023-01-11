import numpy as np
import itertools
import torch
import time

import math

class SimpleGame():
    def __init__(self, num_choices=3):
        self.num_choices = num_choices

    def sample_utilities(self):
        """
        Randomly select utilities for a single game

        Return:
        utilities: np.array of size (num_choices,)
        """
        utilities = np.random.choice(10, size=self.num_choices, replace=False) + 1
        return utilities.astype(np.float32) / utilities.max()

    def generate_batch(self, batch_size=32):
        """
        Generate games for a batch

        Return:
        batch: np.array of size (batch_size, num_choices)
        """
        batch = []
        for _ in range(batch_size):
            batch.append(self.sample_utilities())
        return np.array(batch)

    def compute_rewards(self, utilities, choices):
        """
        Given a batch of utilities and agent choices, compute the rewards for each agent.

        Return
        rewards: np.array of size (batch_size,)
        """
        rewards = utilities.gather(-1, choices)
        return rewards


class SignalingBanditsGame():
    def __init__(self, num_choices=3, 
                    num_colors=3, num_shapes=3, 
                    max_color_val=6, max_shape_val=3,
                    num_games_for_eval=1
                ):
        self.num_choices = num_choices
        self.num_colors = num_colors
        self.num_shapes = num_shapes

        color_utilities = np.linspace(start=-max_color_val, stop=max_color_val, num=num_colors)
        shape_utilities = np.linspace(start=-max_shape_val, stop=max_shape_val, num=num_shapes)
        
        color_orderings = list(itertools.permutations(color_utilities))
        shape_orderings = list(itertools.permutations(shape_utilities))

        self.possible_reward_assignments = list(itertools.product(color_orderings, shape_orderings))
        self.num_reward_assignments = len(self.possible_reward_assignments)

        # all possible listener contexts
        num_unique_objects = num_colors * num_shapes 
        self.combinations = list(itertools.combinations(range(num_unique_objects), r=num_choices))

        # fix the number of unique games on which we evaluate each agent's accuracy
        self.num_games_for_eval = num_games_for_eval

    def sample_reward_matrix(self, inductive_bias, split, train_percent):
        """
        Generate the reward matrix, to which the speaker will gain access

        Arguments:
            inductive_bias (Boolean)
            split (string): 'train' or 'val'
            train_percent (float): size of train split

        Return:
            reward_matrix: np.array of size (self.num_colors*self.num_shapes, self.num_colors + self.num_shapes + 1)
                The first (self.num_colors + self.num_shapes) items in a row are the object embedding
                The last item is the utility associated with that object

        i.e. if the item described by the feature [0, 1, 0, 1, 0, 0] has utility 3,
        then the corresponding row in reward_matrix is [0, 1, 0, 1, 0, 0, 3]
        """
        # determines the reward configuration we are using
        if inductive_bias:
            probs = [(1/24) * 0.2] * 24 + [(1/(self.num_reward_assignments-24)) * 0.8] * (self.num_reward_assignments-24)
            reward_assignment_idx = np.random.choice(self.num_reward_assignments, p=probs)
        else:
            if split == 'train':
                train_upper_bound_idx = int(self.num_reward_assignments * train_percent)
                reward_assignment_idx = np.random.randint(0, train_upper_bound_idx)
            else:
                val_lower_bound_idx = int(self.num_reward_assignments * train_percent)
                reward_assignment_idx = np.random.randint(val_lower_bound_idx, self.num_reward_assignments)

        reward_assignment = self.possible_reward_assignments[reward_assignment_idx]
        color_utilities, shape_utilities = reward_assignment
        
        curr_idx = 0
        reward_matrix = []
        for i in range(self.num_colors):
            for j in range(self.num_shapes):
                color_embedding = np.zeros(shape=(self.num_colors,))
                color_embedding[i] = 1
                color_utility = color_utilities[i]

                shape_embedding = np.zeros(shape=(self.num_shapes,))
                shape_embedding[j] = 1
                shape_utility = shape_utilities[j]

                embedding = np.concatenate((color_embedding, shape_embedding))
                embedding = np.append(embedding, [color_utility+shape_utility])
                
                reward_matrix.append(embedding)

                curr_idx += 1
    
        return np.array(reward_matrix)

    def sample_batch(self, inductive_bias, split, train_percent=0.8, batch_size=32):
        """
        Sample games for a whole batch
        Arguments:
        inductive_bias (Boolean): whether to skew the distribution of examples
        split (string): 'train' or 'val'
        train_percent (float): size of the train split
        batch_size (int)

        Return
        batch_reward_matrices: np.array of size (batch_size, self.num_colors*self.num_shapes, self.num_colors+self.num_shapes+1)
        batch_eval_listener_views: np.array of size (batch_size, self.num_games_for_eval, self.num_choices, self.num_colors+self.num_shapes)
        batch_all_listener_views: np.array of size (batch_size, len(self.combinations), self.num_choices, self.num_colors+self.num_shapes)
        """

        batch_reward_matrices = []
        batch_eval_listener_views = []
        batch_all_listener_views = []
        
        for i in range(batch_size):
            reward_matrix = self.sample_reward_matrix(inductive_bias, split, train_percent)
 
            all_listener_views = reward_matrix[self.combinations][:, :, :-1]

            random_indices = np.random.choice(a=len(self.combinations), size=self.num_games_for_eval, replace=False) 
            eval_listener_views = all_listener_views[random_indices, ...]

            batch_reward_matrices.append(reward_matrix)
            batch_eval_listener_views.append(eval_listener_views)
            batch_all_listener_views.append(all_listener_views)
        
        return np.array(batch_reward_matrices), np.array(batch_eval_listener_views), np.array(batch_all_listener_views)

    def compute_rewards(self, listener_views, reward_matrices):
        """
        Given a batch of games and model predictions, compute the rewards

        Arguments:
        listener_views: np.array of size (batch_size, num_views, self.num_choices, self.num_colors+self.num_shapes)
        reward_matrices: np.array of size (batch_size, self.num_colors*self.num_shapes, self.num_colors+self.num_shapes+2)

        Return:
        reshaped_rewards: torch.Tensor of size (batch_size, num_listener_views, self.num_choices)
        """

        batch_size = reward_matrices.shape[0]
        indices_of_ones = listener_views.nonzero()
        indices_of_ones_combined = indices_of_ones.view(indices_of_ones.shape[0] // 2, -1)
        # an example row is:
        # [batch_i, listener_view_j, object_in_context_k, 2, batch_i, listener_view_j, object_in_context_k, 4]
        # which means that in the ith data point in the batch, for the jth listener_view, for the kth object in the listener context,
        # there is a 1 at index 2 and index 4

        indices_into_batch = indices_of_ones_combined[:, 0]

        # arithmetic trick to derive the index into the reward matrix
        indices_into_reward_matrix_row = self.num_colors*indices_of_ones_combined[:, self.num_colors-1] + \
                                        (indices_of_ones_combined[:, -1]-self.num_shapes)

        # note that we only need the index of the sample in the batch and the row in the reward matrix
        # and then we pull out the LAST element, which is the reward
        rewards = reward_matrices[indices_into_batch, indices_into_reward_matrix_row, -1]

        # note: I verified that .view() yields the correct arrangement
        rewards_reshaped = rewards.view(batch_size, -1, self.num_choices)

        return rewards_reshaped

    def generate_masked_reward_matrix_views(self, reward_matrices, 
                                                    chunks, 
                                                    num_views,
                                                    same_agent_view,
                                                    no_additional_info,
                                                    num_utilities_seen_in_training):
        """
        For a batch of reward matrices, produce partial views

        Arguments:
        reward_matrices: np.array of size (batch_size, self.num_colors*self.num_shapes, self.num_colors+self.num_shapes+2)
        chunks: list(int) representing the indices at which to split the shuffled indices (i.e. [2, 4] would yield [a[:2], a[2:4], a[4:]])
        num_views: int representing the number of partial views to generate
        same_agent_view: bool to indicate if all agents in the chain see the same k objects
        no_additional_info: bool to indicate if there is no new information provided after the first agent
        num_utilities_seen_in training: if not None, this specifies how many utilities the agent sees

        Return:
        reward_matrix_views: np.array of size (num_views, batch_size, self.num_colors*self.num_shapes, self.num_colors+self.num_shapes+2)
        """

        # generate masks
        batch_size = reward_matrices.shape[0]
        batch_masks = []
        
        for _ in range(batch_size):
            shuffled_indices = np.arange(start=0, stop=reward_matrices.shape[1], dtype=int)
            np.random.shuffle(shuffled_indices)
            
            if chunks is not None:

                if num_utilities_seen_in_training is not None:  # view fewer utilities during training, compared to test
                    splits = np.array_split(shuffled_indices, num_utilities_seen_in_training)
                else:
                    splits = np.array_split(shuffled_indices, chunks)
                splits = splits[:num_views]
            else:
                splits = np.array_split(shuffled_indices, num_views)

            masks = []
            for split in splits:
                mask = list(set(range(reward_matrices.shape[1])) - set(split))
                masks.append(mask)
            batch_masks.append(masks)
        
        # apply masks to reward matrices (only to the first n-1 agents)
        reward_matrices_views = torch.stack([reward_matrices for _ in range(num_views)])
        for sample_idx in range(batch_size):
            for agent_idx in range(num_views):
                mask = batch_masks[sample_idx][agent_idx]
                reward_matrices_views[agent_idx, sample_idx, mask, :] = 0

                # make sure that the first 4 objects are guaranteed to be masked (which means that color 0 is never seen
                # during training and we can evaluate OOD performance)
                #if exclude_first_color:
                #    reward_matrices_views[agent_idx, sample_idx, 0:self.num_shapes, :] = 0
        
        if same_agent_view:
            # repeat the view of the first agent for all the agents
            reward_matrices_views = reward_matrices_views[0, :, :, :].unsqueeze(0).repeat(num_views, 1, 1, 1)

        if no_additional_info:
            # just do zeros for all agents after agent 0
            zeros = torch.zeros_like(reward_matrices_views)[:-1, :, :, :]
            reward_matrices_views = torch.cat([reward_matrices_views[0, :, :, :].unsqueeze(0), zeros], dim=0)
        
        return reward_matrices_views
            

            







    