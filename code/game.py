import numpy as np
import torch

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
                    max_color_val=2, max_shape_val=1,
                    randomize_rewards=True
                ):
        self.num_choices = num_choices
        self.num_colors = num_colors
        self.num_shapes = num_shapes

        self.color_utilities = np.linspace(start=-max_color_val, stop=max_color_val, num=num_colors)
        self.shape_utilities = np.linspace(start=-max_shape_val, stop=max_shape_val, num=num_shapes)

    def sample_reward_matrix(self, randomize_rewards=True):
        """
        Generate the reward matrix, to which the speaker will gain access

        Arguments:
        randomize_rewards: True if the utility associated with a particular feature is randomized for each game
            default is [-2, 0, 2] for colors and [-1, 0, 1] for shapes

        Return:
        reward_matrix: np.array of size (self.num_colors*self.num_shapes, self.num_colors + self.num_shapes + 2)
        The first (self.num_colors + self.num_shapes) items in a row are the object embedding
        The next item is the utility associated with that object
        The next item is a Boolean representing whether that object is in the listener's context

        i.e. if the item described by the feature [0, 1, 0, 1, 0, 0] has utility 3 and is present in the listener's
        context, then the corresponding row in reward_matrix is [0, 1, 0, 1, 0, 0, 3, 1]
        """
        if randomize_rewards:
            np.random.shuffle(self.color_utilities)
            np.random.shuffle(self.shape_utilities)
            #breakpoint()
            #if np.random.randint(0, 2):
            #    self.color_utilities = np.flip(self.color_utilities)

            #if np.random.randint(0, 2):
            #    self.shape_utilities = np.flip(self.shape_utilities)

        indices = np.random.choice(self.num_colors*self.num_shapes, size=self.num_choices, replace=False)

        curr_idx = 0
        reward_matrix = []
        for i in range(self.num_colors):
            for j in range(self.num_shapes):
                color_embedding = np.zeros(shape=(self.num_colors,))
                color_embedding[i] = 1
                color_utility = self.color_utilities[i]

                shape_embedding = np.zeros(shape=(self.num_shapes,))
                shape_embedding[j] = 1
                shape_utility = self.shape_utilities[j]

                embedding = np.concatenate((color_embedding, shape_embedding))

                in_listener_context = int(curr_idx in indices)
                #breakpoint()
                embedding = np.append(embedding, [color_utility+shape_utility, in_listener_context])
                reward_matrix.append(embedding)

                curr_idx += 1
        
        return np.array(reward_matrix)

    def get_listener_view(self, reward_matrix):
        """
        Given a reward matrix, get the corresponding listener view

        Arguments:
        reward_matrix: np.array of size (self.num_colors*self.num_shapes, self.num_colors + self.num_shapes + 2)

        Return:
        game: np.array of size (num_choices, self.num_colors + self.num_shapes)
        """
        indices = np.where(reward_matrix[:, -1] == 1)[0]
        listener_view = reward_matrix[indices, :-2]  # lop off the last two elements

        return listener_view


    def sample_batch(self, batch_size=32):
        """
        Sample games for a whole batch

        Return
        batch_reward_matrices: np.array of size (batch_size, self.num_colors*self.num_shapes, self.num_colors+self.num_shapes+2)
        batch_listener_views: np.array of size (batch_size, self.num_choices, self.num_colors+self.num_shapes)
        """
        batch_reward_matrices = []
        batch_listener_views = []
        
        for i in range(batch_size):
            reward_matrix = self.sample_reward_matrix()
            listener_view = self.get_listener_view(reward_matrix)
            
            batch_reward_matrices.append(reward_matrix)
            batch_listener_views.append(listener_view)

        return np.array(batch_reward_matrices), np.array(batch_listener_views)

    def get_rewards_for_single_game(self, listener_view, reward_matrix):
        object_rewards = []

        for i in range(self.num_choices):
            curr_obj = listener_view[i]
            for j in range(self.num_colors*self.num_shapes):
                if torch.equal(curr_obj, reward_matrix[j, :-2]):
                    reward = reward_matrix[j, -2]
                    object_rewards.append(reward)
                    break

        return object_rewards

    def compute_rewards(self, choices, listener_views, reward_matrices):
        """
        Given a batch of games and model predictions, compute the rewards

        Arguments:
        choices: torch.Tensor of size (batch_size)
        listener_views: np.array of size (batch_size, self.num_choices, self.num_colors+self.num_shapes)
        reward_matrices: np.array of size (batch_size, self.num_colors*self.num_shapes, self.num_colors+self.num_shapes+2)

        Return:
        accuracy: np.array of size (batch_size)
        """
        batch_size = listener_views.shape[0]
        batch_rewards = []
        #breakpoint()
        for i in range(batch_size):
            listener_view = listener_views[i]
            reward_matrix = reward_matrices[i]

            rewards = self.get_rewards_for_single_game(listener_view, reward_matrix)

            batch_rewards.append(rewards)
 
        return torch.Tensor(batch_rewards)
            

            







    