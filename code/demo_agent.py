import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from listener import RNNEncoder
import data

class DemoAgent(nn.Module):
    def __init__(self, chain_length,
                object_encoding_len=6, num_objects=9,
                embedding_dim=64, hidden_size=100, 
                num_examples_for_demos=10,
                num_choices_in_listener_context=3):
                
        super().__init__()
        assert embedding_dim % 2 == 0, "input dim must be divisible by 2"
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.num_examples_for_demos = num_examples_for_demos
        self.num_choices_in_listener_context = num_choices_in_listener_context
        self.object_encoding_len = object_encoding_len

        self.games_embedding =  nn.Sequential(
                                    nn.Linear(object_encoding_len, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, embedding_dim)
                                )

        self.reward_matrix_embedding = nn.Sequential(
                                            nn.Linear(object_encoding_len + 1, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, embedding_dim)
                                        )

        self.examples_mlp = nn.Sequential(
                                        nn.Linear(object_encoding_len + 1, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, embedding_dim)
                                )
        # input size: (x, y, embedding_dim * (num_examples_for_demos * num_choices_in_lis_context + num_objects))
        self.composite_emb_compression_mlp = nn.Linear(embedding_dim * (num_examples_for_demos * num_choices_in_listener_context + num_objects),
                                                            embedding_dim)

        self.pedagogical_sampling = False


    def embed_reward_matrix_and_demos(self,
                                        reward_matrices,
                                        demos,
                                        ):
        """
        Produce a composite embedding of the reward matrix and the observed demos

        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+1)
        demos: torch.Tensor of size (batch_size, num_examples_in_demo, num_choices_in_lis_context, object_encoding_length+1)

        Return:
        reduced_composite_emb: torch.Tensor of size (batch_size, embedding_dim)
        """
        batch_size = reward_matrices.shape[0]
        
        reward_matrix_emb = self.reward_matrix_embedding(reward_matrices)  # (batch_size, num_objects, embedding_dim)
        reward_matrix_emb = reward_matrix_emb.view(batch_size, -1)    # (batch_size, embedding_dim * num_objects)

        demos = demos.to(reward_matrices.device)    # move to GPU
        reshaped_demos = demos.view(batch_size, -1, self.object_encoding_len + 1)  # (batch_size, num_examples_for_demos * num_choices_in_lis_context, object_encoding_length+1)
        demos_emb = self.examples_mlp(reshaped_demos)   # (batch_size, num_examples_for_demos * num_choices_in_lis_context, embedding_dim)
        demos_emb = demos_emb.view(batch_size, -1)   # (batch_size, embedding_dim * num_examples_for_demos * num_choices_in_lis_context)

        composite_emb = torch.cat([reward_matrix_emb, demos_emb], dim=1)    # (batch_size, embedding_dim * (num_examples_for_demos * num_choices_in_lis_context + num_objects))
        reduced_composite_emb = self.composite_emb_compression_mlp(composite_emb)   # (batch_size, embedding_dim)

        return reduced_composite_emb

    def forward(self, 
                reward_matrices,
                demos,
                all_possible_games,
                games_for_eval
                ):
        """
        Arguments:
        reward_matrices: 
            The batchwise reward matrices seen by the agent
            torch.Tensor of size (batch_size, num_objects, object_encoding_length+1)
        demos:
            The batchwise demos seen by the agent
            if this is the first agent in the chain, this is None
            torch.Tensor of size (batch_size, num_examples_in_demo, num_choices_in_lis_context, object_encoding_length+1)
        games_for_future_demos: 
            These are the games for which the agent makes a prediction, and the game/prediction combos are used
            as demonstrations for the next generation. This SHOULD be all the possible games (and then we can sample
            it down to a few to pass on, either through random or pedagogical selection)
            torch.Tensor of size (batch_size, num_games_to_eval, num_choices_in_lis_context, object_encoding_len)
        games_for_eval:
            These are the games that are used to evaluate the 

        Return:
        output_lang:
            if use_discrete_comm: torch.Tensor of size (batch_size, max_message_len, vocab_size)
            else: torch.Tensor of size (batch_size, vocab_size)
        output_lang_len:
            if use_discrete_comm: torch.Tensor of size (batch_size, )
            else: None
        scores: torch.Tensor of size (batch_size, num_views, num_choices)

        """
        batch_size = reward_matrices.shape[0]

        # 1. produce an embedding of the demos

        # if this is the first agent in the chain, there is no demo (represented with just zeros)
        if demos is None:
            demos = torch.zeros(size=(batch_size, self.num_examples_for_demos, self.num_choices_in_listener_context, self.object_encoding_len+1))
        
        # this representation is used to "help" agents decide which objects to pick
        reward_matrix_and_demos_emb = self.embed_reward_matrix_and_demos(reward_matrices, demos)
        
        # 2. Produce demos for the next generation
        # sample the games that will be used as demos for the next generation
        if self.pedagogical_sampling:
            pass
        else:
            num_possible_games = all_possible_games.shape[1]
            random_indices = np.random.choice(a=num_possible_games, size=self.num_examples_for_demos, replace=False) 
            games_for_future_demos = all_possible_games[:, random_indices, :, :]

        # move to gpu
        games_for_future_demos = games_for_future_demos.to(reward_matrices.device)

        demo_listener_context_emb = self.games_embedding(games_for_future_demos.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
        demo_scores = torch.einsum('bvce,be->bvc', (demo_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
        demo_scores = F.log_softmax(demo_scores, dim=-1)

        # generate predictions for the demo games
        demo_preds = torch.argmax(demo_scores, dim=-1)
        demo_preds_as_one_hot = F.one_hot(demo_preds)

        # concatenate the predictions for the demo set to the actual games
        demos_for_next_gen = torch.cat([games_for_future_demos, demo_preds_as_one_hot.unsqueeze(-1).float()], dim=-1)

        # 3. produce scores over outputs for the games that are used to evaluate performance
        eval_listener_context_emb = self.games_embedding(games_for_eval.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
        eval_scores = torch.einsum('bvce,be->bvc', (eval_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
        eval_scores = F.log_softmax(eval_scores, dim=-1)
        
        return demos_for_next_gen, eval_scores

    
