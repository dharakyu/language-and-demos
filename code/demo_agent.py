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
                num_examples_in_demo=10,
                num_choices_in_listener_context=3):
                
        super().__init__()
        assert embedding_dim % 2 == 0, "input dim must be divisible by 2"
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.num_examples_in_demo = num_examples_in_demo
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
        self.composite_emb_compression_mlp = nn.Linear(embedding_dim * ((object_encoding_len + 1) * num_choices_in_listener_context + num_objects),
                                                            embedding_dim)


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

        reshaped_demos = demos.view(batch_size, -1)  # (batch_size, num_examples_in_demo * num_choices_in_lis_context, object_encoding_length+1)
        demos_emb = self.examples_mlp(reshaped_demos)   # (batch_size, num_examples_in_demo * num_choices_in_lis_context, embedding_dim)
        demos_emb = demos_emb.view(batch_size, -1)   # (batch_size, num_examples_in_demo * num_choices_in_lis_context * embedding_dim)

        composite_emb = torch.cat([reward_matrix_emb, demos_emb], dim=1)    # (batch_size, embedding_dim * (object_encoding_length+1 * num_choices_in_lis_context + num_objects))
        reduced_composite_emb = self.composite_emb_compression_mlp(composite_emb)   # (batch_size, embedding_dim)

        return reduced_composite_emb

    def forward(self, 
                reward_matrices,
                demos,
                games
                ):
        """
        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+1)
        demos:
            if this is the first agent in the chain, this is None
            torch.Tensor of size (batch_size, num_examples_in_demo, num_choices_in_lis_context, object_encoding_length+1)
        games: torch.Tensor of size (batch_size, num_games_to_eval, num_choices_in_lis_context, object_encoding_len)

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
            demos = torch.zeros(size=(batch_size, self.num_examples_in_demo, self.num_choices_in_listener_context, self.object_encoding_len+1))

        reward_matrix_and_demos_emb = self.embed_reward_matrix_and_demos(reward_matrices, demos)

        # 2. produce scores over outputs
        listener_context_emb = self.games_embedding(games.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
        scores = torch.einsum('bvce,be->bvc', (listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
        scores = F.log_softmax(scores, dim=-1)

        return scores

    
