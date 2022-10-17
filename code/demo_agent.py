import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from listener import RNNEncoder
import data

class DemoAgent(nn.Module):
    def __init__(self, chain_length,
                pedagogical_sampling,
                object_encoding_len=8, num_objects=16,
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

        self.pedagogical_sampling = pedagogical_sampling

        if self.pedagogical_sampling:
            input_dim = (num_choices_in_listener_context * object_encoding_len) + num_choices_in_listener_context
            self.teacher_demo_encoding_mlp = nn.Linear(input_dim, embedding_dim)


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
        
        # composite representation of the reward matrix and the demos that the current agent sees
        reward_matrix_and_demos_emb = self.embed_reward_matrix_and_demos(reward_matrices, demos)
        
        # 2. Produce demos for the next generation
        if self.pedagogical_sampling:
            # in the pedagogical sample condition, our approach is to:
            # a) evaluate the current agent on all possible (560) games
            # b) get the predictions
            # c) tack the games and predictions together to create a tensor of all possible demos
            # d) embed all possible demos with self.teacher_demo_encoder_mlp
            # e) compute a dot product between that embedding, and reward_matrix_and_demos_emb
            # to get a probability score over all the demos
            # f) Gumbel-Softmax the probability score vector

            # part a)
            # move to gpu
            all_possible_games = all_possible_games.to(reward_matrices.device)  # (batch_size, num_examples_for_demo, num_choices_in_listener_context, obj_encoding_len)
            all_games_listener_context_emb = self.games_embedding(all_possible_games.float())
            game_scores = torch.einsum('bvce,be->bvc', (all_games_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
            game_scores = F.log_softmax(game_scores, dim=-1)

            # part b)
            game_preds = torch.argmax(game_scores, dim=-1)
            game_preds_as_one_hot = F.one_hot(game_preds, num_classes=self.num_choices_in_listener_context)

            # part c)
            batch_size = all_possible_games.shape[0]
            num_possible_games = all_possible_games.shape[1]

            # size (batch_size, 560, num_choices_in_listener_context, obj_encoding_len+1)
            all_possible_demos = torch.cat([all_possible_games, game_preds_as_one_hot.unsqueeze(-1).float()], dim=-1)

            # size (batch_size, 560, num_choices_in_listener_context*(obj_encoding_len+1))
            all_possible_demos = all_possible_demos.view(batch_size, num_possible_games, -1)

            # part d)
            # (batch_size, 560, num_choices_in_listener_context*(obj_encoding_len+1)) -> (batch_size, 560, embedding_dim)
            all_demos_emb = self.teacher_demo_encoding_mlp(all_possible_demos)

            # part e)
            # (batch_size, 560, embedding_dim) x (batch_size, embedding_dim) -> (batch_size, 560)
            demo_prob_scores = torch.einsum('bde, be->bd', (all_demos_emb, reward_matrix_and_demos_emb))
            demo_prob_scores = F.log_softmax(demo_prob_scores, dim=-1)

            # part f)
            choice = F.gumbel_softmax(demo_prob_scores, dim=-1, hard=True)  # (batch_size, 560)

            demos_for_next_gen = all_possible_demos * choice.unsqueeze(-1)  # (batch_size, 560, 27) * (batch_size, 560, 1)
            demos_for_next_gen = demos_for_next_gen.view(batch_size, num_possible_games, self.num_choices_in_listener_context, -1)  # (batch_size, 560, 3, 9)

            demos_for_next_gen = demos_for_next_gen[choice.bool()]  # (batch_size, 3, 9)
            demos_for_next_gen = demos_for_next_gen.unsqueeze(1)    # (batch_size, 1, 3, 9)

        else:
            # in the random sample condition, our approach is to:
            # a) sample random games to be used as demos for the next generation
            # b) evaluate the current agent on those random games
            # c) get the predictions
            # d) create the demos by tacking together the game and the choice that the agent made

            # part a)
            num_possible_games = all_possible_games.shape[1]
            random_indices = np.random.choice(a=num_possible_games, size=self.num_examples_for_demos, replace=False) 
            games_for_future_demos = all_possible_games[:, random_indices, :, :]

            # part b)
            # move to gpu
            games_for_future_demos = games_for_future_demos.to(reward_matrices.device)

            game_listener_context_emb = self.games_embedding(games_for_future_demos.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
            game_scores = torch.einsum('bvce,be->bvc', (game_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
            game_scores = F.log_softmax(game_scores, dim=-1)

            # part c)
            # generate predictions for the demo games
            game_preds = torch.argmax(game_scores, dim=-1)
            game_preds_as_one_hot = F.one_hot(game_preds, num_classes=self.num_choices_in_listener_context)
            
            # part d)
            if games_for_future_demos.shape[2] != game_preds_as_one_hot.unsqueeze(-1).shape[2]:
                breakpoint()
            # concatenate the predictions for the demo set to the actual games, is size (32, n, 3, 9)
            demos_for_next_gen = torch.cat([games_for_future_demos, game_preds_as_one_hot.unsqueeze(-1).float()], dim=-1)   # (batch_size, num_views, num_choices, encoding_len)

        # 3. produce scores over outputs for the games that are used to evaluate performance
        eval_listener_context_emb = self.games_embedding(games_for_eval.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
        eval_scores = torch.einsum('bvce,be->bvc', (eval_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
        eval_scores = F.log_softmax(eval_scores, dim=-1)
        
        return demos_for_next_gen, eval_scores

    
