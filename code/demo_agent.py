import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from listener import RNNEncoder
import data

import copy

class DemoAgent(nn.Module):
    def __init__(self, chain_length,
                pedagogical_sampling,
                num_possible_demos,
                object_encoding_len=8, num_objects=16,
                num_intermediate_features=64,
                embedding_dim=64, hidden_size=120,
                num_examples_for_demos=10,
                num_choices_in_listener_context=3):

        super().__init__()
        assert embedding_dim % 2 == 0, "input dim must be divisible by 2"
        self.num_intermediate_features = num_intermediate_features
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.num_examples_for_demos = num_examples_for_demos
        self.num_choices_in_listener_context = num_choices_in_listener_context
        self.object_encoding_len = object_encoding_len

        # instead of using this to represent the game that the agent is evaluated on,
        # we want to recycle the examples_mlp to encourage grounding
        self.games_embedding =  nn.Sequential(
                                    nn.Linear(object_encoding_len, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, embedding_dim)
                                )

        self.reward_matrix_mlp = nn.Sequential(
                                            nn.Linear(object_encoding_len+1, hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(hidden_size, num_intermediate_features)
                                        )
        self.reward_matrix_compression_mlp = nn.Sequential(
                                                    nn.Linear(num_objects*num_intermediate_features, hidden_size),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_size, embedding_dim)
                                                )

        self.examples_mlp = nn.Sequential(
                                        nn.Linear(object_encoding_len+1, hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, num_intermediate_features)
                                )

        examples_compression_mlp_input_dim = num_examples_for_demos * num_choices_in_listener_context * num_intermediate_features
        self.examples_compression_mlp = nn.Sequential(
                                                    nn.Linear(examples_compression_mlp_input_dim, hidden_size),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_size, embedding_dim)
                                                )

        self.composite_emb_compression_mlp = nn.Linear(2 * embedding_dim,
                                                            embedding_dim)

        self.pedagogical_sampling = pedagogical_sampling

        if self.pedagogical_sampling:
            # TODO: make this not hardcoded
            #NUM_POSSIBLE_DEMOS = 560
            self.num_possible_demos = num_possible_demos

            demo_input_dim = (num_choices_in_listener_context * object_encoding_len) + num_choices_in_listener_context
            self.teacher_demo_encoding_mlp = nn.Linear(demo_input_dim, self.hidden_size)
            #self.teacher_demo_encoding_mlp = nn.Linear(demo_input_dim, self.embedding_dim)

            self.onehot_embedding = nn.Linear(num_possible_demos, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
            
            self.init_h = nn.Linear(self.embedding_dim, self.hidden_size)
            self.outputs2demos = nn.Linear(self.embedding_dim, num_possible_demos)

            self.outputs_mlp = nn.Sequential(
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_size, self.hidden_size),
                                )


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

        reward_matrix_emb = self.reward_matrix_mlp(reward_matrices)  # (batch_size, 16, num_intermediate_features)
        reward_matrix_emb = reward_matrix_emb.view(batch_size, -1)
        reward_matrix_emb = self.reward_matrix_compression_mlp(reward_matrix_emb)

        demos = demos.to(reward_matrices.device)    # move to GPU
        demos_emb = self.examples_mlp(demos)   # (batch_size, num_demos, obj_encoding_len+1, num_intermediate_features)
        demos_emb = demos_emb.view(batch_size, -1)
        demos_emb = self.examples_compression_mlp(demos_emb)    # (batch_size, embedding_size)

        composite_emb = torch.cat([reward_matrix_emb, demos_emb], dim=1)    # (batch_size, 2*num_intermediate_features)

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
        all_possible_games:
            These are the games for which the agent makes a prediction, and the game/prediction combos are used
            as demonstrations for the next generation. This SHOULD be all the possible games (and then we can sample
            it down to a few to pass on, either through random or pedagogical selection)
        games_for_eval:
            These are the games that are used to evaluate the performance of the agent
            torch.Tensor of size (batch_size, num_games_to_eval, num_choices_in_lis_context, object_encoding_len)

        Return:
        demos_for_next_gen:
            if use_discrete_comm: torch.Tensor of size (batch_size, max_message_len, vocab_size)
            else: torch.Tensor of size (batch_size, vocab_size)
        eval_scores:
            scores over the possible intended targets in games_for_eval
            torch.Tensor of size (batch_size, num_games_to_eval, num_choices_in_lis_context)
        scores_for_comparison: 
            these are the scores over all possible demos that could be shown to the next gen
            used to compare performance to the Bayesian model
            torch.Tensor of size (batch_size, num_possible_games)

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
            # a) evaluate the current agent on all possible (560) games
            # move to gpu
            all_possible_games = all_possible_games.to(reward_matrices.device)  # (batch_size, num_examples_for_demo, num_choices_in_listener_context, obj_encoding_len)
            all_games_listener_context_emb = self.games_embedding(all_possible_games.float())
            game_scores = torch.einsum('bvce,be->bvc', (all_games_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
            game_scores = F.log_softmax(game_scores, dim=-1)

            # b) get the predictions
            game_preds = torch.argmax(game_scores, dim=-1)
            game_preds_as_one_hot = F.one_hot(game_preds, num_classes=self.num_choices_in_listener_context)

            # c) tack the games and predictions together to create a tensor of all possible demos
            batch_size = all_possible_games.shape[0]
            num_possible_games = all_possible_games.shape[1]

            # size (batch_size, 560, num_choices_in_listener_context, obj_encoding_len+1)
            all_possible_demos = torch.cat([all_possible_games, game_preds_as_one_hot.unsqueeze(-1).float()], dim=-1)

            # d) create an embedding representation of all the demos (this is states) and a representation
            # of the previously selected demo, which is just zeros for now (this is inputs)

            # size (batch_size, 560, num_choices_in_listener_context*(obj_encoding_len+1))
            all_possible_demos = all_possible_demos.view(batch_size, num_possible_games, -1)

            all_demos_emb = self.teacher_demo_encoding_mlp(all_possible_demos)

            # initialize states for the for loop
            states = self.init_h(reward_matrix_and_demos_emb)   # (batch_size, hidden_size)
            states = states.unsqueeze(0)                        # (1, batch_size, hidden_size)

            # this is the dummy input to pass in for the first iteration
            inputs = torch.zeros((1, batch_size, self.embedding_dim))
            inputs = inputs.to(reward_matrices.device)  # move to gpu

            # this is where we store all the demos that will be passed to the next generation of agents
            demos_for_next_gen = torch.zeros((batch_size, self.num_examples_for_demos, self.num_choices_in_listener_context, self.object_encoding_len+1))
            
            mask = torch.zeros(size=(batch_size, num_possible_games)).to(reward_matrices.device)
            
            for j in range(self.num_examples_for_demos):

                # inputs needs to be size (1, batch_size, embedding_dim)
                # states needs to be size (1, batch_size, hidden_size)
                # inputs is the previous demo sampled for this agent (starts with null)
                # states is the state of the agent

                # here we embed the agent's state (states) along with the last demo it produced (inputs)
                outputs, states = self.gru(inputs, states)  # (1, batch_size, hidden_size)
                outputs = outputs.squeeze(0)                # (batch_size, hidden_size)

                # this is taking a dot product between the demo representations and the agent's state
                # to compute a score over every demo
                outputs = torch.einsum('bde, be->bd', (all_demos_emb, outputs)) # (batch_size, 560)

                # this is used for the comparison to the Bayesian model
                # this has to be in log probs bc of how the pytorch KL function works!!
                scores_for_comparison = F.softmax(outputs, dim=-1)

                outputs = F.log_softmax(outputs, dim=-1)   

                # mask the indices of the demos we already selected with -np.inf
                outputs = outputs.masked_fill(mask.bool(), -np.inf)

                # use Gumbel-Softmax to convert outputs to a onehot vector
                predicted_onehot = F.gumbel_softmax(outputs, dim=-1, hard=True) # (batch_size, 560)

                # add all the selected demos to the mask
                mask[predicted_onehot.bool()] = 1

                # Zero out all demos except for the ones corresponding to
                # `predicted_onehot`
                demo_j = all_possible_demos * predicted_onehot.unsqueeze(-1)  # (batch_size, 560, 27) * (batch_size, 560, 1)
                demo_j = demo_j.view(batch_size, num_possible_games, self.num_choices_in_listener_context, -1)  # (batch_size, 560, 3, 9)

                demo_j = demo_j[predicted_onehot.bool()]  # (batch_size, 3, 9)
                demos_for_next_gen[:, j, :, :] = demo_j

                # update inputs by pushing the predicted onehot encoding through self.onehot_embedding
                predicted_onehot_unsqueezed = predicted_onehot.unsqueeze(0) # (1, batch_size, 560)
                new_inputs = self.onehot_embedding(predicted_onehot_unsqueezed)     # (1, batch_size, hidden_size)
                inputs = new_inputs

                # TODO: retrieve demo emb corresponding to what you predicted...
                # state will keep track of past demos, though we may want to
                # have multiple demos as past input

        else:
            # a) sample random games to be used as demos for the next generation
            num_possible_games = all_possible_games.shape[1]
            random_indices = np.random.choice(a=num_possible_games, size=self.num_examples_for_demos, replace=False)
            games_for_future_demos = all_possible_games[:, random_indices, :, :]

            # b) evaluate the current agent on those random games
            # move to gpu
            games_for_future_demos = games_for_future_demos.to(reward_matrices.device)

            game_listener_context_emb = self.games_embedding(games_for_future_demos.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
            game_scores = torch.einsum('bvce,be->bvc', (game_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
            game_scores = F.log_softmax(game_scores, dim=-1)

            # c) generate predictions for the demo games
            game_preds = torch.argmax(game_scores, dim=-1)
            game_preds_as_one_hot = F.one_hot(game_preds, num_classes=self.num_choices_in_listener_context)

            assert games_for_future_demos.shape[2] == game_preds_as_one_hot.unsqueeze(-1).shape[2]
            
            # d) create the demos by tacking together the game and the choice that the agent made
            demos_for_next_gen = torch.cat([games_for_future_demos, game_preds_as_one_hot.unsqueeze(-1).float()], dim=-1)   # (batch_size, num_views, num_choices, encoding_len)

            scores_for_comparison = None

        # 3. produce scores over outputs for the games that are used to evaluate performance
        #eval_listener_context_emb = self.games_embedding(games_for_eval.float()) # (batch_size, num_views, num_choices_in_listener_context, embedding_dim)
        
        # try it with recycled embedding
        # first we need to make the dimensions work by adding an extra zero at the end of each listener context
        zeros_shape = [games_for_eval.shape[0], games_for_eval.shape[1], games_for_eval.shape[2], 1]
        zeros = torch.zeros(size=zeros_shape).to(reward_matrices.device)
        games_for_eval_with_pad = torch.cat([games_for_eval.float(), zeros.float()], dim=-1)

        # then we can embed the padded games
        eval_listener_context_emb = self.examples_mlp(games_for_eval_with_pad)
        
        eval_scores = torch.einsum('bvce,be->bvc', (eval_listener_context_emb, reward_matrix_and_demos_emb))  # (batch_size, num_views, num_choices_in_listener_context)
        eval_scores = F.log_softmax(eval_scores, dim=-1)

        return demos_for_next_gen, eval_scores, scores_for_comparison


