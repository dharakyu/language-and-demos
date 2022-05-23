import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from listener import RNNEncoder
import data

class Agent(nn.Module):
    def __init__(self, object_encoding_len=6, num_objects=9,
                embedding_dim=64, vocab_size=40, hidden_size=100, 
                softmax_temp=1.0, max_message_len=4,
                use_discrete_comm=True, view_listener_context=False):
        super().__init__()
        assert embedding_dim % 2 == 0, "input dim must be divisible by 2"
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.softmax_temp = softmax_temp
        self.hidden_size = hidden_size
        self.max_message_len = max_message_len

        self.use_discrete_comm = use_discrete_comm
        self.view_listener_context = view_listener_context

        extension = 2 if view_listener_context else 1

        self.games_embedding =  nn.Sequential(
                                    nn.Linear(object_encoding_len, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, embedding_dim)
                                )

        if use_discrete_comm:
            # for producing a message
            self.reward_matrix_embedding = nn.Sequential(
                                nn.Linear(object_encoding_len + extension, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, self.embedding_dim)
                            )

            self.init_h = nn.Linear(self.embedding_dim * num_objects, self.hidden_size)
            
            self.onehot_embedding = nn.Linear(self.vocab_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
            self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
            
            # for computing scores over objects
            message_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lang_model = RNNEncoder(message_embedding, hidden_size)
            self.bilinear = nn.Linear(self.lang_model.hidden_size, embedding_dim, bias=False)

        else:
            self.continuous_mlp = nn.Sequential(
                            nn.Linear(object_encoding_len + extension, self.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size, self.vocab_size)
                        )


    def get_continuous_messages(self, reward_matrices):
        """
        Get a continuous message representation of the reward matrix

        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+extension)

        Return:
        messages: torch.Tensor of size (batch_size, vocab_size)
        lang_len: None
        """
        if not self.view_listener_context:
            # trim the reward matrices such that the speaker can no longer see the listener context
            reward_matrices = reward_matrices[:, :, :-1]

        reward_embeddings = self.continuous_mlp(reward_matrices)  # (batch_size, num_objects, vocab_size)
        messages = reward_embeddings.sum(1)    # (batch_size, vocab_size)

        return messages, None


    def get_discrete_messages(self, reward_matrices, greedy):
        """
        Produce a discrete message representation conditioned on the reward matrix embedding

        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+extension)
        greedy: sample predicted tokens in a greedy fashion (i.e. not using Gumbel-Softmax)

        Return:
        lang_tensor: torch.Tensor of size (batch_size, max_message_len, vocab_size)
        lang_len: torch
        """
        batch_size = reward_matrices.shape[0]

        if not self.view_listener_context:
            # trim listener context
            reward_matrices = reward_matrices[:, :, :-1]

        emb = self.reward_matrix_embedding(reward_matrices)  # (batch_size, num_objects, embedding_dim)
        emb = emb.view(batch_size, -1)    # (batch_size, embedding_dim * num_objects)

        states = self.init_h(emb)
        states = states.unsqueeze(0)    # (1, batch_size, hidden_size)

        # This contains the batch of sampled onehot vectors
        lang = []
        # Keep track of message length and whether we've finished sampling
        #lang_length = torch.ones(batch_size, dtype=torch.int64).to(feats.device)
        lang_length = torch.ones(batch_size, dtype=torch.int64).to(reward_matrices.device)
        done_sampling = [False for _ in range(batch_size)]
        
        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(reward_matrices.device)
        inputs_onehot[:, data.SOS_IDX] = 1.0

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        # Add SOS to lang
        lang.append(inputs_onehot)

        # (B,L,D) to (L,B,D)
        inputs_onehot = inputs_onehot.transpose(0, 1)
        
        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = self.onehot_embedding(inputs_onehot)

        for i in range(self.max_message_len - 2):  # Have room for SOS, EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if all(done_sampling):
                break
            
            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)                # outputs: (B,H)
            outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

            if greedy:
                predicted = outputs.max(1)[1]
                predicted = predicted.unsqueeze(1)
            else:
                # do not sample PAD or SOS tokens
                outputs[:, data.PAD_IDX] = -np.inf
                outputs[:, data.SOS_IDX] = -np.inf

                # prevent an empty message
                if i==0:
                    outputs[:, data.EOS_IDX] = -np.inf

                predicted_onehot = F.gumbel_softmax(outputs, tau=1, hard=True)
                # Add to lang
                lang.append(predicted_onehot.unsqueeze(1))

            predicted_npy = predicted_onehot.argmax(1).cpu().numpy()

            # Update language lengths
            for i, pred in enumerate(predicted_npy):
                if not done_sampling[i]:
                    lang_length[i] += 1
                if pred == data.EOS_IDX:
                    done_sampling[i] = True

            # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
            inputs = self.onehot_embedding(predicted_onehot.unsqueeze(0))

        # Add EOS if we've never sampled it
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(reward_matrices.device)
        eos_onehot[:, 0, data.EOS_IDX] = 1.0
        lang.append(eos_onehot)
        # Cut off the rest of the sentences
        for i, _ in enumerate(predicted_npy):
            if not done_sampling[i]:
                lang_length[i] += 1
            done_sampling[i] = True

        # Cat language tensors
        lang_tensor = torch.cat(lang, 1)

        # Trim max length
        max_lang_len = max(lang_length)
        lang_tensor = lang_tensor[:, :max_lang_len, :]

        # convert lang_length from a list to a Tensor
        #lang_length = torch.Tensor(lang_length)

        return lang_tensor, lang_length


    def forward(self, 
                reward_matrices,
                input_lang,
                input_lang_len,
                games,
                greedy=False
                ):
        """
        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+extension)
        input_lang:
            if this is the first agent in the chain, this is None
            if use_discrete_comm: torch.Tensor of size (batch_size, max_message_len, vocab_size)
            else: torch.Tensor of size (batch_size, vocab_size)
        input_lang_length:
            if use_discrete_comm: torch.Tensor of size (batch_size, )
            else: None
        games: torch.Tensor of size (batch_size, num_views, num_choices, object_encoding_len)
        greedy: Boolean determining message sampling strategy

        Return:
        output_lang:
            if use_discrete_comm: torch.Tensor of size (batch_size, max_message_len, vocab_size)
            else: torch.Tensor of size (batch_size, vocab_size)
        output_lang_len:
            if use_discrete_comm: torch.Tensor of size (batch_size, )
            else: None
        scores: torch.Tensor of size (batch_size, num_views, num_choices)

        """

        # 1. produce messages
        
        if self.use_discrete_comm:
            output_lang, output_lang_len = self.get_discrete_messages(reward_matrices, greedy)
        else:
            output_lang, output_lang_len = self.get_continuous_messages(reward_matrices)

        # 2. produce scores over outputs
        #if torch.cuda.is_available():
        #    input_lang = input_lang.cuda()
        #    input_lang_len = input_lang_len.cuda()

        if input_lang is not None:
            
            games_emb = self.games_embedding(games.float()) # (batch_size, num_views, num_choices, embedding_dim)

            if self.use_discrete_comm:
                lang_emb = self.lang_model(input_lang, input_lang_len)   # (batch_size, num_views, hidden_size)
                lang_emb = self.bilinear(lang_emb) # (batch_size, num_views, embedding_dim)
            
            else:
                lang_emb = self.lang_mlp(input_lang) # (batch_size, num_views, embedding_dim)

            #breakpoint()
            scores = torch.einsum('bvce,be->bvc', (games_emb, lang_emb))    # (batch_size, num_views, num_choices)
            scores = F.log_softmax(scores, dim=-1)
        else:
            # simplification for now: don't produce scores for the first agent
            # eventually, I will want to produce an embedding that is the concatanation
            # of the input message embedding and the reward matrix embedding, and dot product that with
            # the game embedding
            scores = None
        
        return output_lang, output_lang_len, scores

    
