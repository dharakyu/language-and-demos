import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from listener import RNNEncoder
import data

class Agent(nn.Module):
    def __init__(self, chain_length,
                object_encoding_len=6, num_objects=9,
                embedding_dim=64, vocab_size=40, hidden_size=100, 
                softmax_temp=1.0, max_message_len=4,
                use_discrete_comm=False,
                ingest_multiple_messages=False):
                
        super().__init__()
        assert embedding_dim % 2 == 0, "input dim must be divisible by 2"
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.softmax_temp = softmax_temp
        self.hidden_size = hidden_size
        self.max_message_len = max_message_len

        self.use_discrete_comm = use_discrete_comm

        self.ingest_multiple_messages = ingest_multiple_messages

        self.num_messages_received = chain_length if ingest_multiple_messages else 1

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
        self.input_message_embedding = nn.Linear(max_message_len * vocab_size * self.num_messages_received, 
                                                    max_message_len * vocab_size)

        self.reduce_reward_matrix_and_input_message = nn.Linear(embedding_dim * num_objects + max_message_len * vocab_size,
                                                                embedding_dim)

        if use_discrete_comm:
            # for producing a message
            self.init_h = nn.Linear((embedding_dim * num_objects) + (max_message_len * vocab_size), hidden_size)
                
            self.onehot_embedding = nn.Linear(self.vocab_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
            self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
                
            # for computing scores over objects
            message_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lang_model = RNNEncoder(message_embedding, hidden_size)
            self.bilinear = nn.Linear(self.lang_model.hidden_size, embedding_dim, bias=False)

        else:
            # produce a continuous message, conditioned on the reward matrix/input message
            self.cont_comm_message_mlp = nn.Sequential(
                                ##nn.Linear(object_encoding_len + extension, self.hidden_size),
                                nn.Linear(((embedding_dim * num_objects) + (max_message_len * vocab_size)) // max_message_len, hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, self.vocab_size)
                            )

            # embed the input message so we can take a dot product of that with the listener context embedding
            self.cont_comm_lang_model_mlp = nn.Sequential(
                                nn.Linear(max_message_len * vocab_size, hidden_size),
                                nn.ReLU(),
                                nn.Linear(hidden_size, embedding_dim)
                            )

    def embed_reward_matrix_and_input_message(self, reward_matrices, input_message):
        """
        Produce a composite embedding of the reward matrix and the input message

        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+extension)
        input_message: torch.Tensor of size (batch_size, max_message_len, vocab_size)
            if args.ingest_multiple_messages is True, then input_message is size (batch_size, max_message_len, vocab_size)

        Return:
        emb: torch.Tensor of size (batch_size, embedding_dim * num_objects + max_message_len * vocab_size)
        """
        batch_size = reward_matrices.shape[0]

        reward_matrix_emb = self.reward_matrix_embedding(reward_matrices)  # (batch_size, num_objects, embedding_dim)
        reward_matrix_emb = reward_matrix_emb.view(batch_size, -1)    # (batch_size, embedding_dim * num_objects)

        input_message_reshaped = input_message.view(batch_size, -1) # (batch_size, max_message_len * vocab_size * num_messages)
        assert(input_message_reshaped.shape == (batch_size, self.max_message_len * self.vocab_size * self.num_messages_received))
        input_message_reshaped = input_message_reshaped.to(reward_matrices.device)
        input_message_emb = self.input_message_embedding(input_message_reshaped) # (batch_size, max_message_len * vocab_size)

        emb = torch.cat([reward_matrix_emb, input_message_emb], dim=1)   # (batch_size, embedding_dim * num_objects + max_message_len * vocab_size)
        return emb

    def get_continuous_messages(self, emb):
        """
        Get a continuous message representation of the reward matrix

        Arguments:
        emb: torch.Tensor of size (batch_size, embedding_dim * num_objects + max_message_len * vocab_size)

        Return:
        messages: torch.Tensor of size (batch_size, max_message_len, vocab_size)
        lang_len: None
        """

        batch_size = emb.shape[0]
        reshaped_emb = emb.view(batch_size, self.max_message_len, -1)   # (batch_size, max_message_len, -1)
        messages = self.cont_comm_message_mlp(reshaped_emb)  # (batch_size, vocab_size)

        return messages, None


    def get_discrete_messages(self, emb, greedy):
        """
        Produce a discrete message representation conditioned on a concatenated embedding 
        of the reward matrix and the input message

        Arguments:
        emb: torch.Tensor of size (batch_size, embedding_dim * num_objects + max_message_len * vocab_size)
        greedy: sample predicted tokens in a greedy fashion (i.e. not using Gumbel-Softmax)

        Return:
        lang_tensor: torch.Tensor of size (batch_size, max_message_len, vocab_size)
        lang_len: torch.Tensor of size (batch_size,)
        """

        batch_size = emb.shape[0]

        #states = self.init_h(game_emb)
        states = self.init_h(emb)
        states = states.unsqueeze(0)    # (1, batch_size, hidden_size)

        # This contains the batch of sampled onehot vectors
        lang = []

        # Keep track of message length and whether we've finished sampling
        lang_length = torch.ones(batch_size, dtype=torch.int64).to(emb.device)
        done_sampling = [False for _ in range(batch_size)]
        
        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size).to(emb.device)
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
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size).to(emb.device)
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
        batch_size = reward_matrices.shape[0]

        # learning from language
        if input_lang is None:  # create a null message
            if self.num_messages_received > 1:
                input_lang = torch.zeros(size=(batch_size, self.max_message_len, self.vocab_size, self.num_messages_received)).to(reward_matrices.device)
            else:
                input_lang = torch.zeros(size=(batch_size, self.max_message_len, self.vocab_size)).to(reward_matrices.device)
            #input_lang_len = torch.full(size=(batch_size,), fill_value=self.max_message_len).to(reward_matrices.device)
        
        # 1. produce messages
        reward_matrix_and_input_message_emb = self.embed_reward_matrix_and_input_message(reward_matrices, input_lang)
        if self.use_discrete_comm:
            output_lang, output_lang_len = self.get_discrete_messages(reward_matrix_and_input_message_emb, greedy)
        else:
            output_lang, output_lang_len = self.get_continuous_messages(reward_matrix_and_input_message_emb)

        # 2. produce scores over outputs
        listener_context_emb = self.games_embedding(games.float()) # (batch_size, num_views, num_choices, embedding_dim)

        reduced_reward_matrix_and_input_message_emb = self.reduce_reward_matrix_and_input_message(reward_matrix_and_input_message_emb)
        scores = torch.einsum('bvce,be->bvc', (listener_context_emb, reduced_reward_matrix_and_input_message_emb))  # (batch_size, num_views, num_choices)
        scores = F.log_softmax(scores, dim=-1)

        return output_lang, output_lang_len, scores

    
