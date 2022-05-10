import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

import data

"""
The speaker agent embeds the game state and the rewards matrix, and produces a message conditioned on that embedding
"""

class Speaker(nn.Module):
    def __init__(self, object_encoding_len=6,
                embedding_dim=64, vocab_size=20, hidden_size=100, 
                softmax_temp=1.0, max_message_len=4):
        super().__init__()
        assert embedding_dim % 2 == 0, "input dim must be divisible by 2"
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.softmax_temp = softmax_temp
        self.hidden_size = hidden_size
        self.max_message_len = max_message_len

        #self.games_embedding = nn.Linear(num_choices * object_encoding_len, self.embedding_dim // 2)
        #self.reward_matrix_embedding = nn.Linear(num_objects * (object_encoding_len + 1), self.embedding_dim // 2)

        self.init_h = nn.Linear(self.embedding_dim, self.hidden_size)

        self.onehot_embedding = nn.Linear(self.vocab_size, self.embedding_dim)

        self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)

        #self.debug_mlp = nn.Linear(self.embedding_dim, self.vocab_size)
        self.debug_mlp = nn.Sequential(
                            nn.Linear(object_encoding_len+1, self.hidden_size),
                            nn.ReLU(),
                            nn.Linear(self.hidden_size, self.vocab_size)
                        )

        self.reward_matrix_embedding = nn.Sequential(
                                nn.Linear(object_encoding_len+1, self.hidden_size),
                                nn.ReLU(),
                                nn.Linear(self.hidden_size, self.embedding_dim)
                            )
        
    def forward(self, reward_matrices, greedy=False):
        """
        Arguments:
        reward_matrices: torch.Tensor of size (batch_size, num_objects, object_encoding_length+2)

        Return:
        lang_tensor: torch.Tensor of size (batch_size, max_message_len)
        lang_len: torch.Tensor of size (batch_size,)
        """

        """
        # trim the reward matrices such that the speaker can no longer see the listener context
        # also need to change self.debug_mlp size
        reward_matrices = reward_matrices[:, :, :-1]
        
        reward_embeddings = self.debug_mlp(reward_matrices)  # (batch_size, num_objects, vocab_size)
        messages = reward_embeddings.sum(1)    # (batch_size, vocab_size)

        return messages, None
        """
        batch_size = reward_matrices.shape[0]

        truncated_reward_matrices = []
        for i in range(batch_size):
            game = reward_matrices[i]
            keep = game[:, -1] == 1
            truncated = game[keep, :-1]
            truncated_reward_matrices.append(truncated)
        #breakpoint()
        truncated_reward_matrices = torch.stack(truncated_reward_matrices)
        

        emb = self.reward_matrix_embedding(truncated_reward_matrices)  # (batch_size, num_objects, embedding_dim)
        emb = emb.sum(1)    # (batch_size, embedding_dim)

        states = self.init_h(emb)
        states = states.unsqueeze(0)    # (1, batch_size, hidden_size)

        # This contains the batch of sampled onehot vectors
        lang = []
        # Keep track of message length and whether we've finished sampling
        #lang_length = torch.ones(batch_size, dtype=torch.int64).to(feats.device)
        lang_length = [1 for _ in range(batch_size)]
        done_sampling = [False for _ in range(batch_size)]
        
        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size)    # may need to move to the GPU
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
        eos_onehot = torch.zeros(batch_size, 1, self.vocab_size)    # may need to move to device
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
        lang_length = torch.Tensor(lang_length)

        return lang_tensor, lang_length