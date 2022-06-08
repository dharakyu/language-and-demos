import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """
    def __init__(self, embedding_module, hidden_size=100):
        super(RNNEncoder, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, seq, length):
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden


class Listener(nn.Module):
    def __init__(self, object_encoding_len=6, 
                embedding_dim=64, vocab_size=40, hidden_size=100,
                use_discrete_comm=True):
        """
        Note: embedding_dim is used as the embedding size for both the message embedding
        and the game embedding
        """
        super(Listener, self).__init__()
        self.use_discrete_comm = use_discrete_comm

        self.games_embedding =  nn.Sequential(
                                    nn.Linear(object_encoding_len, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, embedding_dim)
                                )

        if use_discrete_comm:
            message_embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lang_model = RNNEncoder(message_embedding, hidden_size)
            # self.lang_model will output something of size (batch_size, hidden_size) so 
            # we need to project it to (batch_size, embedding_dim)
            self.bilinear = nn.Linear(self.lang_model.hidden_size, embedding_dim, bias=False)

        else:
            self.lang_mlp = nn.Sequential(
                            nn.Linear(vocab_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, embedding_dim)
                        )

    def forward(self, games, lang, lang_length):
        """
        Produce probability scores over objects in the listener context

        Arguments:
        games: torch.Tensor of size (batch_size, num_choices, object_encoding_len)
        lang:
            if use_discrete_comm: torch.Tensor of size (batch_size, max_message_len, vocab_size)
            else: torch.Tensor of size (batch_size, vocab_size)
        lang_length:
            if use_discrete_comm: torch.Tensor of size (batch_size, )
            else: None

        Return:
        scores: torch.Tensor of size (batch_size, num_choices)
        """

        # Embed games
        games_emb = self.games_embedding(games.float()) # (batch_size, num_choices, embedding_dim)

        if self.use_discrete_comm:
            lang_emb = self.lang_model(lang, lang_length)   # (batch_size, hidden_size)

            # Bilinear term: lang embedding space to game embedding space
            lang_emb = self.bilinear(lang_emb) # (batch_size, embedding_dim)
        
        else:
            lang_emb = self.lang_mlp(lang) # (batch_size, embedding_dim)

        scores = torch.einsum('ijh,ih->ij', (games_emb, lang_emb))
        return F.log_softmax(scores, dim=1)
