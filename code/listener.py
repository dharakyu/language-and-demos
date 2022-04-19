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
        #self.embedding_dim = embedding_module.embedding_dim
        self.embedding_dim = 64
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size)

    def forward(self, seq, length):
        breakpoint()
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
    def __init__(self, 
                num_choices=3, embedding_dim=64, vocab_size=20, hidden_size=100, softmax_temp=1.0, max_message_len=4):
        super(Listener, self).__init__()
        #self.embedding = nn.Linear(vocab_size, hidden_size)
        #self.lang_model = RNNEncoder(self.embedding)
        #self.vocab_size = embedding_module.num_embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # encode the messages
        self.message_encoder = nn.GRU(vocab_size, hidden_size)

        # squish message encoding to size (batch_size, 3)
        self.emb2logits = nn.Linear(hidden_size, num_choices)

        #self.bilinear = nn.Linear(self.lang_model.hidden_size, self.feat_size, bias=False)

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest)  # (batch_size*n_obj, rest)
        feats_emb_flat = self.feat_model(feats_flat)    # (batch_size*n_obj, emb_len)

        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1) # (batch_size, n_obj, emb_len)

        return feats_emb

    def forward(self, lang, lang_length):
        # Embed features
        #feats_emb = self.embed_features(feats) # don't embed images in simple version of game
        # Embed language
        #lang_emb = self.lang_model(lang, lang_length)
        # reorder from (batch_size, seq_len, vocab_size) to (seq_len, batch_size, vocab_size)
        lang = lang.transpose(0, 1)

        _, hidden = self.message_encoder(lang)
        # squeeze hidden which is size (1, batch_size, hidden_size)
        hidden = hidden.squeeze(0)

        logits = self.emb2logits(hidden)
        logits = torch.log_softmax(logits, dim=-1)

        return logits
