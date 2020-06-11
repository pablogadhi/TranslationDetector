import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Embeddings(nn.Module):
    def __init__(self, model_dim, vocab):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.model_dim)


class Generator(nn.Module):
    def __init__(self, model_dim, vocab):
        super(Generator, self).__init__()
        self.project = nn.Linear(model_dim, vocab)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, dropout=0.1):
        super(Transformer, self).__init__()
        self.core = nn.Transformer()
        self.src_embeddings = nn.Sequential(Embeddings(
            self.core.d_model, src_vocab), PositionalEncoding(self.core.d_model, dropout))
        self.tgt_embeddings = nn.Sequential(Embeddings(
            self.core.d_model, tgt_vocab), PositionalEncoding(self.core.d_model, dropout))
        self.generator = Generator(self.core.d_model, tgt_vocab)

    def forward(self, src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask, src_mask=None, mem_mask=None):
        return self.core(self.src_embeddings(src), self.tgt_embeddings(tgt),
                         src_mask, tgt_mask, mem_mask, src_pad_mask, tgt_pad_mask, None)

    def encode(self, src, src_mask=None, src_pad_mask=None):
        return self.core.encoder(self.src_embeddings(src), mask=src_mask,
                                 src_key_padding_mask=src_pad_mask)

    def decode(self, tgt, memory, tgt_mask, tgt_pad_mask, mem_pad_mask=None, mem_mask=None):
        return self.core.decoder(self.tgt_embeddings(tgt), memory, tgt_mask=tgt_mask, memory_mask=mem_mask,
                                 tgt_key_padding_mask=tgt_pad_mask,
                                 memory_key_padding_mask=mem_pad_mask)
