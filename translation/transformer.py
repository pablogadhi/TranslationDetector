import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, model_dim)
        pos = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) *
                             (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
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

    def forward(self, src, tgt, tgt_mask, src_pad_mask, tgt_pad_mask):
        return self.core(self.src_embeddings(src), self.tgt_embeddings(tgt),
                         None, tgt_mask, None, src_pad_mask, tgt_pad_mask, None)
