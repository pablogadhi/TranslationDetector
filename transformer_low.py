"""
Implementation of the Transformer architecture, heavily based on Alexander Rush's article
The Annotated Transformer found at http://nlp.seas.harvard.edu/2018/04/03/attention.html.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def clones(module, amount):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(amount)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attention = dropout(p_attention)
    return torch.matmul(p_attention, value), p_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, att_heads, model_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_k = model_dim // att_heads
        self.att_heads = att_heads
        self.linears = clones(nn.Linear(model_dim, model_dim), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.usqueeze(1)
        n_batches = query.size(0)

        query, key, value = [l(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x, self.attention = attention(
            query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(
            n_batches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PosWiseFeedForward(nn.Module):
    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(PosWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, ff_dim)
        self.w_2 = nn.Linear(ff_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class NormLayer(nn.Module):
    def __init__(self, features, epsilon=1e-6):
        super(NormLayer, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.epsilon = epsilon

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.epsilon) + self.b_2


class SubLayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SubLayerConnection, self).__init__()
        self.norm = NormLayer(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, ff_dim, att_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(att_heads, size)
        self.feed_forward = PosWiseFeedForward(size, ff_dim, dropout)
        self.sublayer = clones(SubLayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, size, ff_dim, att_heads, dropout, N):
        super(Encoder, self).__init__()
        self.layers = clones(EncoderLayer(size, ff_dim, att_heads, dropout), N)
        self.norm = NormLayer(size)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, ff_dim, att_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(att_heads, size, dropout)
        self.src_attention = MultiHeadAttention(att_heads, size, dropout)
        self.feed_forward = PosWiseFeedForward(size, ff_dim, dropout)
        self.sublayer = clones(SubLayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        mem = memory
        x = self.sublayer[0](
            x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayer[1](
            x, lambda x: self.src_attention(x, mem, mem, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, size, ff_dim, att_heads, dropout, N):
        super(Decoder, self).__init__()
        self.layers = clones(DecoderLayer(size, ff_dim, att_heads, dropout), N)
        self.norm = NormLayer(size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encodings = torch.zeros(max_length, model_dim)
        pos = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) *
                             (math.log(10000.0) / model_dim) * (-1))
        pos_encodings[:, 0::2] = torch.sin(pos * div_term)
        pos_encodings[:, 1::2] = torch.cos(pos * div_term)
        pos_encodings.unsqueeze(0)
        self.register_buffer('pos_encodings', pos_encodings)

    def forward(self, x):
        x = x + \
            Variable(self.pos_encodings[:, :x.size(1)], requires_grad=False)
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
    def __init__(self, src_vocab, tgt_vocab, n_layers=6, model_dim=512, ff_dim=2048, att_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(model_dim, ff_dim, att_heads, dropout, n_layers)
        self.decoder = Decoder(model_dim, ff_dim, att_heads, dropout, n_layers)
        self.src_embeddings = nn.Sequential(Embeddings(
            model_dim, src_vocab), PositionalEncoding(model_dim, dropout))
        self.tgt_embeddings = nn.Sequential(Embeddings(
            model_dim, tgt_vocab), PositionalEncoding(model_dim, dropout))
        self.generator = Generator(model_dim, tgt_vocab)

        # Init paramets with xavier_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embeddings(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embeddings(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
