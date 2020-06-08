import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer, LayerNorm


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, model_dim)
        pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() *
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


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, src_pad, tgt_pad, n_layers=6, model_dim=512, ff_dim=2048, att_heads=8, dropout=0.1):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = TransformerEncoder(TransformerEncoderLayer(
            model_dim, att_heads, ff_dim, dropout), n_layers, LayerNorm(model_dim))
        self.decoder = TransformerDecoder(TransformerDecoderLayer(
            model_dim, att_heads, ff_dim, dropout), n_layers, LayerNorm(model_dim))
        self.src_embeddings = nn.Sequential(Embeddings(
            model_dim, src_vocab), PositionalEncoding(model_dim, dropout))
        self.tgt_embeddings = nn.Sequential(Embeddings(
            model_dim, tgt_vocab), PositionalEncoding(model_dim, dropout))
        self.src_pad = src_pad
        self.tgt_pad = tgt_pad
        # self.generator = Generator(model_dim, tgt_vocab)

        # Init paramets with xavier_uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def encode(self, src, src_mask, src_pad_mask):
        return self.encoder(self.src_embeddings(src), src_mask, src_key_padding_mask=src_pad_mask)

    def decode(self, tgt, memory, tgt_mask, mem_mask, tgt_pad_mask, mem_pad_mask):
        return self.decoder(self.tgt_embeddings(tgt), memory,
                            tgt_mask=tgt_mask, memory_mask=mem_mask,
                            tgt_key_padding_mask=tgt_pad_mask,
                            memory_key_padding_mask=mem_pad_mask)

    def forward(self, src, tgt):
        device = src.device
        # src_mask = self.generate_square_subsequent_mask(len(src)).to(device)
        tgt_mask = self.generate_square_subsequent_mask(len(tgt)).to(device)
        src_pad_mask = (src == self.src_pad).T.to(device)
        tgt_pad_mask = (tgt == self.tgt_pad).T.to(device)
        memory = self.encode(src, None, src_pad_mask)
        return self.decode(tgt, memory, tgt_mask, None, tgt_pad_mask, src_pad_mask)
