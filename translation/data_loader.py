import math
import spacy
import torch
import dill
import numpy as np
from torchtext import data, datasets

spacy_es = spacy.load('es')
spacy_en = spacy.load('en')

global max_src_in_batch, max_tgt_in_batch


def tokenize_es(text):
    return [tok.text for tok in spacy_es.tokenizer(text)]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def load_field(path):
    file = open(path, "rb")
    return dill.load(file)


def load_data(lang_dir, src_ext, tgt_ext, src_path=None, tgt_path=None):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<pad>'
    SRC = data.Field(tokenize=tokenize_es, init_token=BOS_WORD, eos_token=EOS_WORD,
                     pad_token=BLANK_WORD) if src_path is None else load_field(src_path)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD, eos_token=EOS_WORD,
                     pad_token=BLANK_WORD) if tgt_path is None else load_field(tgt_path)

    print("Loading data...")
    dataset = datasets.TranslationDataset(
        lang_dir, (src_ext, tgt_ext), (SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= 100 and
        len(vars(x)['trg']) <= 100)
    print("Data loaded!")

    train, valid, test = dataset.split(
        split_ratio=[0.7, 0.15, 0.15])

    if src_path is None:
        SRC.build_vocab(train.src, min_freq=2, max_size=39996)
    if tgt_path is None:
        TGT.build_vocab(train.trg, min_freq=2, max_size=39996)

    return SRC, TGT, train, valid, test


def batch_size_func(new, count, sofar):
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


def make_iters(train, valid, test, device, batch_size=12):
    train_iter = TranslationIterator(train, batch_size=batch_size, device=device,
                                     repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                     batch_size_fn=batch_size_func, train=True)

    valid_iter = TranslationIterator(valid, batch_size=batch_size, device=device,
                                     repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                     batch_size_fn=batch_size_func, train=True)

    test_iter = TranslationIterator(test, batch_size=batch_size, device=device,
                                    repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                                    batch_size_fn=batch_size_func, train=True)

    return train_iter, valid_iter, test_iter


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    def __init__(self, src, tgt, src_pad=0, tgt_pad=0, device=None):
        self.src = src
        self.tgt = tgt[:-1, :]
        self.tgt_y = tgt[1:, :]
        self.src_pad_mask = (src == src_pad).T.to(device)
        self.tgt_pad_mask = (self.tgt == tgt_pad).T.to(device)
        self.tgt_mask = self.generate_square_subsequent_mask(
            len(self.tgt), len(self.tgt)).to(device)
        self.n_tokens = (self.tgt_y != tgt_pad).data.sum()

    @staticmethod
    def generate_square_subsequent_mask(l_sz, r_sz):
        mask = (torch.triu(torch.ones(l_sz, r_sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


def make_batch(batch, src_pad, tgt_pad, device):
    return Batch(batch.src, batch.trg, src_pad, tgt_pad, device)


class TranslationIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
