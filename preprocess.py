import dill
import torch
from torchtext.datasets import TranslationDataset
from torchtext.data import Field, BucketIterator

SRC = Field(tokenize="spacy", tokenizer_language="en",
            init_token="<sos>", eos_token="<eos>", lower=True)

TGT = Field(tokenize="spacy", tokenizer_language="es",
            init_token="<sos>", eos_token="<eos>", lower=True)

print("Preprocessing data...")

train_data, valid_data, test_data = TranslationDataset(
    "data/en-es/en-es_", ("en_val.txt", "es_val.txt"), (SRC, TGT)).split(split_ratio=[0.7, 0.15, 0.15])

SRC.build_vocab(train_data, min_freq=2)
TGT.build_vocab(train_data, min_freq=2)

train_file = open("data/train.pt", "wb")
test_file = open("data/test.pt", "wb")
valid_file = open("data/valid.pt", "wb")

pickle.dump(train_data, train_file)
pickle.dump(valid_data, valid_file)
pickle.dump(test_data, test_file)

print("Data generated!")
