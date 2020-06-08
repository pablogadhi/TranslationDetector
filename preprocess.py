import time
import dill
from torch.utils.data import DataLoader
from torchtext.datasets import TranslationDataset
from torchtext.data import Field

start = time.time()
SRC = Field(tokenize="spacy", tokenizer_language="en",
            init_token="<sos>", eos_token="<eos>", lower=True)

TGT = Field(tokenize="spacy", tokenizer_language="es",
            init_token="<sos>", eos_token="<eos>", lower=True)

print("Field decl: {}".format(time.time() - start))

print("Preprocessing data...")

start = time.time()
dataset = TranslationDataset(
    "data/en-es/en-es_", ("en.txt", "es.txt"), (SRC, TGT))

print("Dataset decl: {}".format(time.time() - start))

train_data, valid_data, test_data = dataset.split(
    split_ratio=[0.7, 0.15, 0.15])
SRC.build_vocab(train_data, min_freq=2)
TGT.build_vocab(train_data, min_freq=2)

src_field_file = open("data/src_field.pt", "wb")
tgt_field_file = open("data/tgt_field.pt", "wb")

dill.dump(SRC, src_field_file)
dill.dump(TGT, tgt_field_file)

print("Data generated!")
