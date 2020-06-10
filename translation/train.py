import math
import time
import torch
import torch.nn as nn
from torchtext.datasets import TranslationDataset
from torchtext.data import BucketIterator, Field
from .transformer import Transformer
from .test import eval_model
from .utils import LabelSmoothing, NoamOpt, SimpleLossCompute
from .data_loader import load_data as create_data
from .data_loader import make_iters, rebatch


def load_data(lang_dir, src_ext, tgt_ext, device, batch_size=12):
    SRC = Field(tokenize="spacy", tokenizer_language="en",
                init_token="<sos>", eos_token="<eos>", lower=True)

    TGT = Field(tokenize="spacy", tokenizer_language="es",
                init_token="<sos>", eos_token="<eos>", lower=True)

    dataset = TranslationDataset(
        lang_dir, (src_ext, tgt_ext), (SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= 100 and
        len(vars(x)['trg']) <= 100)

    train_data, valid_data, test_data = dataset.split(
        split_ratio=[0.7, 0.15, 0.15])

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size, device=device)

    SRC.build_vocab(train_data.src, min_freq=2, max_size=40000)
    TGT.build_vocab(train_data.trg, min_freq=2, max_size=40000)

    return SRC, TGT, train_iterator, valid_iterator, test_iterator


def run_epoch(model, iterator, compute_loss, log_interval, device):
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    for i, batch in enumerate(iterator):
        output = model(batch.src, batch.tgt, batch.tgt_mask,
                       batch.src_pad_mask, batch.tgt_pad_mask)
        loss = compute_loss(output, batch.tgt_y, batch.n_tokens)

        total_loss += loss.item()
        total_tokens += batch.n_tokens

        if i % log_interval == 0 and i > 0:
            cur_loss = loss / batch.n_tokens
            elapsed = time.time() - start_time
            print('Step {:5d} : '
                  '{:5.2f} ms/batch | '
                  'loss {:5.2f}'.format(i, elapsed * 1000 / log_interval, cur_loss))
            start_time = time.time()

    return total_loss / float(total_tokens)


def train_model(model, train_itr, valid_itr, src, tgt, device, epochs=100, checkpoint_f="data/model_checkpoint.pt", save_at=100):
    src_pad = src.vocab.stoi['<pad>']
    tgt_pad = tgt.vocab.stoi['<pad>']
    criterion = LabelSmoothing(
        size=len(tgt.vocab), padding_idx=tgt_pad, smoothing=0.1).to(device)
    optimizer = NoamOpt(model.core.d_model, 1, 2000,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    compute_loss = SimpleLossCompute(model.generator, criterion, optimizer)

    for epoch in range(epochs):

        model.train()
        train_loss = run_epoch(
            model, (rebatch(b, src_pad, tgt_pad, device) for b in train_itr), compute_loss, 50, device)

        model.eval()
        valid_loss = run_epoch(
            model, (rebatch(b, src_pad, tgt_pad, device) for b in valid_itr),
            SimpleLossCompute(model.generator, criterion, None), 50, device)

        print(f'Epoch: {epoch+1:02}')
        print(
            f'\tTrain Loss: {train_loss:.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f}')

        if epoch % save_at == 0:
            checkpoint = open(checkpoint_f, "wb")
            torch.save(model.state_dict(), checkpoint)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC, TGT, train, valid, test = create_data(
        "data/en-es/en-es_", "en.txt", "es.txt")
    train_itr, valid_itr, test_itr = make_iters(
        train, valid, test, device, batch_size=700)
    model = Transformer(len(SRC.vocab), len(TGT.vocab)).to(device)

    train_model(model, train_itr, valid_itr, SRC, TGT, device, 10, save_at=1)
