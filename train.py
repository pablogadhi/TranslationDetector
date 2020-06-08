import math
import time
import dill
import torch
import torch.nn as nn
from torchtext.datasets import TranslationDataset
from torchtext.data import BucketIterator
from transformer import Transformer
from test import eval_model


def load_data(src_field, tgt_field, lang_dir, src_ext, tgt_ext):
    src_field_file = open(src_field, "rb")
    tgt_field_file = open(tgt_field, "rb")

    SRC = dill.load(src_field_file)
    TGT = dill.load(tgt_field_file)

    dataset = TranslationDataset(
        lang_dir, (src_ext, tgt_ext), (SRC, TGT))

    return SRC, TGT, dataset


def iter_from_dataset(dataset, device, batch_size=12):
    train_data, valid_data, test_data = dataset.split(
        split_ratio=[0.7, 0.15, 0.15])

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device=device)

    return train_iterator, valid_iterator, test_iterator


def run_epoch(model, iterator, optimizer, criterion, log_interval):
    model.train()

    epoch_loss = 0
    start_time = time.time()
    for i, batch in enumerate(iterator):
        src = batch.src
        tgt = batch.trg

        optimizer.zero_grad()

        output = model(src, tgt)
        output = output[1:].view(-1, output.shape[-1])
        tgt = tgt[1:].view(-1)

        loss = criterion(output, tgt)
        loss.backward()

        # TODO Test with a clip of 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        epoch_loss += loss.item()

        if i % log_interval == 0 and i > 0:
            cur_loss = epoch_loss / log_interval
            elapsed = time.time() - start_time
            print('{:5d}/{:5d} batches | '
                  'ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(i, len(
                      iterator), elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            epoch_loss = 0
            start_time = time.time()

    return epoch_loss / len(iterator)


def train_model(model, train_itr, valid_itr, epochs=100, checkpoint_f="data/model_checkpoint.pt", save_at=100):
    # Ignore padding when computing the loss
    PAD_IDX = TGT.vocab.stoi['<pad>']

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for epoch in range(epochs):

        train_loss = run_epoch(
            model, train_itr, optimizer, criterion, 50)
        valid_loss = eval_model(model, valid_itr, criterion)

        print(f'Epoch: {epoch+1:02}')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        if epoch % save_at == 0 and epoch > 0:
            checkpoint = open(checkpoint_f, "wb")
            torch.save(model.state_dict(), checkpoint)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC, TGT, dataset = load_data(
        "data/src_field.pt", "data/tgt_field.pt", "data/en-es/en-es_", "en_val.txt", "es_val.txt")
    train_itr, valid_itr, test_itr = iter_from_dataset(dataset, device=device)

    # Test with stoi
    model = Transformer(len(SRC.vocab), len(
        TGT.vocab), SRC.vocab.stoi['<pad>'], TGT.vocab.stoi['<pad>']).to(device)

    train_model(model, train_itr, valid_itr, 10)
