import time
import torch
from .utils import LabelSmoothing, DynamicOptimizer, LossCompute
from .data_loader import make_batch


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
    optimizer = DynamicOptimizer(model.core.d_model, 1, 2000,
                                 torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    compute_loss = LossCompute(model.generator, criterion, optimizer)

    for epoch in range(epochs):

        model.train()
        train_loss = run_epoch(
            model, (make_batch(b, src_pad, tgt_pad, device) for b in train_itr), compute_loss, 50, device)

        model.eval()
        valid_loss = run_epoch(
            model, (make_batch(b, src_pad, tgt_pad, device)
                    for b in valid_itr),
            LossCompute(model.generator, criterion, None), 50, device)

        print(f'Epoch: {epoch+1:02}')
        print(
            f'\tTrain Loss: {train_loss:.3f}')
        print(
            f'\t Val. Loss: {valid_loss:.3f}')

        if epoch % save_at == 0:
            torch.save(model.state_dict(), checkpoint_f)
            print("Model saved!")
