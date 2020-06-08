import torch


def eval_model(model, iterator, criterion):
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            tgt = batch.trg

            output = model(src, tgt)
            output = output[1:].view(-1, output.shape[-1])

            tgt = tgt[1:].view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
