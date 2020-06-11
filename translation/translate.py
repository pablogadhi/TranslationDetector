import torch
from .data_loader import Batch


def translate_sentence(model, src, src_pad, tgt_pad, sos_idx, eos_idx, device, beams=10, max_iters=10):
    model.eval()

    src_pad_mask = (src == src_pad).T.to(device)
    src_mask = Batch.generate_square_subsequent_mask(
        len(src), len(src)).type_as(src.data).to(device)
    memory = model.encode(src, src_mask, src_pad_mask=src_pad_mask)
    pred = [(torch.ones(1, 1).fill_(sos_idx).type_as(src.data).to(device), 0)]
    candidates = []

    iter_num = 0
    while(iter_num < max_iters):
        for _ in range(len(pred)):
            y_pred, score = pred.pop()
            y_pad_mask = (y_pred == tgt_pad).T.to(device)
            y_mask = Batch.generate_square_subsequent_mask(
                len(y_pred), len(y_pred)).type_as(src.data).to(device)
            output = model.decode(y_pred, memory, y_mask, y_pad_mask)
            prob = model.generator(output[:, -1])

            max_probs, next_cand = torch.topk(prob, beams, dim=1)

            for i in range(beams):
                new_pred = torch.cat([y_pred, torch.ones(1, 1).type_as(
                    src.data).fill_(next_cand[0, i])], dim=0)
                candidates.append((new_pred, score + max_probs[0, i].item()))

        all_scores = torch.FloatTensor([x[1] for x in candidates])
        _, chosen = torch.topk(all_scores, beams)

        for i in chosen:
            pred.append(candidates[i])
        candidates = []

        iter_num += 1

    return pred
