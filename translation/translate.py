import torch
from .data_loader import Batch


def encode_sentence(sentence, SRC, sos, eos, device):
    encoded = [SRC.vocab.stoi[x] for x in sentence]
    encoded.insert(0, sos)
    encoded.append(eos)
    encoded = torch.tensor(
        encoded, dtype=torch.int64).unsqueeze(0).T.to(device)
    return encoded


def decode_sentence(sentence, TGT):
    return [TGT.vocab.itos[x] for x in sentence[0]], sentence[1]


def translate_sentence(model, src, src_pad, tgt_pad, sos_idx, eos_idx, device, beams=5, max_iters=50):
    model.eval()

    src_pad_mask = (src == src_pad).T.to(device)
    src_mask = Batch.generate_square_subsequent_mask(
        len(src), len(src)).type_as(src.data).to(device)
    memory = model.encode(src, None, src_pad_mask=src_pad_mask)
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


def translate_dataset(model, dataset, SRC, TGT, sos, eos, pad, device, n_words=999999999999):
    for i, sentence in enumerate(dataset.examples):
        encoded = encode_sentence(sentence.src, SRC, sos, eos, device)
        translation_candidates = translate_sentence(
            model, encoded, pad, pad, sos, eos, device, max_iters=len(sentence.src))
        original_sentence = '<s>' + \
            ', '.join([word for word in sentence.src]) + '</s>'
        print("Src: ", original_sentence)
        print("Candidates:")
        for candidate in translation_candidates:
            decoded, score = decode_sentence(candidate, TGT)
            decoded_str = ', '.join(decoded)
            print("Translation: {} - score: {}".format(decoded_str, score))
        print()

        if i + 1 >= n_words:
            break
