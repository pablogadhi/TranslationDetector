import dill
import torch
from torchtext.datasets import TranslationDataset
from translation.transformer import Transformer
from translation.translate import translate_dataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    src_field_f = open("data/SRC_Field.pt", "rb")
    tgt_field_f = open("data/TGT_Field.pt", "rb")
    SRC = dill.load(src_field_f)
    TGT = dill.load(tgt_field_f)

    data = TranslationDataset(
        "data/en-es/en-es_", ("en_test.txt", "es_test.txt"), (SRC, TGT))
    model_data = torch.load("data/en-es_checkpoint_2.pt")
    model = Transformer(len(SRC.vocab), len(TGT.vocab)).to(device)
    model.load_state_dict(model_data)
    pad = SRC.vocab.stoi['<pad>']
    sos = SRC.vocab.stoi['<s>']
    eos = SRC.vocab.stoi['</s>']

    translate_dataset(model, data, SRC, TGT, sos, eos, pad, device, 5)
