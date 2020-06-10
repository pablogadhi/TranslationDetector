import torch
from translation.data_loader import load_data, make_iters
from translation.transformer import Transformer
from translation.train import train_model

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SRC, TGT, train, valid, test = load_data(
        "data/en-es/en-es_", "en_test.txt", "es_test.txt", "data/SRC_Field.pt", "data/TGT_Field.pt")
    print("Vocab: ", len(SRC.vocab.stoi))
    train_itr, valid_itr, test_itr = make_iters(
        train, valid, test, device, batch_size=500)
    model = Transformer(len(SRC.vocab), len(TGT.vocab)).to(device)

    train_model(model, train_itr, valid_itr, SRC, TGT, device, 10, save_at=1)
