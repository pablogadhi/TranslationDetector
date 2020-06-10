import dill
from torchtext import datasets, data
from translation.data_loader import tokenize_en, tokenize_es

if __name__ == "__main__":
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<pad>'
    SRC = data.Field(tokenize=tokenize_es, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    dataset = datasets.TranslationDataset(
        'data/en-es/en-es_', ('en.txt', 'es.txt'), (SRC, TGT),
        filter_pred=lambda x: len(vars(x)['src']) <= 100 and
        len(vars(x)['trg']) <= 100)

    SRC.build_vocab(dataset.src, min_freq=2, max_size=39996)
    TGT.build_vocab(dataset.trg, min_freq=2, max_size=39996)

    src_file = open("data/SRC_Field.pt", "wb")
    tgt_file = open("data/TGT_Field.pt", "wb")
    dill.dump(SRC, src_file)
    dill.dump(TGT, tgt_file)

    print("Field files generated!")
