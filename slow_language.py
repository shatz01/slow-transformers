# Desc: SLM Stands for Slow Language Model!
import torch
from data import AGNewsDataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torchtext

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset


def build_vocab(tokenizer, datasets):
    for dataset in datasets:
        for _, text in dataset:
            yield tokenizer(text)

def vectorize_batch(batch):
    Y, X = list(zip(*batch))
    X = vectorizer.transform(X).todense()
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y) - 1 ## We have deducted 1 from target names to get them in range [0,1,2,3] from [1,2,3,4]


if __name__ == '__main__':
    print("🐌 running slow language classification...")

    train_dataset, test_dataset  = torchtext.datasets.AG_NEWS()
    target_classes = ["World", "Sports", "Business", "Sci/Tec"]

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(build_vocab(tokenizer, [train_dataset, test_dataset]), specials=["<UNK>"])
    vocab.set_default_index(vocab["<UNK>"])

    vectorizer = CountVectorizer(vocabulary=vocab.get_itos(), tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=256, collate_fn=vectorize_batch)
    test_loader  = DataLoader(test_dataset, batch_size=256, collate_fn=vectorize_batch)

    x, y = next(iter(train_loader))