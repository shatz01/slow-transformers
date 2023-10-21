import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math

from transformers import DataCollatorWithPadding
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding

import urllib.request

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx):
        # idx and targets are both (B,T) tensors of ints
        logits = self.token_embedding_table(idx) # (B,T,C)
        return logits
    
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        # get logits
        logits = model(idx) # (B,T,C)
        # get only the last token/time step 
        logits = logits[:, -1, :] # (B,C)
        # take softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B,C)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
        # append sampled index to the running sequence
        idx = torch.cat([idx, idx_next], dim=-1) # (B,T+1)
    return idx

if __name__ == "__main__":
    print("🐌 running slow_bigram.py slowly!...")

    bs = 32
    vocab_size = 1000
    block_size = 64

    ### Test BigramLanguageModel
    model = BigramLanguageModel(vocab_size)
    sample = torch.randint(0, vocab_size, (bs, block_size)) # (B,T)
    out = model(sample) # (B,T,C)
    assert out.shape == (bs, block_size, vocab_size), "🚨 ERROR"; print("✅ BigramLanguageModel test passed!")
    del model

    ### download dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    filename = url.split("/")[-1]
    urllib.request.urlretrieve(url, filename)
    print(f"Downloaded {filename} from {url}")
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print("length of dataset in characters: ", len(text))
    
    # get vocabulary
    chars = sorted(list(set(text)))
    vocab_size = len(chars); print("vocabulary size: ", vocab_size)

    # create a mapping from characters to integers
    str_to_int = { ch:i for i,ch in enumerate(chars) }
    int_to_str = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [str_to_int[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([int_to_str[i] for i in l]) # decoder: take a list of integers, output a string

    # prepare datasets/dataloaders
    data = torch.tensor(encode(text)) # (L,)
    k = int(0.9*len(data)) # first 90% of data will be train
    train_data = data[:k]
    val_data = data[k:]

    def get_batch(split):
        # generate a small batch of data of inputs x and targets y
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - block_size, (bs,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        return x, y

    ### Test generate
    sample = get_batch("train")[0][0].unsqueeze(0) # (1,T)
    model = BigramLanguageModel(vocab_size)
    generated = generate(model, sample, 64)
    print("GENERATING TEXT:", decode(generated[0].tolist()))
    print("✅ generate test passed!")

    ### Training
    num_epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for iter in range(num_epochs):
        train_loss = 0.0
        train_total = 0
        for _ in tqdm(range(1000)):
            x, y = get_batch("train")
            optimizer.zero_grad()
            logits = model(x) # (B,T,C)
            logits = logits.view(-1, vocab_size) # (B*T,C)
            y = y.flatten() # (B*T,)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * bs
            train_total += bs
        
        val_loss = 0.0
        val_total = 0
        with torch.no_grad():
            for _ in range(100):
                x, y = get_batch("val")
                logits = model(x)
                logits = logits.view(-1, vocab_size)
                y = y.flatten()
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item() * bs
                val_total += bs

        print(f"iter {iter} | train loss: {train_loss/train_total:.3f} | val loss: {val_loss/val_total:.3f}")
        generated = generate(model, torch.zeros((1,1), dtype=torch.long), 128)
        print("GENERATING TEXT:", decode(generated[0].tolist()))

    