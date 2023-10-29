import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import math

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from transformers import BertTokenizer, DataCollatorWithPadding

# device = 'cpu'
device = 'cuda:0'
# device = torch.device("mps")

class MSA(nn.Module):

  def __init__(self, input_dim, embed_dim, num_heads):
    '''
    Multi Headed Self Attention
    input_dim: Dimension of input token embeddings
    embed_dim: Dimension of internal key, query, and value embeddings
    num_heads: Number of self-attention heads
    '''

    super().__init__()

    self.input_dim = input_dim
    self.embed_dim = embed_dim
    self.num_heads = num_heads

    self.K_embed = nn.Linear(input_dim, embed_dim, bias=False)
    self.Q_embed = nn.Linear(input_dim, embed_dim, bias=False)
    self.V_embed = nn.Linear(input_dim, embed_dim, bias=False)
    self.out_embed = nn.Linear(embed_dim, embed_dim, bias=False)

  def forward(self, x):
    '''
    x: input of shape (batch_size, max_length, input_dim)
    return: output of shape (batch_size, max_length, embed_dim)
    '''

    batch_size, max_num_tokens, given_input_dim = x.shape
    assert given_input_dim == self.input_dim
    assert self.embed_dim % self.num_heads == 0

    # Calculate K, Q, V (remember broacasting occurs over the first dim)
    K = self.K_embed(x) # (batch_size, max_num_tokens, embed_dim)
    Q = self.Q_embed(x)
    V = self.V_embed(x)

    # split embedding dim into heads
    indiv_dim = self.embed_dim // self.num_heads
    K = K.reshape(batch_size, max_num_tokens, self.num_heads, indiv_dim)
    Q = Q.reshape(batch_size, max_num_tokens, self.num_heads, indiv_dim)
    V = V.reshape(batch_size, max_num_tokens, self.num_heads, indiv_dim)

    # swap middle dims so it goes (batch_size, max_num_tokens, num_heads, indiv_dim) -> (batch_size, num_heads, max_num_tokens, indiv_div)
    K = K.permute(0, 2, 1, 3)
    Q = Q.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)

    # transpose and Batch Matrix Multiply (bmm) (broadcasting over the first 2 dims)
    K_T = K.permute(0, 1, 3, 2) # batch_size, num_heads, indiv_dim, max_num_tokens
    QK = Q@K_T # (batch_size, num_heads, max_num_tokens, indiv_dim) @ (batch_size, num_heads, indiv_dim, max_num_tokens) -> (batch_size, num_heads, max_num_tokens, max_num_tokens)

    # Calculate weights by dividing everything by the square root of d (self.embed_dim)
    weights = QK / self.embed_dim # still a matrix of (batch_size, num_heads, max_num_tokens, max_num_tokens) 

    # Take softmax over the last dim
    weights = torch.nn.functional.softmax(weights, dim=-1)

    # Get weighted average (use bmm)
    w_V = weights@V # (batch_size, num_heads, max_num_tokens, max_num_tokens) @ (batch_size, num_heads, max_num_tokens, indiv_div) -> (batch_size, num_heads, max_num_tokens, indiv_dim)

    # Rejoin Heads! (permute middle two dims back and re-combine last two dims)
    w_V = w_V.permute(0, 2, 1, 3) # (batch_size, max_num_tokens, num_heads, indiv_dim)
    w_V = w_V.reshape(batch_size, max_num_tokens, self.embed_dim)

    out = self.out_embed(w_V)
    return out

class TransformerLayer(nn.Module):

  def __init__(self, num_heads, input_dim, embed_dim, mlp_hidden_dim, dropout=0.1):
    assert input_dim == embed_dim
    super().__init__()
    self.layernorm1 = nn.LayerNorm(input_dim)
    self.msa = MSA(input_dim, embed_dim, num_heads)
    self.w_o_dropout = nn.Dropout(dropout)
    self.layernorm2 = nn.LayerNorm(input_dim)
    self.mlp = nn.Sequential(nn.Linear(input_dim, mlp_hidden_dim),
                             nn.GELU(),
                             nn.Dropout(dropout),
                             nn.Linear(mlp_hidden_dim, embed_dim),
                             nn.Dropout(dropout))

  def forward(self, x):
    identity = x
    out = self.layernorm1(x)
    out = self.msa(out)
    out = self.w_o_dropout(out)
    out += identity
    identity = out
    out2 = self.layernorm2(out)
    out2 = self.mlp(out2)
    out2 += identity
    return out2

class LanguageTransformer(nn.Module):

  def __init__(self, vocab_size, num_layers, num_heads, embed_dim, mlp_hidden_dim, dropout):
    super().__init__()

    # make sure input length (max_num_tokens) is multiple of num_heads
    assert embed_dim % num_heads == 0, "ERROR: input length (max_num_tokens) must be multiple of num_heads"

    self.num_heads = num_heads
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(self.vocab_size, embed_dim)
    # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.embedding_dropout = nn.Dropout(dropout)
    self.encoder_layers = nn.ModuleList([])
    for i in range(num_layers):
      self.encoder_layers.append(TransformerLayer(num_heads, embed_dim, embed_dim, mlp_hidden_dim, dropout))
    self.mlp_head = nn.Linear(embed_dim, vocab_size)
    self.layernorm = nn.LayerNorm(embed_dim)

  def forward(self, x):
    """x: encoded sentences (batch_size, max_num_tokens)"""
    out = self.embedding(x)

    out = self.embedding_dropout(out)

    # run through encoder layers
    for layer in self.encoder_layers:
      out = layer(out)
    
    out = self.layernorm(out)

    logits = self.mlp_head(out)
    # logits = logits.view(-1, logits.size(-1)) # (batch_size * max_num_tokens, vocab_size)

    # if we only want to generate the next token, we can just return the last token's logits
    # logits = logits[:, -1, :]
    # even better, we didnt even need to run mlp on the whole sequence, we can just run it on the last token's embedding
    # logits = self.mlp_head(out[:, [-1], :]) # [-1] to preserve dim
    
    return logits

def get_tiny_model(vocab_size=100):
    return LanguageTransformer(vocab_size=vocab_size, num_layers=12, num_heads=3,
              embed_dim=192, mlp_hidden_dim=768, dropout=0.1)

def get_small_model(vocab_size=100):
    return LanguageTransformer(vocab_size=vocab_size, num_layers=12, num_heads=6,
               embed_dim=384, mlp_hidden_dim=1536, dropout=0.1)

def generate(model, tokenizer, max_length, start_word):
    model.eval()
    start_token = tokenizer.encode(start_word, return_tensors="pt").to(device)
    with torch.no_grad():
        input_ids = torch.tensor([start_token]).unsqueeze(0).to(device)  # Batch size 1
        for i in range(max_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            new_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat((input_ids, new_token.unsqueeze(0)), dim=-1)
            if new_token == tokenizer.eos_token_id:
                break
        return input_ids, tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    
if __name__ == "__main__":
    print("🐌 running slowly!...")

    bs = 16
    max_num_tokens = 50
    input_dim = 32
    embed_dim = 100
    num_heads = 5

    ### Test MSA (Multiheaded Self Attention)
    sample = torch.randn(bs, max_num_tokens, input_dim, device=device)
    msa = MSA(input_dim = input_dim, embed_dim = embed_dim, num_heads=num_heads).to(device)
    msa_out_shape = msa(sample).shape
    assert msa_out_shape == (bs, max_num_tokens, embed_dim), "🚨 ERROR"; print("✅ MSA test passed!")
    del msa

    ### Test TransformerLayer
    mlp_hidden_dim = 128
    sample = torch.randn(bs, max_num_tokens, embed_dim, device=device)
    vitlayer = TransformerLayer(num_heads=num_heads, input_dim=embed_dim, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim).to(device)
    vitlayer_out_shape = vitlayer(sample).shape
    assert vitlayer_out_shape == (bs, max_num_tokens, embed_dim), "🚨 ERROR"; print("✅ TransformerLayer test passed!")
    del vitlayer

    ### Test LanguageTransformer
    vocab_size = 5000 # <- just for tests
    model = get_tiny_model(vocab_size=vocab_size).to(device)
    sample = torch.randint(0, 100, (bs, max_num_tokens), device=device)
    out = model(sample)
    assert out.shape == (bs, max_num_tokens, vocab_size), "🚨 ERROR"; print("✅ LanguageTransformer test passed!")
    del model


    ########### ######### TRAIN Model!!! ########## ##########
    # prepare dataset/dataloader
    wikitext_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token='<pad>')
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=False)
        # return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=500)
    tokenized_datasets = wikitext_dataset.map(tokenize_function, batched=True, num_proc=32)
    tokenized_datasets = tokenized_datasets.remove_columns(['text'])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(tokenized_datasets["train"], batch_size=bs, collate_fn=data_collator, num_workers=10, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(tokenized_datasets["test"], batch_size=bs, collate_fn=data_collator, num_workers=10)

    # hyperparams
    lr = 3e-4
    num_epochs = 10

    # model
    model = get_tiny_model(vocab_size=len(tokenizer.get_vocab())).to(device)
    criterion = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []

    # test generate
    generated_ids, generated_words = generate(model, tokenizer, max_length=50, start_word="hello")
    print("------ testing generate ------")
    print(generated_words)
    print("----------- done ----------")

    # train 
    for epoch in range(num_epochs):
      train_loss = 0.0
      # train_acc = 0.0
      train_total = 0
      model.train()
      for batch in tqdm(train_dataloader):
        inputs = batch['input_ids'][:, :-1].to(device)
        labels = batch['input_ids'][:, 1:].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, outputs.size(-1))
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.shape[0]
        # train_acc += torch.sum((torch.argmax(outputs, dim=1) == labels)).item()
        train_total += inputs.shape[0]
      train_loss = train_loss / train_total
      # train_acc = train_acc / train_total
      train_losses.append(train_loss)

      test_loss = 0.0
      # test_acc = 0.0
      test_total = 0
      model.eval()
      with torch.no_grad():
          for batch in test_dataloader:
              inputs = batch['input_ids'][:, :-1].to(device)
              labels = batch['input_ids'][:, 1:].to(device)

              outputs = model(inputs)
              # loss = criterion(outputs, labels.long())
              loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
              test_loss += loss.item() * inputs.shape[0]
              # test_acc += torch.sum((torch.argmax(outputs, dim=1) == labels)).item()
              test_total += inputs.shape[0]
      test_loss = test_loss / test_total
      # test_acc = test_acc / test_total
      test_losses.append(test_loss)

      print(f'[{epoch + 1:2d}] train loss: {train_loss:.3f} | test_loss: {test_loss:.3f} ')
      generated_ids, generated_words = generate(model, tokenizer, max_length=50, start_word="hello")
      print("------ testing generate ------")
      print(generated_words)
      print("----------- done ----------")

    print("Finished Training")