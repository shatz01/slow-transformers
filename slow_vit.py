import torch.nn.functional as F
from torch import nn
import torch
from data import Cifar10Dataset
from torch.utils.data import DataLoader, Dataset
import math
from tqdm import tqdm

device = 'cpu'
# device = 'cuda'
# device = torch.device("mps")

class MSA(nn.Module):

  def __init__(self, input_dim, embed_dim, num_heads):
    '''
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
    assert max_num_tokens % self.num_heads == 0

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


class ViTLayer(nn.Module):

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

  def forward(self, x, db=False):
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


class ViT(nn.Module):

  def __init__(self, patch_dim, image_dim, num_layers, num_heads, embed_dim, mlp_hidden_dim, num_classes, dropout):
    super().__init__()
    self.num_heads = num_heads
    self.image_dim = image_dim
    self.patch_dim = patch_dim
    self.input_dim = patch_dim * patch_dim * 3 
    self.patch_embedding = nn.Linear(self.input_dim, embed_dim)
    self.positional_embedding = nn.Parameter(torch.zeros(1, (image_dim//patch_dim) ** 2 + 1, embed_dim))
    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.embedding_dropout = nn.Dropout(dropout)
    self.encoder_layers = nn.ModuleList([])
    for i in range(num_layers):
      self.encoder_layers.append(ViTLayer(num_heads, embed_dim, embed_dim, mlp_hidden_dim, dropout))
    self.mlp_head = nn.Linear(embed_dim, num_classes)
    self.layernorm = nn.LayerNorm(embed_dim)

  def forward(self, x):
    """x: raw image data (batch_size, channels, rows, cols)"""

    h = w = self.image_dim // self.patch_dim
    bs = x.shape[0]
    x = x.reshape(bs, 3, h, self.patch_dim, w, self.patch_dim) # unroll last 2 dims
    x = torch.einsum("nchpwq -> nhwpqc", x) # make channels last and put other dims together
    x = x.reshape(bs, h*w, self.input_dim) # now we have patches (8, 3, 32, 32) -> (8, 64, 48)

    patch_embeddings = self.patch_embedding(x)
    patch_embeddings = torch.cat([torch.tile(self.cls_token, (bs, 1, 1)),
                                 patch_embeddings], dim=1) # (8, 64, 192) -> (8, 65, 192)
    out = patch_embeddings + torch.tile(self.positional_embedding, (bs, 1, 1)) # add (not concat) positional embedding
    out = self.embedding_dropout(out)

    # we must pad s.t. input length is multiple of num_heads
    add_len = (self.num_heads - out.shape[1]) % self.num_heads
    out = torch.cat([out, torch.zeros(bs, add_len, out.shape[2], device=device)], dim=1)

    # run through encoder layers
    for l in self.encoder_layers:
      out = l(out, db=True)
    
    # pop off and read our classification token, see what the value is
    cls_head = self.layernorm(torch.squeeze(out[:, 0], dim=1))
    logits = self.mlp_head(cls_head)
    return logits

def get_vit_tiny(num_classes=10, patch_dim=4, image_dim=32):
    return ViT(patch_dim=patch_dim, image_dim=image_dim, num_layers=12, num_heads=3,
              embed_dim=192, mlp_hidden_dim=768, num_classes=num_classes, dropout=0.1)

def get_vit_small(num_classes=10, patch_dim=4, image_dim=32):
    return ViT(patch_dim=patch_dim, image_dim=image_dim, num_layers=12, num_heads=6,
               embed_dim=384, mlp_hidden_dim=1536, num_classes=num_classes, dropout=0.1)
    
if __name__ == "__main__":
    print("🐌 running slowly!...")

    bs = 8
    max_num_tokens = 50
    input_dim = 32
    embed_dim = 100
    num_heads = 5

    ### Test MSA (Multiheaded Self Attention)
    sample = torch.randn(bs, max_num_tokens, input_dim).to(device)
    msa = MSA(input_dim = input_dim, embed_dim = embed_dim, num_heads=num_heads)
    msa_out_shape = msa(sample).shape
    assert msa_out_shape == (bs, max_num_tokens, embed_dim), "🚨 ERROR"; print("✅ MSA test passed!")

    ### Test ViTLayer
    mlp_hidden_dim = 128
    sample = torch.randn(bs, max_num_tokens, embed_dim)
    vitlayer = ViTLayer(num_heads=num_heads, input_dim=embed_dim, embed_dim=embed_dim, mlp_hidden_dim=mlp_hidden_dim)
    vitlayer_out_shape = vitlayer(sample).shape
    assert vitlayer_out_shape == (bs, max_num_tokens, embed_dim), "🚨 ERROR"; print("✅ ViTLayer test passed!")

    ### Test ViT
    num_classes=10
    bs = 256
    vit = get_vit_tiny().to(device)
    sample = torch.randn(bs, 3, 32, 32) # example batch of images from CIFAR
    vit_out_shape = vit(sample).shape
    assert vit_out_shape == (bs, num_classes), "🚨 ERROR"; print("✅ Full ViT test passed!")

    ### TRAIN VIT!!!
    train_dataset = Cifar10Dataset(True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=bs, num_workers=12) # TODO: Pass our dataset trainset into a torch Dataloader object, with shuffle = True and the batch_size=batch_size, num_workers=2
    test_dataset = Cifar10Dataset(False)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=bs, num_workers=12) # TODO: create a test dataset the same as the train loader but with shuffle=False and the test dataset
    lr = 5e-4 * bs / 256
    num_epochs = 10
    warmup_frac = 0.1
    weight_decay = 0.1
    total_steps = math.ceil(len(train_dataset) * num_epochs)
    warmup_steps = total_steps * warmup_frac
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vit.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        vit.train()
        for inputs, labels in tqdm(train_dataloader):
            """TODO:
            1. Set inputs and labels to be on device
            2. zero out our gradients
            3. pass our inputs through the ViT
            4. pass our outputs / labels into our loss / criterion
            5. backpropagate
            6. step our optimizeer
            """
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = vit(inputs)
            
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.shape[0]
            train_acc += torch.sum((torch.argmax(outputs, dim=1) == labels)).item()
            train_total += inputs.shape[0]
        train_loss = train_loss / train_total
        train_acc = train_acc / train_total
        train_losses.append(train_loss)

        test_loss = 0.0
        test_acc = 0.0
        test_total = 0
        vit.eval()
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = vit(inputs)
                loss = criterion(outputs, labels.long())

                test_loss += loss.item() * inputs.shape[0]
                test_acc += torch.sum((torch.argmax(outputs, dim=1) == labels)).item()
                test_total += inputs.shape[0]
        test_loss = test_loss / test_total
        test_acc = test_acc / test_total
        test_losses.append(test_loss)

        print(f'[{epoch + 1:2d}] train loss: {train_loss:.3f} | train accuracy: {train_acc:.3f} | test_loss: {test_loss:.3f} | test_accuracy: {test_acc:.3f}')

    print('Finished Training')