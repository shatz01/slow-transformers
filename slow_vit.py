# device = 'cpu'
device = 'cuda'
import torch.nn.functional as F
from torch import nn
import torch
from data import Cifar10Dataset
from torch.utils.data import DataLoader, Dataset

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

  def forward(self, x, v=False):
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
    w_V = w_V.reshape(batch_size, max_num_tokens, embed_dim)

    out = self.out_embed(w_V)
    return out


class ViTLayer(nn.Module):

  def __init__(self, num_heads, input_dim, embed_dim, mlp_hidden_dim, dropout=0.1):
    assert input_dim == embed_dim
    super().__init__()
    self.layernorm1 = nn.LayerNorm(input_dim)
    self.msa = MSA(input_dim, input_dim, num_heads)
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


class ViT(nn.Module):

  def __init__(self, patch_dim, image_dim, num_layers, num_heads, embed_dim, mlp_hidden_dim, num_classes, dropout):
    super().__init__()
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
    pass

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
    sample = torch.randn(bs, max_num_tokens, input_dim)
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
    vit = get_vit_tiny()
    sample = torch.randn(bs, 3, 32, 32) # example batch of images from cifar
    # assert vitlayer_out_shape == (bs, max_num_tokens, embed_dim), "🚨 ERROR"; print("✅ ViTLayer test passed!")

    ### TRAIN VIT!!!
    trainset = Cifar10Dataset(True)
    trainloader = DataLoader(trainset, shuffle=True, batch_size=bs, num_workers=24) # TODO: Pass our dataset trainset into a torch Dataloader object, with shuffle = True and the batch_size=batch_size, num_workers=2
    testset = Cifar10Dataset(False)
    testloader = DataLoader(testset, shuffle=True, batch_size=bs, num_workers=24) # TODO: create a test dataset the same as the train loader but with shuffle=False and the test dataset
    sample = next(iter(trainloader))
    print(sample[0].shape)