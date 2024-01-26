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

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, embed_dim, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_dim, embed_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(embed_dim),
            )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.embed_dim = embed_dim
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size = 7, stride = 2, padding = 3)
        self.res1 = ResidualBlock(64, 64)
        self.res2 = ResidualBlock(64, 64)
        self.res_layers = nn.Sequential(*self.res_layers)


    
if __name__ == "__main__":
    print("🐌 running slowly!...")

    bs = 16
    max_num_tokens = 50
    input_dim = 32
    embed_dim = 100
    num_heads = 5

    ### Test ResidualBlock
    sample = torch.randn(bs, max_num_tokens, embed_dim) # torch.Size([16, 50, 32]) (B, L, input_dim)
    sample = sample.permute(0, 2, 1)
    residual_block = ResidualBlock(embed_dim, embed_dim)
    out = residual_block(sample)
    out = out.permute(0, 2, 1) # torch.Size([16, 50, 100]) (B, L, embed_dim)
    assert out.shape == (bs, max_num_tokens, embed_dim)
    print("✅ ResidualBlock passed!")

    # ### Test ResNet
    # num_classes = 10
    # model = ResNet(num_classes, embed_dim)
