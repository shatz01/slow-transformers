import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision
import math
import torch.optim as optim

class Cifar10Dataset(Dataset):
    def __init__(self, train):
        self.cifar10_mean = torch.tensor([0.49139968, 0.48215827, 0.44653124])
        self.cifar10_std = torch.tensor([0.24703233, 0.24348505, 0.26158768])
        self.transform = transforms.Compose([
                                                transforms.Resize(40),
                                                transforms.RandomCrop(32),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.cifar10_mean, self.cifar10_std)
                                            ])
        self.dataset = torchvision.datasets.CIFAR10(root='./SSL-Vision/data',
                                                    train=train,
                                                    download=True)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label