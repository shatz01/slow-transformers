import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision

class Cifar10Dataset(Dataset):
    def __init__(self, train):
        # self.cifar10_mean = torch.tensor([0.49139968, 0.48215827, 0.44653124])
        # self.cifar10_std = torch.tensor([0.24703233, 0.24348505, 0.26158768])
        self.cifar10_mean = torch.tensor([0.49139968, 0.48215841, 0.44653091])
        self.cifar10_std = torch.tensor([0.24703223, 0.24348513, 0.26158784])
        self.transform = transforms.Compose([
                                                transforms.Resize(40),
                                                transforms.RandomCrop(32),
                                                transforms.RandomResizedCrop((32,32),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize(self.cifar10_mean, self.cifar10_std)
                                            ])
        self.dataset = torchvision.datasets.CIFAR10(root='./cifar_data',
                                                    train=train,
                                                    download=True)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label