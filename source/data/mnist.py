import numpy as np
from typing import Tuple

import torch
from torch.utils.data import Dataset
import torchvision as tv
import torchvision.transforms as transforms

from ..constants import MNIST_PATH, FMNIST_PATH

transform = transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_mnist_test():
    dataset = tv.datasets.MNIST(MNIST_PATH, train=False, download=True, transform=transform)
    return _get_tensors(dataset)
        
def get_mnist_train():
    dataset = tv.datasets.MNIST(MNIST_PATH, train=True, download=True, transform=transform)
    return _get_tensors(dataset)

def get_fmnist_test():
    dataset = tv.datasets.FashionMNIST(FMNIST_PATH, train=False, download=True, transform=transform)
    return _get_tensors(dataset)
        
def get_fmnist_train():
    dataset = tv.datasets.FashionMNIST(FMNIST_PATH, train=True, download=True, transform=transform)
    return _get_tensors(dataset)

def _get_tensors(dataset):
    images, labels = next(iter(torch.utils.data.DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)))
    return images, labels


class TensorDataset(Dataset):

    def __init__(self, x_test_t, y_test_t):
        super(TensorDataset, self).__init__()
        self.inputs = x_test_t
        self.targets = y_test_t

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    

class UpsampleDataset(Dataset):

    def __init__(self, dataset: Dataset, upsample: int, seed: int = 42) -> None:
        super().__init__()

        self.dataset = dataset
        self.upsample = upsample

        self.rng = np.random.default_rng(seed=seed)
        self.indices = self.rng.integers(low=0, high=len(self.dataset), size=(max(upsample, len(dataset)), ))
        self.indices[:len(self.dataset)] = np.asarray(range(len(self.dataset)))

    def __len__(self) -> int:
        return self.upsample
    
    def __getitem__(self, index) -> Tuple:
        return self.dataset[self.indices[index]]
