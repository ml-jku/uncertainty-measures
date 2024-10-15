import os

import torchvision as tv

from source.constants import CIFAR10_PATH, CIFAR100_PATH, SVHN_PATH, TIN_PATH, LSUN_PATH
from source.data.utils import train_transform, transform, tin_transform, download_tiny_imagenet, download_lsun
from source.data.imagenet import get_imagenet_test, get_imagenet_o, get_imagenet_a

def load_train_dataset(dataset_name):
    if dataset_name == "cifar10":
        n_classes = 10
        dataset = tv.datasets.CIFAR10(CIFAR10_PATH, train=True, download=True, transform=train_transform)
    elif dataset_name == "cifar100":
        n_classes = 100
        dataset = tv.datasets.CIFAR100(CIFAR100_PATH, train=True, download=True, transform=train_transform)
    elif dataset_name == "svhn":
        n_classes = 10
        dataset = tv.datasets.SVHN(SVHN_PATH, split="train", download=True, transform=train_transform)
    elif dataset_name == "tin":
        n_classes = 200
        download_tiny_imagenet(TIN_PATH)
        dataset = tv.datasets.ImageFolder(root=os.path.join(TIN_PATH, "tiny-imagenet-200", "train"), 
                                        transform=tin_transform(train_transform))
    # LSUN is only test dataset
    # elif args.dataset == "lsun":
    #     n_classes = 10
    #     download_lsun(LSUN_PATH)
    #     dataset = tv.datasets.ImageFolder(root=os.path.join(LSUN_PATH, "LSUN_resize"), transform=train_transform)
    else:
        raise NotImplementedError("Dataset not supported")

    return dataset, n_classes

def load_test_dataset(dataset_name, provide_n_classes=False, custom_transform=None):

    if custom_transform is not None:
        t = custom_transform
    else:
        t = transform

    if dataset_name == "cifar10":
        n_classes = 10
        dataset = tv.datasets.CIFAR10(CIFAR10_PATH, train=False, download=True, transform=t)
    elif dataset_name == "cifar100":
        n_classes = 100
        dataset = tv.datasets.CIFAR100(CIFAR100_PATH, train=False, download=True, transform=t)
    elif dataset_name == "svhn":
        n_classes = 10
        dataset = tv.datasets.SVHN(SVHN_PATH, split="test", download=True, transform=t)
    elif dataset_name == "tin":
        n_classes = 200
        download_tiny_imagenet(TIN_PATH)
        # use validation set as test set -> has labels
        dataset = tv.datasets.ImageFolder(root=os.path.join(TIN_PATH, "tiny-imagenet-200", "val"), 
                                          transform=tin_transform(t))
    elif dataset_name == "lsun":
        n_classes = 10
        download_lsun(LSUN_PATH)
        dataset = tv.datasets.ImageFolder(root=os.path.join(LSUN_PATH, "LSUN_resize"), transform=t)
    elif dataset_name == "imagenet":
        n_classes = 1000
        dataset = get_imagenet_test()
    elif dataset_name == "imagenet-o":
        n_classes = 1000
        dataset = get_imagenet_o()
    elif dataset_name == "imagenet-a":
        n_classes = 1000
        dataset = get_imagenet_a()
    else:
        raise NotImplementedError("Dataset not supported")
    
    if provide_n_classes:
        return dataset, n_classes
    return dataset
