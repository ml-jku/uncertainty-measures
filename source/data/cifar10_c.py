import os
import wget
import shutil
import numpy as np

import torch
from torch.utils.data import TensorDataset
from torchvision import datasets

from source.constants import CIFAR10_PATH
from .utils import transform

"""
* This file contains all necessary methods to create datasets for experiments on CIFAR10-C
"""

corruptions = [
    # test corruptions
    "brightness",
    "contrast",
    "defocus_blur",
    "elastic_transform",
    "fog",
    "frost",
    "gaussian_blur",
    "gaussian_noise",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "shot_noise",
    "snow",
    "zoom_blur",
    # hyperparameter selection validation corruptions (https://arxiv.org/abs/1903.12261)
    "speckle_noise",
    "glass_blur",
    "spatter",
    "saturate",
]

def get_cifar10_c(corruption:int=0, severity:int=1):
    assert 0 <= severity <= 5, f"'severity' must be in [0, 5], but was {severity}."
    assert 0 <= corruption <= len(corruptions), f"'corruption' must be in [0, {len(corruptions)}], was {corruption}"
    
    if severity == 0:
        # severity 0 is original testset
        return datasets.CIFAR10(CIFAR10_PATH, train=False, download=True, transform=transform)
    
    _download_cifar10_c(CIFAR10_PATH)

    test_x = np.load(os.path.join(CIFAR10_PATH, "CIFAR-10-C", f"{corruptions[corruption]}.npy"))
    test_y = np.load(os.path.join(CIFAR10_PATH, "CIFAR-10-C", "labels.npy"))

    test_x = test_x[10_000 * (severity - 1) : 10_000 * severity]

    transformed_test_x = list()
    for i in range(len(test_x)):
        transformed_test_x.append(transform(test_x[i]))
        if transform(test_x[i]).shape != (3, 32, 32):
            print(transform(test_x[i]).shape)

    test_y = test_y[10_000 * (severity - 1) : 10_000 * severity]

    return TensorDataset(torch.stack(transformed_test_x, dim=0), torch.Tensor(test_y))


def _download_cifar10_c(path:str):
    if not (os.path.exists(os.path.join(path, "CIFAR-10-C.tar")) or os.path.exists(os.path.join(path, "CIFAR-10-C"))):
        wget.download("https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1", os.path.join(path, "CIFAR-10-C.tar"))
    if not os.path.exists(os.path.join(path, "CIFAR-10-C")):
        print("unpacking...")
        shutil.unpack_archive(os.path.join(path, "CIFAR-10-C.tar"), os.path.join(path))
        print("unpacked")
