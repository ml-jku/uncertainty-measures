import os
from typing import Final

# general paths
DATASETS_PATH: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "datasets"))
RESULTS_PATH: Final[str] =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "results"))
RESULTS_PATH_AL: Final[str] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "results_al"))
PLOTS_PATH: Final[str] =  os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "plots"))

# Dataset paths
CIFAR10_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "CIFAR10"))
CIFAR100_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "CIFAR100"))
SVHN_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "SVHN"))
TIN_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "TIN"))
LSUN_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "LSUN"))
MNIST_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "MNIST"))
FMNIST_PATH: Final[str] = os.path.abspath(os.path.join(DATASETS_PATH, "FMNIST"))

# Statistics
CIFAR_MEAN: Final[list] = [0.4914, 0.4822, 0.4465]
CIFAR_STD: Final[list] = [0.2023, 0.1994, 0.2010]
MNIST_MEAN = 0.1307
MNIST_VAR = 0.3081