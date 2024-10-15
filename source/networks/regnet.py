import torch.nn as nn
import torchvision as tv
from torchvision.ops.misc import Conv2dNormActivation


def get_regnet_y_800mf(num_classes=10):
    network = tv.models.regnet_y_800mf(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network)

def _adapt_for_small_images(network: nn.Module):
    # adapt network for 32x32 images
    network.stem = Conv2dNormActivation(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    return network
