import torch.nn as nn
import torchvision as tv


def get_densenet121(num_classes: int):
    network = tv.models.densenet121(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network)

def get_densenet161(num_classes: int):
    network = tv.models.densenet161(weights=None, num_classes=num_classes)
    return _adapt_for_small_images

def get_densenet169(num_classes: int):
    network = tv.models.densenet169(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network)

def get_densenet201(num_classes: int):
    network = tv.models.densenet201(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network)

def _adapt_for_small_images(network: nn.Module):
    # adapt network for 32x32 images
    network.features.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    network.features.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    return network