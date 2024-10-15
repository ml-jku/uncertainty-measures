import torch.nn as nn
import torchvision as tv


def get_resnet18(num_classes=10):
    network = tv.models.resnet18(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network) 

def get_resnet34(num_classes=10):
    network = tv.models.resnet34(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network)

def get_resnet50(num_classes=10):
    network = tv.models.resnet50(weights=None, num_classes=num_classes)
    return _adapt_for_small_images(network)

def get_resnet18_d(num_classes=10, p_drop=0.2):
    network = get_resnet18(num_classes=num_classes)
    return _add_dropout(network, p_drop)

def _adapt_for_small_images(network: nn.Module):
    # adapt network for 32x32 images
    network.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    network.maxpool = nn.Identity()
    return network

def _add_dropout(network: nn.Module, p_drop: float):
    for name, module in network.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(network, name, nn.Sequential(module, nn.Dropout2d(p=p_drop)))
        if isinstance(module, nn.AdaptiveAvgPool2d):
            setattr(network, name, nn.Sequential(module, nn.Dropout(p=p_drop)))
        else:
            _add_dropout(module, p_drop)  # Recursively apply to submodules
    return network
