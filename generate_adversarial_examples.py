import os
import glob
import argparse
from tqdm import tqdm

import torch
import torchvision as tv
from torch.utils.data import DataLoader

from foolbox import PyTorchModel
from foolbox.attacks import FGSM, LinfPGD
from foolbox.criteria import Misclassification

from source.constants import RESULTS_PATH, CIFAR_MEAN, CIFAR_STD
from source.networks.resnet import get_resnet18, get_resnet18_d
from source.networks.densenet import get_densenet169
from utils import load_test_dataset


###############
### Parsing ###
###############

parser = argparse.ArgumentParser()
# general
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--method", default="de")
parser.add_argument("--p_drop", default=0.2, type=float)
parser.add_argument("--network", default="resnet18")
parser.add_argument("--seed", default=42, type=int) 
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--aa", default="linfpgd", type=str)
# Attack parameters
parser.add_argument("--eps", default=8/255, type=float)
parser.add_argument("--batch_size", default=512, type=int)
parser.add_argument("--runs", default=5, type=float)

# parse
args = parser.parse_args()

# convinience
seed, device = args.seed, args.device
print("Computation executed on >", device)

# check arguments
assert args.dataset in ["cifar10", "cifar100", "svhn", "tin"], "Dataset not supported"
assert args.network in ["resnet18", "densenet169"], "Network not supported"
assert args.method in ["de", "la", "mcd"], "Method not supported"
assert args.aa in ["fgsm", "linfpgd"], "Attack not supported"

#################
### Load Data ###
#################

dataset, n_classes = load_test_dataset(args.dataset, provide_n_classes=True, custom_transform=tv.transforms.ToTensor())

if args.method == "de":
    path = os.path.join(RESULTS_PATH, f"{args.dataset}_{args.network}_seed{seed}")
elif args.method == "la":
    path = os.path.join(RESULTS_PATH, f"{args.dataset}_{args.network}_seed{seed}_laplace")
elif args.method == "mcd":
    path = os.path.join(RESULTS_PATH, f"{args.dataset}_{args.network}_dropout{args.p_drop}_seed{seed}")


#################
## Generate AE ##
#################

save_path = os.path.join(path, f"adversarial_examples")
os.makedirs(save_path, exist_ok=True)

model_files = glob.glob(os.path.join(path, "models", "*.pt"))

for run in range(args.runs):

    model_file = os.path.join(path, "models", f"model_{run * (len(model_files) // args.runs)}.pt")
    print(model_file)

    # Load model
    if args.network == "resnet18":
        if args.method == "mcd":
            network = get_resnet18_d(num_classes=n_classes, p_drop=args.p_drop)
        else:
            network = get_resnet18(num_classes=n_classes) 
    elif args.network == "densenet169":
        if args.method == "mcd":
            raise NotImplementedError("MCD not implemented for DenseNet169")
        else:
            network = get_densenet169(num_classes=n_classes)

    network.load_state_dict(torch.load(model_file, map_location=device))
    network.eval()
    network.to(device)

    preprocessing = dict(mean=CIFAR_MEAN, std=CIFAR_STD, axis=-3)
    # Create model
    fmodel = PyTorchModel(network, bounds=(0, 1), device=device, preprocessing=preprocessing)

    # Attack
    if args.aa == "fgsm":
        attack = FGSM()
    elif args.aa == "linfpgd":
        attack = LinfPGD()

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Evaluate
    all_aes = list()
    pb = tqdm(dataloader)
    for images, labels in pb:
        images, labels = images.to(device), labels.to(device)

        # Attack

        _, clipped_advs, success = attack(fmodel, images, epsilons=[args.eps], criterion=Misclassification(labels))

        all_aes.append((clipped_advs[0] * 255).to(torch.uint8).detach().cpu())

        pb.set_description(f"Success rate: {success.float().mean().item() * 100:.1f}%")

    all_aes = torch.cat(all_aes, dim=0)

    # Save
    torch.save(all_aes, os.path.join(save_path, f"{args.aa}_{run}.pt"))

