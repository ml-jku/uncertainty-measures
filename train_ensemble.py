import os
import argparse
import numpy as np

import torch
from torch.utils.data import Subset, DataLoader

from source.constants import RESULTS_PATH
from source.networks.resnet import get_resnet18
from source.networks.densenet import get_densenet169
from source.networks.regnet import get_regnet_y_800mf
from source.utils.seeding import fix_seeds
from source.utils.train_utils import fit
from utils import load_train_dataset


###############
### Parsing ###
###############

parser = argparse.ArgumentParser()
# general
parser.add_argument("--dataset", default="cifar10")
parser.add_argument("--network", default="resnet18")
parser.add_argument("--seed", default=42, type=int) 
parser.add_argument("--device", default="cuda:0")
# Network
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--patience", default=0, type=int)
parser.add_argument("--num_workers", default=4, type=int)
# Ensemble
parser.add_argument("--num_networks", default=50, type=int)

# parse
args = parser.parse_args()

# convinience
seed, device = args.seed, args.device
print("Computation executed on >", device)

# check network
assert args.network in ["resnet18", "densenet169", "regnet", "cnn"], "Network not supported"

run_path = os.path.join(RESULTS_PATH, f"{args.dataset}_{args.network}_seed{seed}")
os.makedirs(run_path, exist_ok=True)

# save command line arguments
formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
with open(os.path.join(run_path, "args.txt"), "w") as file:
    file.write(formatted_args)

#################
### LOAD DATA ###
#################

dataset, n_classes = load_train_dataset(args.dataset)

# partition train / val
rng = np.random.default_rng(seed=seed)
splitting = 6

val_inds = rng.choice(np.arange(len(dataset)), size=len(dataset) // splitting, replace=False)
train_inds = np.delete(np.arange(len(dataset)), (val_inds))

print(len(train_inds), len(val_inds))

# for training just train and val datasets necessary
train_ds = Subset(dataset, indices=train_inds)
val_ds = Subset(dataset, indices=val_inds)

# save val indices for reproducibility
torch.save(torch.LongTensor(val_inds), os.path.join(run_path, "val_inds.pt"))

####################
### LEARN MODELS ###
####################

fix_seeds(seed=seed)

for n in range(args.num_networks):

    if args.network == "resnet18":
        network = get_resnet18(num_classes=n_classes) 
    elif args.network == "densenet169":
        network = get_densenet169(num_classes=n_classes)
    elif args.network == "regnet":
        network = get_regnet_y_800mf(num_classes=n_classes)
    elif args.network == "cnn":
        network = SimpleCNN(n_classes)
    else:
        raise NotImplementedError("Network not supported")
    
    network.to(device)
    network.train()

    network, val_perf = fit(network = network, 
                            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers), 
                            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers),
                            epochs = args.epochs,
                            lr = args.lr, 
                            weight_decay = args.weight_decay, 
                            use_adam = False, 
                            patience = args.patience, 
                            use_auroc= False,
                            verbose = False)

    os.makedirs(os.path.join(run_path, "models"), exist_ok=True)
    torch.save(network.state_dict(), os.path.join(run_path, "models", f"model_{n}.pt"))
    
    # save val_perf to file as text file & remove if existed previously
    if n == 0 and os.path.exists(os.path.join(run_path, f"val_perfs.txt")):
        os.remove(os.path.join(run_path, f"val_perfs.txt"))
    with open(os.path.join(run_path, f"val_perfs.txt"), "a") as file:
        file.write(f"{n}: {(max(val_perf) * 100):.2f}%\n")

    # print highest val_acc  
    print(f"Model {n} trained with performance: {(max(val_perf) * 100):.2f}%")
    