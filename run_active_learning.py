import os
import copy
from typing import Dict
import numpy as np
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from source.utils.seeding import fix_seeds
from source.utils.uncertainty_measures import calculate_uncertainties
from source.utils.train_utils import fit
from source.data.mnist import get_mnist_train, get_mnist_test, get_fmnist_train, get_fmnist_test, TensorDataset, UpsampleDataset
from source.networks.mnist_cnn import MNISTConvNet
from source.constants import RESULTS_PATH_AL


# prediction methods
def predict_ensemble(
        train_loader: DataLoader,
        val_loader: DataLoader,
        x_test_t: torch.Tensor,
        n_samples: int,
        lr: float,
        epochs: int,
        weight_decay: float,
        device: str
) -> Dict:
    
    preds, state_dicts = list(), list()

    for _ in range(n_samples):
        model = MNISTConvNet()
        model.to(device=device)

        model, _ = fit(model, train_loader, val_loader, epochs, lr, weight_decay, use_adam=True, patience=50, verbose=False)

        with torch.no_grad():
            preds.append(torch.softmax(model.forward(x_test_t), dim=1).cpu())
        state_dicts.append(model.cpu().state_dict())

    preds = torch.stack(preds, dim=1)

    return {
        "preds": preds, # [n_test_samples, n_samples, n_classes]
        "state_dicts": state_dicts,
    }

def predict_mc_dropout(
        train_loader: DataLoader,
        val_loader: DataLoader,
        x_test_t: torch.Tensor,
        n_samples: int,
        p_drop: float,
        lr: float,
        epochs: int,
        weight_decay: float,
        device: str
) -> Dict:
    
    model = MNISTConvNet(p_drop=p_drop)
    model.to(device=device)

    model, _ = fit(model, train_loader, val_loader, epochs, lr, weight_decay, use_adam=True, patience=50, verbose=False)

    with torch.no_grad():
        # prediction of non-dropped-out model first
        model.eval()
        preds = [torch.softmax(model.forward(x_test_t), dim=1).cpu()]

        model.train()
        for _ in range(n_samples - 1):
            preds.append(torch.softmax(model.forward(x_test_t), dim=1).cpu())

        preds = torch.stack(preds, dim=1)

    return {
        "preds": preds, # [n_test_samples, n_samples, n_classes]
        "state_dicts": copy.deepcopy(model).cpu().state_dict(),
    }


###############
### Parsing ###
###############

parser = argparse.ArgumentParser()
# general
parser.add_argument("--dataset", default="mnist")
parser.add_argument("--method", default="mc_dropout")
parser.add_argument("--acquisition_function", default="tu_bc3")
parser.add_argument("--start_samples_per_class", default=2, type=int)
parser.add_argument("--n_iterations", default=57, type=int)
parser.add_argument("--n_samples_per_iteration", default=5, type=int)
parser.add_argument("--train_size", default=0, type=int)
    # Note: for now seed only used for everything method specific, dataset uses indep. fixed seed
parser.add_argument("--seed", default=42, type=int) 
parser.add_argument("--device", default="cpu")
parser.add_argument("--num_workers", default=0, type=int)
# method general
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=50, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
    # Note: n_samples refers to number of posterior samples (models).
parser.add_argument("--n_samples", default=50, type=int)
# method specific
## Ensembles
    # n_samples
## MCD
    # n_samples
parser.add_argument("--p_drop", default=0.2, type=float)
# parse
args = parser.parse_args()


# convinience
seed, device = args.seed, args.device
print("Computation executed on >", device)
# if no desired fixed train size is given, use the final train size
final_train_size = args.start_samples_per_class * 10 + (args.n_iterations - 1) * args.n_samples_per_iteration if args.train_size == 0 else args.train_size
print("Train size >", final_train_size)
print("Acquisition function >", args.acquisition_function)


# check dataset
assert args.dataset in ["mnist", "fmnist"], "Dataset not supported"
# check method
assert args.method in ["mc_dropout", "ensemble"], "Method not supported"
# check acquisition function
assert args.acquisition_function in ["tu_bc3", "tu_bc2_au_b", "au_c", "eu_c3", "eu_c2", "eu_b3", "random"], "acquisition function not supported"


# define path for results
now = datetime.now().strftime('%Y_%m_%d__%H_%M')
run_path = os.path.join(RESULTS_PATH_AL, f"{args.dataset}_{args.method}_{args.acquisition_function}_seed_{seed}_{now}")
os.makedirs(run_path, exist_ok=True)

# save command line arguments
formatted_args = "\n".join(f"{key}: {value}" for key, value in vars(args).items())
with open(os.path.join(run_path, "args.txt"), "w") as file:
    file.write(formatted_args)

#################
### LOAD DATA ###
#################

if args.dataset == "mnist":
    x_test_t, y_test_t = get_mnist_test()
    x_test_t = x_test_t.to(device)

    x_train_t, y_train_t = get_mnist_train()
    x_train_t = x_train_t.to(device)
    y_train_t = y_train_t.to(device)
elif args.dataset == "fmnist":
    x_test_t, y_test_t = get_fmnist_test()
    x_test_t = x_test_t.to(device)

    x_train_t, y_train_t = get_fmnist_train()
    x_train_t = x_train_t.to(device)
    y_train_t = y_train_t.to(device)
else:
    raise NotImplementedError("Dataset not implemented")

rng = np.random.default_rng(seed=42) # dataset split constant over runs
val_inds = rng.choice(np.arange(len(x_train_t)), size=len(x_train_t) // 10, replace=False)
train_inds = torch.as_tensor(np.delete(np.arange(len(x_train_t)), val_inds))

full_train_ds = TensorDataset(x_train_t[train_inds], y_train_t[train_inds])
val_ds = TensorDataset(x_train_t[val_inds], y_train_t[val_inds])

#################
### MAIN LOOP ###
#################

test_perfs = list()
# seeding
fix_seeds(seed)

for i in range(args.n_iterations):

    print(f"Iteration {i + 1} / {args.n_iterations}")

    if i == 0:
        indices = list()
        for cls in range(10):
            class_indices = (full_train_ds.targets == cls).nonzero(as_tuple=True)[0].cpu()
            random_indices = class_indices[torch.randperm(len(class_indices))[:args.start_samples_per_class]]
            indices.append(random_indices)
        indices = torch.cat(indices, dim=0)
        remaining_indices = torch.as_tensor(np.delete(np.arange(len(full_train_ds)), indices.numpy()))

        train_ds = TensorDataset(full_train_ds.inputs[indices], full_train_ds.targets[indices])
        pool_ds = TensorDataset(full_train_ds.inputs[remaining_indices], full_train_ds.targets[remaining_indices])
    else:
        train_ds = TensorDataset(torch.cat((train_ds.inputs, pool_ds.inputs[acquired_samples_indices]), dim=0),
                                 torch.cat((train_ds.targets, pool_ds.targets[acquired_samples_indices]), dim=0))
        remaining_indices = torch.as_tensor(np.delete(np.arange(len(pool_ds)), acquired_samples_indices.cpu().numpy()))
        pool_ds = TensorDataset(pool_ds.inputs[remaining_indices], pool_ds.targets[remaining_indices])

    print(len(train_ds), len(pool_ds))

    # create dataloader for training
    train_loader = DataLoader(UpsampleDataset(train_ds, upsample=final_train_size),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # create tensor for pool
    x_pool_t = pool_ds.inputs.to(device)

    if args.method == "ensemble":
        results = predict_ensemble(train_loader, 
                                   val_loader, 
                                   torch.cat((x_pool_t, x_test_t), dim=0),
                                   args.n_samples, 
                                   args.lr, 
                                   args.epochs, 
                                   args.weight_decay, 
                                   device)
    elif args.method == "mc_dropout":
        results = predict_mc_dropout(train_loader, 
                                     val_loader, 
                                     torch.cat((x_pool_t, x_test_t), dim=0), 
                                     args.n_samples, 
                                     args.p_drop, args.lr, 
                                     args.epochs, 
                                     args.weight_decay, 
                                     device)
    else:
        raise NotImplementedError()
    
    # get indices of samples to be acquired according to acquisition function
    pool_preds = results["preds"][:len(pool_ds)]

    # calculate uncertainties
    if args.acquisition_function != "random":
        uncertainties = calculate_uncertainties(pool_preds, pool_preds)
        
    if args.acquisition_function == "tu_bc3":
        uncertainties = uncertainties["C3"][0]
    elif args.acquisition_function == "tu_bc2_au_b":
        uncertainties = uncertainties["C2"][0]
    elif args.acquisition_function == "au_c":
        uncertainties = uncertainties["C3"][1]
    elif args.acquisition_function == "eu_c3":
        uncertainties = uncertainties["C3"][2]
    elif args.acquisition_function == "eu_c2":
        uncertainties = uncertainties["C2"][2]
    elif args.acquisition_function == "eu_b3":
        uncertainties = uncertainties["B3"][2]
    elif args.acquisition_function == "random":
        uncertainties = torch.rand(size=(len(pool_preds), ))
    else:
        raise NotImplementedError("acquisition function not implemented")

    _, acquired_samples_indices = torch.topk(uncertainties, k=args.n_samples_per_iteration, dim=0, largest=True, sorted=False)

    test_preds = results["preds"][-len(x_test_t):].cpu()
    test_perf = (test_preds.mean(dim=1).argmax(dim=1) == y_test_t).float().mean().item() * 100
    print(f"test performance: {test_perf:.02f}%")
    test_perfs.append(test_perf)
    
    # save test preds and state dicts
    torch.save(test_preds.to(dtype=torch.float16), os.path.join(run_path, f"preds_{i}.pt"))
    torch.save(results["state_dicts"], os.path.join(run_path, f"state_dicts_{i}.pt"))

# save test performances as text file
with open(os.path.join(run_path, "test_perfs.txt"), "w") as file:
    file.write("\n".join(str(perf) for perf in test_perfs))
