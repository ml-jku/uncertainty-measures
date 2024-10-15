import copy
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import Accuracy, AUROC

@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss_fn: nn.Module, opt: torch.optim.Optimizer) -> list:

    network.train()

    device = next(network.parameters()).device

    losses = list()
    for x, y in data:
        x, y = [t.to(device) for t in [x, y]]

        y_hat = network.forward(x)
        loss = loss_fn(y_hat, y)

        losses.append(loss.item())
        opt.zero_grad()
        try:
            loss.backward()
            opt.step()
        except:
            print('Exception in update step')

    return losses

@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> float:

    network.eval()

    device = next(network.parameters()).device

    y_hats, ys = list(), list()
    for x, y in data:
        x = x.to(device)

        y_hat = network.forward(x).cpu()

        y_hats.append(y_hat)
        ys.append(y)
    
    return metric(torch.concat(y_hats, dim=0), torch.concat(ys, dim=0)).item()

def fit(network: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        epochs: int, lr: float, weight_decay: float, use_adam=False, patience=20, use_auroc=False,
        verbose: bool = False):

    if use_adam:
        optimizer = torch.optim.Adam(params=network.parameters(), lr=lr, weight_decay=weight_decay)
        # do not scale the lr -> factor 1 , just to make the overall code easier
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=epochs)
    else:
        optimizer = torch.optim.SGD(params=network.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1., end_factor=0.1, total_iters=epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        ])

    loss_fn = nn.CrossEntropyLoss()
    if use_auroc:
        metric = AUROC()
    else:
        metric = Accuracy()

    train_losses, val_perfs = list(), list()

    pbar = tqdm(range(epochs))
    for ep in pbar:
        # update network
        tl = update(network=network, data=train_loader, loss_fn=loss_fn, opt=optimizer)
        train_losses.extend(tl)
        vp = evaluate(network=network, data=val_loader, metric=metric)

        if len(val_perfs) == 0 or vp > max(val_perfs):
            best_model = copy.deepcopy(network)

        val_perfs.append(vp)

        if verbose:
            tp = evaluate(network=network, data=train_loader, metric=metric)
            vl = evaluate(network=network, data=val_loader, metric=loss_fn)
            print(f"train loss: {round(np.mean(tl), 4):.4f}, "
                  f"train performance: {round(tp * 100, 2):.2f}%, "
                  f"val loss: {round(vl, 4):.4f}, "
                  f"val performance: {round(vp * 100, 2):.2f}%")
            pbar.set_description_str(desc=f"Epoch {ep+1}")
        else:
            pbar.set_description_str(desc=f"Train loss {round(np.mean(tl), 4):.4f}, " + 
                                     f"val performance: {round(vp * 100, 2):.2f}%")
            
        # stop search if no improvements for defined number of epochs
        if patience > 0 and \
            len(val_perfs) > patience and \
                np.argmax(val_perfs) < len(val_perfs) - patience:
            break
        
        scheduler.step()

    return best_model, val_perfs
