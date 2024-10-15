import numpy as np

import torch
import torch.nn as nn

from sklearn.metrics import roc_curve, roc_auc_score, auc, precision_recall_curve

"""
Pytorch Modules
! These metrics should only be used on a full dataset using the evaluate method in utils/train_utils.py or similar.
! Otherwise, averaging could introduce adverse effects.
"""

class Accuracy(nn.Module):

    def forward(self, x, y):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        y = y.cpu()

        return accuracy(y_pred, y)
    
class BalancedAccuracy(nn.Module):

    def forward(self, x, y):

        y_pred = torch.softmax(x, dim=1).argmax(dim=1).cpu()
        y = y.cpu()

        return balanced_accuracy(y_pred, y)

class AUROC(nn.Module):

    def forward(self, x, y):

        y_probs = torch.softmax(x, dim=1)[:, 1].cpu()
        y = y.cpu()

        return auroc(y_probs, y)
    
class NLL(nn.Module):

    def forward(self, x, y):
        nll = list()
        for i in range(len(x)):
            nll.append(- torch.log(x[i, y[i]]))
        return torch.mean(torch.as_tensor(nll))
    
"""
Functions
"""

def accuracy(y_preds, ys):
    return torch.mean((y_preds == ys).float())

def nll(y_probs, ys, eps=0):
    return -torch.mean(torch.log(y_probs[range(len(y_probs)), ys] + eps))

def balanced_accuracy(y_preds, ys):
    # True Positives
    TP = torch.sum((ys == 1) & (y_preds == 1)).float()
    # True Negatives
    TN = torch.sum((ys == 0) & (y_preds == 0)).float()
    # False Positives
    FP = torch.sum((ys == 0) & (y_preds == 1)).float()
    # False Negatives
    FN = torch.sum((ys == 1) & (y_preds == 0)).float()

    # Sensitivity (True Positive Rate)
    TPR = TP / (TP + FN)
    # Specificity (True Negative Rate)
    TNR = TN / (TN + FP)

    return 0.5 * (TPR + TNR)

def auroc(scores, ys):
    return torch.as_tensor(roc_auc_score(ys, scores))

def aupr(scores, ys):
    precision, recall, _ = precision_recall_curve(ys.numpy(), scores.numpy())
    return torch.as_tensor(auc(recall, precision))

def fpr_at_tpr_x(scores, ys, x=0.95):
    fpr, tpr, _ = roc_curve(ys.numpy(), scores.numpy())
    return torch.as_tensor(fpr[np.argmin(np.abs(tpr - x))])
