from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    __name__ = 'FocalLoss'

    def __init__(self, activation='sigmoid', alpha=1, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        if activation == 'sigmoid':
            self.ce = F.binary_cross_entropy_with_logits
        elif activation == 'softmax':
            self.ce = F.cross_entropy
        else:
            raise NotImplementedError

    def forward(self, input, target):
        loss = self.ce(input, target, reduction="none")
        pt = torch.exp(-loss)
        loss = self.alpha * loss * (1 - pt) ** self.gamma  # focal loss

        return loss.mean()


class AutoFocalLoss(nn.Module):  # https://arxiv.org/abs/1904.09048
    __name__ = 'AutoFocalLoss'

    def __init__(self, activation='sigmoid', alpha=1, h=0.7, eps=1e-7):
        super(AutoFocalLoss, self).__init__()
        self.h = h
        self.alpha = alpha
        self.pc = 0
        self.eps = eps
        if activation == 'sigmoid':
            self.ce = F.binary_cross_entropy_with_logits
            self.activation = F.sigmoid
        elif activation == 'softmax':
            self.ce = F.cross_entropy
            self.activation = partial(F.softmax, dim=1)
        else:
            raise NotImplementedError

    def forward(self, input, target):
        logit = self.activation(input)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = self.ce(input, target, reduction="none")
        with torch.no_grad():
            self.pc = self.pc * 0.95 + logit.mean(axis=1) * 0.05
            k = self.h * self.pc + (1 - self.h)
            gamma = torch.log(1 - k) / torch.log(1 - self.pc) - 1
        pt = torch.exp(-loss)
        loss = self.alpha * loss * (1 - pt) ** gamma  # focal loss

        return loss.mean()

class ReducedFocalLoss(nn.Module): # https://arxiv.org/abs/1903.01347
    __name__ = 'ReducedFocalLoss'

    def __init__(self, activation='sigmoid', alpha=1, gamma=2, eps=1e-7, threshold=0.5):
        super(ReducedFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.threshold = threshold
        if activation == 'sigmoid':
            self.ce = F.binary_cross_entropy_with_logits
        elif activation == 'softmax':
            self.ce = F.cross_entropy
        else:
            raise NotImplementedError

    def forward(self, input, target):
        loss = self.ce(input, target, reduction="none")
        pt = torch.exp(-loss)
        fr = torch.where(pt < self.threshold, pt**self.gamma,  (pt / self.threshold) **self.gamma)
        loss = self.alpha * loss * fr  # focal loss

        return loss.mean()