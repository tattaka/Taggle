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
            self.activation = F.sigmoid
        elif activation == 'softmax':
            self.ce = F.cross_entropy
            self.activation = F.softmax
        else:
            raise NotImplementedError

    def forward(self, input, target):
        logit = self.activation(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = self.ce(input, target, reduction="none")
        view = target.size() + (1,)
        index = target.view(*view)
        loss = self.alpha * loss * \
            (1 - logit.gather(1, index).squeeze(1)) ** self.gamma  # focal loss

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
            self.activation = F.softmax
        else:
            raise NotImplementedError

    def forward(self, input, target):
        logit = self.activation(input, dim=1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = self.ce(input, target, reduction="none")
        view = target.size() + (1,)
        with torch.no_grad():
            self.pc = self.pc * 0.95 + logit.mean(axis=1) * 0.05
            k = self.h * self.pc + (1 - self.h)
            gamma = torch.log(1 - k) / torch.log(1 - self.pc) - 1
        index = target.view(*view)
        loss = self.alpha * loss * \
            (1 - logit.gather(1, index).squeeze(1)) ** gamma  # focal loss

        return loss.mean()
