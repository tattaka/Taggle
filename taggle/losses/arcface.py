# https://github.com/BG2CRW/top_k_optimization/blob/master/losses/ArcFace.py
import math

import torch
from torch import nn


class ArcFaceLoss(nn.Module):
    __name__ = 'ArcFaceLoss'

    def __init__(self, s=30, margin=0.5, easy_margin=True, reduction="mean"):
        super().__init__()
        self.m = margin
        self.s = s
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, labels):
        sine = torch.sqrt(1.0 - torch.pow(logits, 2))
        phi = logits * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(logits > 0, phi, logits)
        else:
            phi = torch.where(logits > self.th, phi, logits - self.mm)
        one_hot = torch.zeros(logits.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * logits)
        output *= self.s
        return self.criterion(output, labels)
