import math

import torch
from torch import nn


class AdaCosLoss(nn.Module):
    __name__ = 'AdaCosLoss'

    def __init__(self, num_classes, m=0.50, reduction="mean"):
        super(AdaCosLoss, self).__init__()
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.classify_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, labels):
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(
                self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / logits.size(0)
            # print(B_avg)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(
                B_avg) / torch.cos(torch.min(math.pi / 4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        loss = self.classify_loss(output, labels)
        return loss
