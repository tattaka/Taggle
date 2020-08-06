# https://github.com/BG2CRW/top_k_optimization/blob/master/losses/ArcFace.py
import math

import torch
from torch import nn


class ArcFaceLoss(nn.Module):
    __name__ = 'ArcFaceLoss'

    def __init__(self, input_dim, output_dim, margin=0.5, easy_margin=True):
        super().__init__()
        self.m = margin
        self.s = 30
        self.weight = nn.Parameter(
            torch.FloatTensor(output_dim, input_dim)).cuda()
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m
        self.criterion = nn.CrossEntropyLoss()

    # input [batch, embed_size],weight [embed_size, class_number]
    def forward(self, logits, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        sine = torch.sqrt(1.0 - torch.pow(logits, 2))
        phi = logits * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(logits > 0, phi, logits)
        else:
            phi = torch.where(logits > self.th, phi, logits - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(logits.size(), device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * logits)
        output *= self.s
        return self.criterion(output, labels)
