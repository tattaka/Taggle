import torch
from torch import nn

from ..sync_batchnorm import DataParallelWithCallback


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def convert_model_ReLU2Swish(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model_ReLU2Swish(mod)
        mod = DataParallelWithCallback(mod)
        return mod

    mod = module
    if isinstance(module, torch.nn.ReLU):
        mod = Swish()
    for name, child in module.named_children():
        mod.add_module(name, convert_model_ReLU2Swish(child))
    return mod
