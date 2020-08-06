import torch
import torch.nn.functional as F
from torch import nn

from ..sync_batchnorm import DataParallelWithCallback


class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x))
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))


class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)


def convert_model_ReLU2Mish(module):
    if isinstance(module, torch.nn.DataParallel):
        mod = module.module
        mod = convert_model_ReLU2Mish(mod)
        mod = DataParallelWithCallback(mod)
        return mod

    mod = module
    if isinstance(module, torch.nn.ReLU):
        mod = Mish()
    for name, child in module.named_children():
        mod.add_module(name, convert_model_ReLU2Mish(child))
    return mod
