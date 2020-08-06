from torch import nn


class Flatten(nn.Module):
    """
    Simple class for flattening layer.
    """

    def forward(self, x):
        return x.view(x.size()[0], -1)
