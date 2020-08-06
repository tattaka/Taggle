from torch import nn

from ..utils import metric_functions as mf


class DiceLoss(nn.Module):
    __name__ = 'DiceLoss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - mf.f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)
