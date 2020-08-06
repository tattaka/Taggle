from torch import nn

from ..utils import metric_functions as mf


class JaccardLoss(nn.Module):
    __name__ = 'JaccardLoss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - mf.jaccard(y_pr, y_gt, eps=self.eps, threshold=None, activation=self.activation)
