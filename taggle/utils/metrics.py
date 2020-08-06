import torch.nn as nn

from . import metric_functions as mf


class IoUMetric(nn.Module):

    __name__ = 'iou'

    def __init__(self, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold

    def forward(self, y_pr, y_gt):
        return mf.iou(y_pr, y_gt, self.eps, self.threshold, self.activation)


class FscoreMetric(nn.Module):

    __name__ = 'f-score'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps
        self.threshold = threshold
        self.beta = beta

    def forward(self, y_pr, y_gt):
        return mf.f_score(y_pr, y_gt, self.beta, self.eps, self.threshold, self.activation)


class AccucacyMetric(nn.Module):

    __name__ = 'accuracy'

    def __init__(self, topk=(1,)):
        self.topk = topk

    def forward(self, y_pr, y_gt):
        return mf.accuracy(y_pr, y_gt, self.topk)
