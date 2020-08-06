import torch
import torch.nn as nn

from .smooth_ce import SmoothCrossEntropyLoss


class OHEMLoss(nn.Module):
    # https://www.kaggle.com/c/bengaliai-cv19/discussion/128637
    __name__ = 'OHEMLoss'

    def __init__(self, rate=0.7, smooth=0.1, criterion=None):
        super(OHEMLoss, self).__init__()
        self.rate = rate
        if criterion is None:
            self.criterion = SmoothCrossEntropyLoss(
                reduction="none", smoothing=smooth)
        else:
            self.criterion = criterion

    def forward(self, input, target):
        batch_size = input.size(0)
        ohem_cls_loss = self.criterion.forward(input, target)

        sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
        keep_num = min(sorted_ohem_loss.size()[0], int(batch_size * self.rate))
        if keep_num < sorted_ohem_loss.size()[0]:
            keep_idx_cuda = idx[:keep_num]
            ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
        loss = ohem_cls_loss.sum() / keep_num
        return loss
