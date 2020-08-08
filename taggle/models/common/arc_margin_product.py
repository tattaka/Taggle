import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features, out_features, extract_feature=False):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        self.extract_feature = extract_feature
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feats):
        # cosine = F.linear(F.normalize(feats), F.normalize(self.weight.cuda()))
        if self.extract_feature:
            return feats
        else:
            cosine = F.linear(F.normalize(feats), F.normalize(self.weight.to(feats.device)))
            return cosine
