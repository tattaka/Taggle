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

    def __init__(self, in_features, out_features):
        super(ArcMarginProduct, self).__init__()
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, feats):
        # cosine = F.linear(F.normalize(feats), F.normalize(self.weight.cuda()))
        cosine = F.linear(F.normalize(feats), F.normalize(self.weight.to(feats.device)))
        return cosine
