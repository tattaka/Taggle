from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class SpatiallyAttentiveOutputHead(nn.Module):
    __name__ = 'SpatiallyAttentiveOutputHead'

    def __init__(self, encoder_channels, mid_channel=512, p=0.2, last_activation=None, num_class=10):
        super(SpatiallyAttentiveOutputHead, self).__init__()
        self.sa_layers = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], mid_channel, 3, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, 1, 3, 1, 1))
        self.interpolate = partial(
            F.interpolate, mode='bilinear', align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv2d(encoder_channels[-1], mid_channel, 3, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(
            nn.Conv2d(encoder_channels[-2], mid_channel, 3, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),)
        self.conv3 = nn.Sequential(
            nn.Conv2d(encoder_channels[-3], mid_channel, 3, 1, 1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),)
        self.conv4 = nn.Conv2d(mid_channel * 3, num_class, 3, 1, 1)
        self.softmax2d = nn.Softmax(2)

        if last_activation is None:
            self.activation = last_activation
        elif last_activation == 'LogSoftmax':
            self.activation = nn.LogSoftmax(dim=1)
        elif last_activation == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        elif last_activation == 'Sigmoid':
            self.activation = nn.Sigmoid(dim=1)
        else:
            raise ValueError(
                'Activation should be "LogSoftmax"/"Softmax"/"Sigmoid"/None')

    def forward(self, feats):
        _, _, h, w = feats[-1].size()
        spatial_attention_map = self.sa_layers(feats[-1])
        spatial_attention_map = self.softmax2d(spatial_attention_map.view(
            *spatial_attention_map.size()[:2], -1)).view_as(spatial_attention_map)
        feat1 = self.conv1(feats[-1])
        feat2 = self.conv2(self.interpolate(feats[-2], size=(h, w)))
        feat3 = self.conv3(self.interpolate(feats[-3], size=(h, w)))
        spatial_logits = self.conv4(torch.cat([feat1, feat2, feat3], dim=1))
        logits = (spatial_attention_map * spatial_logits).sum(axis=(-1, -2))

        if self.activation:
            logits = self.activation(logits)

        return logits
