import torch
import torch.nn.functional as F
from torch import nn


class SeparableConv2d(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size,
                               stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512):
        super(JPU, self).__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[0], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3 * width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3 * width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3 * width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3 * width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[0]), self.conv4(
            inputs[1]), self.conv3(inputs[2])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], size=(h, w),
                                  mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], size=(h, w),
                                  mode='bilinear', align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(
            feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
        return feat
