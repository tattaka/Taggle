import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import ASPP, JPU, Conv2dReLU


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3,
                       padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3,
                       padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        x = self.block(x)
        return x


class FastFCNHead(nn.Module):
    __name__ = 'FastFCNHead'

    def __init__(
            self,
            encoder_channels,
            num_class=1,
            use_batchnorm=True,
            mid_channel=128,
    ):
        super().__init__()
        self.jpu = JPU([encoder_channels[-1], encoder_channels[-2],
                        encoder_channels[-3]], mid_channel)

        self.aspp = ASPP(mid_channel * 4, mid_channel,
                         dilations=[1, (1, 4), (2, 8), (3, 12)])

        self.final_conv = nn.Conv2d(mid_channel, num_class, kernel_size=(1, 1))

    def forward(self, x):
        x = self.jpu(x[-1], x[-2], x[-3])
        x = self.aspp(x)
        x = nn.functional.interpolate(
            x, scale_factor=8, mode='bilinear', align_corners=True)
        x = self.final_conv(x)
        return x


class FastFCNImproveHead(nn.Module):
    __name__ = 'FastFCNImproveHead'

    def __init__(
            self,
            encoder_channels,
            num_class=1,
            use_batchnorm=True,
            mid_channel=128,
    ):
        super().__init__()
        self.jpu = JPU([encoder_channels[-1], encoder_channels[-2],
                        encoder_channels[-3]], mid_channel)

        self.aspp = ASPP(mid_channel * 4, mid_channel,
                         dilations=[1, (1, 4), (2, 8), (3, 12)])

        self.decoder1 = DecoderBlock(encoder_channels[-4] + mid_channel, 128)
        self.decoder2 = DecoderBlock(encoder_channels[-5] + 128, 64)
        self.decoder3 = DecoderBlock(64, 32)

        self.final_conv = nn.Conv2d(32, num_class, kernel_size=(1, 1))

    def forward(self, x):
        skips = x
        x = self.jpu(skips[-1], skips[-2], skips[-3])
        x = self.aspp(x)
        x = self.decoder1([x, skips[-4]])
        x = self.decoder2([x, skips[-5]])
        x = self.decoder3([x, None])
        x = self.final_conv(x)
        return x
