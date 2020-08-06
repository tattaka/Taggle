import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import (
    AdaptiveConcatPool2d,
    CBAMModule,
    Conv2dReLU,
    Flatten,
    SCSEModule
)


class CenterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None, upsample=True):
        super().__init__()
        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=1,
                       use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=1,
                       use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        return self.block(x)


class FPA(nn.Module):
    def __init__(self, channels):
        """
        Feature Pyramid Attention
        https://github.com/JaveyWang/Pyramid-Attention-Networks-pytorch/blob/master/networks.py
        :type channels: int
        """
        super(FPA, self).__init__()
        channels_mid = int(channels / 4)

        self.channels_cond = channels

        # Master branch
        self.conv_master = nn.Conv2d(
            self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_master = nn.BatchNorm2d(channels)

        # Global pooling branch
        self.conv_gpb = nn.Conv2d(
            self.channels_cond, channels, kernel_size=1, bias=False)
        self.bn_gpb = nn.BatchNorm2d(channels)

        # C333 because of the shape of last feature maps is (16, 16).
        self.conv7x7_1 = nn.Conv2d(self.channels_cond, channels_mid, kernel_size=(
            7, 7), stride=2, padding=3, bias=False)
        self.bn1_1 = nn.BatchNorm2d(channels_mid)
        self.conv5x5_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(
            5, 5), stride=2, padding=2, bias=False)
        self.bn2_1 = nn.BatchNorm2d(channels_mid)
        self.conv3x3_1 = nn.Conv2d(channels_mid, channels_mid, kernel_size=(
            3, 3), stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(channels_mid)

        self.conv7x7_2 = nn.Conv2d(channels_mid, channels, kernel_size=(
            7, 7), stride=1, padding=3, bias=False)
        self.bn1_2 = nn.BatchNorm2d(channels)
        self.conv5x5_2 = nn.Conv2d(channels_mid, channels, kernel_size=(
            5, 5), stride=1, padding=2, bias=False)
        self.bn2_2 = nn.BatchNorm2d(channels)
        self.conv3x3_2 = nn.Conv2d(channels_mid, channels, kernel_size=(
            3, 3), stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Master branch
        h, w = x.size(2), x.size(3)
        x_master = self.conv_master(x)
        x_master = self.bn_master(x_master)

        # Global pooling branch
        x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(
            x.shape[0], self.channels_cond, 1, 1)
        x_gpb = self.conv_gpb(x_gpb)
        x_gpb = self.bn_gpb(x_gpb)

        # Branch 1
        x1_1 = self.conv7x7_1(x)
        x1_1 = self.bn1_1(x1_1)
        x1_1 = self.relu(x1_1)
        x1_2 = self.conv7x7_2(x1_1)
        x1_2 = self.bn1_2(x1_2)
        x1_2 = self.relu(x1_2)

        # Branch 2
        x2_1 = self.conv5x5_1(x1_1)
        x2_1 = self.bn2_1(x2_1)
        x2_1 = self.relu(x2_1)
        x2_2 = self.conv5x5_2(x2_1)
        x2_2 = self.bn2_2(x2_2)
        x2_2 = self.relu(x2_2)

        # Branch 3
        x3_1 = self.conv3x3_1(x2_1)
        x3_1 = self.bn3_1(x3_1)
        x3_1 = self.relu(x3_1)
        x3_2 = self.conv3x3_2(x3_1)
        x3_2 = self.bn3_2(x3_2)
        x3_2 = self.relu(x3_2)

        # Merge branch 1 and
        x3_upsample = nn.Upsample(
            size=(h // 4, w // 4), mode='bilinear', align_corners=True)(x3_2)
        x2_merge = x2_2 + x3_upsample
        x2_upsample = nn.Upsample(
            size=(h // 2, w // 2), mode='bilinear', align_corners=True)(x2_merge)
        x1_merge = x1_2 + x2_upsample
        x_master = x_master * \
            nn.Upsample(size=(h, w), mode='bilinear',
                        align_corners=True)(x1_merge)

        out = x_master + x_gpb

        return out


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        planes = int(planes)
        self.atrous_conv = nn.Conv2d(inplanes, int(
            planes), kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        self.aspp1 = _ASPPModule(
            inplanes, mid_c, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(
            inplanes, mid_c, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(
            inplanes, mid_c, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(
            inplanes, mid_c, 3, padding=dilations[3], dilation=dilations[3])
        mid_c = int(mid_c)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(
                                                 inplanes, mid_c, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mid_c),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mid_c * 5, mid_c, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_c)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[
                           2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)
        elif attention_type == 'cbam':
            self.attention1 = CBAMModule(in_channels)
            self.attention2 = CBAMModule(out_channels)

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
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x


class UNetHead(nn.Module):
    __name__ = "UNetHead"

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            num_class=1,
            use_batchnorm=True,
            center=None,
            attention_type=None,
            classification=False,
    ):
        super().__init__()

        if center == 'normal':
            channels = encoder_channels[0]
            self.center = CenterBlock(
                channels, channels, use_batchnorm=use_batchnorm)
        elif center == 'fpa':
            channels = encoder_channels[0]
            self.center = FPA(channels)
        elif center == 'aspp':
            channels = encoder_channels[0]
            self.center = ASPP(channels, channels, dilations=[
                               1, (1, 6), (2, 12), (3, 18)])
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(
            out_channels[4], num_class, kernel_size=(1, 1))

        self.classification = classification
        if self.classification:
            self.linear_feature = nn.Sequential(
                nn.Conv2d(encoder_channels[0], 64, kernel_size=1),
                AdaptiveConcatPool2d(1),
                Flatten(),
                nn.Dropout(),
                nn.Linear(128, num_class)
            )

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)
        if self.classification:
            class_refine = self.linear_feature(encoder_head)[:, :, None, None]
            x = x * class_refine
        return x


class HyperColumnsHead(nn.Module):
    __name__ = "HyperColumnsHead"

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            num_class=1,
            use_batchnorm=True,
            center=None,
            attention_type=None
    ):
        super().__init__()

        if center == 'normal':
            channels = encoder_channels[0]
            self.center = CenterBlock(
                channels, channels, use_batchnorm=use_batchnorm)
        elif center == 'fpa':
            channels = encoder_channels[0]
            self.center = FPA(channels)
        elif center == 'aspp':
            channels = encoder_channels[0]
            self.center = ASPP(inplanes=channels,
                               mid_c=channels / 2, dilations=[1, 6, 12, 18])
        else:
            self.center = None

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.connect_conv1_2 = nn.Sequential(
            nn.Conv2d(out_channels[0], num_class, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_class))

        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv2_3 = nn.Sequential(
            nn.Conv2d(out_channels[1], num_class, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_class))

        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv3_4 = nn.Sequential(
            nn.Conv2d(out_channels[2], num_class, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_class))

        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv4_5 = nn.Sequential(
            nn.Conv2d(out_channels[3], num_class, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_class))

        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.connect_conv5_6 = nn.Sequential(
            nn.Conv2d(out_channels[4], num_class, kernel_size=(1, 1)),
            nn.BatchNorm2d(num_class))

        self.final_conv = nn.Conv2d(
            out_channels[4], num_class, kernel_size=(1, 1))

        self.hc_conv = nn.Sequential(
            Conv2dReLU(6 * num_class, out_channels[4], kernel_size=3,
                       padding=1, stride=1, use_batchnorm=use_batchnorm),
            nn.Conv2d(out_channels[4], num_class, kernel_size=(1, 1)))

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        xc = self.connect_conv1_2(F.interpolate(
            x, scale_factor=16, mode='nearest'))
        x = self.layer2([x, skips[1]])
        xc = torch.cat((xc, self.connect_conv2_3(
            F.interpolate(x, scale_factor=8, mode='nearest'))), 1)
        x = self.layer3([x, skips[2]])
        xc = torch.cat((xc, self.connect_conv3_4(
            F.interpolate(x, scale_factor=4, mode='nearest'))), 1)
        x = self.layer4([x, skips[3]])
        xc = torch.cat((xc, self.connect_conv4_5(
            F.interpolate(x, scale_factor=2, mode='nearest'))), 1)
        x = self.layer5([x, None])
        xc = torch.cat((xc, self.connect_conv5_6(x)), 1)
        x = self.final_conv(x)
        x = torch.cat((x, xc), 1)
        x = self.hc_conv(x)

        return x
