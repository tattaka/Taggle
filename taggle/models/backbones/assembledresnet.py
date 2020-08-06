# follow https://arxiv.org/abs/2001.06268
# sk-conv -> se-conv and without biglittlenet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dropblock import DropBlock2D, LinearScheduler


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AADownsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(AADownsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)),
                          int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt)

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt[None, None, :, :].repeat((inp.shape[1], 1, 1, 1)), stride=self.stride, groups=inp.shape[1])


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dropblock_fn=None, antialias=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dropblock_fn is None:
            dropblock_fn = nn.Identity()
        self.antialias = antialias
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
            conv1x1(inplanes, planes),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            dropblock_fn,
        )

        if antialias:
            self.conv2 = nn.Sequential(
                conv3x3(planes, planes, groups=groups),
                nn.ReLU(inplace=True),
                dropblock_fn,
            )
            self.aa_downsample = nn.Sequential(
                conv1x1(planes, planes),
                norm_layer(planes),
                nn.ReLU(inplace=True),
                AADownsample(filt_size=3, stride=stride),
            )
        else:
            self.conv2 = nn.Sequential(
                conv3x3(planes, planes, groups=groups, stride=stride),
                norm_layer(planes),
                nn.ReLU(inplace=True),
                dropblock_fn,
            )
            self.aa_downsample = nn.Identity()

        self.conv3 = nn.Sequential(
            conv1x1(planes, planes),
            norm_layer(planes),
            nn.ReLU(inplace=True),
            dropblock_fn
        )
        self.se_module = SEModule(planes, reduction=16)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.aa_downsample(self.conv2(out))

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_module(out) + identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, dropblock_fn=None, antialias=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dropblock_fn is None:
            dropblock_fn = nn.Identity()
        self.antialias = antialias
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Sequential(
            conv1x1(inplanes, width),
            norm_layer(width),
            nn.ReLU(inplace=True),
            dropblock_fn,
        )

        if antialias:
            self.conv2 = nn.Sequential(
                conv3x3(width, width, groups=groups),
                nn.ReLU(inplace=True),
                dropblock_fn,
            )
            self.aa_downsample = nn.Sequential(
                conv1x1(width, width),
                norm_layer(width),
                nn.ReLU(inplace=True),
                AADownsample(filt_size=3, stride=stride),
            )
        else:
            self.conv2 = nn.Sequential(
                conv3x3(width, width, groups=groups, stride=stride),
                norm_layer(width),
                nn.ReLU(inplace=True),
                dropblock_fn,
            )
            self.aa_downsample = nn.Identity()

        self.conv3 = nn.Sequential(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion),
            nn.ReLU(inplace=True),
            dropblock_fn,
        )
        self.se_module = SEModule(planes, reduction=16)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        out = self.aa_downsample(self.conv2(out))

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
#         print(x.size(), identity.size(), out.size())
        out = self.se_module(out) + identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dropbrock = LinearScheduler(
            DropBlock2D(block_size=7, drop_prob=0.),
            start_value=0.,
            stop_value=0.1,
            nr_steps=5
        )
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes // 2, kernel_size=3,
                      stride=2, padding=1, bias=False),
            norm_layer(self.inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // 2, self.inplanes // 2,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(self.inplanes // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inplanes // 2, self.inplanes,
                      kernel_size=3, padding=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], antialias=True)

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], dropblock_fn=self.dropbrock, antialias=True)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], dropblock_fn=self.dropbrock, antialias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, dropblock_fn=None, antialias=False,):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dropblock_fn is None:
            dropblock_fn = nn.Identity()
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=stride,
                                 padding=0, ceil_mode=True),
                    conv1x1(self.inplanes, planes * block.expansion),
                    norm_layer(planes * block.expansion),
                    dropblock_fn)
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                    dropblock_fn)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, dropblock_fn=dropblock_fn, antialias=antialias))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, dropblock_fn=dropblock_fn, antialias=False))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        self.dropblock.step()
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetBackbone(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        if 'fc.bias' in state_dict and 'fc.weight' in state_dict:
            state_dict.pop('fc.bias')
            state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


assembled_resnet_backbones = {
    'aresnet18': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'aresnet34': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'aresnet50': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'aresnet101': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'aresnet152': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 8, 36, 3],
        },
    },

    'aresnext50_32x4d': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
            'groups': 32,
            'width_per_group': 4
        },
    },

    'aresnext101_32x8d': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8
        },
    },

    'aresnext101_32x16d': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 16
        },
    },

    'aresnext101_32x32d': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 32
        },
    },

    'aresnext101_32x48d': {
        'backbone': ResNetBackbone,
        'pretrained_settings': None,
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 48
        },
    },
}
