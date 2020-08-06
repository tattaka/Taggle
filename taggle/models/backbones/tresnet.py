from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from .tresnet_utils import (
    ABN,
    AntiAliasDownsampleLayer,
    FastGlobalAvgPool2d,
    SEModule,
    SpaceToDepthModule,
    url_map,
    url_map_448
)

try:
    from inplace_abn import InPlaceABN
    inplace_enable = True
except ImportError:
    inplace_enable = False


def IABN2Float(module: nn.Module) -> nn.Module:
    "If `module` is IABN don't use half precision."
    if isinstance(module, InPlaceABN):
        module.float()
    for child in module.children():
        IABN2Float(child)
    return module


def conv2d_ABN(ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1, inplace=True):
    if inplace and inplace_enable:
        inplace_abn = InPlaceABN(
            num_features=nf, activation=activation, activation_param=activation_param)
    elif inplace_enable:
        raise NotImplementedError
    else:
        inplace_abn = ABN(num_features=nf, activation=activation,
                          activation_param=activation_param)
    return nn.Sequential(
        nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                  bias=False),
        inplace_abn
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None, inplace=True):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = conv2d_ABN(
                inplanes, planes, stride=1, activation_param=1e-3, inplace=inplace)
        else:
            if anti_alias_layer is None:
                self.conv1 = conv2d_ABN(
                    inplanes, planes, stride=2, activation_param=1e-3, inplace=inplace)
            else:
                self.conv1 = nn.Sequential(conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = conv2d_ABN(
            planes, planes, stride=1, activation="identity", inplace=inplace)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion,
                           reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None, inplace=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3, inplace=inplace)
        if stride == 1:
            self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3, inplace=inplace)
        else:
            if anti_alias_layer is None:
                self.conv2 = conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3, inplace=inplace)
            else:
                self.conv2 = nn.Sequential(conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3, inplace=inplace),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity", inplace=inplace)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)

        out = self.conv3(out)
        out = out + residual  # no inplace
        out = self.relu(out)

        return out


class TResNet(nn.Module):

    def __init__(self, layers, num_classes=1000, width_factor=1.0, remove_aa_jit=False, inplace=True):
        super(TResNet, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = partial(
            AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = conv2d_ABN(48, self.planes, stride=1,
                           kernel_size=3, inplace=inplace)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer, inplace=inplace)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer, inplace=inplace)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer, inplace=inplace)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer, inplace=inplace)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict(
            [('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, InPlaceABN):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(
                    torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(
                    torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None, inplace=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2,
                                           ceil_mode=True, count_include_pad=False))
            layers += [conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity", inplace=inplace)]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer, inplace=inplace))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer, inplace=inplace))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits


class TResNetBackbone(TResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.head
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.body._modules['SpaceToDepth'](x)
        x0 = self.body._modules['conv1'](x0)
        # print(x0.size())
        x1 = self.body._modules['layer1'](x0)
        # print(x1.size())
        x2 = self.body._modules['layer2'](x1)
        # print(x2.size())
        x3 = self.body._modules['layer3'](x2)
        # print(x3.size())
        x4 = self.body._modules['layer4'](x3)
        # print(x4.size())
        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = state_dict["model"]
        # # "*filt"
        # pop_keys = [k for k in state_dict.keys() if 'filt' in k]
        # for k in pop_keys:
        #     state_dict.pop(k)
        if 'head.fc.bias' in state_dict and 'head.fc.weight' in state_dict:
            state_dict.pop('head.fc.bias')
            state_dict.pop('head.fc.weight')
        super().load_state_dict(state_dict, **kwargs)


def _get_pretrained_settings(backbone, type="default"):
    if type == "default":
        urls = url_map
    elif type == "448":
        urls = url_map_448
    pretrained_settings = {
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'url': urls[backbone],
            'input_space': 'RGB',
            'input_range': [0, 1]
        }
    }
    return pretrained_settings


tresnet_backbones = {
    'tresnet-m': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-m'),
        'out_shapes': (2048, 1024, 128, 64, 64),
        'params': {
            'layers': [3, 4, 11, 3],
            'remove_aa_jit': True,
            'inplace': False
        },
    },
    'tresnet-l': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-l'),
        'out_shapes': (2432, 1216, 152, 76, 76),
        'params': {
            'layers': [4, 5, 18, 3],
            'width_factor': 1.2,
            'remove_aa_jit': True,
            'inplace': False
        },
    },
    'tresnet-xl': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-xl'),
        'out_shapes': (2656, 1328, 166, 83, 83),
        'params': {
            'layers': [4, 5, 24, 3],
            'width_factor': 1.3,
            'remove_aa_jit': True,
            'inplace': False
        },
    },
    'tresnet-m-448': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-m', '448'),
        'out_shapes': (2048, 1024, 128, 64, 64),
        'params': {
            'layers': [3, 4, 11, 3],
            'remove_aa_jit': True,
            'inplace': False
        },
    },
    'tresnet-l-448': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-l', '448'),
        'out_shapes': (2432, 1216, 152, 76, 76),
        'params': {
            'layers': [4, 5, 18, 3],
            'width_factor': 1.2,
            'remove_aa_jit': True,
            'inplace': False
        },
    },
    'tresnet-xl-448': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-xl', '448'),
        'out_shapes': (2656, 1328, 166, 83, 83),
        'params': {
            'layers': [4, 5, 24, 3],
            'width_factor': 1.3,
            'remove_aa_jit': True,
            'inplace': False
        },
    },
    'tresnet-m-inplace': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-m'),
        'out_shapes': (2048, 1024, 128, 64, 64),
        'params': {
            'layers': [3, 4, 11, 3],
            'remove_aa_jit': True,
        },
    },
    'tresnet-l-inplace': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-l'),
        'out_shapes': (2432, 1216, 152, 76, 76),
        'params': {
            'layers': [4, 5, 18, 3],
            'width_factor': 1.2,
            'remove_aa_jit': True
        },
    },
    'tresnet-xl-inplace': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-xl'),
        'out_shapes': (2656, 1328, 166, 83, 83),
        'params': {
            'layers': [4, 5, 24, 3],
            'width_factor': 1.3,
            'remove_aa_jit': True
        },
    },
    'tresnet-m-inplace-448': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-m', '448'),
        'out_shapes': (2048, 1024, 128, 64, 64),
        'params': {
            'layers': [3, 4, 11, 3],
            'remove_aa_jit': True
        },
    },
    'tresnet-l-inplace-448': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-l', '448'),
        'out_shapes': (2432, 1216, 152, 76, 76),
        'params': {
            'layers': [4, 5, 18, 3],
            'width_factor': 1.2,
            'remove_aa_jit': True
        },
    },
    'tresnet-xl-inplace-448': {
        'backbone': TResNetBackbone,
        'pretrained_settings': _get_pretrained_settings('tresnet-xl', '448'),
        'out_shapes': (2656, 1328, 166, 83, 83),
        'params': {
            'layers': [4, 5, 24, 3],
            'width_factor': 1.3,
            'remove_aa_jit': True
        },
    },
}
