from pretrainedmodels.models.torchvision_models import pretrained_settings
from torch import nn

from .attention_modules.bam import BAM
from .attention_modules.cbam import CBAM
from .resnet import Bottleneck, ResNet


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class CBAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM(planes, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class CBAMBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(CBAMBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM(planes * 4, 16)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out


class CBAMResNetBackbone(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        self.bam1 = BAM(64 * (kwargs["block"].expansion))
        self.bam2 = BAM(128 * (kwargs["block"].expansion))
        self.bam3 = BAM(256 * (kwargs["block"].expansion))
        del self.fc
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
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x1 = self.bam1(x1)

        x2 = self.layer2(x1)
        x2 = self.bam2(x2)
        x3 = self.layer3(x2)
        x3 = self.bam3(x3)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        if 'fc.bias' in state_dict and 'fc.weight' in state_dict:
            state_dict.pop('fc.bias')
            state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


cbam_resnet_backbones = {
    'cbam_resnet18': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': CBAMBasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },

    'cbam_resnet34': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': pretrained_settings['resnet34'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': CBAMBasicBlock,
            'layers': [3, 4, 6, 3],
        },
    },

    'cbam_resnet50': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': pretrained_settings['resnet50'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 4, 6, 3],
        },
    },

    'cbam_resnet101': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': pretrained_settings['resnet101'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 4, 23, 3],
        },
    },

    'cbam_resnet152': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': pretrained_settings['resnet152'],
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 8, 36, 3],
        },
    },

    'cbam_resnext50_32x4d': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': {
            'imagenet': {
                'url': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': Bottleneck,
            'layers': [3, 4, 6, 3],
            'groups': 32,
            'width_per_group': 4
        },
    },

    'cbam_resnext101_32x8d': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': {
            'imagenet': {
                'url': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            },
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 8
        },
    },

    'cbam_resnext101_32x16d': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': {
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 16
        },
    },

    'cbam_resnext101_32x32d': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': {
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 32
        },
    },

    'cbam_resnext101_32x48d': {
        'backbone': CBAMResNetBackbone,
        'pretrained_settings': {
            'instagram': {
                'url': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
                'input_space': 'RGB',
                'input_size': [3, 224, 224],
                'input_range': [0, 1],
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225],
                'num_classes': 1000
            }
        },
        'out_shapes': (2048, 1024, 512, 256, 64),
        'params': {
            'block': CBAMBottleneck,
            'layers': [3, 4, 23, 3],
            'groups': 32,
            'width_per_group': 48
        },
    }
}
