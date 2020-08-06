import torch.nn as nn
from pretrainedmodels.models.inceptionresnetv2 import (
    InceptionResNetV2,
    pretrained_settings
)


class InceptionResNetV2Backbone(InceptionResNetV2):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pretrained = False

        # correct paddings
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.kernel_size == (3, 3):
                    m.padding = (1, 1)
            if isinstance(m, nn.MaxPool2d):
                m.padding = (1, 1)

        # remove linear layers
        del self.avgpool_1a
        del self.last_linear

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x0 = x

        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x1 = x

        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x2 = x

        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x3 = x

        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        x4 = x

        features = [x4, x3, x2, x1, x0]
        return features
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_state_dict(self, state_dict, **kwargs):
        if 'last_linear.bias' in state_dict and 'last_linear.weight' in state_dict:
            state_dict.pop('last_linear.bias')
            state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)


inception_backbones = {
    'inceptionresnetv2': {
        'backbone': InceptionResNetV2Backbone,
        'pretrained_settings': pretrained_settings['inceptionresnetv2'],
        'out_shapes': (1536, 1088, 320, 192, 64),
        'params': {
            'num_classes': 1000,
        }

    }
}
