import functools

import torch.utils.model_zoo as model_zoo

from ._preprocessing import preprocess_input
from .assembledresnet import assembled_resnet_backbones
from .cbamresnet import cbam_resnet_backbones
from .densenet import densenet_backbones
from .dpn import dpn_backbones
from .efficientnet import efficient_net_backbones
from .inceptionresnetv2 import inception_backbones
from .resnest import resnest_backbones
from .resnet import resnet_backbones
from .senet import senet_backbones
from .tresnet import tresnet_backbones
from .vgg import vgg_backbones

backbones = {}
backbones.update(resnet_backbones)
backbones.update(dpn_backbones)
backbones.update(vgg_backbones)
backbones.update(senet_backbones)
backbones.update(densenet_backbones)
backbones.update(inception_backbones)
backbones.update(efficient_net_backbones)
backbones.update(cbam_resnet_backbones)
backbones.update(assembled_resnet_backbones)
backbones.update(tresnet_backbones)
backbones.update(resnest_backbones)


def get_backbone(backbones, name, backbone_weights=None):
    Backbone = backbones[name]['backbone']
    backbone = Backbone(**backbones[name]['params'])
    backbone.out_shapes = backbones[name]['out_shapes']

    if backbone_weights is not None:
        settings = backbones[name]['pretrained_settings'][backbone_weights]
        backbone.load_state_dict(model_zoo.load_url(settings['url']))

    return backbone


def get_backbone_names():
    return list(backbones.keys())


def get_preprocessing_params(backbone_name, pretrained='imagenet'):
    settings = backbones[backbone_name]['pretrained_settings']

    if pretrained not in settings.keys():
        raise ValueError(
            'Avaliable pretrained options {}'.format(settings.keys()))

    formatted_settings = {}
    formatted_settings['input_space'] = settings[pretrained].get('input_space')
    formatted_settings['input_range'] = settings[pretrained].get('input_range')
    formatted_settings['mean'] = settings[pretrained].get('mean')
    formatted_settings['std'] = settings[pretrained].get('std')
    return formatted_settings


def get_preprocessing_fn(backbone_name, pretrained='imagenet'):
    params = get_preprocessing_params(backbone_name, pretrained=pretrained)
    return functools.partial(preprocess_input, **params)
