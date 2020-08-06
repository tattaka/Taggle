import torch

from .backbones import backbones, get_backbone
from .base import BaseModel
from .common import *
from .heads import get_head, heads
from .sync_batchnorm import *

######### Example Config #########
# model:
#     backbone:
#       type: resnet18
#       params:
#         backbone_weight: imagenet
#     heads:
#       output1:
#         type: SimpleHead
#         params:
#           num_class: 16
#           last_activation: None
#       output2:
#         type: SimpleHead
#         params:
#           num_class: 16
#           last_activation: None
#           metric_branch:True
#       output3:
#         type: UNetHead
#         params:
#           num_class: 16
#           last_activation: Softmax
######### Example Config #########


class ModelProvider:
    def __init__(self):
        self.backbones = backbones
        self.heads = heads

    def get_model(self, model_config: dict):
        backbone_cfg = model_config["backbone"]
        heads_cfg = model_config["heads"]
        backbone = get_backbone(
            self.backbones, backbone_cfg["type"], **backbone_cfg["params"])
        heads = {}
        for name in heads_cfg:
            heads.update({name: get_head(
                self.heads, heads_cfg[name]["type"], heads_cfg[name]["params"], backbone.out_shapes)})
        model = BaseModel(backbone, heads)
        if "mid_activation" in model_config:
            if model_config["mid_activation"] == "Mish":
                model = convert_model_ReLU2Mish(model)
            elif model_config["mid_activation"] == "Swish":
                model = convert_model_ReLU2Swish(model)
            elif model_config["mid_activation"] == "ReLU":
                pass
            else:
                raise NotImplementedError
        return model


def _test():
    import yaml
    with open("models/example.yaml", "r+") as f:
        setting = yaml.load(f)
    model_provider = ModelProvider()
    model = model_provider.get_model(setting['model'])
    x = torch.randn(2, 3, 256, 128)
    y = model(x)
    for key in y:
        print(key, y[key].size())
    return model
