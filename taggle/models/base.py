import torch
from torch import nn


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class BaseModel(Model):
    def __init__(self, backbone, heads: dict):
        super().__init__()
        self.heads = nn.ModuleDict()
        for key in heads:
            self.heads[key] = heads[key]
        self.initialize()
        self.backbone = backbone

    def forward(self, x):
        y = {}
        x = self.backbone(x)
        for key in self.heads:
            y.update({key: self.heads[key](x)})
        return y

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
        return x

    def freeze(self, target='backbone'):
        if target == "backbone":
            target_nn = getattr(self, target)
        else:
            target_nn = self.heads[target]
        if isinstance(target_nn, nn.Module):
            for parameter in target_nn.parameters():
                parameter.requires_grad = False
        else:
            print(f'Not found module name {target}')
            raise Exception

    def unfreeze(self, target='backbone'):
        if target == "backbone":
            target_nn = getattr(self, target)
        else:
            target_nn = self.heads[target]
        if isinstance(target_nn, nn.Module):
            for parameter in target_nn.parameters():
                parameter.requires_grad = True
        else:
            print(f'Not found module name {target}')
            raise Exception
