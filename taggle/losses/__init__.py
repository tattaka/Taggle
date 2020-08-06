from torch import nn

from .adacos import AdaCosLoss
from .arcface import ArcFaceLoss
from .dice import DiceLoss
from .focal import AutoFocalLoss, FocalLoss
from .jaccard import JaccardLoss
from .lovasz import BinaryLovaszHingeLoss, SoftmaxLovaszLoss
from .ohem import OHEMLoss
from .smooth_ce import SmoothCrossEntropyLoss

losses = {}

losses.update({"CrossEntropyLoss": nn.CrossEntropyLoss})
losses.update({"MSELoss": nn.MSELoss})
losses.update({"L1Loss": nn.L1Loss})
losses.update({"BCEWithLogitsLoss": nn.BCEWithLogitsLoss})
losses.update({"SmoothL1Loss": nn.SmoothL1Loss})
losses.update({ArcFaceLoss.__name__: ArcFaceLoss})
losses.update({AdaCosLoss.__name__: AdaCosLoss})
losses.update({DiceLoss.__name__: DiceLoss})
losses.update({FocalLoss.__name__: FocalLoss})
losses.update({AutoFocalLoss.__name__: AutoFocalLoss})
losses.update({JaccardLoss.__name__: JaccardLoss})
losses.update({SoftmaxLovaszLoss.__name__: SoftmaxLovaszLoss})
losses.update({BinaryLovaszHingeLoss.__name__: BinaryLovaszHingeLoss})
losses.update({OHEMLoss.__name__: OHEMLoss})
losses.update({SmoothCrossEntropyLoss.__name__: SmoothCrossEntropyLoss})


def get_loss(name, **kwargs):
    Loss = losses[name]
    loss = Loss(**kwargs)
    return loss


def get_losses_dict(cfg: dict):
    criterions = {}
    for key in cfg:
        if cfg[key]["params"] is None:
            criterions.update({key: get_loss(cfg[key]["name"])})
        else:
            criterions.update(
                {key: get_loss(cfg[key]["name"], **cfg[key]["params"])})
    return criterions
