from .fastfcn import FastFCNHead, FastFCNImproveHead
from .fpn import FPNHead
from .jpu import JPUHead
from .linknet import LinkNetHead
from .pspnet import PSPNetHead
from .refinenet import RefineNetHead, RefineNetPoolingImproveHead
from .saol import SpatiallyAttentiveOutputHead
from .simple import SimpleHead
from .unet import HyperColumnsHead, UNetHead

heads = {}
heads.update({SimpleHead.__name__: SimpleHead})
heads.update({JPUHead.__name__: JPUHead})
heads.update(
    {SpatiallyAttentiveOutputHead.__name__: SpatiallyAttentiveOutputHead})
heads.update({FastFCNHead.__name__: FastFCNHead})
heads.update({FastFCNImproveHead.__name__: FastFCNImproveHead})
heads.update({FPNHead.__name__: FPNHead})
heads.update({LinkNetHead.__name__: LinkNetHead})
heads.update({PSPNetHead.__name__: PSPNetHead})
heads.update({RefineNetHead.__name__: RefineNetHead})
heads.update(
    {RefineNetPoolingImproveHead.__name__: RefineNetPoolingImproveHead})
heads.update({UNetHead.__name__: UNetHead})
heads.update({HyperColumnsHead.__name__: HyperColumnsHead})


def get_head(heads, name, params, encoder_channels):
    Head = heads[name]
    if params is not None:
        head = Head(encoder_channels=encoder_channels, **params)
    return head
