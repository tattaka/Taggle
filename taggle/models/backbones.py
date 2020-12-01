from functools import partial

import timm


def get_backbone(name, **kwargs):
    
    backbone = timm.create_model(name, features_only=True, **kwargs)
    return backbone

