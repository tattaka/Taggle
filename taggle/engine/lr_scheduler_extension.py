import os

import requests
from functools import partial

from .base_engine import BaseEngine
from .base_extension import BaseExtension

from torch.optim.lr_scheduler import (
    LambdaLR, 
    MultiplicativeLR, 
    StepLR, 
    MultiStepLR, 
    ExponentialLR, 
    CosineAnnealingLR, 
    ReduceLROnPlateau, 
    CyclicLR, 
    OneCycleLR,
    CosineAnnealingWarmRestarts
)

class LRSchedulerExtension(BaseExtension):
    def __init__(self, scheduler_type, key="default", **kwargs):
        
        self.key = key
        self.scheduler_type = scheduler_type
        
        if scheduler_type == "LambdaLR":
            self.scheduler = partial(LambdaLR, **kwargs)
        elif scheduler_type == "MultiplicativeLR":
            self.scheduler = partial(MultiplicativeLR, **kwargs)
        elif scheduler_type == "StepLR":
            self.scheduler = partial(StepLR, **kwargs)
        elif scheduler_type == "MultiStepLR":
            self.scheduler = partial(MultiStepLR, **kwargs)
        elif scheduler_type == "ExponentialLR":
            self.scheduler = partial(ExponentialLR, **kwargs)
        elif scheduler_type == "CosineAnnealingLR":
            self.scheduler = partial(CosineAnnealingLR, **kwargs)
        elif scheduler_type == "ReduceLROnPlateau":
            self.scheduler = partial(ReduceLROnPlateau, **kwargs)
        elif scheduler_type == "CyclicLR":
            self.scheduler = partial(CyclicLR, **kwargs)
        elif scheduler_type == "OneCycleLR":
            self.scheduler = partial(OneCycleLR, **kwargs)
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            self.scheduler = partial(CosineAnnealingWarmRestarts, **kwargs)
        else:
            raise NotImplementedError            

    def on_initialize(self, engine: BaseEngine):
        if engine.schedulers is None:
            engine.schedulers = {}
        engine.schedulers.update({self.key: self.scheduler})
        engine.schedulers[self.key] = engine.schedulers[self.key](engine.optimizers[self.key])
        if engine.init_checkpoints:
            engine.schedulers[self.key].load_state_dict(engine.init_checkpoints[self.key + "_scheduler_state_dict"])

    def on_epoch_end(self, engine: BaseEngine):
        if self.scheduler_type in ("LambdaLR", "MultiplicativeLR", "StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR"):
            engine.schedulers[self.key].step()
        elif self.scheduler_type == "ReduceLROnPlateau":
            engine.schedulers[self.key].step(engine.valid_results["loss"])

    def on_batch_end(self, engine: BaseEngine):
        if engine.mode == "train":
            if self.scheduler_type in ("CyclicLR", "OneCycleLR"):
                engine.schedulers[self.key].step()
            elif self.scheduler_type == "CosineAnnealingWarmRestarts":
                engine.schedulers[self.key].step(engine.iterator / len(engine.train_loader))

