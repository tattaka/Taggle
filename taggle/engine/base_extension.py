from .base_engine import BaseEngine


class BaseExtension(object):
    def __init__(self):
        pass

    def on_initialize(self, engine: BaseEngine):
        pass

    def on_epoch_start(self, engine: BaseEngine):
        pass

    def on_epoch_end(self, engine: BaseEngine):
        pass

    def on_batch_start(self, engine: BaseEngine):
        pass

    def on_batch_end(self, engine: BaseEngine):
        pass

    def on_train_start(self, engine: BaseEngine):
        pass

    def on_train_end(self, engine: BaseEngine):
        pass
