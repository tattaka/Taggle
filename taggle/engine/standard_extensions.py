import os

import requests

from .base_engine import BaseEngine
from .base_extension import BaseExtension

try:
    from tensorboardX import SummaryWriter
except ImportError:
    print("Not installed tensorboardX")


class CSVLoggerExtension(BaseExtension):

    def on_initialize(self, engine: BaseEngine):
        if not os.path.exists(os.path.join(engine.output_dir, "logs")):
            os.makedirs(os.path.join(engine.output_dir, "logs"))
        if not os.path.exists(os.path.join(engine.output_dir, "logs", "train_log.csv")):
            with open(os.path.join(engine.output_dir, "logs", "train_log.csv"), mode="w") as f:
                f.write("")
        if not os.path.exists(os.path.join(engine.output_dir, "logs", "valid_log.csv")):
            with open(os.path.join(engine.output_dir, "logs", "valid_log.csv"), mode="w") as f:
                f.write("")

    def on_epoch_end(self, engine: BaseEngine):
        isfirst_write = True
        if os.path.getsize(os.path.join(engine.output_dir, "logs", "valid_log.csv")) > 0:
            isfirst_write = False
        else:
            isfirst_write = True
        with open(os.path.join(engine.output_dir, "logs", "valid_log.csv"), mode="a") as f:
            if isfirst_write:
                header = "epoch," + \
                    ",".join(map(str, engine.lr.keys())) + "," + \
                    ",".join(map(str, engine.valid_results.keys())) + "\n"
                f.write(header)
            f.write(str(engine.epoch) + "," + ",".join(map(str, engine.lr.values()))
                    + "," + ",".join(map(str, engine.valid_results.values())) + "\n")

        if os.path.getsize(os.path.join(engine.output_dir, "logs", "train_log.csv")) > 0:
            isfirst_write = False
        else:
            isfirst_write = True
        with open(os.path.join(engine.output_dir, "logs", "train_log.csv"), mode="a") as f:
            if isfirst_write:
                header = "epoch," + \
                    ",".join(map(str, engine.lr.keys())) + "," + \
                    ",".join(map(str, engine.train_results.keys())) + "\n"
                f.write(header)
            f.write(str(engine.epoch) + "," + ",".join(map(str, engine.lr.values()))
                    + "," + ",".join(map(str, engine.train_results.values())) + "\n")


class TensorBoardExtension(BaseExtension):

    def on_initialize(self, engine: BaseEngine):
        engine.train_writer = SummaryWriter(
            os.path.join(engine.output_dir, "train"))
        engine.val_writer = SummaryWriter(
            os.path.join(engine.output_dir, "valid"))

    def on_epoch_end(self, engine: BaseEngine):
        for key in engine.save_losses:
            engine.train_writer.add_scalar(
                key + "/EpochLoss", engine.train_results[key], engine.epoch)
        if engine.calc_train_metrics:
            for key in engine.save_metrics:
                engine.train_writer.add_scalar(
                    key + "/EpochMetric", engine.train_results[key], engine.epoch)
        for key in engine.save_losses:
            engine.val_writer.add_scalar(
                key + "/EpochLoss", engine.valid_results[key], engine.epoch)
        for key in engine.save_metrics:
            engine.val_writer.add_scalar(
                key + "/EpochMetric", engine.valid_results[key], engine.epoch)


class LineNotifyExtension(BaseExtension):

    def on_train_end(self, engine: BaseEngine):
        message = engine.result_info
        line_token = os.environ["LINE_TOKEN"]
        endpoint = "https://notify-api.line.me/api/notify"
        message = "\n{}".format(message)
        payload = {"message": message}
        headers = {"Authorization": "Bearer {}".format(line_token)}
        requests.post(endpoint, data=payload, headers=headers)
