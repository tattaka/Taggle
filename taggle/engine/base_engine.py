import copy
import os
import re
from typing import Dict, List, Union

import numpy as np
import torch
from torch import nn
from torch._six import container_abcs, int_classes, string_classes
from tqdm import tqdm

from ..utils import helper_functions

try:
    from apex import amp
    amp_enable = True
except ImportError:
    print("Not installed apex")
    amp_enable = False


class BaseEngine(object):

    def __init__(self,
                 models: Union[nn.Module, Dict[str, nn.Module]],
                 optimizers: Union[torch.optim.Optimizer, Dict[str, torch.optim.Optimizer]],
                 schedulers: Union[torch.optim.lr_scheduler._LRScheduler, Dict[str, torch.optim.lr_scheduler._LRScheduler]],
                 criterions: dict,
                 output_dir: str,
                 save_metrics: List[str],
                 save_losses: List[str],
                 train_loader,
                 valid_loader,
                 device_ids: Union[int, List[int]] = None,
                 init_epoch: int = 0,
                 save_interval: int = 1,
                 accumulation_steps: int = 1,
                 use_amp: bool = False,
                 opt_level: str = "O2",
                 weights_path: str = None,
                 apply_fn: dict = None,
                 calc_train_metrics: bool = True,
                 calc_metrics_mode: str = "epoch",
                 requierd_eval_data: List[str] = None,
                 extensions: list = None,
                 static_data=None):
        '''
        If optimizer, criterion, model, and scheduler are not dict,
        they are converted to dict inside engine and default keys are assigned.
        '''
        self.models = helper_functions.input2dict(models, "default")
        self.schedulers = helper_functions.input2dict(schedulers, "default")
        self.optimizers = helper_functions.input2dict(optimizers, "default")
        self.criterions = helper_functions.input2dict(criterions, "default")
        self.static_data = static_data
        if device_ids is None:
            self.device_ids = list(range(torch.cuda.device_count()))
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ids = helper_functions.input2list(device_ids)
            self.device = torch.device(f'cuda:{self.device_ids[0]}')

        if weights_path:
            checkpoints = torch.load(weights_path)
        else:
            checkpoints = None

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.output_dir = output_dir
        self.epoch = init_epoch
        self.max_epoch = None
        self.iterator = init_epoch * len(self.train_loader)
        self.save_metrics = helper_functions.input2list(save_metrics)
        self.save_losses = helper_functions.input2list(save_losses)
        if "loss" in self.save_losses:
            self.save_losses.append("loss")
        self.requierd_eval_data = helper_functions.input2list(
            requierd_eval_data)
        self.save_interval = save_interval

        self.use_amp = use_amp

        if self.optimizers is not None:
            self.initialize_models(opt_level, checkpoints, apply_fn)
            self.initialize_optimizers(opt_level, checkpoints)
        else:
            self.initialize_models(opt_level, checkpoints, apply_fn)
        self.parallel_model()
        self.mode = "train"
        self.accumulation_steps = accumulation_steps
        if self.accumulation_steps <= 0:
            self.accumulation_steps = 1
        self.accumulation_counter = 0
        self.calc_train_metrics = calc_train_metrics
        self.calc_metrics_mode = calc_metrics_mode
        self.initialize_logger()
        self.extensions = helper_functions.input2list(extensions)
        self.run_extensions("on_initialize")

    def initialize_models(self, opt_level, checkpoints, apply_fn):
        for key in self.models:
            if checkpoints:
                self.models[key].load_state_dict(
                    checkpoints[key + "_model_state_dict"])
            self.models[key] = self.cuda(self.models[key], self.device)
            if self.use_amp and amp_enable and self.optimizers is None:
                self.models[key], _ = amp.initialize(
                    self.models[key], None, opt_level=opt_level)
            if apply_fn and apply_fn[key]:
                self.models[key].apply(apply_fn[key])

    def parallel_model(self):
        if len(self.device_ids) > 1:
            for key in self.models:
                self.models[key] = nn.DataParallel(
                    self.models[key], device_ids=self.device_ids)

    def initialize_optimizers(self, opt_level, checkpoints):
        for key in self.optimizers:
            if self.schedulers is not None:
                self.schedulers[key] = self.schedulers[key](
                    self.optimizers[key])
            if self.use_amp and amp_enable:
                self.models[key], self.optimizers[key] = amp.initialize(
                    self.models[key], self.optimizers[key], opt_level=opt_level)
            if checkpoints:
                self.optimizers[key].load_state_dict(
                    checkpoints[key + "_optimizer_state_dict"])
                self.schedulers[key].load_state_dict(
                    checkpoints[key + "_scheduler_state_dict"])

    def initialize_logger(self):
        self.train_results = {}
        self.valid_results = {}
        self.lr = {}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(os.path.join(self.output_dir, "checkpoints")):
            os.makedirs(os.path.join(self.output_dir, "checkpoints"))

        for metric in self.save_metrics:
            self.valid_results["best_" + metric] = -float("inf")
        self.valid_results["best_loss"] = float("inf")
        self.result_info = ""

    def run(self):
        self.run_extensions("on_epoch_start")
        for key in self.models:
            self.models[key].train()
        if self.schedulers is not None:
            for key in self.schedulers:
                self.schedulers[key].step()

        losses, metrics, outputs = self.epoch_stepper()

        for key in losses:
            self.train_results[key] = losses[key]
        for key in metrics:
            self.train_results[key] = metrics[key]

        losses, metrics, outputs = self.epoch_validator()
        for key in losses:
            self.valid_results[key] = losses[key]
        for key in metrics:
            self.valid_results[key] = metrics[key]

        for metric in self.save_metrics:
            if self.valid_results["best_" + metric] <= self.valid_results[metric]:
                self.valid_results["best_"
                                   + metric] = self.valid_results[metric]
                self.save_checkpoint(metric=metric)
        if self.valid_results["best_loss"] >= self.valid_results["loss"]:
            self.valid_results["best_loss"] = self.valid_results["loss"]
            self.save_checkpoint(metric="loss")
        if ((self.epoch + 1) % self.save_interval == 0) and self.epoch != 0 and self.save_interval > 0:
            self.save_checkpoint()
        self.lr = {}
        for key in self.optimizers:
            if key != "default":
                self.lr[key + "_lr"] = [group["lr"]
                                        for group in self.optimizers[key].param_groups][0]
            else:
                self.lr["lr"] = [group["lr"]
                                 for group in self.optimizers[key].param_groups][0]
        self.result_info = ""
        for result_key, result_value in zip(self.valid_results.keys(), self.valid_results.values()):
            self.result_info = self.result_info + result_key + \
                ":" + str(round(result_value, 4)) + " "
        for lr_key, lr_value in zip(self.lr.keys(), self.lr.values()):
            self.result_info = self.result_info + \
                lr_key + ":" + str(round(lr_value, 4)) + " "
        print("Epoch %i" % self.epoch, self.result_info)

        self.run_extensions("on_epoch_end")
        self.epoch += 1

    def repeated_run(self, max_epoch):
        self.run_extensions("on_train_start")
        self.max_epoch = max_epoch
        for _ in range(max_epoch):
            self.run()
        self.run_extensions("on_train_end")

    def epoch_stepper(self):
        self.mode = "train"
        data = []
        outputs = {}
        losses = {}
        metrics = {}
        for model_key in self.models:
            self.models[model_key].train()
        with tqdm(self.train_loader, position=0, leave=True, ascii=" ##", dynamic_ncols=True) as t:
            for batch_idx, batch_data in enumerate(t):
                self.run_extensions("on_batch_start")
                self.accumulation_counter += 1
                if self.calc_train_metrics and self.calc_metrics_mode == "epoch":
                    if isinstance(batch_data, list):
                        batch_data = helper_functions.list2dict(batch_data)
                    if len(self.requierd_eval_data) == 0:
                        self.requierd_eval_data = list(batch_data.keys())
                    data.append(copy.deepcopy(
                        {key: batch_data[key] for key in self.requierd_eval_data}))
                batch_data = self.cuda(batch_data)
                batch_losses, batch_outputs = self.batch_ones(
                    batch_data)
                if self.calc_train_metrics and self.calc_metrics_mode == "batch":
                    batch_metrics = self.calc_metrics(
                        batch_outputs, batch_data)
                    for key in batch_metrics:
                        if key in metrics.keys():
                            metrics[key].append(batch_metrics[key].item())
                        else:
                            metrics[key] = [batch_metrics[key].item()]
                for key in batch_losses:
                    self.train_writer.add_scalar(
                        key + "/BatchLoss", batch_losses[key].item(), self.iterator)
                    if key in losses.keys():
                        losses[key].append(batch_losses[key].item())
                    else:
                        losses[key] = [batch_losses[key].item()]
                if self.calc_train_metrics and self.calc_metrics_mode == "epoch":
                    for key in batch_outputs:
                        if key in outputs.keys():
                            outputs[key] = torch.cat(
                                [outputs[key], batch_outputs[key].clone().detach().cpu()], dim=0)
                        else:
                            outputs[key] = batch_outputs[key].clone(
                            ).detach().cpu()
                t.set_description("Epoch %i Training" % self.epoch)
                print_losses = {}
                for key in self.save_losses:
                    print_losses[key] = batch_losses[key].item()
                t.set_postfix(ordered_dict=dict(**print_losses))
                self.run_extensions("on_batch_end")
                self.iterator += 1

        for key in losses:
            losses[key] = sum(losses[key]) / len(losses[key])
        if self.calc_train_metrics and self.calc_metrics_mode == "batch":
            for key in metrics:
                metrics[key] = sum(metrics[key]) / len(metrics[key])
        if len(losses.values()) > 1 and not ("loss" in losses.keys()):
            losses["loss"] = sum(losses.values())
        if self.calc_train_metrics and self.calc_metrics_mode == "epoch":
            data = helper_functions.concat_data(data)
            metrics = self.calc_metrics(outputs, data)
            for key in metrics:
                metrics[key] = metrics[key].item()
        return losses, metrics, outputs

    def epoch_validator(self):
        self.mode = "valid"
        data = []
        losses = {}
        outputs = {}
        metrics = {}
        for model_key in self.models:
            self.models[model_key].eval()
        with tqdm(self.valid_loader, position=0, leave=True, ascii=" ##", dynamic_ncols=True) as t:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(t):
                    self.run_extensions("on_batch_start")
                    if self.calc_metrics_mode == "epoch":
                        if isinstance(batch_data, list):
                            batch_data = helper_functions.list2dict(batch_data)
                        if len(self.requierd_eval_data) == 0:
                            self.requierd_eval_data = list(batch_data.keys)
                        data.append(copy.deepcopy(
                            {key: batch_data[key] for key in self.requierd_eval_data}))
                    batch_data = self.cuda(batch_data)
                    batch_losses, batch_outputs = self.batch_ones(
                        batch_data)
                    if self.calc_metrics_mode == "batch":
                        batch_metrics = self.calc_metrics(
                            batch_outputs, batch_data)
                        for key in batch_metrics:
                            if key in metrics.keys():
                                metrics[key].append(batch_metrics[key].item())
                            else:
                                metrics[key] = [batch_metrics[key].item()]
                    for key in batch_losses:
                        if key in losses.keys():
                            losses[key].append(batch_losses[key].item())
                        else:
                            losses[key] = [batch_losses[key].item()]
                    if self.calc_metrics_mode == "epoch":
                        for key in batch_outputs:
                            if key in outputs.keys():
                                outputs[key] = torch.cat(
                                    [outputs[key], batch_outputs[key].clone().detach().cpu()], dim=0)
                            else:
                                outputs[key] = batch_outputs[key].clone(
                                ).detach().cpu()
                    t.set_description("Epoch %i Validation" % self.epoch)
                    print_losses = {}
                    for key in self.save_losses:
                        print_losses[key] = batch_losses[key].item()
                    t.set_postfix(ordered_dict=dict(
                        **print_losses))
                    self.run_extensions("on_batch_end")

        for key in losses:
            losses[key] = sum(losses[key]) / len(losses[key])
        if self.calc_metrics_mode == "batch":
            for key in metrics:
                metrics[key] = sum(metrics[key]) / len(metrics[key])
        if len(losses.values()) > 1 and not ("loss" in losses.keys()):
            losses["loss"] = sum(losses.values())
        if self.calc_metrics_mode == "epoch":
            data = helper_functions.concat_data(data)
            metrics = self.calc_metrics(outputs, data)
            for key in metrics:
                metrics[key] = metrics[key].item()
        return losses, metrics, outputs

    def test(self):
        self.mode = "test"
        outputs = {}
        for model_key in self.models:
            self.models[model_key].eval()
        with tqdm(self.valid_loader, position=0, leave=True, ascii=" ##", dynamic_ncols=True) as t:
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(t):
                    self.run_extensions("on_batch_start")
                    batch_data = self.cuda(batch_data)
                    batch_outputs = self.forward(batch_data)
                    for key in batch_outputs:
                        if key in outputs.keys():
                            outputs[key] = np.concatenate(
                                [outputs[key], batch_outputs[key].cpu().numpy()], axis=0)
                        else:
                            outputs[key] = batch_outputs[key].cpu().numpy()
                    t.set_description("Epoch %i Test" % self.epoch)
                    self.run_extensions("on_batch_end")
        return outputs

    def save_checkpoint(self, metric=None):
        checkpoint = {}
        if metric is None:
            file_path = os.path.join(
                self.output_dir, "checkpoints", "epoch_" + str(self.epoch) + ".pth")
        else:
            file_path = os.path.join(
                self.output_dir, "checkpoints", "best_" + metric + ".pth")
            checkpoint = {
                "best_epoch": self.epoch,
                "best_" + metric: self.valid_results["best_" + metric]
            }
        for key in self.models:
            if isinstance(self.models[key], nn.DataParallel):
                checkpoint[key
                           + "_model_state_dict"] = self.models[key].module.state_dict()
                self.models[key].module.state_dict()
            else:
                checkpoint[key
                           + "_model_state_dict"] = self.models[key].state_dict()
        for key in self.optimizers:
            checkpoint[key + "_optimizer_state_dict"] = self.optimizers[key].state_dict()
        if self.schedulers is not None:
            for key in self.schedulers:
                checkpoint[key + "_scheduler_state_dict"] = self.schedulers[key].state_dict()

        torch.save(checkpoint, file_path)

    def cuda(self, x, device=None):
        np_str_obj_array_pattern = re.compile(r'[SaUO]')
        if torch.cuda.is_available():
            if isinstance(x, torch.Tensor):
                x = x.cuda(non_blocking=True, device=device)
                return x
            elif isinstance(x, nn.Module):
                x = x.cuda(device=device)
                return x
            elif isinstance(x, np.ndarray):
                if x.shape == ():
                    if np_str_obj_array_pattern.search(x.dtype.str) is not None:
                        return x
                    return self.cuda(torch.as_tensor(x), device=device)
                return self.cuda(torch.from_numpy(x), device=device)
            elif isinstance(x, float):
                return self.cuda(torch.tensor(x, dtype=torch.float64), device=device)
            elif isinstance(x, int_classes):
                return self.cuda(torch.tensor(x), device=device)
            elif isinstance(x, string_classes):
                return x
            elif isinstance(x, container_abcs.Mapping):
                return {key: self.cuda(x[key], device=device) for key in x}
            elif isinstance(x, container_abcs.Sequence):
                return [self.cuda(np.array(xi), device=device)
                        if isinstance(xi, container_abcs.Sequence) else self.cuda(xi, device=device)
                        for xi in x]

    def batch_ones(self, data):
        outputs = self.forward(data)
        losses = self.calc_losses(outputs, data)
        self.backward(losses)

        return losses, outputs

    def calc_losses(self, outputs, data):
        # user area
        # You need implementation of calicurated losses used `self.criterions`.

        # Example
        # losses = {}
        # losses["loss"] = self.criterions["default"](
        #     outputs["default"], data["label"])
        # return losses

        raise NotImplementedError

    def calc_metrics(self, outputs, data):
        # user area
        # You need implementation of calicurated metrics used your metric functions.

        # Example
        # metrics = {}
        # metrics["acc-top1"], metrics["acc-top5"] = self.accuracy(
        #     outputs["default"], data["label"], topk=(1, 5))
        # return metrics

        raise NotImplementedError

    def forward(self, data):
        # user area
        # You need implementation of forward process.

        # Example
        # outputs["default"] = self.models["default"](data["image"])
        # return outputs

        raise NotImplementedError

    def backward(self, losses):
        # user area
        # You need implementation of backward used `self.update_model` function.

        # Example
        # self.update_model(self.models["default"],
        #                   losses["loss"], self.optimizers["default"])
        raise NotImplementedError

    def update_model(self, model, loss, optimizer):
        if self.mode == "train":
            need_gradient_step = (
                self.accumulation_counter + 1) % self.accumulation_steps == 0
            model.zero_grad()
            if self.use_amp and amp_enable:
                delay_unscale = not need_gradient_step
                with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if need_gradient_step:
                optimizer.step()
                self.accumulation_counter = 0

    def run_extensions(self, stage):
        [getattr(e, stage)(self) for e in self.extensions]
