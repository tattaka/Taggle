from __future__ import print_function

import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import taggle
from taggle.engine import (
    BaseEngine,
    CSVLoggerExtension,
    LineNotifyExtension,
    TensorBoardExtension,
    LRSchedulerExtension
)
from taggle.optimizers import get_optimizer

torch.backends.cudnn.benchmark = True


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class Engine(BaseEngine):
    '''
    If optimizer, criterion, model, and scheduler are not dict,
    they are converted to dict inside engine and default keys are assigned.
    '''

    def calc_losses(self, outputs, data):
        losses = {}
        losses["loss"] = self.criterions["default"](
            outputs["default"], data[1])
        return losses

    def calc_metrics(self, outputs, data):
        metrics = {}
        metrics["acc-top1"], metrics["acc-top5"] = taggle.utils.metric_functions.accuracy(
            outputs["default"], data[1], topk=(1, 5))
        return metrics

    def forward(self, data):
        outputs = {}
        outputs["default"] = self.models["default"](data[0])
        return outputs

    def backward(self, losses):
        self.update_model(self.models["default"],
                          losses["loss"], self.optimizers["default"])


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='ables amp training')
    parser.add_argument('--accumulation_steps', type=int, default=1, metavar='S',
                        help='Amount of gradient accumelation')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    optimizer = get_optimizer("Adadelta", model=model, lr=args.lr)
    scheduler_extension = LRSchedulerExtension(scheduler_type="StepLR", step_size=1, gamma=args.gamma)
    criterion = nn.NLLLoss()

    engine = Engine(
        models=model,
        optimizers=optimizer,
        criterions=criterion,
        output_dir="output/fold0",
        save_metrics=["acc-top1", "acc-top5"],
        save_losses=["loss"],
        train_loader=train_loader,
        valid_loader=test_loader,
        # Choosing gpu_ids -> ex. device_ids = [0, 1], None is all device
        device_ids=None,
        init_epoch=0,
        save_interval=1,
        accumulation_steps=args.accumulation_steps,
        use_amp=args.use_amp,
        opt_level='O2',
        weights_path=None,
        apply_fn=None,
        calc_train_metrics=True,
        calc_metrics_mode="batch",
        requierd_eval_data=None,
        extensions=[CSVLoggerExtension(), TensorBoardExtension(), LineNotifyExtension(start_message=f"training start!\n""logdir: output/fold0"), scheduler_extension])

    engine.repeated_run(args.epochs)


if __name__ == '__main__':
    main()