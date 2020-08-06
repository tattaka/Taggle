Taggle
=====

The Customizable wrapper for PyTorch training loop and The helper function for getting major criterion, computer vision model, optimizer

## Intstallation
Clone:  
`$ git clone --recursive https://github.com/tattaka/Taggle.git`

Using pip:  
`$ cd Taggle`  
`$ pip install .`

You can also use it by passing the path in the script:
```python
sys.path.append('/YOUR_DOWNLOAD_PATH/Taggle')
```

## Features

### `Engine` (Wrapper class for training loop)
Implement only process specific to task as below:

```python
from taggle.engine import BaseEngine

class Engine(BaseEngine):

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
```
`Engine` class can process multiple models, loss, optimizer and scheduler in dict format.  
If optimizer, criterion, model, and scheduler are not dict, they are converted to dict inside engine and "default" keys are assigned.  
You can also use the `Extension` class to perform additional processing such as logging.  
`CSVLoggerExtension`, `TensorBoardExtension`, `LineNotifyExtension` have already been implemented.  
The new processing can be implemented by inheriting the `BaseExtension` class.  

For detail, please see [here](https://github.com/tattaka/Taggle/blob/master/example/train_mnist_example.py)


### `ModelProvider` (Helper class for building computer vision models)
It supports major backbone model and Heads used for classification and segmentation.   
This feature is inspired by [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch).  
usage:
```yaml example.yaml
model:
    backbone: 
        type: resnet18
        params:
            backbone_weights: imagenet
    heads:
        output1: 
            type: SimpleHead
            params:
                num_class: 16
        output2: 
            type: UNetHead
            params:
                num_class: 10
```
```python
import yaml
from taggle.models import ModelProvider
with open("/PATH/config.yaml", "r+") as f:
        config = yaml.load(f)
model_provider = ModelProvider()
model = model_provider.get_model(config['model'])
```
Also, you can easily customize it by extending `ModelProvider` class and `BaseModel` class.

### `get_optimizer` 
Get the latest deep learning optimizers from the fork of [Best-Deep-Learning-Optimizers](https://github.com/lessw2020/Best-Deep-Learning-Optimizers) using `get_optimizer` function:
``` yaml
optimizer: 
    type: Adam
    params:
        separate_head: True
        lr: 1.e-3
        lr_e: 1.e-3
```

```python
from taggle.optimizers import get_optimizer
with open("/PATH/config.yaml", "r+") as f:
        config = yaml.load(f)
optimizer = get_optimizer(model=model, 
                          optimizer=config["optimizer"]["type"], 
                          **config["optimizer"]["params"])
```

### `get_losses_dict`
Get the major loss functions with dict format:
```yaml
loss_fn:
    classification:
        name: SmoothCrossEntropyLoss
        params: 
            smoothing: 0.1
```

``` python
from taggle.losses import get_loss_dict
with open("/PATH/config.yaml", "r+") as f:
        config = yaml.load(f)
criterions = get_losses_dict(config["loss_fn"])
```

## License
Project is distributed under [MIT License](https://github.com/tattaka/Taggle/blob/master/LICENSE).  
This software includes the work that is distributed in the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0).