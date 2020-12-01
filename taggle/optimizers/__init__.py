import warnings

import torch.optim as optim
import torch_optimizer as optim2

from torch.optim.optimizer import Optimizer

warnings.filterwarnings("once")


def get_optimizer(optimizer: str = 'Adam',
                  lookahead: bool = False,
                  model=None,
                  params_rule: dict = None,
                  lr: float = 1e-3,
                  **kwargs):

    if params_rule is None:
        params = [{'params': model.parameters(), 'lr': lr}]
    else:
        params = []
        for key in params_rule:
            params.append({'params': getattr(model, key).parameters(), 'lr': params_rule[key]})
    if optimizer in optim.__dir__():
        optimizer = getattr(optim, optimizer)(params, lr=lr, **kwargs)
    elif optimizer in optim2.__dir__():
        optimizer = getattr(optim2, optimizer)(params, lr=lr, **kwargs)
    else:
        raise ValueError('unknown base optimizer type')

    if lookahead:
        optimizer = Lookahead(base_optimizer=optimizer, k=5, alpha=0.5)

    return optimizer


class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        self.optimizer = base_optimizer
        self.param_groups = self.optimizer.param_groups
        self.alpha = alpha
        self.k = k
        for group in self.param_groups:
            group["step_counter"] = 0
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                             for group in self.param_groups]

        for w in it.chain(*self.slow_weights):
            w.requires_grad = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        loss = self.optimizer.step()
        for group, slow_weights in zip(self.param_groups, self.slow_weights):
            group['step_counter'] += 1
            if group['step_counter'] % self.k != 0:
                continue
            for p, q in zip(group['params'], slow_weights):
                if p.grad is None:
                    continue
                q.data.add_(self.alpha, p.data - q.data)
                p.data.copy_(q.data)
        return loss
