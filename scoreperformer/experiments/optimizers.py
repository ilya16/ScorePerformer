""" Optimizers and Learning Rate Schedulers """

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict

import torch
from torch.cuda import amp

_optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW
}


def switch_optimizer(optimizer_name: str):
    optimizer = getattr(torch.optim, optimizer_name, None)
    if optimizer is not None:
        return optimizer

    if optimizer_name in _optimizers:
        optimizer = _optimizers[optimizer_name]
    else:
        raise ValueError(
            f"There is no such name for optimizers: {optimizer_name}. "
            f"Valid optimizers: {list(_optimizers.keys())}"
        )

    return optimizer


def get_optimizer(
        optimizer_name: str,
        optimizer_params: dict,
        lr: float,
        model: torch.nn.Module = None,
        parameters: List = None,
) -> torch.optim.Optimizer:
    """Find, initialize and return an optimizer.
    Args:
        optimizer_name (str): Optimizer name.
        optimizer_params (dict): Optimizer parameters.
        lr (float): Initial learning rate.
        model (torch.nn.Module): Model to pass to the optimizer.
    Returns:
        torch.optim.Optimizer: Functional optimizer.
    """
    optimizer = switch_optimizer(optimizer_name)
    if model is not None:
        parameters = model.parameters()
    return optimizer(parameters, lr=lr, **optimizer_params)


_lr_schedulers = {
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
}


def switch_lr_scheduler(lr_scheduler_name: str):
    lr_scheduler = getattr(torch.optim, lr_scheduler_name, None)
    if lr_scheduler is not None:
        return lr_scheduler

    if lr_scheduler_name in _lr_schedulers:
        lr_scheduler = _lr_schedulers[lr_scheduler_name]
    else:
        raise ValueError(
            f"There is no such name for lr_schedulers: {lr_scheduler_name}. "
            f"Valid lr_schedulers: {list(_lr_schedulers.keys())}"
        )

    return lr_scheduler


def get_lr_scheduler(
        lr_scheduler: str, lr_scheduler_params: Dict, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:  # pylint: disable=protected-access
    """Find, initialize and return a scheduler.
    Args:
        lr_scheduler (str): Scheduler name.
        lr_scheduler_params (Dict): Scheduler parameters.
        optimizer (torch.optim.Optimizer): Optimizer to pass to the scheduler.
    Returns:
        torch.optim.lr_scheduler._LRScheduler: Functional scheduler.
    """
    if lr_scheduler is None:
        return None
    scheduler = switch_lr_scheduler(lr_scheduler)
    return scheduler(optimizer, **lr_scheduler_params)


@dataclass
class OptimizerConfig:
    lr: Union[float, List[float]] = field(
        default=0.001, metadata={"help": "Learning rate for each optimizer. Defaults to 0.001"}
    )
    optimizer: Union[str, List[str]] = field(
        default="Adam", metadata={"help": "Optimizer(s) to use. Defaults to None"}
    )
    optimizer_params: Union[Dict, List[Dict]] = field(
        default_factory=dict, metadata={"help": "Optimizer(s) arguments. Defaults to {}"}
    )
    lr_scheduler: Union[str, List[str]] = field(
        default=None, metadata={"help": "Learning rate scheduler(s) to use. Defaults to None"}
    )
    lr_scheduler_params: Dict = field(
        default_factory=dict, metadata={"help": "Learning rate scheduler(s) arguments. Defaults to {}"}
    )
    grad_clip: Optional[float] = field(
        default=None, metadata={"help": "Gradient clipping threshold. Defaults to None"}
    )
    grad_accum_steps: Optional[int] = field(
        default=1, metadata={"help": "Gradient accumulation steps. Defaults to 1"}
    )
    mixed_precision: bool = field(
        default=False, metadata={"help": "Use mixed precision for training. Defaults to False"}
    )


class Optimizer:
    """ Combined Optimizer and Scheduler class. """

    def __init__(
            self,
            config: OptimizerConfig,
            model: torch.nn.Module = None,
            parameters: List = None
    ):
        self.config = config

        self.optimizer = get_optimizer(
            config.optimizer,
            config.optimizer_params or {},
            config.lr,
            model=model,
            parameters=parameters
        )

        self.lr_scheduler = get_lr_scheduler(
            config.lr_scheduler,
            config.lr_scheduler_params,
            optimizer=self.optimizer
        )

        self._scaler = amp.GradScaler(enabled=config.mixed_precision)
        self.grad_clip = config.grad_clip
        self.grad_accum_steps = config.grad_accum_steps
        self.mixed_precision = config.mixed_precision

    def step(self, loss_value, step_optimizer=True):
        self._scaler.scale(loss_value / self.grad_accum_steps).backward()

        grad_norm = None
        if step_optimizer:
            self._scaler.unscale_(self.optimizer)

            if self.grad_clip is not None:
                parameters = self.optimizer.param_groups[0]["params"]
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    grad_norm = None

            self._scaler.step(self.optimizer)
            self._scaler.update()

            self.optimizer.zero_grad()

        return grad_norm

    def anneal_on_epoch_end(self, *args):
        if self.lr_scheduler is not None:
            args = args if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else []
            self.lr_scheduler.step(*args)

    def anneal_on_iter_end(self, *args):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def get_last_lr(self):
        return self.lr_scheduler.get_last_lr()

    def load_state_dict(self, state_dict, restore_lr=True):
        self.optimizer.load_state_dict(state_dict["optimizer"])

        if restore_lr:
            lr_scheduler_params = state_dict.get("lr_scheduler", None)
            if lr_scheduler_params is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_params)
        else:
            base_lr = self.lr_scheduler.get_last_lr()
            for lr, param_group in zip(base_lr, self.optimizer.param_groups):
                param_group["lr"] = lr

    def state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }

    def set_progress(self, iteration, epoch):
        for key, value in self.optimizer.state.items():
            if "step" in value:
                self.optimizer.state[key]["step"] = iteration

        self.lr_scheduler.last_epoch = epoch
        self.lr_scheduler._step_count = epoch + 1

    def __repr__(self):
        return f"Optimizer(optimizer={self.optimizer}, lr_scheduler={self.lr_scheduler})"
