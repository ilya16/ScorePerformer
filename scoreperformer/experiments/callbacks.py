"""
Callbacks to use with the Trainer class and customize the training loop.

Adapted from: https://github.com/huggingface/transformers/blob/main/src/transformers/trainer_callback.py
"""

import dataclasses
import json
import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np
import torch
from loguru import logger
from tqdm.auto import tqdm

from .trainer_config import TrainerConfig
from .trainer_utils import IntervalStrategy


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    Note: one step is one update step which might contain multiple gradient accumulation steps.
    """

    epoch: Optional[float] = field(
        default=0, metadata={"help": "The epoch the training is at "
                                     "(the decimal part is percentage of the current epoch completed)."}
    )
    global_step: int = field(
        default=0, metadata={"help": "The number of training steps completed."}
    )
    max_steps: int = field(
        default=0, metadata={"help": "The maximum number of training steps to be completed."}
    )
    num_train_epochs: int = field(
        default=0, metadata={"help": "The number of training epochs to run."}
    )
    epoch_step: int = field(
        default=0, metadata={"help": "The number of training/evaluation epoch steps completed."}
    )
    log_history: List[Dict[str, float]] = field(
        default=None, metadata={"help": "The list of logs done since the beginning of training."}
    )
    best_metric: Optional[float] = field(
        default=None, metadata={"help": "The current best metric value monitored during the training."}
    )
    last_model_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to the last saved model checkpoint."}
    )
    best_model_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to the checkpoint related to the best metric."}
    )
    is_local_process_zero: bool = field(
        default=True, metadata={"help": "Whether the current process is the local main process."}
    )

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json_string(cls, json_string):
        """
        Builds an instance from a JSON string.
        """
        return cls(**json.loads(json_string))

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = self.to_json_string()
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


@dataclass
class TrainerControl:
    """
    A class that handles the [`Trainer`] control flow.

    This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.
    """

    should_training_stop: bool = field(
        default=False,
        metadata={"help": ("Whether the training should be interrupted.",
                           "If `True`, this variable will not be set back to `False`. The training will just stop.")}
    )
    should_epoch_stop: bool = field(
        default=False,
        metadata={"help": ("Whether the current epoch should be interrupted.",
                           "If `True`, this variable will be set back to `False` at the beginning of the next epoch.")}
    )
    should_save: bool = field(
        default=False,
        metadata={"help": ("Whether the model should be saved at this step.",
                           "If `True`, this variable will be set back to `False` at the beginning of the next step.")}
    )
    should_evaluate: bool = field(
        default=False,
        metadata={"help": ("Whether the the model should be evaluated at this step.",
                           "If `True`, this variable will not be set back to `False`. The training will just stop.")}
    )
    should_log: bool = field(
        default=False,
        metadata={"help": ("Whether the logs should be reported at this step.",
                           "If `True`, this variable will be set back to `False` at the beginning of the next step.")}
    )
    is_train: bool = field(
        default=False,
        metadata={"help": ("Whether the Trainer is in training mode or not.",
                           "Used to distinguish evaluation epochs inside training epochs.")}
    )

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False


class TrainerCallback:
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions.
    At each of those events the following arguments are available:

    Args:
        config ([`TrainerConfig`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the data.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.data.DataLoader`, *optional*):
            The current dataloader used for training.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `config`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformer.PrinterCallback`].

    """

    def on_init_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_epoch_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_substep_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_step_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_save(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, logs, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass


class CallbackHandler(TrainerCallback):
    """ Internal class that just calls the list of callbacks in order. """

    def __init__(self, callbacks, model, optimizer):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            logger.warning(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def has_callback(self, callback_class):
        return callback_class in [c.__class__ for c in self.callbacks]

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if self.has_callback(cb_class):
            logger.warning(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_init_end", config, state, control, **kwargs)

    def on_train_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_training_stop = False
        return self.call_event("on_train_begin", config, state, control, **kwargs)

    def on_train_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_train_end", config, state, control, **kwargs)

    def on_epoch_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", config, state, control, **kwargs)

    def on_epoch_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_epoch_end", config, state, control, **kwargs)

    def on_step_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", config, state, control, **kwargs)

    def on_substep_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_substep_end", config, state, control, **kwargs)

    def on_step_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        return self.call_event("on_step_end", config, state, control, **kwargs)

    def on_evaluate(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        control.should_evaluate = False
        return self.call_event("on_evaluate", config, state, control, metrics=metrics, **kwargs)

    def on_save(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        control.should_save = False
        return self.call_event("on_save", config, state, control, **kwargs)

    def on_log(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, logs, **kwargs):
        control.should_log = False
        return self.call_event("on_log", config, state, control, logs=logs, **kwargs)

    def call_event(self, event, config, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                config,
                state,
                control,
                model=self.model,
                optimizer=self.optimizer,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if state.global_step == 1 and config.log_first_step:
            control.should_log = True
        if config.log_strategy == IntervalStrategy.STEPS and state.global_step % config.log_steps == 0:
            control.should_log = True

        if control.is_train:
            # Evaluate
            if config.eval_strategy == IntervalStrategy.STEPS and state.global_step % config.eval_steps == 0:
                print('should_evaluate')
                control.should_evaluate = True

            # Save
            if (
                    config.save_strategy == IntervalStrategy.STEPS
                    and config.save_steps > 0
                    and state.global_step % config.save_steps == 0
            ):
                print('should_save')
                control.should_save = True

            # End training
            if state.global_step >= state.max_steps:
                control.should_training_stop = True
        else:
            # End evaluation epoch
            if config.eval_batches and state.epoch_step == config.eval_batches:
                control.should_epoch_stop = True

        return control

    def on_epoch_end(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if config.log_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        if control.is_train:
            # Evaluate
            if config.eval_strategy == IntervalStrategy.EPOCH:
                control.should_evaluate = True

            # Save
            if config.save_strategy == IntervalStrategy.EPOCH:
                control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    """

    def __init__(self):
        self.training_bar = None
        self.evaluation_bar = None

    def on_train_begin(self, config, state, control, **kwargs):
        if config.eval_first_step and state.global_step == 0:
            control.should_evaluate = True

    def on_epoch_begin(self, config: TrainerConfig, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero:
            if control.is_train:
                self.training_bar = tqdm(total=kwargs.get("steps_in_epoch", None))
            else:
                self.evaluation_bar = tqdm(total=kwargs.get("steps_in_epoch", None),
                                           leave=False)  # self.training_bar is None)

    def on_step_end(self, config, state, control, **kwargs):
        if state.is_local_process_zero:
            _bar = self.training_bar if control.is_train else self.evaluation_bar
            _bar.update(1)

            if state.epoch_step % config.progress_steps == 0 or state.epoch_step == 1:
                desc = f" epoch: {state.epoch:.3f}"
                desc += self._get_metrics_message(config, metrics=kwargs.get("epoch_stats", {}))

                if control.is_train:
                    grad_norm = kwargs.get("grad_norm", None)
                    desc += f", GN: {grad_norm}" if grad_norm is None else f", GN: {grad_norm:.3f}"

                _bar.set_description(desc)

    def on_epoch_end(self, config, state, control, **kwargs):
        if state.is_local_process_zero:
            _bar = self.training_bar if control.is_train else self.evaluation_bar
            if _bar is not None:
                _bar.set_description(None)
                _bar.close()
            _bar = None

            prefix = "[TRAIN] " if control.is_train else "[EVAL]  "
            step_message = f"epoch: {int(state.epoch):3d}/{config.epochs:3d} (step: {state.global_step})"
            metrics_message = self._get_metrics_message(config, metrics=kwargs.get("metrics", {}))
            logger.info(prefix + step_message + metrics_message)
            if config.log_raw_to_console:
                metrics = {key: round(value, 5) for key, value in kwargs.get("metrics", {}).items()}
                logger.info(str(metrics))
            logger.complete()

    @staticmethod
    def _get_metrics_message(config, metrics):
        message = ''
        if metrics and config.progress_metrics:
            for metric in config.progress_metrics:
                message += f", {metric}: {float(metrics[metric]):6.5f}" if metric in metrics else ""
        return message


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, config, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(logs)


class EpochReproducibilityCallback(TrainerCallback):
    """
    A [`TrainerCallback`] to make each training epoch reproducible.
    """

    def on_epoch_begin(self, config, state, control, **kwargs):
        if control.is_train:
            seed = (math.ceil(state.epoch) + 1) * config.seed
            logger.debug(f"Set random seed to {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
