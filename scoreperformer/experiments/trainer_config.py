""" Trainer Configuration. """

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Union, List

from omegaconf import OmegaConf

from scoreperformer.utils import Config
from .optimizers import OptimizerConfig
from .trainer_utils import resolve_path


OmegaConf.register_new_resolver(
    "date", lambda: datetime.now().strftime("%y-%m-%d")
)


@dataclass
class TrainerConfig(Config):
    # general
    output_dir: Union[str, List[str]] = field(
        default="results",
        metadata={"help": "Either a string with full path or a list of path elements constructing the path."}
    )

    do_train: bool = field(
        default=False,
        metadata={"help": "Whether to run training or not."}
    )
    do_eval: bool = field(
        default=False,
        metadata={"help": "Whether to run evaluation or not."}
    )
    eval_mode: bool = field(
        default=False,
        metadata={"help": "Enable evaluation mode (no model optimization). "
                          "Run a single evaluation run on the datasets."}
    )

    seed: int = field(
        default=0,
        metadata={"help": "The training seed for reproducibility."}
    )
    device: str = field(
        default="cuda:0",
        metadata={"help": "The device to put the model on to."}
    )

    # logging
    log_dir: str = field(
        default="logs",
        metadata={"help": "The directory name under the `output_dir` to put logs into."}
    )
    log_to_file: bool = field(
        default=False,
        metadata={"help": "Whether to log to file in addition to console or not."}
    )

    dashboard_logger: Optional[str] = field(
        default="tensorboard",
        metadata={"help": "The optional dashboard logger. Supported options: `tensorboard`."}
    )
    log_strategy: str = field(
        default="steps",
        metadata={"help": "The logging strategy to use during training. Possible values are:\n"
                          "  - `no`: no logging is done during training.\n"
                          "  - `epoch`: logging is done at the end of each epoch.\n"
                          "  - `steps`: logging is done every `logging_steps`"}
    )
    log_steps: int = field(
        default=1,
        metadata={"help": "How frequently (in epochs/steps) model evaluation should be performed."}
    )
    log_first_step: bool = field(
        default=False,
        metadata={"help": "Whether to log the first `global_step` or not."}
    )
    log_raw_to_console: bool = field(
        default=False,
        metadata={"help": "Whether to log raw metrics value to console or not."}
    )

    disable_tqdm: bool = field(
        default=False,
        metadata={"help": "Whether to disable tqdm progress bars or not."}
    )
    progress_steps: int = field(
        default=5,
        metadata={"help": "How frequently in steps model the progress bar should updated."}
    )
    progress_metrics: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The list of metrics to log in the progress bar."}
    )

    ignore_data_skip: bool = field(
        default=False,
        metadata={"help": "When resuming training, whether or not to skip the epochs and batches "
                          "to get the data loading at the same stage as in the previous training."
                          "If set to `True`, the training will begin faster will not yield the same results "
                          "as the interrupted training would have."}
    )

    # data
    num_workers: int = field(
        default=1,
        metadata={"help": "The number of training workers to use for the DataLoader."}
    )
    pin_memory: bool = field(
        default=False,
        metadata={"help": "Whether to pin memory in DataLoader or not."}
    )
    shuffle: bool = field(
        default=True,
        metadata={"help": "Whether to shuffle the data for the training. Applies only to `train_dataloader`."}
    )

    # training & evaluation
    epochs: int = field(
        default=100,
        metadata={"help": "The number of training epochs to run."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "The number of training update steps to run. Overrides `epochs`."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "The training batch size."}
    )

    eval_batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "The evaluation batch size. If not specified, `batch_size` is used."}
    )
    eval_batches: Optional[Union[int, float]] = field(
        default=None,
        metadata={"help": "The number of batches to use for the evaluation."
                          "If not specified, the whole `eval_dataset` is used."}
    )

    eval_strategy: str = field(
        default="epoch",
        metadata={"help": "The evaluation strategy to use during training. Possible values are:\n"
                          "  - `no`: no evaluation is done during training.\n"
                          "  - `epoch`: evaluation is done at the end of each epoch.\n"
                          "  - `steps`: evaluation is done every `logging_steps`"}
    )
    eval_steps: int = field(
        default=1,
        metadata={"help": "How frequently (in epochs/steps) model evaluation should be performed."}
    )
    eval_first_step: bool = field(
        default=True,
        metadata={"help": "Whether to run evaluation on the first `global_step` or not."}
    )

    optimization: OptimizerConfig = field(
        default=OptimizerConfig(
            lr=1e-3,
            optimizer="adam",
            optimizer_params={"weight_decay": 1e-6},
            lr_scheduler="exponential",
            lr_scheduler_params={"gamma": 0.99},
            grad_clip=1.0,
            mixed_precision=False
        ),
        metadata={"help": "The Optimizer config, and instance of `OptimizerConfig`."}
    )

    # checkpointing
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The model checkpointing strategy to use during training. Possible values are:\n"
                          "  - `no`: no model checkpoints are saved during training.\n"
                          "  - `epoch`: model checkpoints are saved at the end of each epoch.\n"
                          "  - `steps`: model checkpoints are saved every `logging_steps`"}
    )
    save_steps: int = field(
        default=1,
        metadata={"help": "How frequently (in epochs/steps) model checkpoints should be saved."}
    )
    save_optimizer: bool = field(
        default=True,
        metadata={"help": "Whether to save optimizer in the model checkpoint or not."}
    )
    save_best_only: bool = field(
        default=False,
        metadata={"help": "Whether to save only the best performing model checkpoints or not."}
    )
    save_rewrite_checkpoint: bool = field(
        default=False,
        metadata={"help": "Whether to always rewrite last (best) checkpoint or not."}
    )

    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to define and monitor the best performing model."}
    )
    metric_maximize: bool = field(
        default=True,
        metadata={"help": "Whether to the `metric_for_best_model` should be maximized or not."}
    )

    resume_from_checkpoint: Optional[Union[str, bool]] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for the model, "
                          "or a path to a model checkpoint itself."},
    )
    warm_start: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use checkpoint as a pretraining step, "
                          "or continue the training from the checkpoint."},
    )
    ignore_layers: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of model keys to be ignored during the checkpoint loading. "
                          "Matches all model keys containing any of the specified layers."},
    )
    ignore_mismatched_keys: bool = field(
        default=True,
        metadata={"help": "Whether to automatically ignore the mismatched model checkpoint keys,"
                          "or raise a RuntimeError and warn user."},
    )
    finetune_layers: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of model layers to be finetuned during the training. "
                          "Matches all model keys starting with any of the specified layers. "
                          "All other layers will be freezed."},
    )
    restore_lr: bool = field(
        default=True,
        metadata={"help": "Whether to restore the learning rate on the continuation of training."},
    )

    def __post_init__(self):
        self.output_dir = str(resolve_path(self.output_dir))

        if self.log_dir is None:
            self.log_dir = "logs"
        self.log_dir = os.path.join(self.output_dir, self.log_dir)

        self.do_train = self.do_train and not self.eval_mode
        self.eval_batch_size = self.eval_batch_size or self.batch_size
