"""
External Trainer integraions.

Adapted from: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations.py
"""
import json

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from .callbacks import TrainerCallback


class TensorBoardCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [TensorBoard](https://www.tensorflow.org/tensorboard).
    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self, tb_writer=None):
        self.tb_writer = tb_writer

    def _init_summary_writer(self, config, log_dir=None):
        log_dir = log_dir or config.log_dir
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def on_train_begin(self, config, state, control, **kwargs):
        if self.tb_writer is None:
            self._init_summary_writer(config)

        if self.tb_writer is not None:
            self.tb_writer.add_text("trainer_config", config.to_json_string())
            exp_config = kwargs.get("exp_config", None)
            if exp_config is not None:
                def to_json_string(cfg):
                    return json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)
                self.tb_writer.add_text("data_config", to_json_string(exp_config.data))
                self.tb_writer.add_text("model_config", to_json_string(exp_config.model))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.tb_writer is None:
            self._init_summary_writer(args)

        if self.tb_writer is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, state.global_step)
            self.tb_writer.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None
