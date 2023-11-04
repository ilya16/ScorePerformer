""" Base Model class. """

from abc import abstractmethod
from typing import Optional, List, Dict, Union

import torch
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import Dataset

from scoreperformer.modules.constructor import Constructor, ModuleConfig


class Model(nn.Module, Constructor):
    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def prepare_inputs(self, inputs) -> Dict[str, Tensor]:
        ...

    @staticmethod
    def allocate_inputs(inputs_dict, device):
        return {key: value.to(device, non_blocking=True) for key, value in inputs_dict.items()}

    @staticmethod
    def inject_data_config(
            config: Optional[Union[DictConfig, ModuleConfig]],
            dataset: Optional[Dataset]
    ) -> Optional[Union[DictConfig, ModuleConfig]]:
        return config

    @staticmethod
    def cleanup_config(
            config: Optional[Union[DictConfig, ModuleConfig]]
    ) -> Optional[Union[DictConfig, ModuleConfig]]:
        return config

    @classmethod
    def from_pretrained(cls, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model_cfg = OmegaConf.create(checkpoint['model']['config'])
        model = cls.init(model_cfg)

        state_dict = checkpoint['model']['state_dict']
        model.load_state_dict(state_dict, strict=True)

        return model

    def load(
            self,
            state_dict: Dict[str, Tensor],
            ignore_layers: Optional[List] = None,
            ignore_mismatched_keys: bool = False
    ):
        ignore_layers = ignore_layers or []

        model_state = self.state_dict()

        extra_keys = [k for k in state_dict.keys() if k not in model_state]
        if extra_keys:
            logger.warning(f"The following checkpoint keys are not presented in the model "
                           f"and will be ignored: {extra_keys}")
            state_dict = {k: v for k, v in state_dict.items() if k not in extra_keys}

        ignored_keys = []
        if ignore_mismatched_keys:
            auto_ignore_layers = []
            for k, v in state_dict.items():
                if v.data.shape != model_state[k].data.shape:
                    auto_ignore_layers.append(k)
            logger.info(f"Automatically found the checkpoint keys "
                        f"incompatible with the model: {auto_ignore_layers}")
            ignored_keys.extend(auto_ignore_layers)

        if ignore_layers:
            for k, v in state_dict.items():
                if any(layer in k for layer in ignore_layers):
                    ignored_keys.append(k)

        if ignored_keys:
            state_dict = {k: v for k, v in state_dict.items()
                          if all(k != key for key in ignored_keys)}
            logger.info(f"The following checkpoint keys were ignored: {ignored_keys}")

        model_state.update(state_dict)
        self.load_state_dict(model_state)

        return self

    def freeze(self, exception_list=None):
        not_frozen = []
        exception_list = exception_list or []
        for name, param in self.named_parameters():
            param.requires_grad = any((name.startswith(layer) for layer in exception_list))
            if param.requires_grad:
                not_frozen.append(name)
        logger.info(f"The model graph has been frozen, except for the following parameters: {not_frozen}")
