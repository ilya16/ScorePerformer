""" Config-based Module Constructor. """
import copy
from dataclasses import dataclass
from inspect import signature
from typing import Optional, Union, Callable

import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf, MISSING


@dataclass
class ModuleConfig:
    def update(self, **kwargs):
        kwargs = {key: kwargs.get(key, MISSING) for key in kwargs if not key.startswith("_")}  # get rid of service keys
        invalid_keys = [key for key in kwargs if key not in self.__dict__]
        if invalid_keys:
            logger.warning(f"The following params are incompatible with the config {self.__name__}, "
                           f"so they will be ignored: {invalid_keys}.")
            kwargs = {key: value for key, value in kwargs.items() if key not in invalid_keys}

        for key, value in kwargs.items():
            self.__setattr__(key, value)

        return self

    def to_dict(self, check_missing=False, make_copy=True):
        if check_missing:
            missing = [key for key, value in self.__dict__.items() if value == MISSING]
            if missing:
                raise RuntimeError(f"The following params are mandatory to set: {missing}")

        return copy.deepcopy(self.__dict__) if make_copy else {k: v for k, v in self.__dict__.items()}


class Constructor:
    @classmethod
    def _pre_init(cls, config: Optional[Union[DictConfig, ModuleConfig]] = None, **parameters):
        module_parameters = {key: value for key, value in parameters.items() if isinstance(value, torch.nn.Module)}
        parameters = {key: value for key, value in parameters.items() if key not in module_parameters}
        config = merge(config or {}, parameters)
        for key, value in module_parameters.items():
            config[key] = value
        config = {key: value for key, value in config.items() if not key.startswith("_")}

        return config

    @classmethod
    def init(cls, config: Optional[Union[DictConfig, ModuleConfig]] = None, **parameters):
        config = cls._pre_init(config, **parameters)

        signature_ = dict(signature(cls.__init__).parameters)

        if "kwargs" not in signature_:
            invalid_keys = [key for key in config if key not in signature_]
            if invalid_keys:
                logger.warning(f"The following params are incompatible with the {cls.__name__} constructor, "
                               f"so they will be ignored: {invalid_keys}.")
                config = {key: value for key, value in config.items() if key not in invalid_keys}

        missing = [key for key, value in config.items() if value == MISSING]
        if missing:
            raise RuntimeError(f"The following params are mandatory to set: {missing}")

        return cls(**config)


def merge(*containers: Union[DictConfig, ModuleConfig, dict], as_omega=False) -> Union[dict, DictConfig]:
    readonly = False
    _containers = []
    for cont in containers:
        if isinstance(cont, ModuleConfig):
            cont = cont.to_dict(make_copy=False)
        elif isinstance(cont, DictConfig):
            readonly = cont._get_flag("readonly")
        elif not isinstance(cont, dict):
            raise TypeError
        _containers.append(cont)

    merged = OmegaConf.merge(*_containers)
    OmegaConf.set_readonly(merged, readonly)

    if not as_omega:
        merged = dict(merged)

    return merged


@dataclass
class VariableModuleConfig(ModuleConfig):
    _target_: str


class Registry:
    """
    Parent class for different registries.
    """

    def __init__(self):
        self._objects = {}

    def register(
            self,
            name: str,
            module: Optional[Callable] = None
    ):
        if not isinstance(name, str):
            raise TypeError(f"`name` must be a str, got {name}")

        def _register(reg_obj: Callable):
            self._objects[name] = reg_obj
            return reg_obj

        return _register if module is None else _register(module)

    def instantiate(self, config: Union[VariableModuleConfig, DictConfig], **kwargs):
        module: Constructor = self.get(config._target_)
        return module.init(config, **kwargs)

    def get(self, key: str):
        try:
            return self._objects[key]
        except KeyError:
            raise KeyError(f"'{key}' not found in registry. Available names: {self.available_names}")

    def remove(self, name):
        self._objects.pop(name)

    @property
    def objects(self):
        return self._objects

    @property
    def available_names(self):
        return tuple(self._objects.keys())

    def __str__(self) -> str:
        return f"Objects={self.available_names}"
