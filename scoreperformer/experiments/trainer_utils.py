""" Trainer utility functions. """

from pathlib import Path
from typing import Union, List

from omegaconf import ListConfig

from scoreperformer.utils import ExplicitEnum


def resolve_path(constructor: Union[Path, str, List[str]]):
    return Path(*constructor) if isinstance(constructor, (List, ListConfig)) else Path(constructor)


class Accumulator:
    def __init__(self):
        self._sums = {}
        self._counts = {}

    def __getitem__(self, key):
        return self._sums[key] / self._counts[key]

    @property
    def sums(self):
        return self._sums

    @property
    def counts(self):
        return self._counts

    @property
    def mean_values(self):
        return {key: self._sums[key] / self._counts[key] for key in self._sums if self._counts[key] > 0}

    def items(self):
        return self.mean_values.items()

    def add_value(self, name, value):
        self._sums[name] = value
        self._counts[name] = 1

    def update_value(self, name, value):
        if name not in self._sums:
            self.add_value(name, value)
        else:
            self._sums[name] += value
            self._counts[name] += 1

    def add_values(self, name_dict):
        for key, value in name_dict.items():
            self.add_value(key, value)

    def update_values(self, value_dict):
        for key, value in value_dict.items():
            self.update_value(key, value)

    def reset(self):
        for key in self._sums:
            self._sums[key] = 0
            self._counts[key] = 0

    def clear(self):
        self._sums = {}
        self._counts = {}


class IntervalStrategy(ExplicitEnum):
    NO = "no"
    STEPS = "steps"
    EPOCH = "epoch"
