""" A set of utility classes and functions used throughout the repository. """

import random
import sys
from enum import Enum
from inspect import isfunction

import numpy as np
from tqdm.asyncio import tqdm


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class equals:
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head


def prob2bool(prob):
    return random.choices([True, False], weights=[prob, 1 - prob])[0]


def find_closest(array, values):
    """Finds indices of the values closest to `values` in a given array."""
    ids = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    arr_values = array[np.minimum(ids, len(array) - 1)]
    prev_values = array[np.maximum(ids - 1, 0)]
    prev_idx_is_less = (ids == len(array)) | (np.fabs(values - prev_values) < np.fabs(values - arr_values))

    if isinstance(ids, np.ndarray):
        ids[prev_idx_is_less] -= 1
    elif prev_idx_is_less:
        ids -= 1

    ids = np.maximum(0, ids)

    return ids


def tqdm_iterator(iterable, desc=None, position=0, leave=False, file=sys.stdout, **kwargs):
    return tqdm(iterable, desc=desc, position=position, leave=leave, file=file, **kwargs)


def apply(seqs, func, tqdm_enabled=True, desc=None):
    """ Apply a given `func` over a list of sequences `seqs`."""
    iterator = tqdm_iterator(seqs, desc=desc) if tqdm_enabled else seqs
    return [func(seq) for seq in iterator]


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
