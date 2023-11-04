from dataclasses import dataclass
from typing import Union

import numpy as np
import torch


@dataclass
class SeqInputs:
    tokens: Union[np.ndarray, torch.Tensor]
    mask: Union[np.ndarray, torch.Tensor]
    lengths: Union[np.ndarray, torch.Tensor]
