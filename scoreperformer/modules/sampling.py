""" Sampling functions. """

import math
from typing import Callable, Optional, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from scoreperformer.utils import default


# nucleus

def top_p(logits: Tensor, thres: float = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres
    sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value=False)

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk

def top_k(logits: Tensor, thres: float = 0.9, k: Optional[int] = None):
    k = default(k, math.ceil((1 - thres) * logits.shape[-1]))
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# top_a

def top_a(logits: Tensor, min_p_pow: float = 2.0, min_p_ratio: float = 0.02):
    probs = F.softmax(logits, dim=-1)
    limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
    return torch.where(probs < limit, float('-inf'), logits)


# sampling

def filter_logits_and_sample(
        logits: Tensor,
        filter_logits_fn: Callable,
        filter_kwargs: Optional[Dict[str, object]] = None,
        temperature: float = 1.,
        sample: bool = True
):
    filter_kwargs = filter_kwargs or {}
    filtered_logits = filter_logits_fn(logits, **filter_kwargs)

    probs = F.softmax(filtered_logits / temperature, dim=-1)
    if not sample:
        return probs
    return torch.multinomial(probs, 1)
