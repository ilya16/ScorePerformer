""" Performance data collators. """

import math
from dataclasses import dataclass
from functools import reduce
from typing import Optional, List

import numpy as np
import torch

from .common import SeqInputs


@dataclass
class PerformanceInputs:
    performances: SeqInputs


class PerformanceCollator:
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1
    ):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def pad_len(self, length):
        # pad to a multiple of `pad_to_multiple`
        if self.pad_to_multiple_of > 0:
            pad_size = self.pad_to_multiple_of - length % self.pad_to_multiple_of
            length += pad_size if 0 < pad_size < self.pad_to_multiple_of else 0
        return length

    def get_max_lengths(self, batch, inference=False):
        lens_perf = np.array(list(map(lambda sample: len(sample.perf), batch))).T

        return {
            'performance': np.max(lens_perf) if inference else self.pad_len(np.max(lens_perf))
        }

    def _init_seq_data(self, batch_size, max_len, compound_factor=1):
        if compound_factor > 1:
            seq_data = SeqInputs(
                tokens=torch.full((batch_size, max_len, compound_factor), self.pad_token_id, dtype=torch.long),
                mask=torch.zeros(batch_size, max_len, dtype=torch.bool),
                lengths=torch.zeros(batch_size, dtype=torch.long)
            )
        else:
            seq_data = SeqInputs(
                tokens=torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long),
                mask=torch.zeros(batch_size, max_len, dtype=torch.bool),
                lengths=torch.zeros(batch_size, dtype=torch.long)
            )
        return seq_data

    def init_data(self, batch, inference=False):
        max_lens = self.get_max_lengths(batch, inference=inference)

        return PerformanceInputs(
            performances=self._init_seq_data(
                len(batch), max_lens['performance'],
                compound_factor=batch[0].perf.shape[-1]
            )
        )

    @staticmethod
    def _process_sequence(i, seq, seq_data):
        seq = torch.from_numpy(seq)
        seq_len = len(seq)

        seq_data.tokens[i, :seq_len] = seq
        seq_data.mask[i, :seq_len] = True
        seq_data.lengths[i] = seq_len

    def process_sample(self, i, sample, data, inference=False):
        self._process_sequence(i, seq=sample.perf, seq_data=data.performances)

    def __call__(self, batch, inference=False, return_tensors=True):
        data = self.init_data(batch, inference=inference)
        for i, sample in enumerate(batch):
            self.process_sample(i, sample, data)

        return data


# FOR LANGUAGE MODELING

def prob_mask_like(t: torch.Tensor, prob: float):
    if t.ndim == 2:
        return torch.zeros_like(t).float().uniform_(0, 1) < prob
    else:
        mask = torch.zeros(*t.shape[:2], dtype=torch.float32, device=t.device).uniform_(0, 1) < prob
        return mask[..., None]


def mask_with_tokens(t: torch.Tensor, token_ids, squeeze=True):
    if t.ndim == 2 or not squeeze:
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    else:
        init_no_mask = torch.full(t.shape[:2], False, dtype=torch.bool, device=t.device)
        mask = reduce(lambda acc, el: acc | torch.any(t == el, dim=-1), token_ids, init_no_mask)
    return mask


def mask_with_token_dims(t: torch.Tensor, token_dims):
    if t.ndim == 2:
        return torch.zeros_like(t)
    else:
        mask = torch.full_like(t, False, dtype=torch.bool, device=t.device)
        if token_dims:
            mask[..., token_dims] = True
    return mask


def get_mask_subset_with_prob(mask: torch.Tensor, prob):
    batch, seq_len, device = *mask.shape[:2], mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


@dataclass
class LMPerformanceInputs(PerformanceInputs):
    labels: SeqInputs


class LMPerformanceCollator(PerformanceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mlm: bool = False,
            mask_prob: float = 0.15,
            replace_prob: float = 0.9,
            mask_token_id: int = 1,
            mask_ignore_token_ids: Optional[List[int]] = None,
            mask_ignore_token_dims: Optional[List[int]] = None,
            label_pad_ignored_dims: bool = True,
            label_pad_token_id: int = -100
    ):
        super().__init__(pad_token_id=pad_token_id, pad_to_multiple_of=pad_to_multiple_of)

        self.mlm = mlm
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = {*(mask_ignore_token_ids or []), pad_token_id}
        self.mask_ignore_token_dims = mask_ignore_token_dims or []
        self.label_pad_ignored_dims = label_pad_ignored_dims
        self.label_pad_token_id = label_pad_token_id

    def mask_sequence(self, seq: torch.Tensor):
        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([sos], [eos])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # possibly expand mask
        if seq.ndim == 3:
            mask = mask[..., None].expand(-1, -1, seq.size(2))

        # mask input with [mask] tokens with probability of `replace_prob`
        dim_mask = mask_with_token_dims(seq, self.mask_ignore_token_dims)
        replace_prob = prob_mask_like(seq, self.replace_prob)

        token_mask = mask * replace_prob * (~dim_mask)
        masked_seq = seq.clone().detach().masked_fill(token_mask, self.mask_token_id)

        # derive labels to predict
        label_mask = mask
        if self.label_pad_ignored_dims:
            label_mask = label_mask * (~dim_mask)
        labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        return masked_seq, labels, label_mask

    def __call__(self, batch, inference=False, return_tensors=True):
        data = super().__call__(batch, inference=inference)

        if self.mlm:
            masked_seq, labels, label_mask = self.mask_sequence(data.performances.tokens)
            data.performances.tokens = masked_seq
        else:
            labels = data.performances.tokens.clone().detach()
            labels[labels == self.pad_token_id] = self.label_pad_token_id
            label_mask = data.performances.mask.clone().detach()

        data = LMPerformanceInputs(
            performances=data.performances,
            labels=SeqInputs(
                tokens=labels,
                mask=label_mask,
                lengths=data.performances.lengths
            )
        )

        return data


@dataclass
class MixedLMPerformanceInputs(LMPerformanceInputs):
    masked_performances: SeqInputs


class MixedLMPerformanceCollator(PerformanceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1,

            mask_token_id: int = 1,
            mask_ignore_token_ids: Optional[List[int]] = None,
            mask_ignore_token_dims: Optional[List[int]] = None,
            label_pad_ignored_dims: bool = True,
            label_pad_token_id: int = -100
    ):
        super().__init__(pad_token_id=pad_token_id, pad_to_multiple_of=pad_to_multiple_of)

        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = {*(mask_ignore_token_ids or []), pad_token_id}
        self.mask_ignore_token_dims = mask_ignore_token_dims or []
        self.label_pad_ignored_dims = label_pad_ignored_dims
        self.label_pad_token_id = label_pad_token_id

    def mask_sequence(self, seq: torch.Tensor):
        # do not mask positions for ignored tokens
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids, squeeze=False)

        # mask only non-ignored token dimension
        dim_mask = mask_with_token_dims(seq, self.mask_ignore_token_dims)

        token_mask = (~no_mask) * (~dim_mask)
        masked_seq = seq.clone().detach().masked_fill(token_mask, self.mask_token_id)

        # derive labels to predict
        label_mask = ~no_mask
        if self.label_pad_ignored_dims:
            label_mask = label_mask * (~dim_mask)
        labels = seq.clone().detach().masked_fill(~label_mask, self.label_pad_token_id)

        return masked_seq, labels

    def __call__(self, batch, inference=False, return_tensors=True):
        data = super().__call__(batch, inference=inference)

        masked_performances, labels = self.mask_sequence(data.performances.tokens)
        label_mask = data.performances.mask.clone().detach()

        data = MixedLMPerformanceInputs(
            performances=data.performances,
            masked_performances=SeqInputs(
                tokens=masked_performances,
                mask=label_mask,
                lengths=data.performances.lengths
            ),
            labels=SeqInputs(
                tokens=labels,
                mask=label_mask,
                lengths=data.performances.lengths
            )
        )

        return data
