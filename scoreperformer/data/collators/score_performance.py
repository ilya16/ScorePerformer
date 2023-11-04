""" Score-Performance data collators. """

from dataclasses import dataclass
from typing import Optional, List, Union, Dict

import numpy as np
import torch

from .common import SeqInputs
from .performance import (
    PerformanceCollator,
    LMPerformanceCollator,
    MixedLMPerformanceCollator
)
from ..datasets.score_performance import ScorePerformanceSample


@dataclass
class SeqSegments:
    bar: Optional[Union[np.ndarray, torch.Tensor]] = None
    beat: Optional[Union[np.ndarray, torch.Tensor]] = None
    onset: Optional[Union[np.ndarray, torch.Tensor]] = None


@dataclass
class ScorePerformanceInputs:
    scores: SeqInputs
    performances: SeqInputs
    noisy_performances: Optional[SeqInputs] = None
    segments: Optional[SeqSegments] = None
    directions: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None
    deadpan_mask: Optional[torch.Tensor] = None


class ScorePerformanceCollator(PerformanceCollator):
    def __init__(
            self,
            pad_token_id: int = 0,
            pad_to_multiple_of: int = 1
    ):
        super().__init__(pad_token_id, pad_to_multiple_of)

    def get_max_lengths(self, batch: List[ScorePerformanceSample], inference: bool = False):
        max_lens = super().get_max_lengths(batch, inference=inference)

        lens_score = np.array(list(map(lambda sample: len(sample.score), batch))).T
        max_lens['score'] = self.pad_len(np.max(lens_score))

        if all((sample.noisy_perf is not None for sample in batch)):
            lens_noisy_perf = np.array(list(map(lambda sample: len(sample.noisy_perf), batch))).T
            max_lens['noisy_perf'] = self.pad_len(np.max(lens_noisy_perf))

        return max_lens

    def init_data(self, batch: List[ScorePerformanceSample], inference: bool = False):
        data = super().init_data(batch, inference=inference)
        max_lens = self.get_max_lengths(batch, inference=inference)

        sample, batch_size = batch[0], len(batch)
        return ScorePerformanceInputs(
            scores=self._init_seq_data(
                batch_size, max_lens['score'],
                compound_factor=sample.score.shape[-1]
            ),
            performances=data.performances,
            noisy_performances=self._init_seq_data(
                batch_size, max_lens['noisy_perf'],
                compound_factor=sample.noisy_perf.shape[-1]
            ) if 'noisy_perf' in max_lens else None,
            segments=SeqSegments(
                bar=torch.zeros(len(batch), max_lens['score'], dtype=torch.long),
                beat=torch.zeros(len(batch), max_lens['score'], dtype=torch.long),
                onset=torch.zeros(len(batch), max_lens['score'], dtype=torch.long)
            ) if sample.segments is not None else None,
            directions=torch.zeros(
                batch_size, max_lens['score'], len(sample.directions), dtype=torch.long
            ) if sample.directions is not None else None,
            deadpan_mask=torch.zeros(batch_size, dtype=torch.bool)
        )

    def process_sample(self, i: int, sample: ScorePerformanceSample, data: ScorePerformanceInputs,
                       inference: bool = False):
        # process performance
        super().process_sample(i, sample, data, inference=inference)

        # process score
        self._process_sequence(i, seq=sample.score, seq_data=data.scores)

        # process noisy performance is present
        if sample.noisy_perf is not None:
            self._process_sequence(i, seq=sample.noisy_perf, seq_data=data.noisy_performances)

        # process note segments if present
        seq_len = len(sample.score)
        if sample.segments is not None:
            data.segments.bar[i, :seq_len] = torch.from_numpy(sample.segments.bar)
            data.segments.beat[i, :seq_len] = torch.from_numpy(sample.segments.beat)
            data.segments.onset[i, :seq_len] = torch.from_numpy(sample.segments.onset)

        # process directions if present
        if sample.directions is not None:
            for j, (group_name, group_directions) in enumerate(sample.directions.items()):
                for (label, key), direction_map in group_directions.items():
                    mask = direction_map != 0.
                    if np.any(mask):
                        data.directions[i, :seq_len, j][mask] = label * torch.from_numpy(direction_map[mask])

        data.deadpan_mask[i] = sample.is_deadpan

    def __call__(self, batch: List[ScorePerformanceSample], inference: bool = False, return_tensors: bool = True):
        data = self.init_data(batch, inference=inference)
        for i, sample in enumerate(batch):
            self.process_sample(i, sample, data)

        return data


# FOR LANGUAGE MODELING
@dataclass
class LMScorePerformanceInputs(ScorePerformanceInputs):
    labels: Optional[SeqInputs] = None


class LMScorePerformanceCollator(ScorePerformanceCollator, LMPerformanceCollator):
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
        LMPerformanceCollator.__init__(
            self,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            mlm=mlm,
            mask_prob=mask_prob,
            replace_prob=replace_prob,
            mask_token_id=mask_token_id,
            mask_ignore_token_ids=mask_ignore_token_ids,
            mask_ignore_token_dims=mask_ignore_token_dims,
            label_pad_ignored_dims=label_pad_ignored_dims,
            label_pad_token_id=label_pad_token_id
        )

    def __call__(self, batch: List[ScorePerformanceSample], inference: bool = False, return_tensors: bool = True):
        data = super().__call__(batch, inference=inference)

        if self.mlm:
            masked_seq, labels, label_mask = self.mask_sequence(data.performances.tokens)
            data.performances.tokens = masked_seq
        else:
            labels = data.performances.tokens.clone().detach()
            labels[labels == self.pad_token_id] = self.label_pad_token_id
            label_mask = data.performances.mask.clone().detach()

        data = LMScorePerformanceInputs(
            scores=data.scores,
            performances=data.performances,
            noisy_performances=data.noisy_performances,
            segments=data.segments,
            directions=data.directions,
            deadpan_mask=data.deadpan_mask,
            labels=SeqInputs(
                tokens=labels,
                mask=label_mask,
                lengths=data.performances.lengths
            )
        )

        return data


@dataclass
class MixedLMScorePerformanceInputs(LMScorePerformanceInputs):
    masked_performances: Optional[SeqInputs] = None


class MixedLMScorePerformanceCollator(ScorePerformanceCollator, MixedLMPerformanceCollator):
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
        MixedLMPerformanceCollator.__init__(
            self,
            pad_token_id=pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
            mask_token_id=mask_token_id,
            mask_ignore_token_ids=mask_ignore_token_ids,
            mask_ignore_token_dims=mask_ignore_token_dims,
            label_pad_ignored_dims=label_pad_ignored_dims,
            label_pad_token_id=label_pad_token_id
        )

    def __call__(self, batch: List[ScorePerformanceSample], inference: bool = False, return_tensors: bool = True):
        data = super().__call__(batch, inference=inference)

        masked_performances, labels = self.mask_sequence(data.performances.tokens)
        label_mask = data.performances.mask.clone().detach()

        data = MixedLMScorePerformanceInputs(
            scores=data.scores,
            performances=data.performances,
            noisy_performances=data.noisy_performances,
            segments=data.segments,
            directions=data.directions,
            deadpan_mask=data.deadpan_mask,
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
