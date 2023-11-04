""" Performance token sequence dataset. """

import copy
import os
from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from scoreperformer.utils import apply, load_json
from .token_sequence import load_token_sequence, LocalTokenSequenceDataset
from .utils import load_tokens_np, get_num_bars, compute_bar_sample_positions, get_end_bar
from ..helpers import (
    TOKEN_SEQUENCE_PROCESSORS,
    TOKEN_SEQUENCE_INDEXERS,
    TokenSequenceAugmentations
)
from ..tokenizers import TOKENIZERS


@dataclass
class PerformanceSampleMeta:
    idx: Optional[int]
    perf_idx: int
    start_bar: int
    end_bar: Optional[int]
    bar_offset: int = 0
    augmentations: Optional[TokenSequenceAugmentations] = None


@dataclass
class PerformanceSample:
    perf: np.ndarray
    meta: PerformanceSampleMeta


class PerformanceDataset(Dataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            encoding: str = 'OctupleM',

            max_seq_len: int = 512,
            max_bar: int = 256,
            bar_sliding_window: int = 16,

            fit_to_max_bar: bool = False,
            fit_to_zero_bar: bool = False,
            sample_bars: bool = False,

            add_sos_eos: bool = False,

            sample: bool = False,
            seed: int = 23,

            augment_performance: bool = False,
            pitch_shift_range: Tuple[int, int] = (-3, 3),
            velocity_shift_range: Tuple[int, int] = (-2, 2),
            tempo_shift_range: Tuple[int, int] = (-2, 2),

            cache: bool = True,
            **kwargs
    ):

        self.root = root
        self.split = split

        # load metadata
        metadata_file = os.path.join(self.root, 'metadata.json')
        metadata = load_json(metadata_file)

        if all(key not in metadata for key in ['all', 'train', 'eval', 'val', 'test']):
            self.metadata = metadata  # metadata of names
        else:
            self.metadata = metadata[self.split]

        self.performance_names = list(self.metadata)

        # load tokenizer
        if encoding not in TOKENIZERS:
            raise ValueError(f"Encoding {encoding} is not a valid encoding, "
                             f"supported types are: {list(TOKENIZERS.keys())}")

        self.encoding = encoding
        self.tokenizer = TOKENIZERS[encoding](params=os.path.join(self.root, 'config.json'))

        # load sequences
        load_tokens = partial(load_tokens_np, tokenizer=self.tokenizer)
        perf_load_fn = partial(load_token_sequence, load_fn=load_tokens)
        self.performances = LocalTokenSequenceDataset(
            root=self.root,
            files=self.performance_names,
            load_fn=perf_load_fn,
            cache=cache
        )

        # configurations
        self.max_seq_len = max_seq_len
        self.max_bar = max_bar
        self.bar_sliding_window = bar_sliding_window
        self.add_sos_eos = add_sos_eos
        assert max_bar <= self.tokenizer.max_bar_embedding

        # bar indexer and indices arrays
        self.indexer = TOKEN_SEQUENCE_INDEXERS[encoding](self.tokenizer)
        self._bar_indices = [None] * len(self.performances)

        # load or compute number of bars in performances used to build samples
        bars_file = os.path.join(self.root, 'bars.json')
        if os.path.exists(bars_file):
            _num_bars = load_json(bars_file)
            _perf_num_bars = np.array([_num_bars[perf] for perf in self.performance_names])
        else:
            _perf_num_bars = np.array(apply(self.performances, partial(get_num_bars, tokenizer=self.tokenizer)))

        # compute sample positions
        self._length, self._sample_positions, self._sample_ids = compute_bar_sample_positions(
            seq_num_bars=_perf_num_bars, bar_sliding_window=self.bar_sliding_window
        )

        # random effects they do not advertise
        self.sample = sample
        if self.sample:
            np.random.seed(seed)

        # bar sampling
        assert not (fit_to_max_bar and fit_to_zero_bar), \
            "Only one of `fit_to_max_bar`/`fit_to_zero_bar` could be set to True"
        self.fit_to_max_bar = fit_to_max_bar
        self.fit_to_zero_bar = fit_to_zero_bar
        self.sample_bars = sample and sample_bars

        # augmentations
        self.augment_performance = sample and augment_performance

        if not self.augment_performance:
            pitch_shift_range = velocity_shift_range = tempo_shift_range = (0, 0)

        # sequence processor
        self.processor = TOKEN_SEQUENCE_PROCESSORS[encoding](
            tokenizer=self.tokenizer,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_shift_range=tempo_shift_range,
        )

    def _get_augmentations(self, meta):
        if meta is None:
            if self.augment_performance:
                return self.processor.sample_augmentations()
            else:
                return None
        else:
            return meta.augmentations

    def _augment_sequence(self, seq, augmentations):
        if augmentations is None:
            return seq

        seq = self.processor.augment_sequence(seq, augmentations)
        mask = self.processor.compute_valid_pitch_mask(seq)
        return seq[mask]

    def get(self, idx: Optional[int] = None, meta: Optional[PerformanceSampleMeta] = None):
        assert idx is not None or meta is not None, 'one of `idx`/`meta` should be provided as an argument'

        # get performance
        if meta is None:
            perf_idx = np.where(idx >= self._sample_ids)[0][-1]
        else:
            idx, perf_idx = meta.idx, meta.perf_idx

        bar_indices = self._bar_indices[perf_idx]
        if bar_indices is None:
            bar_indices = self._bar_indices[perf_idx] = self.indexer.compute_bar_indices(self.performances[perf_idx])

        total_bars = bar_indices.shape[0] - 1

        # compute start bar index
        if meta is None:
            start_bar = self._sample_positions[idx]
            start_bar = min(start_bar, bar_indices.shape[0] - self.bar_sliding_window // 2)  # bars of silent notes
            if self.sample:
                low = max(0, start_bar - self.bar_sliding_window // 2)
                high = min(total_bars - self.bar_sliding_window // 4, start_bar + self.bar_sliding_window // 2)
                high = max(low + 1, high)
                start_bar = np.random.randint(low, high)
        else:
            start_bar = meta.start_bar

        # compute start index
        perf_start = bar_indices[start_bar]

        # compute end bar index
        if meta is None or meta.end_bar is None:
            end_bar = get_end_bar(bar_indices, start_bar, self.max_seq_len, self.max_bar)
        else:
            end_bar = meta.end_bar

        # compute end index
        perf_end = bar_indices[end_bar + 1]

        # get token sequences
        perf_seq = copy.copy(self.performances[perf_idx][perf_start:perf_end])

        min_bar = perf_seq[:, 0].min() - self.tokenizer.zero_token
        max_bar = perf_seq[:, 0].max() - self.tokenizer.zero_token

        # shift bar indices
        bar_offset = 0
        if meta is None:
            if self.fit_to_max_bar:
                # to make bar index distribute in [0, bar_max)
                if self.sample_bars:
                    bar_offset = np.random.randint(-min_bar, self.max_bar - max_bar)
                elif end_bar >= self.max_bar:
                    # move in proportion to `score_total_bars`
                    _end_bar = int((self.max_bar - 1) * max_bar / total_bars)
                    bar_offset = _end_bar - max_bar
            elif self.fit_to_zero_bar:
                bar_offset = -min_bar
        else:
            bar_offset = meta.bar_offset

        if bar_offset != 0:
            perf_seq[:, self.tokenizer.vocab_types_idx['Bar']] += bar_offset

        # augmentations
        augmentations = self._get_augmentations(meta)
        perf_seq = self._augment_sequence(perf_seq, augmentations)

        if self.add_sos_eos:
            if start_bar == 0:
                perf_seq = self.processor.add_sos_token(perf_seq)
            if end_bar + 1 == total_bars:
                perf_seq = self.processor.add_eos_token(perf_seq)

        # build sample metadata
        meta = PerformanceSampleMeta(
            idx=idx,
            perf_idx=perf_idx,
            start_bar=start_bar,
            end_bar=end_bar,
            bar_offset=bar_offset,
            augmentations=augmentations
        )

        return PerformanceSample(
            perf=perf_seq,
            meta=meta
        )

    def __getitem__(self, idx: int):
        return self.get(idx=idx)

    def __len__(self):
        return self._length
