""" Score-Performance token sequence datasets. """

import copy
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

import numpy as np
from omegaconf import ListConfig, DictConfig
from torch.utils.data import Dataset

from scoreperformer.utils import exists, prob2bool, load_json, dump_json
from .token_sequence import load_token_sequence, TokenSequenceDataset, LocalTokenSequenceDataset
from .utils import load_tokens_np, get_num_bars, compute_bar_sample_positions, get_end_bar
from ..helpers import (
    TupleTokenSequenceProcessor,
    TupleTokenSequenceIndexer,
    TokenSequenceAugmentations
)
from ..tokenizers import TOKENIZERS, SPMupleBase, TokSequence


@dataclass
class NoteSegments:
    bar: np.ndarray
    beat: np.ndarray
    onset: np.ndarray


@dataclass
class ScorePerformanceSampleMeta:
    idx: Optional[int]
    score_idx: int
    perf_idx: int
    start_bar: int
    end_bar: Optional[int]
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    bar_offset: int = 0
    note_shifts: Tuple[int, int] = (0, 0)
    augmentations: Optional[TokenSequenceAugmentations] = None
    noisy_augmentations: Optional[TokenSequenceAugmentations] = None
    is_deadpan: bool = False


@dataclass
class ScorePerformanceSample:
    score: np.ndarray
    perf: np.ndarray
    meta: ScorePerformanceSampleMeta
    noisy_perf: Optional[np.ndarray] = None
    segments: Optional[NoteSegments] = None
    directions: Optional[Dict[str, Dict[Tuple[int, str], np.ndarray]]] = None
    is_deadpan: bool = False


class SequenceTypes(str, Enum):
    score = "score"
    performance = "performance"
    noisy_performance = "noisy_performance"


class ScorePerformanceDataset(Dataset):
    def __init__(
            self,
            scores: TokenSequenceDataset,
            performances: TokenSequenceDataset,
            metadata: Dict[str, List[str]],
            tokenizer: Union[SPMupleBase, Dict[str, object]],
            alignments: Optional[Dict[str, np.ndarray]] = None,
            auxiliary_data: Optional[Dict[str, object]] = None,
            performance_directions: Optional[Union[str, Path, List[str], Dict[str, List[str]]]] = None,
            score_directions_dict: Optional[Union[str, Path, Dict[str, List[dict]]]] = None,

            max_seq_len: int = 512,
            max_bar: int = 256,
            bar_sliding_window: int = 16,

            sample_bars: Union[bool, float] = False,
            sample_note_shift: Union[bool, float] = False,
            force_max_seq_len: Union[bool, float] = False,

            fit_to_max_bar: bool = False,
            fit_to_zero_bar: bool = False,
            sample_bar_offset: Union[bool, float] = False,

            add_sos_eos: bool = False,

            sample: bool = False,
            seed: int = 23,

            augment_performance: Union[bool, float] = False,
            pitch_shift_range: Tuple[int, int] = (-3, 3),
            velocity_shift_range: Tuple[int, int] = (-2, 2),
            tempo_shift_range: Tuple[int, int] = (-2, 2),

            noisy_performance: bool = False,
            noise_strength: float = 0.5,
            noisy_random_bars: Union[bool, float] = 0.5,

            deadpan_performance: Union[bool, float] = False,

            **kwargs
    ):
        self.metadata = metadata

        self.performance_names = list(sorted(set(chain.from_iterable(self.metadata.values()))))
        self.score_names = list(sorted(self.metadata.keys()))

        self._performance_map = {
            perf: (score, idx)
            for score, performances in self.metadata.items()
            for idx, perf in enumerate(performances)
        }

        self.scores = scores
        self.performances = performances

        # perf-to-score alignments
        self.alignments = alignments

        # load tokenizer
        if isinstance(tokenizer, dict):
            encoding = TOKENIZERS[tokenizer['tokenization']]
            self.tokenizer = encoding(params=tokenizer)
        else:
            self.tokenizer = tokenizer
        self.encoding = self.tokenizer.__class__.__name__

        # augmentations
        self.augment_performance = augment_performance
        self.noisy_performance = noisy_performance

        if self.augment_performance == 0. and not self.noisy_performance:
            pitch_shift_range = velocity_shift_range = tempo_shift_range = (0, 0)

        self.noise_strength = noise_strength
        self.noisy_random_bars = noisy_random_bars

        # sequence processor
        self.processor = TupleTokenSequenceProcessor(
            tokenizer=self.tokenizer,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_shift_range=tempo_shift_range,
        )

        # set up auxiliary data
        if auxiliary_data is not None:
            for key, data in auxiliary_data.items():
                setattr(self, key, data)

        # configurations
        self.max_seq_len = max_seq_len
        self.max_bar = max_bar
        self.bar_sliding_window = bar_sliding_window
        self.add_sos_eos = add_sos_eos
        assert max_bar <= self.tokenizer.config.additional_params["max_bar_embedding"]

        # bar indexer and indices arrays
        self.indexer = TupleTokenSequenceIndexer(self.tokenizer)
        self._score_indices = [None] * len(self.scores)
        self._perf_indices = [None] * len(self.performances)

        # load or compute number of bars in performances used to build samples
        self.bars = getattr(self, "bars", {})
        for perf_idx, perf in enumerate(self.performance_names):
            if perf not in self.bars:
                self.bars[perf] = get_num_bars(self.performances[perf_idx], tokenizer=self.tokenizer)
        _perf_num_bars = np.array([self.bars[perf] for perf in self.performance_names])

        # compute sample positions
        self._length, self._sample_positions, self._sample_ids = compute_bar_sample_positions(
            seq_num_bars=_perf_num_bars, bar_sliding_window=self.bar_sliding_window
        )

        # compute beat note maps
        self._beat_maps, self._onset_maps = [], []
        for score_seq in self.scores:
            ticks_data = self.tokenizer.compute_ticks(score_seq, compute_beat_ticks=True)
            self._beat_maps.append(np.searchsorted(ticks_data['beat'], ticks_data['note_on'], side='right') - 1)

            unique_onsets, onset_notes = np.unique(ticks_data['note_on'], return_counts=True)
            self._onset_maps.append(np.arange(len(unique_onsets)).repeat(onset_notes))

        # random effects they do not advertise
        self.sample = sample
        if self.sample:
            random.seed(seed)
            np.random.seed(seed)

        # sequence sampling
        self.sample_bars = sample_bars
        self.sample_note_shift = sample_note_shift
        self.force_max_seq_len = force_max_seq_len

        # bar of the first note
        assert not (fit_to_max_bar and fit_to_zero_bar), \
            "Only one of `fit_to_max_bar`/`fit_to_zero_bar` could be set to True"
        self.fit_to_max_bar = fit_to_max_bar
        self.fit_to_zero_bar = fit_to_zero_bar
        self.sample_bar_offset = sample_bar_offset

        # occasional score-based deadpan performances
        self.deadpan_performance = deadpan_performance

        # performance directions data
        if isinstance(performance_directions, (str, Path)):
            with open(performance_directions, 'r') as f:
                performance_directions = json.load(f)

        performance_direction_sizes = None
        if performance_directions is not None:
            assert score_directions_dict is not None, \
                "`score_directions_dict` should be provided with `performance_directions`"
            if isinstance(performance_directions, (list, ListConfig)):
                performance_directions = {"directions": list(performance_directions)}
            elif isinstance(performance_directions, DictConfig):
                performance_directions = dict(performance_directions)

            performance_direction_sizes = {
                key: len(performance_directions[key]) + 1 for key in performance_directions
            }

        self.performance_directions = performance_directions
        self.performance_direction_sizes = performance_direction_sizes

        # score-direction maps
        if isinstance(score_directions_dict, (str, Path)):
            with open(score_directions_dict, 'r') as f:
                score_directions_dict = json.load(f)

        self.score_direction_maps = None
        if score_directions_dict is not None:
            from .directions import build_score_direction_maps
            performance_directions = [
                item for group_keys in self.performance_directions.values() for item in group_keys
            ]
            self.score_direction_maps = build_score_direction_maps(
                self, score_directions_dict, direction_keys=performance_directions
            )['score']['note']

    def get_direction_class_weights(self):
        directions_nums = {}
        for group_name, group_directions in self.performance_directions.items():
            directions_nums[group_name] = defaultdict(int)

        none_key = (0, 'none')
        total_notes = 0
        for score_idx, score in enumerate(self.score_names):
            score_direction_note_maps = self.score_direction_maps[score_idx]
            total_notes += len(self.scores[score_idx]) * len(self.metadata[score])
            for group_name, group_directions in self.performance_directions.items():
                directions_nums[group_name][none_key] += len(self.scores[score_idx]) * len(self.metadata[score])
                for i, key in enumerate(group_directions):
                    if key in score_direction_note_maps:
                        num_notes = int(score_direction_note_maps[key].sum())
                    else:
                        num_notes = 0
                    directions_nums[group_name][(i + 1, key)] += num_notes * len(self.metadata[score])

        weights = {}
        for group_name, group_directions in self.performance_directions.items():
            not_empty = sum(directions_nums[group_name].values()) - directions_nums[group_name][none_key]
            directions_nums[group_name][none_key] = (total_notes - not_empty) / total_notes

            for i, key in enumerate(group_directions):
                directions_nums[group_name][(i + 1, key)] /= total_notes

            weights[group_name] = list(directions_nums[group_name].values())

        return directions_nums, weights

    def _get_augmentations(self, meta: ScorePerformanceSampleMeta, is_noisy_perf: bool = False):
        if meta is None:
            if self.sample and prob2bool(self.augment_performance) and not is_noisy_perf:
                return self.processor.sample_augmentations()
            elif self.sample and self.noisy_performance and is_noisy_perf:
                return self.processor.sample_augmentations(multiplier=self.noise_strength)
            else:
                return None
        elif is_noisy_perf:
            return meta.noisy_augmentations
        else:
            return meta.augmentations

    def _augment_sequence(
            self,
            seq: np.ndarray,
            augmentations: Optional[TokenSequenceAugmentations] = None,
            is_perf: bool = True
    ):
        if augmentations is None:
            return seq, np.ones_like(seq[:, 0]).astype(bool)

        if not is_perf:
            augmentations = copy.deepcopy(augmentations)
            augmentations.velocity_shift = 0
            augmentations.tempo_shift = 0

        seq = self.processor.augment_sequence(seq, augmentations)
        mask = self.processor.compute_valid_pitch_mask(seq)
        return seq[mask], mask

    def get(self, idx: Optional[int] = None, meta: Optional[ScorePerformanceSampleMeta] = None):
        assert exists(idx) or exists(meta), 'one of `idx`/`meta` should be provided as an argument'

        # get performance
        if meta is None:
            perf_idx = np.where(idx >= self._sample_ids)[0][-1]
        else:
            idx, perf_idx = meta.idx, meta.perf_idx
        perf = self.performance_names[perf_idx]

        # get score
        score, score_perf_idx = self._performance_map[perf]
        score_idx = self.scores._name_to_idx[score]

        score_indices = self._score_indices[score_idx]
        if score_indices is None:
            score_indices = self._score_indices[score_idx] = self.indexer.compute_bar_indices(self.scores[score_idx])
        perf_indices = self._perf_indices[perf_idx]
        if perf_indices is None:
            perf_indices = self._perf_indices[perf_idx] = self.indexer.compute_bar_indices(self.performances[perf_idx])

        score_total_bars = score_indices.shape[0] - 1
        perf_total_bars = perf_indices.shape[0] - 1
        score_total_notes = self.scores[score_idx].shape[0]

        # compute start bar index
        if meta is None:
            start_bar = self._sample_positions[idx]
            start_bar = min(start_bar, perf_indices.shape[0] - self.bar_sliding_window // 2)  # bars of silent notes
            if self.sample and prob2bool(self.sample_bars):
                low = max(0, start_bar - self.bar_sliding_window // 2)
                high = min(min(score_total_bars, perf_total_bars) - self.bar_sliding_window // 4,
                           start_bar + self.bar_sliding_window // 2)
                high = max(low + 1, high)
                start_bar = np.random.randint(low, high)
        else:
            start_bar = meta.start_bar

        # compute start indices
        score_start = score_indices[start_bar]
        perf_start = perf_indices[start_bar]

        # compute end bar index
        if meta is None or meta.end_bar is None:
            end_bar = get_end_bar(score_indices, start_bar, self.max_seq_len, self.max_bar)
        else:
            end_bar = meta.end_bar

        # compute end indices
        score_end = score_indices[end_bar + 1]
        perf_end = perf_indices[min(end_bar + 1, perf_total_bars)]

        # if bar does not fit or overfits `max_seq_len`
        if score_start == score_end or score_end - score_start > self.max_seq_len:
            score_end = min(score_end, score_start + self.max_seq_len)
            perf_end = min(perf_end, perf_start + self.max_seq_len)

        # sample note shifts to avoid sequence starting from the 1st bar note
        if meta is None:
            start_note_shift = end_note_shift = 0
            if self.sample and prob2bool(self.sample_note_shift):
                low = max(-score_start, -self.max_seq_len // 4)
                high = min(score_total_notes - score_start - self.max_seq_len // 4, self.max_seq_len // 4)
                start_note_shift = end_note_shift = np.random.randint(low, high)
                end_note_shift = min(end_note_shift, score_total_notes - score_end)

            # force `max_seq_len` even if the sequence is shorter (note: for the sequence tail might be no-op)
            if prob2bool(self.force_max_seq_len):
                end_note_shift += min(
                    self.max_seq_len - score_end + score_start,
                    score_total_notes - score_end - end_note_shift
                )
        else:
            start_note_shift, end_note_shift = meta.note_shifts

        score_start, perf_start = map(lambda x: x + start_note_shift, (score_start, perf_start))
        score_end, perf_end = map(lambda x: x + end_note_shift, (score_end, perf_end))

        # get token sequences
        score_seq = copy.copy(self.scores[score_idx][score_start:score_end])

        if self.alignments is not None:
            alignment = self.alignments[perf]
            perf_indices = alignment[np.arange(score_start, score_end)]
            perf_seq = copy.copy(self.performances[perf_idx][perf_indices])
        else:
            perf_seq = copy.copy(self.performances[perf_idx][perf_start:perf_end])

        min_bar = perf_seq[:, 0].min() - self.tokenizer.zero_token
        min_bar = min(min_bar, score_seq[:, 0].min() - self.tokenizer.zero_token)

        max_bar = perf_seq[:, 0].max() - self.tokenizer.zero_token
        max_bar = max(max_bar, score_seq[:, 0].max() - self.tokenizer.zero_token)

        # bar/beat note maps
        bar_segments = score_seq[:, 0] - self.tokenizer.zero_token
        beat_segments = self._beat_maps[score_idx][score_start:score_end]
        onset_segments = self._onset_maps[score_idx][score_start:score_end]
        bar_segments, beat_segments, onset_segments = map(
            lambda s: s - s[0] + self.tokenizer.zero_token,
            (bar_segments, beat_segments, onset_segments)
        )

        # shift bar indices
        bar_offset = 0
        if meta is None:
            if self.fit_to_max_bar:
                # to make bar index distribute in [0, bar_max)
                if self.sample and self.sample_bar_offset:
                    bar_offset = np.random.randint(-min_bar, self.max_bar - max_bar)
                elif end_bar >= self.max_bar:
                    # move in proportion to `score_total_bars`
                    _end_bar = int((self.max_bar - 1) * max_bar / score_total_bars)
                    bar_offset = _end_bar - max_bar
            elif self.fit_to_zero_bar:
                bar_offset = -min_bar
        else:
            bar_offset = meta.bar_offset

        if bar_offset != 0:
            score_seq[:, self.tokenizer.vocab_types_idx['Bar']] += bar_offset
            perf_seq[:, self.tokenizer.vocab_types_idx['Bar']] += bar_offset

        # augmentations
        augmentations = self._get_augmentations(meta)
        score_seq, mask = self._augment_sequence(score_seq, augmentations, is_perf=False)
        perf_seq, _ = self._augment_sequence(perf_seq, augmentations, is_perf=True)

        # select subset of segments for left notes
        bar_segments, beat_segments, onset_segments = map(
            lambda s: s[mask], (bar_segments, beat_segments, onset_segments)
        )

        # noisy performance
        noisy_perf_seq = noisy_augmentations = None
        if self.noisy_performance:
            noisy_augmentations = self._get_augmentations(meta, is_noisy_perf=True)
            noisy_perf_seq = perf_seq.copy()
            noisy_perf_seq, _ = self._augment_sequence(noisy_perf_seq, noisy_augmentations, is_perf=True)
            if noisy_perf_seq.shape[0] < perf_seq.shape[0]:
                noisy_perf_seq = perf_seq.copy()  # pitch overflow, omit by reverting changes for now

            if prob2bool(self.noisy_random_bars):
                bar_ids = np.arange(self.max_bar)
                np.random.shuffle(bar_ids)
                bar_0 = self.tokenizer.zero_token
                noisy_perf_seq[:, 0] = bar_ids[noisy_perf_seq[:, 0] - bar_0] + bar_0

        # deadpan performance
        use_deadpan = self.sample and prob2bool(self.deadpan_performance) if meta is None else meta.is_deadpan
        if use_deadpan:
            # all previous performance processing made no sense, we love some deadpan performance
            perf_seq = np.array(self.tokenizer.score_tokens_as_performance(TokSequence(ids=score_seq.tolist())).ids)

        # sequence boundaries
        if self.add_sos_eos:
            if score_start == 0:
                score_seq = self.processor.add_sos_token(score_seq)
                perf_seq = self.processor.add_sos_token(perf_seq)
                noisy_perf_seq = self.processor.add_sos_token(noisy_perf_seq) if exists(noisy_perf_seq) else None
                bar_segments, beat_segments, onset_segments = map(
                    lambda s: np.concatenate([[s[0]], s]), (bar_segments, beat_segments, onset_segments)
                )
            if score_end == score_total_notes:
                score_seq = self.processor.add_eos_token(score_seq)
                perf_seq = self.processor.add_eos_token(perf_seq)
                noisy_perf_seq = self.processor.add_eos_token(noisy_perf_seq) if exists(noisy_perf_seq) else None
                bar_segments, beat_segments, onset_segments = map(
                    lambda s: np.concatenate([s, [s[-1]]]), (bar_segments, beat_segments, onset_segments)
                )

        # note performance direction labels
        directions = {}
        if self.performance_directions is not None:
            score_direction_note_maps = self.score_direction_maps[score_idx]
            for group_name, group_directions in self.performance_directions.items():
                directions[group_name] = {}
                for i, key in enumerate(group_directions):
                    if key in score_direction_note_maps:
                        note_map = copy.copy(score_direction_note_maps[key][score_start:score_end])[mask]
                        if self.add_sos_eos:
                            note_map = np.concatenate([[0], note_map]) if score_start == 0 else note_map
                            note_map = np.concatenate([note_map, [0]]) if score_end == score_total_notes else note_map
                    else:
                        note_map = np.zeros(score_seq.shape[0])
                    directions[group_name][(i + 1, key)] = note_map.astype(int)  # 0 is for None

        # build sample metadata
        meta = ScorePerformanceSampleMeta(
            idx=idx,
            score_idx=score_idx,
            perf_idx=perf_idx,
            start_bar=start_bar,
            end_bar=end_bar,
            start_idx=score_start,
            end_idx=score_end,
            bar_offset=bar_offset,
            note_shifts=(start_note_shift, end_note_shift),
            augmentations=augmentations,
            noisy_augmentations=noisy_augmentations,
            is_deadpan=use_deadpan
        )

        return ScorePerformanceSample(
            score=score_seq,
            perf=perf_seq,
            meta=meta,
            noisy_perf=noisy_perf_seq,
            segments=NoteSegments(
                bar=bar_segments,
                beat=beat_segments,
                onset=onset_segments
            ),
            directions=directions,
            is_deadpan=use_deadpan
        )

    def __getitem__(self, idx: int):
        return self.get(idx=idx)

    def __len__(self):
        return self._length


class LocalScorePerformanceDataset(ScorePerformanceDataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
            use_alignments: bool = False,
            auxiliary_data_keys: Optional[List[str]] = None,
            save_auxiliary_data: bool = True,
            performance_directions: Optional[Union[str, Path, List[str], Dict[str, List[str]], Path]] = None,
            score_directions_dict: Optional[Union[str, Path]] = None,

            max_seq_len: int = 512,
            max_bar: int = 256,
            bar_sliding_window: int = 16,

            sample_bars: Union[bool, float] = False,
            sample_note_shift: Union[bool, float] = False,
            force_max_seq_len: Union[bool, float] = False,

            fit_to_max_bar: bool = False,
            fit_to_zero_bar: bool = False,
            sample_bar_offset: Union[bool, float] = False,

            add_sos_eos: bool = False,

            sample: bool = False,
            seed: int = 23,

            augment_performance: Union[bool, float] = False,
            pitch_shift_range: Tuple[int, int] = (-3, 3),
            velocity_shift_range: Tuple[int, int] = (-2, 2),
            tempo_shift_range: Tuple[int, int] = (-2, 2),

            noisy_performance: bool = False,
            noise_strength: float = 0.5,
            noisy_random_bars: Union[bool, float] = 0.5,

            deadpan_performance: Union[bool, float] = False,

            zero_out_silent_durations: bool = True,
            delete_silent_notes: bool = False,

            preload: bool = False,
            cache: bool = True,
            **kwargs
    ):

        self.root = root
        self.split = split

        # load metadata
        metadata_file = os.path.join(self.root, 'metadata.json')
        metadata = load_json(metadata_file)

        if any(key in metadata for key in ['all', 'train', 'eval', 'val', 'test']):
            metadata = metadata[self.split]

        self.performance_names = list(sorted(set(chain.from_iterable(metadata.values()))))
        self.score_names = list(sorted(metadata.keys()))

        self._performance_map = {
            perf: (score, idx)
            for score, performances in metadata.items()
            for idx, perf in enumerate(performances)
        }

        # perf-to-score alignments
        alignments = None
        if use_alignments:
            alignment_file = os.path.join(self.root, 'alignments.json')
            if os.path.exists(alignment_file):
                alignments = {
                    key: np.array(values) for key, values in load_json(alignment_file).items()
                    if key in self._performance_map
                }

        # load tokenizer
        params_path = os.path.join(self.root, 'config.json')
        with open(params_path) as f:
            params = json.load(f)
        encoding = TOKENIZERS[params['tokenization']]
        tokenizer = encoding(params=params_path)

        # sequence processor for sequence loading
        processor = TupleTokenSequenceProcessor(tokenizer=tokenizer)

        # load sequences
        load_tokens = partial(load_tokens_np, tokenizer=tokenizer)
        seq_proc_funcs, perf_seq_proc_funcs = [], []
        if zero_out_silent_durations:  # silent notes have non-zero duration
            seq_proc_funcs.append(processor.zero_out_durations)
        if delete_silent_notes:  # remove silent notes from performances
            perf_seq_proc_funcs.append(processor.remove_silent_notes)

        score_load_fn = partial(load_token_sequence, load_fn=load_tokens, processing_funcs=seq_proc_funcs)
        scores = LocalTokenSequenceDataset(
            root=self.root,
            files=self.score_names,
            load_fn=score_load_fn,
            preload=preload,
            cache=cache
        )

        perf_load_fn = partial(
            load_token_sequence, load_fn=load_tokens, processing_funcs=seq_proc_funcs + perf_seq_proc_funcs
        )
        performances = LocalTokenSequenceDataset(
            root=self.root,
            files=self.performance_names,
            load_fn=perf_load_fn,
            preload=preload,
            cache=cache
        )

        # load auxiliary data
        auxiliary_data = {}
        if auxiliary_data_keys is not None:
            for key in auxiliary_data_keys:
                data_file = os.path.join(self.root, f'{key}.json')
                if os.path.exists(data_file):
                    auxiliary_data[key] = load_json(data_file)

        super().__init__(
            scores=scores,
            performances=performances,
            metadata=metadata,
            tokenizer=tokenizer,
            alignments=alignments,
            auxiliary_data=auxiliary_data,
            performance_directions=performance_directions,
            score_directions_dict=score_directions_dict,
            max_seq_len=max_seq_len,
            max_bar=max_bar,
            bar_sliding_window=bar_sliding_window,
            sample_bars=sample_bars,
            sample_note_shift=sample_note_shift,
            force_max_seq_len=force_max_seq_len,
            fit_to_max_bar=fit_to_max_bar,
            fit_to_zero_bar=fit_to_zero_bar,
            sample_bar_offset=sample_bar_offset,
            add_sos_eos=add_sos_eos,
            sample=sample,
            seed=seed,
            augment_performance=augment_performance,
            pitch_shift_range=pitch_shift_range,
            velocity_shift_range=velocity_shift_range,
            tempo_shift_range=tempo_shift_range,
            noisy_performance=noisy_performance,
            noise_strength=noise_strength,
            noisy_random_bars=noisy_random_bars,
            deadpan_performance=deadpan_performance
        )

        if save_auxiliary_data:
            for key in auxiliary_data_keys:
                data_file = os.path.join(self.root, f'{key}.json')
                data = getattr(self, key, None)
                if data is not None and (not os.path.exists(data_file) or len(data) != len(load_json(data_file))):
                    dump_json(data, data_file)

        for score in self.score_names:
            assert score in self.scores._name_to_idx, score
