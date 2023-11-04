""" Performance direction dataset and utilities. """

import json
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from scoreperformer.utils import find_closest
from .score_performance import ScorePerformanceDataset


def build_score_direction_maps(
        sp_dataset: ScorePerformanceDataset,
        score_directions_dict: Dict[str, List[dict]],
        direction_keys: Optional[List[str]] = None,
        bar_stretch: Optional[int] = None,
        time_division: int = 480,
        disable_tqdm: bool = True
):
    score_direction_bar_maps, score_direction_note_maps = [], []
    direction_score_bar_maps, direction_score_note_maps = {}, {}

    pbar = sp_dataset.score_names if disable_tqdm else tqdm(sp_dataset.score_names)
    for score_idx, score in enumerate(pbar):
        score_seq = sp_dataset.scores[score_idx]
        ticks_data = sp_dataset.tokenizer.compute_ticks(score_seq, time_division=time_division)
        score_notes, score_bars = ticks_data['note_on'], ticks_data['bar']

        score_directions = score_directions_dict[score]
        if len(score_directions) and isinstance(score_directions[0], list):
            score_directions = [d for part_directions in score_directions for d in part_directions]

        direction_bar_maps, direction_note_maps = {}, {}
        for d in score_directions:
            key = d['type']

            if direction_keys is not None and key not in direction_keys:
                continue

            if d.get('end', None) is not None:  # dynamic/tempo markings, note and bar maps
                # bar map
                if key not in direction_bar_maps:
                    direction_bar_maps[key] = np.zeros(score_bars.shape[0] - 1)
                bar_map = direction_bar_maps[key]

                start = find_closest(score_bars, d['start'])
                end = find_closest(score_bars, d['end'])
                end = max(end, start + 1)
                if bar_stretch is not None:
                    end = min(end, start + 1 + bar_stretch)

                bar_map[start:end] = 1

                # note map
                if key not in direction_note_maps:
                    direction_note_maps[key] = np.zeros(score_seq.shape[0])
                note_map = direction_note_maps[key]

                note_map[np.where(np.logical_and(d['start'] <= score_notes, d['end'] >= score_notes))] = 1

            elif d.get('offset', None) is not None:  # note articulation
                if key not in direction_note_maps:
                    direction_note_maps[key] = np.zeros(score_seq.shape[0])
                direction_note_maps[key][d['offset']] = 1

        score_direction_bar_maps.append(dict(sorted(direction_bar_maps.items())))
        score_direction_note_maps.append(dict(sorted(direction_note_maps.items())))

        # store in global map
        for key, bar_map in direction_bar_maps.items():
            if key not in direction_score_bar_maps:
                direction_score_bar_maps[key] = []
            if np.any(bar_map):
                direction_score_bar_maps[key].append((score_idx, bar_map))

        for key, note_map in direction_note_maps.items():
            if key not in direction_score_note_maps:
                direction_score_note_maps[key] = []
            if np.any(note_map):
                direction_score_note_maps[key].append((score_idx, note_map))

    direction_score_bar_maps = dict(sorted(direction_score_bar_maps.items()))
    direction_score_note_maps = dict(sorted(direction_score_note_maps.items()))

    return {
        'score': {
            'bar': score_direction_bar_maps,
            'note': score_direction_note_maps
        },
        'direction': {
            'bar': direction_score_bar_maps,
            'note': direction_score_note_maps
        }
    }


def get_all_direction_embeddings(
        sp_dataset: ScorePerformanceDataset,
        score_direction_maps: List[Dict[str, np.ndarray]],
        embeddings: Union[np.ndarray, torch.Tensor],
        performance_ids: Union[np.ndarray, torch.Tensor],
        key: str
):
    direction_embeddings = []
    for score_idx, score in enumerate(sp_dataset.score_names):
        dir_map = score_direction_maps[score_idx].get(key, None)
        if dir_map is None:
            continue

        for perf in sp_dataset.metadata[score]:
            perf_idx = sp_dataset.performances._name_to_idx[perf]
            perf_embs = embeddings[performance_ids == perf_idx]
            perf_dir_embs = perf_embs[dir_map == 1.]
            direction_embeddings.append(perf_dir_embs)

    return torch.cat(direction_embeddings, dim=0)


def get_direction_performances_map(
        sp_dataset: ScorePerformanceDataset,
        score_direction_maps: List[Dict[str, np.ndarray]],
        key: str,
        level: str = 'bar'
):
    dir_perf_map = []
    for score_idx, score in enumerate(sp_dataset.score_names):
        score_dir_map = score_direction_maps[score_idx]
        if key not in score_dir_map:
            score_seq = sp_dataset.scores[score_idx]
            if level == 'bar':
                dir_map = np.zeros(score_seq[-1, 0] - sp_dataset.tokenizer.zero_token + 1)
            else:
                dir_map = np.zeros(score_seq.shape[0])
        else:
            dir_map = score_dir_map[key]

        for _ in sp_dataset.metadata[score]:
            dir_perf_map.append(dir_map)

    return np.concatenate(dir_perf_map, axis=0).astype(bool)


def get_performance_idx_map(sp_dataset: ScorePerformanceDataset, level='bar'):
    perf_ids = []
    for score_idx, score in enumerate(sp_dataset.score_names):
        score_seq = sp_dataset.scores[score_idx]
        for perf in sp_dataset.metadata[score]:
            num_ids = score_seq[-1, 0] - sp_dataset.tokenizer.zero_token + 1 if level == 'bar' else score_seq.shape[0]
            perf_ids.append(np.full(num_ids, sp_dataset.performances._name_to_idx[perf]))

    return np.concatenate(perf_ids, axis=0)


class DirectionBarEmbeddingDataset(Dataset):
    def __init__(
            self,
            sp_dataset: ScorePerformanceDataset,
            direction_keys: List[str],

            embedding_data_dict: Optional[Union[str, Path]] = None,
            split: Optional[str] = None,
            embeddings: Optional[Union[np.ndarray, torch.Tensor]] = None,

            score_directions_dict: Optional[Union[str, Path, Dict[str, List[dict]]]] = None,
            direction_bar_stretch: Optional[int] = None,

            remove_multi_label: bool = False,
            negative_samples: float = 1.0,
            num_prev_embeddings: int = 0
    ):
        self.sp_dataset = sp_dataset
        self.direction_keys = direction_keys

        # embeddings
        assert embedding_data_dict is not None or embeddings is not None, \
            "One of `embedding_data_dict` and `embeddings` should be provided to DirectionEmbeddingDataset"

        if embeddings is None:
            embedding_data_dict = torch.load(embedding_data_dict, map_location="cpu")
            embeddings = embedding_data_dict[split]["perf_embs"]

        self.embeddings = embeddings

        # score-direction maps
        if isinstance(score_directions_dict, (str, Path)):
            with open(score_directions_dict, 'r') as f:
                score_directions_dict = json.load(f)

        self.score_direction_maps = build_score_direction_maps(
            sp_dataset, score_directions_dict, bar_stretch=direction_bar_stretch
        )['score']['bar']

        # build performance idx maps
        self.perf_ids = get_performance_idx_map(sp_dataset=sp_dataset)

        # build indices for each direction
        direction_maps = [
            (key, get_direction_performances_map(sp_dataset, self.score_direction_maps, key, level='bar'))
            for key in direction_keys
        ]

        # empty (no direction) map
        nodir_map = np.ones(embeddings.shape[0]).astype(bool)
        for _, dir_map in direction_maps:
            nodir_map[dir_map] = False
        direction_maps.insert(0, (None, nodir_map))

        # count multi-label embeddings and remove them
        if remove_multi_label:
            counts = np.zeros(embeddings.shape[0])
            for _, dir_map in direction_maps:
                counts[dir_map] += 1
            direction_maps = [
                (key, np.logical_and(dir_map, counts == 1.))
                for key, dir_map in direction_maps
            ]

        self.direction_maps = dict(direction_maps)

        # build labels ids
        self.labels = {key: i for i, key in enumerate(self.direction_maps.keys())}
        self.inv_labels = {i: key for i, key in enumerate(self.direction_maps.keys())}

        # compute number of direction embeddings and dataset length
        direction_numbers = {
            key: dir_perf_map.sum()
            for key, dir_perf_map in self.direction_maps.items()
        }

        # limit number of no-direction embeddings
        num_dir_embs = sum([num for key, num in direction_numbers.items() if key is not None])
        direction_numbers[None] = min(self.direction_maps[None].sum(), int(negative_samples * num_dir_embs))

        self.direction_numbers = direction_numbers
        self._length = sum(self.direction_numbers.values())

        # build sample ids
        sample_keys, sample_ids = [], []
        for key, dir_map in self.direction_maps.items():
            if key is None:
                continue

            sample_keys.extend([key] * direction_numbers[key])
            sample_ids.append(np.where(dir_map)[0])

        sample_keys.extend([None] * direction_numbers[None])
        sample_ids.append([-1] * direction_numbers[None])

        self._sample_keys = sample_keys
        self._sample_ids = np.concatenate(sample_ids)
        self._nodir_ids = np.where(self.direction_maps[None])[0]

        self.num_prev_embeddings = num_prev_embeddings

    def get_emb_by_idx(self, emb_idx):
        if self.num_prev_embeddings > 0:
            start_idx = emb_idx  # - self.num_prev_embeddings
            for _ in range(self.num_prev_embeddings):
                if start_idx == 0 or self.perf_ids[start_idx - 1] != self.perf_ids[emb_idx]:
                    break
                start_idx -= 1
            emb = self.embeddings[start_idx:emb_idx + 1]
        else:
            emb = self.embeddings[emb_idx]

        return emb

    def __getitem__(self, idx):
        label = self._sample_keys[idx]

        if label is None:
            emb_idx = self._nodir_ids[np.random.randint(0, self.direction_numbers[None])]  # sample arbitrary embedding
        else:
            emb_idx = self._sample_ids[idx]

        emb = self.get_emb_by_idx(emb_idx=emb_idx)
        label = self.labels[label]

        return emb_idx, emb, label

    def __len__(self):
        return self._length
