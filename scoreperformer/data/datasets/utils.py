""" Common datasets' utils. """

import random
from typing import Optional, Dict, List

import numpy as np

from ..tokenizers import OctupleM, TokSequence


def load_tokens_data(path, tokenizer):
    data = tokenizer.load_tokens(path)
    if isinstance(data, list):  # backward compatibility with old datasets (miditok<=1.2.3)
        data = {'ids': data[0], 'programs': data[1] if len(data) > 1 else []}
    elif 'ids' not in data:  # backward compatibility with old datasets (miditok<=2.0.0)
        data['ids'] = data['tokens']
        del data['tokens']
    return data


def load_tokens_np(path, tokenizer):
    return np.array(load_tokens_data(path, tokenizer)['ids'])


def load_token_sequence(path, tokenizer):
    data = load_tokens_data(path, tokenizer)
    return TokSequence(ids=data['ids'], meta=data.get('meta', {}))


def get_num_bars(seq, tokenizer):
    if isinstance(tokenizer, OctupleM):
        bar_idx = tokenizer.vocab_types_idx['Bar']
        return int(seq[-1, bar_idx] - tokenizer.zero_token + 1)
    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer.__class__.__name__}")


def compute_bar_sample_positions(seq_num_bars, bar_sliding_window):
    bar_shift = bar_sliding_window
    length, sample_positions = 0, []
    for num_bars in seq_num_bars:
        back_shift = -bar_shift // 4 if (num_bars - bar_shift // 2) % bar_shift == 0 else 0
        positions = np.concatenate([
            np.arange(0, num_bars - bar_shift // 2, bar_shift),
            np.arange(num_bars - bar_shift // 2 - back_shift, -1 + bar_shift // 2, -bar_shift)
        ])
        length += len(positions)
        sample_positions.append(positions)

    sample_ids = np.concatenate([[0], np.cumsum(list(map(len, sample_positions)))[:-1]])
    sample_positions = np.concatenate(sample_positions)

    return length, sample_positions, sample_ids


def get_end_bar(score_indices, start_bar=0, max_seq_len=512, max_bar=256):
    end_bar = np.where(score_indices <= score_indices[start_bar] + max_seq_len)[0][-1] - 1
    return min(max(start_bar, end_bar), start_bar + max_bar - 1)


def split_composer_metadata(
        reference_metadata: Dict[str, List[str]],
        splits: Dict[str, float],
        seed: Optional[int] = None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    data_ = {split: dict() for split in splits}

    for comp, score_perf in reference_metadata.items():
        comp_meta_rep = []

        score_perf = list(score_perf.items())
        np.random.shuffle(score_perf)
        score_perf = dict(score_perf)

        for score, perfs in score_perf.items():
            comp_meta_rep.extend([score] * len(perfs))

        if len(comp_meta_rep) > 10:
            start = 0
            for i, (split, ratio) in enumerate(splits.items()):
                end = min(len(comp_meta_rep), start + round(ratio * len(comp_meta_rep)))

                if i == len(splits) - 1:
                    end = len(comp_meta_rep)

                if end < len(comp_meta_rep) and comp_meta_rep[end - 1] == comp_meta_rep[len(comp_meta_rep) - 1]:
                    while end > 0 and comp_meta_rep[end] == comp_meta_rep[end - 1]:
                        end -= 1
                else:
                    while end < len(comp_meta_rep) and comp_meta_rep[end - 1] == comp_meta_rep[end]:
                        end += 1

                split_scores = np.unique(comp_meta_rep[start:end]).tolist()
                for score in split_scores:
                    data_[split][score] = score_perf[score]
                start = end
        else:
            for score, perfs in score_perf.items():
                _split = np.random.choice(np.array(list(splits.keys())), p=np.array(list(splits.values())))
                data_[_split][score] = perfs

    for _split in data_:
        data_[_split] = dict(sorted(data_[_split].items()))

    return data_
