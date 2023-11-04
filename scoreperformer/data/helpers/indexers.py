import numpy as np


class TokenSequenceIndexer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute_bar_indices(self, seq: np.ndarray) -> np.ndarray:
        ...


class TupleTokenSequenceIndexer(TokenSequenceIndexer):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def compute_bar_indices(self, seq: np.ndarray) -> np.ndarray:
        bar_index = self.tokenizer.vocab_types_idx['Bar']

        min_bar = seq[0, bar_index] - self.tokenizer.zero_token
        total_bars = seq[-1, bar_index] - self.tokenizer.zero_token + 1

        bar_diff = np.concatenate([[min_bar], np.diff(seq[:, bar_index])])
        bar_changes = np.where(bar_diff > 0)[0]

        bars = np.concatenate([[0], np.cumsum(bar_diff[bar_changes]), [total_bars]])
        bar_changes = np.concatenate([[0], bar_changes, [seq.shape[0]]])

        bar_indices = np.full(bars[-1] + 1, -1, dtype=np.int16)
        bar_indices[bars] = bar_changes

        for idx in range(len(bar_indices) - 1, 0, -1):
            if bar_indices[idx] == -1:
                bar_indices[idx] = bar_indices[idx + 1]

        return bar_indices
