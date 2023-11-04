from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np

from ..tokenizers import OctupleM
from ..tokenizers.constants import SOS_TOKEN, EOS_TOKEN


def sample_integer_shift(low=-6, high=6):
    return np.random.randint(low, high + 1)


@dataclass
class TokenSequenceAugmentations:
    pitch_shift: Optional[int] = 0
    velocity_shift: Optional[int] = 0
    tempo_shift: Optional[int] = 0


class TokenSequenceProcessor:
    def __init__(
            self,
            pitch_shift_range=(-3, 3),
            velocity_shift_range=(-2, 2),
            tempo_shift_range=(-2, 2),
    ):
        self.pitch_shift_fn = partial(sample_integer_shift, *pitch_shift_range)
        self.velocity_shift_fn = partial(sample_integer_shift, *velocity_shift_range)
        self.tempo_shift_fn = partial(sample_integer_shift, *tempo_shift_range)

    def sample_augmentations(self, multiplier=1.0):
        return TokenSequenceAugmentations(
            pitch_shift=int(multiplier * self.pitch_shift_fn()),
            velocity_shift=int(multiplier * self.velocity_shift_fn()),
            tempo_shift=int(multiplier * self.tempo_shift_fn())
        )

    def augment_sequence(
            self,
            seq: np.ndarray,
            augmentations: TokenSequenceAugmentations
    ) -> np.ndarray:
        ...

    def sort_sequence(self, seq: np.ndarray) -> np.ndarray:
        ...

    def add_sos_token(self, seq: np.ndarray) -> np.ndarray:
        ...

    def add_eos_token(self, seq: np.ndarray) -> np.ndarray:
        ...


class TupleTokenSequenceProcessor(TokenSequenceProcessor):
    def __init__(
            self,
            tokenizer: OctupleM,
            pitch_shift_range=(-3, 3),
            velocity_shift_range=(-2, 2),
            tempo_shift_range=(-2, 2)
    ):
        super().__init__(pitch_shift_range, velocity_shift_range, tempo_shift_range)

        self.tokenizer = tokenizer

    def augment_sequence(
            self,
            seq: np.ndarray,
            augmentations: TokenSequenceAugmentations
    ) -> np.ndarray:
        ...
        if augmentations.pitch_shift != 0:
            pitch_index = self.tokenizer.vocab_types_idx['Pitch']
            seq[:, pitch_index] += augmentations.pitch_shift

        if augmentations.velocity_shift != 0:
            vel_index = self.tokenizer.vocab_types_idx['Velocity']
            vel_min, vel_max = self.tokenizer.zero_token, len(self.tokenizer.vocab[vel_index]) - 1

            seq[:, vel_index] += augmentations.velocity_shift
            seq[:, vel_index] = np.maximum(vel_min, np.minimum(vel_max, seq[:, vel_index]))

        if augmentations.tempo_shift != 0:
            tempo_index = self.tokenizer.vocab_types_idx['Tempo']
            tempo_min, tempo_max = self.tokenizer.zero_token, len(self.tokenizer.vocab[tempo_index]) - 1

            seq[:, tempo_index] += augmentations.tempo_shift
            seq[:, tempo_index] = np.maximum(tempo_min, np.minimum(tempo_max, seq[:, tempo_index]))

        return seq

    def sort_sequence(self, seq: np.ndarray) -> np.ndarray:
        seq = seq[np.lexsort((seq[:, self.tokenizer.vocab_types_idx['Pitch']],
                              seq[:, self.tokenizer.vocab_types_idx['Position']],
                              seq[:, self.tokenizer.vocab_types_idx['Bar']]))]
        return seq

    def add_sos_token(self, seq: np.ndarray, initial_tempo: Optional[int] = None) -> np.ndarray:
        sos_token_id = self.tokenizer[0, SOS_TOKEN]
        seq = np.concatenate((np.full_like(seq[:1], sos_token_id), seq), axis=0)

        return seq

    def add_eos_token(self, seq: np.ndarray) -> np.ndarray:
        eos_token_id = self.tokenizer[0, EOS_TOKEN]
        seq = np.concatenate((seq, np.full_like(seq[:1], eos_token_id)), axis=0)
        return seq

    # Auxiliary processing functions

    def zero_out_durations(self, seq: np.ndarray) -> np.ndarray:
        tto = self.tokenizer.vocab_types_idx
        velocity_index = tto['Velocity']
        if 'PerfDuration' in tto and seq.shape[-1] == len(tto):
            duration_index = tto['PerfDuration']
        else:
            duration_index = tto['Duration']

        silent_mask = seq[:, velocity_index] == self.tokenizer.zero_token
        seq[silent_mask, duration_index] = self.tokenizer.zero_token

        return seq

    def remove_silent_notes(self, seq: np.ndarray) -> np.ndarray:
        velocity_index = self.tokenizer.vocab_types_idx['Velocity']

        silent_mask = seq[:, velocity_index] == self.tokenizer.zero_token
        seq = seq[~silent_mask]

        return seq

    def compute_valid_pitch_mask(self, seq: np.ndarray) -> np.ndarray:
        pitch_index = self.tokenizer.vocab_types_idx['Pitch']
        pitch_min, pitch_max = self.tokenizer.zero_token, len(self.tokenizer.vocab[pitch_index]) - 1
        mask = np.logical_and(seq[:, pitch_index] >= pitch_min, seq[:, pitch_index] <= pitch_max)
        return mask
