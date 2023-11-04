""" SPMuple (ScorePerformanceMusic-tuple) encoding for aligned score-performance music. """

from abc import abstractmethod
from typing import List, Optional, Union, Any

import numpy as np
from miditok.midi_tokenizer import _in_as_seq
from miditoolkit import MidiFile, Note

from ..classes import TokSequence
from ..common import OctupleM
from ..constants import TIME_DIVISION, SCORE_KEYS


class SPMupleBase(OctupleM):
    r"""SPMupleBase: a base class for a family of ScorePerformanceMusic-tuple encodings.

    An extended OctupleM encoding with performance-specific tokens for performance MIDIs.
    """

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        # token vocabulary bins
        self.config.additional_params["token_bins"] = self.config.additional_params.get("token_bins", {})

        # midi postprocessing
        self.config.additional_params["cut_overlapping_notes"] = True

    def preprocess_midi(self, midi: MidiFile, is_score: bool = True):
        r"""Preprocess a MIDI file to be used by SPMuple encoding.

        :param midi: MIDI object to preprocess
        :param is_score: whether MIDI object is a score MIDI or not
        """
        super().preprocess_midi(midi)

    def preprocess_score_midi(self, midi: MidiFile):
        r"""Preprocess a score MIDI file to be used by SPMuple encoding.

        :param midi: MIDI object to preprocess
        """
        self.preprocess_midi(midi, is_score=True)

    def preprocess_performance_midi(self, midi: MidiFile):
        r"""Preprocess a performance MIDI file to be used by SPMuple encoding.

        :param midi: MIDI object to preprocess
        """
        self.preprocess_midi(midi, is_score=False)

    def score_midi_to_tokens(self, midi: MidiFile) -> TokSequence:
        r"""Converts a MIDI file to a score tokens representation, a sequence of "time steps" of tokens.

        A time step is a list of tokens where:
            (list index: token type)
            0: Bar
            1: Position
            2: Pitch
            3: Velocity
            4: Duration
            (5: Tempo)
            (6: TimeSignature)
            (7: Program)

        :param midi: the MIDI objet to convert
        :return: a :class:`miditok.TokSequence`.
        """
        return super().midi_to_tokens(midi)

    def performance_midi_to_tokens(
            self,
            midi: MidiFile,
            score_tokens: TokSequence,
            alignment: Optional[np.ndarray] = None
    ) -> TokSequence:
        r"""Tokenizes a performance MIDI file in to :class:`miditok.TokSequence`.

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param alignment: optional alignment between performance and score tokens.
        :return: a :class:`miditok.TokSequence`.
        """
        # Check if the durations values have been calculated before for this time division
        if midi.ticks_per_beat not in self._durations_ticks:
            self._durations_ticks[midi.ticks_per_beat] = np.array(
                [
                    (beat * res + pos) * midi.ticks_per_beat // res
                    for beat, pos, res in self.durations
                ]
            )

        # Preprocess the MIDI file
        self.preprocess_performance_midi(midi)

        # Register MIDI metadata
        self._current_midi_metadata = {
            "time_division": midi.ticks_per_beat,
            "max_tick": midi.max_tick,
            "tempo_changes": midi.tempo_changes,
            "time_sig_changes": midi.time_signature_changes,
            "key_sig_changes": midi.key_signature_changes,
        }

        tokens = self._performance_midi_to_tokens(midi, score_tokens, alignment)

        return tokens

    @abstractmethod
    def _performance_midi_to_tokens(
            self,
            midi: MidiFile,
            score_tokens: TokSequence,
            alignment: Optional[np.ndarray] = None
    ) -> TokSequence:
        r"""Converts a MIDI file to a performance tokens representation, a sequence of "time steps"
        of score tokens stacked with performance specific features (e.g., OnsetDeviation).

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param alignment: optional alignment between performance and score tokens.
        :return: the performance token representation, i.e. tracks converted into sequences of tokens
        """
        ...

    @_in_as_seq()
    def score_tokens_to_midi(
            self,
            tokens: Union[TokSequence, List, np.ndarray, Any],
            output_path: Optional[str] = None,
            time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts score tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        return self.tokens_to_midi(tokens, output_path=output_path, time_division=time_division)

    @abstractmethod
    @_in_as_seq()
    def performance_tokens_to_midi(
            self,
            tokens: Union[TokSequence, List, np.ndarray, Any],
            output_path: Optional[str] = None,
            time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts performance tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        ...

    @abstractmethod
    @_in_as_seq()
    def score_tokens_as_performance(self, score_tokens: Union[TokSequence, List, np.ndarray, Any]) -> TokSequence:
        r"""Converts a sequence of score tokens into a sequence of performance tokens,
        the tokens corresponding to a deadpan performance with no variation from score notes.
        """
        ...

    def _quantize_notes(self, notes: List[Note], time_division: int, is_score: bool = True):
        r"""Quantize the notes attributes: their pitch, velocity, start and end values.
        It shifts the notes so that they start at times that match the time resolution
        (e.g. 16 samples per bar).
        Notes with pitches outside of self.pitch_range will be deleted.

        :param notes: notes to quantize.
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed).
        :param is_score: whether the notes are from score MIDI or not
        """
        super()._quantize_notes(notes, time_division)

    @abstractmethod
    def _create_base_vocabulary(self) -> List[List[str]]:
        r"""Creates the vocabulary, as a list of string tokens.

        :return: the vocabulary as a list of string.
        """
        return super()._create_base_vocabulary()

    @abstractmethod
    def _get_token_types(self) -> List[str]:
        r"""Creates an ordered list of available token types."""
        return super()._get_token_types()

    @property
    def score_sizes(self):
        return {
            key: value for key, value in self.sizes.items()
            if key in SCORE_KEYS
        }

    @property
    def performance_sizes(self):
        return self.sizes
