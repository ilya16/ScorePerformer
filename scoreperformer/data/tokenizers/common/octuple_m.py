"""
OctupleM encoding method, a modified Octuple encoding,
introduced in MusicBERT https://arxiv.org/abs/2106.05630

Reimagines the Octuple tokenizer in MidiTok package (https://github.com/Natooz/MidiTok)
"""

from math import ceil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union, Any

import numpy as np
from miditok import Event
from miditok.constants import MIDI_INSTRUMENTS, TIME_SIGNATURE, TEMPO
from miditok.midi_tokenizer import _in_as_seq
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from ..classes import TokSequence
from ..constants import TIME_DIVISION
from ..midi_tokenizer import MIDITokenizer


class OctupleM(MIDITokenizer):
    r"""OctupleM: a modified Octuple encoding method.

    Each token is list of the form:
    * 0: Bar
    * 1: Position
    * 2: Pitch
    * 3: Velocity
    * 4: Duration
    * (+ Optional) Tempo
    * (+ Optional) TimeSignature
    * (+ Optional) Program
    """

    def _tweak_config_before_creating_voc(self):
        self.config.use_chords = False
        self.config.use_rests = False
        self.config.use_sustain_pedals = False
        self.config.use_pitch_bends = False
        self.config.delete_equal_successive_tempo_changes = True
        self.config.delete_equal_successive_time_sig_changes = True
        self.one_token_stream = self.config.one_token_stream_for_programs  # override miditok's configuration

        # used in place of positional encoding
        # max embedding as seen outside the tokenizer and used by the model
        self.config.additional_params["max_bar_embedding"] = self.config.additional_params.get("max_bar_embedding", 64)

        # max embedding used by tokenizer, might increase over tokenizations, if the tokenizer encounter longer MIDIs
        self.config.additional_params["real_max_bar_embedding"] = self.config.additional_params["max_bar_embedding"]

        # data preprocessing
        self.config.additional_params["fill_unperformed_notes"] = True
        self.config.additional_params["remove_duplicates"] = False

        self._duration_values = None

    def fill_unperformed_notes(self, midi: MidiFile):
        r"""Adds unperformed notes encoded as markers on a separate track
        unless those notes are already added to MIDI object.

        :param midi: MIDI object to preprocess
        """
        if self.config.additional_params["fill_unperformed_notes"] and midi.instruments[-1].name != "Unperformed Notes":
            notes = []
            for m in midi.markers:
                if m.text.startswith("NoteS"):
                    pitch, start_tick, end_tick = map(int, m.text.split("_")[1:])
                    notes.append(Note(0, pitch, start_tick, end_tick))
            if notes:
                midi.instruments.append(Instrument(0, False, "Unperformed Notes"))
                midi.instruments[-1].notes = notes

    def preprocess_midi(self, midi: MidiFile):
        r"""Pre-process (in place) a MIDI file to quantize its time and note attributes
        before tokenizing it. Its notes attribute (times, pitches, velocities) will be
        quantized and sorted, duplicated notes removed, as well as tempos. Empty tracks
        (with no note) will be removed from the MIDI object. Notes with pitches outside
        of self.pitch_range will be deleted.

        :param midi: MIDI object to preprocess.
        """
        # Insert unperformed notes on a new track
        self.fill_unperformed_notes(midi)

        # Do base preprocessing
        super().preprocess_midi(midi)

    def _add_time_events(self, events: List[Event]) -> List[List[Event]]:
        r"""
        Takes a sequence of note events (containing optionally Chord, Tempo and TimeSignature tokens),
        and insert (not inplace) time tokens (TimeShift, Rest) to complete the sequence.
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

        :param events: note events to complete.
        :return: the same events, with time events inserted.
        """
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / max(self.config.beat_res.values())

        # Add time events
        all_events = []
        current_bar = 0
        current_bar_from_ts_time = 0
        current_tick_from_ts_time = 0
        current_pos = 0
        previous_tick = 0
        current_time_sig = TIME_SIGNATURE
        current_tempo = TEMPO
        current_program = None
        ticks_per_bar = self._compute_ticks_per_bar(
            TimeSignature(*current_time_sig, 0), time_division
        )
        for e, event in enumerate(events):
            # Set current bar and position
            # This is done first, as we need to compute these values with the current ticks_per_bar,
            # which might change if the current event is a TimeSig
            if event.time != previous_tick:
                elapsed_tick = event.time - current_tick_from_ts_time
                current_bar = current_bar_from_ts_time + elapsed_tick // ticks_per_bar
                current_pos = int((elapsed_tick % ticks_per_bar) / ticks_per_sample)
                previous_tick = event.time

            if event.type == "TimeSig":
                current_time_sig = list(map(int, event.value.split("/")))
                current_bar_from_ts_time = current_bar
                current_tick_from_ts_time = previous_tick
                ticks_per_bar = self._compute_ticks_per_bar(
                    TimeSignature(*current_time_sig, event.time), time_division
                )
            elif event.type == "Tempo":
                current_tempo = event.value
            elif event.type == "Program":
                current_program = event.value
            elif event.type == "Pitch" and e + 2 < len(events):
                new_event = [
                    Event(type="Bar", value=current_bar, time=event.time),
                    Event(type="Position", value=current_pos, time=event.time),
                    Event(type="Pitch", value=event.value, time=event.time),
                    Event(type="Velocity", value=events[e + 1].value, time=event.time),
                    Event(type="Duration", value=events[e + 2].value, time=event.time),
                ]
                if self.config.use_tempos:
                    new_event.append(Event(type="Tempo", value=current_tempo))
                if self.config.use_time_signatures:
                    new_event.append(
                        Event(
                            type="TimeSig",
                            value=f"{current_time_sig[0]}/{current_time_sig[1]}",
                        )
                    )
                if self.config.use_programs:
                    new_event.append(Event("Program", current_program))
                all_events.append(new_event)

        return all_events

    def _midi_to_tokens(
            self, midi: MidiFile, *args, **kwargs
    ) -> TokSequence:
        r"""Converts a preprocessed MIDI object to a sequence of tokens.
        The workflow of this method is as follows: the events (Pitch, Velocity, Tempo, TimeSignature...) are
        gathered into a list, then the time events are added. All events of all tracks are treated all at once.
        
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

        :param midi: the MIDI object to convert
        :return: a :class:`miditok.TokSequence`
        """
        # Check bar embedding limit, update if needed
        min_ticks_per_bar = min([
            self._compute_ticks_per_bar(time_sig, midi.ticks_per_beat)
            for time_sig in midi.time_signature_changes
        ])
        nb_bars = ceil(midi.max_tick / min_ticks_per_bar)
        if self.config.additional_params["real_max_bar_embedding"] < nb_bars:
            for i in range(self.config.additional_params["real_max_bar_embedding"], nb_bars):
                self.add_to_vocab(f"Bar_{i}", self.vocab_types_idx["Bar"])
            self.config.additional_params["real_max_bar_embedding"] = nb_bars

        return super()._midi_to_tokens(midi, *args, **kwargs)

    @_in_as_seq()
    def tokens_to_midi(
            self,
            tokens: Union[TokSequence, List, np.ndarray, Any],
            output_path: Optional[str] = None,
            time_division: int = TIME_DIVISION,
    ) -> MidiFile:
        r"""Converts tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        assert time_division % self._max_beat_res == 0, \
            f"Invalid time division, please give one divisible by {self._max_beat_res}"
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_sample = time_division // self._max_beat_res

        tokens = np.array(tokens.ids)

        # Compute NoteON, Time Signature, Bar and Beat ticks
        ticks_data = self.compute_ticks(tokens, time_division, compute_beat_ticks=True)

        # Note attributes
        pitches = self.decode_token_type(tokens, "Pitch")
        velocities = self.decode_token_type(tokens, "Velocity")
        durations = self.decode_token_type(tokens, "Duration") * ticks_per_sample

        # Compute note positions in ticks
        note_on_ticks = ticks_data["note_on"].astype(int)
        note_off_ticks = (note_on_ticks + durations).astype(int)

        # Build Time Signature changes
        time_sigs, time_sig_ticks = ticks_data["time_sig"]
        midi.time_signature_changes = [
            TimeSignature(int(time_sigs[i][0]), int(time_sigs[i][1]), int(time_sig_ticks[i]))
            for i in range(len(time_sigs))
        ]

        # Process Tempo changes
        tempo_indices = np.concatenate([[0], np.where(np.diff(tokens[:, self.vocab_types_idx["Tempo"]]))[0] + 1])
        tempos = self.decode_token_type(tokens[tempo_indices], "Tempo")

        if len(tempos) > 0:
            # Get beat ticks to tie Tempo change to them
            beat_ticks = ticks_data["beat"]
            tempo_ticks = note_on_ticks[tempo_indices]  # Note: position at the start of the beat
            tempo_ticks = beat_ticks[np.minimum(np.searchsorted(beat_ticks, tempo_ticks), beat_ticks.shape[0] - 1)]
            tempo_ticks[0] = 0
        else:
            tempo_ticks = [0]

        midi.tempo_changes = [
            TempoChange(round(tempos[i], 3), int(tempo_ticks[i]))
            for i in range(len(tempos))
        ]

        # Process Programs and Create Notes
        instruments: Dict[int, Instrument] = {}
        if self.config.use_programs:
            programs = self.decode_token_type(tokens, "Program")
        else:
            programs = np.zeros_like(tokens[:, 0])

        for program in np.unique(programs):
            program = int(program)
            instruments[program] = Instrument(
                program=0 if program == -1 else program,
                is_drum=program == -1,
                name="Drums"
                if program == -1
                else MIDI_INSTRUMENTS[program]["name"],
            )

            program_ids = np.where(programs == program)[0]
            instruments[program].notes = [
                Note(vel, pitch, start, end)
                for vel, pitch, start, end in zip(
                    velocities[program_ids], pitches[program_ids],
                    note_on_ticks[program_ids], note_off_ticks[program_ids]
                )
            ]

        midi.instruments = list(instruments.values())
        midi.max_tick = note_off_ticks.max() + 1

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    def _create_base_vocabulary(self) -> List[List[str]]:
        r"""Creates the vocabulary, as a list of string tokens.
        Each token as to be given as the form of "Type_Value", separated with an underscore.
        Example: Pitch_58
        The :class:`miditok.MIDITokenizer` main class will then create the "real" vocabulary as
        a dictionary.
        Special tokens have to be given when creating the tokenizer, and
        will be added to the vocabulary by :class:`miditok.MIDITokenizer`.

        :return: the vocabulary as a list of string.
        """
        vocab = []

        # BAR
        vocab.append([f"Bar_{i}" for i in range(self.config.additional_params["real_max_bar_embedding"])])

        # POSITION
        self._max_beat_res = max(self.config.beat_res.values())
        self._max_nb_beats = max(map(lambda ts: ceil(4 * ts[0] / ts[1]), self.time_signatures))
        nb_positions = self._max_nb_beats * self._max_beat_res
        vocab.append([f"Position_{i}" for i in range(nb_positions)])

        # PITCH
        vocab.append([f"Pitch_{i}" for i in range(*self.config.pitch_range)])

        # VELOCITY
        self.velocities = np.concatenate(([0], self.velocities))  # allow 0 velocity (unperformed note)
        vocab.append([f"Velocity_{i}" for i in self.velocities])

        # DURATION
        self.durations = [(0, 0, self.durations[0][-1])] + self.durations  # allow 0 duration
        vocab.append([f'Duration_{".".join(map(str, duration))}' for duration in self.durations])

        # TEMPO
        if self.config.use_tempos:
            vocab.append([f"Tempo_{i}" for i in self.tempos])

        # TIME_SIGNATURE
        if self.config.use_time_signatures:
            vocab.append([f"TimeSig_{i[0]}/{i[1]}" for i in self.time_signatures])

        # PROGRAM
        if self.config.use_programs:
            vocab.append([f"Program_{i}" for i in self.config.programs])

        token_types = self._get_token_types()
        self.vocab_types_idx = {
            type_: idx for idx, type_ in enumerate(token_types)
        }

        return vocab

    def _get_token_types(self) -> List[str]:
        r"""Creates an ordered list of available token types."""
        token_types = ["Bar", "Position", "Pitch", "Velocity", "Duration"]

        if self.config.use_tempos:
            token_types.append("Tempo")

        if self.config.use_time_signatures:
            token_types.append("TimeSig")

        if self.config.use_programs:
            token_types.append("Program")

        return token_types

    def _create_token_types_graph(self) -> Dict[str, List[str]]:
        r"""Returns a graph (as a dictionary) of the possible token
        types successions.
        Not relevant for Octuple.

        :return: the token types transitions dictionary
        """
        return {}  # not relevant for Octuple

    def decode_token_type(self, tokens: np.ndarray, token_type: str) -> np.ndarray:
        r"""Decodes values from tokens for given token_type.

        :param tokens: a sequence of tokens
        :param token_type: tokens' dimension to compute values
        :return: values for a provided type
        """
        type_tokens = tokens[:, self.vocab_types_idx[token_type]] - self.zero_token
        if token_type == "Pitch":
            return type_tokens + self.config.pitch_range[0]
        elif token_type == "Velocity":
            return self.velocities[type_tokens]
        elif token_type == "Duration":
            return self.duration_values[type_tokens] * self._max_beat_res
        elif token_type == "Tempo":
            return self.tempos[type_tokens]
        elif token_type == "TimeSig":
            return np.array(self.time_signatures)[type_tokens]
        else:
            return type_tokens

    def token_values(
            self,
            normalize: Union[bool, List[str]] = False,
            special_tokens: bool = True
    ) -> Dict[str, np.ndarray]:
        r"""Returns a dictionary of all token types values.

        :param normalize: whether to normalize the token values
        :param special_tokens: whether to prepend values for special tokens
        :return: dictionary of token types values
        """
        if isinstance(normalize, bool):
            normalize = list(self.vocab_types_idx.keys()) if normalize else []

        token_values = {}
        for key in self.vocab_types_idx.keys():
            token_values[key] = self.token_type_values(
                token_type=key, normalize=key in normalize, special_tokens=special_tokens
            )

        return token_values

    def token_type_values(
            self,
            token_type: str,
            normalize: bool = False,
            special_tokens: bool = True
    ) -> np.ndarray:
        r"""Returns token values for given token type.

        :param token_type: vocabulary token type
        :param normalize: whether to normalize the token values
        :param special_tokens: whether to prepend values for special tokens
        :return: array of token values
        """
        if token_type == "Bar":
            values = np.arange(1, self.config.additional_params["max_bar_embedding"] + 1)
            if normalize:
                values = values / self.config.additional_params["max_bar_embedding"]
        elif token_type == "Position":
            values = np.arange(self._max_nb_beats * self._max_beat_res)
            if normalize:
                values = values / self._max_beat_res / 4
        elif token_type == "Pitch":
            values = np.arange(*self.config.pitch_range)
            if normalize:
                values = values % 127
        elif token_type == "Velocity":
            values = self.velocities
            if normalize:
                values = values / self.velocities[-1]
        elif token_type in ("Duration", "PerfDuration"):
            values = self.duration_values
            if normalize:
                values = np.log2(values + 1)
        elif token_type == "Tempo":
            values = self.tempos
            if normalize:
                values = np.log2(values / self.tempos[0])
        elif token_type == "TimeSig":
            values = np.array([x[0] / x[1] for x in self.time_signatures])
        else:
            values = np.zeros(len(self.vocab[self.vocab_types_idx[token_type]]))

        if special_tokens:
            values = np.concatenate([np.zeros(self.zero_token), values])
        return values

    def compute_ticks(
            self,
            tokens: np.ndarray,
            time_division: int = TIME_DIVISION,
            compute_beat_ticks: bool = False
    ) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray]]]:
        r"""Computes tick positions for time signatures, note onsets, bars and beats.

        NOTE: use only for full-length or single time-signature sequences,
        or note/bar/beat ticks might be computed incorrectly due to missing past time signature changes

        :param tokens: a sequence of tokens
        :param time_division: MIDI time division / resolution, in ticks/beat
        :param compute_beat_ticks: whether to compute beat ticks or not
        :return: a dictionary of ticks data
        """
        ticks_per_sample = time_division / self._max_beat_res
        bars = self.decode_token_type(tokens, "Bar")
        positions = self.decode_token_type(tokens, "Position")

        # Compute Time Signature change positions
        time_sig_indices = np.where(np.diff(tokens[:, self.vocab_types_idx["TimeSig"]]))[0] + 1
        time_sig_indices = np.concatenate([[0], time_sig_indices])

        # Get time signatures
        time_sigs = self.decode_token_type(tokens[time_sig_indices], "TimeSig")

        # Compute time signature ticks
        ticks_per_bar = (time_division * 4 * time_sigs[:, 0] / time_sigs[:, 1])
        time_sig_bars = bars[time_sig_indices]
        time_sig_ticks = np.concatenate([[0], np.cumsum(ticks_per_bar[:-1] * np.diff(time_sig_bars))])

        # Compute ticks for each bar
        bar_time_sig_ids = np.maximum(0, np.searchsorted(time_sig_bars, np.arange(bars[-1] + 1), side="right") - 1)
        bar_ticks = np.concatenate([[0], np.cumsum(ticks_per_bar[bar_time_sig_ids])])

        # Compute note ticks
        note_on_ticks = bar_ticks[bars] + positions * ticks_per_sample

        # Combine ticks data
        ticks_data = {
            "note_on": note_on_ticks,
            "time_sig": (time_sigs, time_sig_ticks),
            "bar": bar_ticks
        }

        if compute_beat_ticks:
            # Compute ticks for each beat
            num_beats_in_bar = time_sigs[:, 0]
            num_beats_in_bar[num_beats_in_bar == 6] = 2
            num_beats_in_bar[np.isin(num_beats_in_bar, (9, 18))] = 3
            num_beats_in_bar[np.isin(num_beats_in_bar, (12, 24))] = 4
            ticks_per_beat = ticks_per_bar // num_beats_in_bar

            max_beat = np.sum(np.diff(np.concatenate([time_sig_bars, [bars[-1] + 1]])) * num_beats_in_bar)
            beat_time_sig_ids = np.maximum(0, np.searchsorted(time_sig_bars, np.arange(max_beat + 1), side="right") - 1)
            beat_ticks = np.concatenate([[0], np.cumsum(ticks_per_beat[beat_time_sig_ids])])

            ticks_data["beat"] = beat_ticks

        return ticks_data

    @property
    def sizes(self):
        sizes = {k: len(v) for k, v in zip(self.vocab_types_idx, self.vocab)}
        sizes["Bar"] -= (
                self.config.additional_params["real_max_bar_embedding"]
                - self.config.additional_params["max_bar_embedding"]
        )
        return sizes

    @property
    def zero_token(self):
        return len(self.special_tokens)

    @property
    def duration_values(self):
        if self._duration_values is None:
            self._duration_values = np.array([
                (beat * res + pos) / res if res > 0 else 0
                for beat, pos, res in self.durations
            ])
        return self._duration_values
