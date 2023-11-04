"""
SPMuple (ScorePerformanceMusic-tuple) encoding method,
an OctupleM-based score-performance encoding with position shifts,
relative onset deviations and relative performed durations.
"""

from pathlib import Path
from typing import List, Optional, Union, Any, Dict

import numpy as np
from miditok import Event
from miditok.constants import MIDI_INSTRUMENTS
from miditok.midi_tokenizer import _in_as_seq
from miditok.utils import remove_duplicated_notes
from miditoolkit import MidiFile, Instrument, Note, TempoChange, TimeSignature

from scoreperformer.utils import find_closest
from .base import SPMupleBase
from ..classes import TokSequence
from ..constants import TIME_DIVISION, MASK_TOKEN
from ...midi.utils import cut_overlapping_notes


class SPMuple(SPMupleBase):
    r"""SPMuple: ScorePerformanceMusic-tuple encoding.

    An extended OctupleM encoding with performance-specific tokens ((Rel)OnsetDeviation, (Rel)PerformedDuration)
    and new score-specific tokens (PositionShift).

    Uses OctupleM encoding for score MIDIs and adds performance tokens for performance MIDIs.
    Supports bar and beat local tempos.
    """

    def _tweak_config_before_creating_voc(self):
        super()._tweak_config_before_creating_voc()

        additional_params = self.config.additional_params

        # optional tokens and their value bins
        additional_params["use_position_shifts"] = additional_params.get("use_position_shifts", False)
        additional_params["onset_position_shifts"] = additional_params.get("onset_position_shifts", True)
        self.position_shifts = None

        additional_params["use_onset_indices"] = additional_params.get("use_onset_indices", False)
        additional_params["max_notes_in_onset"] = additional_params.get("max_notes_in_onset", 12)

        additional_params["rel_onset_dev"] = additional_params.get("rel_onset_dev", False)
        additional_params["nb_onset_devs"] = additional_params.get("nb_onset_devs", 129)
        self.rel_onset_deviations = additional_params.get("rel_onset_deviations", None)

        additional_params["rel_perf_duration"] = additional_params.get("rel_perf_duration", False)
        additional_params["nb_perf_durations"] = additional_params.get("nb_perf_durations", 65)
        self.rel_performed_durations = additional_params.get("rel_performed_durations", None)

        # local tempo configuration
        additional_params["bar_tempos"] = additional_params.get("bar_tempos", False)

    def preprocess_midi(self, midi: MidiFile, is_score: bool = True):
        r"""Preprocess a MIDI file to be used by SPMuple encoding.

        :param midi: MIDI object to preprocess
        :param is_score: whether MIDI object is a score MIDI or not
        """
        # Insert unperformed notes on a new track
        self.fill_unperformed_notes(midi)

        # Do note preprocessing
        t = 0
        while t < len(midi.instruments):
            # quantize note attributes
            self._quantize_notes(midi.instruments[t].notes, midi.ticks_per_beat, is_score=is_score)
            midi.instruments[t].notes.sort(key=lambda x: (x.start, x.pitch, x.end))  # sort notes
            if self.config.additional_params["remove_duplicates"]:
                remove_duplicated_notes(midi.instruments[t].notes)  # remove possible duplicated notes
            if len(midi.instruments[t].notes) == 0:
                del midi.instruments[t]
                continue
            t += 1

        # Recalculate max_tick is this could have changed after notes quantization
        if len(midi.instruments) > 0:
            midi.max_tick = max([max([note.end for note in track.notes]) for track in midi.instruments])
            midi.tempo_changes = [tempo for tempo in midi.tempo_changes if tempo.time < midi.max_tick]

        if self.config.use_tempos:
            self._quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

        # Not needed for performance MIDIs (copy time signatures from scores)
        if is_score:
            if self.config.use_time_signatures:
                self._quantize_time_signatures(midi.time_signature_changes, midi.ticks_per_beat)

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
            (8: PositionShift)
            (9: NotesInOnset)
            (10: PositionInOnset)

        :param midi: the MIDI objet to convert
        :return: the scores token representation, i.e. tracks converted into sequences of tokens
        """
        tok_sequence: TokSequence = super().score_midi_to_tokens(midi)

        # Add new score tokens if they are present in the encoding
        if any(self.config.additional_params[key] for key in ("use_position_shifts", "use_onset_indices")):
            token_ids = np.array(tok_sequence.ids)
            tokens = tok_sequence.tokens

            # Prepare constants used for calculations
            time_division = self._current_midi_metadata["time_division"]
            ticks_per_sample = time_division / self._max_beat_res

            # Compute score positions
            ticks_data = self.compute_ticks(token_ids, time_division, compute_beat_ticks=True)
            score_positions = ticks_data["note_on"] / ticks_per_sample

            if self.config.additional_params["use_position_shifts"]:
                # Compute position shifts and PositionShift tokens
                pos_shifts = self.compute_position_shifts(score_positions)
                pos_shifts = self.position_shifts[find_closest(self.position_shifts, pos_shifts)]
                for i, pos_shift in enumerate(pos_shifts):
                    tokens[i].append(f"PositionShift_{pos_shift}")

            _, notes_in_onset, pos_in_onset = self.compute_onset_values(score_positions)
            if self.config.additional_params["use_onset_indices"]:
                for i, num in enumerate(notes_in_onset):
                    tokens[i].append(f"NotesInOnset_{num}")

                for i, pos in enumerate(pos_in_onset):
                    tokens[i].append(f"PositionInOnset_{pos}")

            tok_sequence = TokSequence(tokens=tokens)
            self.complete_sequence(tok_sequence)

        return tok_sequence

    def _performance_midi_to_tokens(
            self,
            midi: MidiFile,
            score_tokens: TokSequence,
            alignment: Optional[np.ndarray] = None
    ) -> TokSequence:
        r"""Converts a MIDI file to a performance tokens representation, a sequence of "time steps"
        of score tokens stacked with performance specific features (e.g., OnsetDeviation).

        A time step is a list of tokens where:
            (list index: token type)
            0: Bar
            1: Position
            2: Pitch
            3: Velocity
            4: Duration
            5: Tempo
            (6: TimeSignature)
            (7: Program)
            (8: PositionShift (score token))
            (9: NotesInOnset (score token))
            (10: PositionInOnset (score token))
            *11: (Relative)OnsetDeviation (performance token)
            *12: (Relative)PerformedDuration (performance token)

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param alignment: optional alignment between performance and score tokens.
        :return: the performance token representation, i.e. tracks converted into sequences of tokens
        """
        additional_params = self.config.additional_params

        # Prepare constants used for calculations
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / self._max_beat_res

        # Convert each track to tokens
        tokens = []
        for track in midi.instruments:
            tokens += self._performance_track_to_tokens(track)

        # Save performance position and duration ticks
        perf_positions = np.array([t[self.vocab_types_idx["Position"]].value for t in tokens])
        perf_durations = np.array([t[self.vocab_types_idx["Duration"]].value for t in tokens])

        pitch_idx = self.vocab_types_idx["Pitch"]
        tokens.sort(
            key=lambda x: (x[pitch_idx].time, x[pitch_idx].desc, x[pitch_idx].value)
        )  # Sort by time, track, pitch

        # Convert pitch, position and durations events into tokens
        for time_step in tokens:
            time_step[pitch_idx] = str(time_step[pitch_idx])
            time_step[self.vocab_types_idx["Position"]] = MASK_TOKEN
            time_step[self.vocab_types_idx["Duration"]] = MASK_TOKEN

        # Convert tokens to ids
        tokens = np.array(self._tokens_to_ids(tokens))

        # Process score tokens
        score_tokens = np.array(score_tokens.ids)

        # Compute NoteON, Time Signature, Bar and Beat ticks
        ticks_data = self.compute_ticks(score_tokens, time_division, compute_beat_ticks=True)
        note_on_ticks = ticks_data["note_on"]
        beat_ticks = ticks_data["bar"] if self.config.additional_params["bar_tempos"] else ticks_data["beat"]

        # Map note ticks to beats
        note_beats = beat_ticks[np.minimum(np.searchsorted(beat_ticks, note_on_ticks), beat_ticks.shape[0] - 1)]

        # Process tempos and their jumping positions
        # Record beat tempos before applying alignment
        if alignment is not None:
            note_beats = note_beats[np.argsort(alignment)]

        note_beat_tempo = np.stack([
            note_beats,
            tokens[:, self.vocab_types_idx["Tempo"]].astype(float)
        ], axis=1)
        un_beat_tempos, counts = np.unique(note_beat_tempo, return_counts=True, axis=0)
        beat_tempo_data = np.concatenate([un_beat_tempos, counts[:, None]], axis=1)

        beat_tempos = []
        while len(beat_tempo_data) > 0:
            beat_tempos_ = beat_tempo_data[beat_tempo_data[:, 0] == beat_tempo_data[0, 0]]
            beat_tempos.append(beat_tempos_[beat_tempos_[:, 2].argmax(), :2])
            beat_tempo_data = beat_tempo_data[len(beat_tempos_):]
        beat_tempos = np.stack(beat_tempos).astype(int)

        # Apply alignment
        if alignment is not None:
            tokens, perf_positions, perf_durations = map(
                lambda x: x[alignment], (tokens, perf_positions, perf_durations)
            )

        # Put back correct beat tempos
        tokens[:, self.vocab_types_idx["Tempo"]] = beat_tempos[np.searchsorted(beat_tempos[:, 0], note_beats)][:, 1]

        # Copy score tokens to midi tokens
        token_types = ["Bar", "Position", "Duration", "TimeSig"]
        if additional_params["use_position_shifts"]:
            token_types.append("PositionShift")
        if additional_params["use_onset_indices"]:
            token_types.extend(["NotesInOnset", "PositionInOnset"])
        for token_type in token_types:
            idx = self.vocab_types_idx[token_type]
            tokens[:, idx] = score_tokens[:, idx]

        # Compute score positions and durations
        score_positions = ticks_data["note_on"] / ticks_per_sample
        score_durations = self.decode_token_type(score_tokens, "Duration")

        # Compute OnsetDeviation and PerformanceDuration tokens
        onset_devs = perf_positions - score_positions

        # Scale onset deviations based on score durations and convert to tokens
        if additional_params["rel_onset_dev"]:
            if additional_params["use_position_shifts"] and additional_params["onset_position_shifts"]:
                pos_shifts = self.position_shifts[tokens[:, self.vocab_types_idx["PositionShift"]] - self.zero_token]
            else:
                pos_shifts = self.compute_position_shifts(score_positions, onset_shift=True)
            pos_shifts[pos_shifts == 0] = 1
            rel_onset_devs = onset_devs / pos_shifts
            onset_dev_tokens = find_closest(self.rel_onset_deviations, rel_onset_devs)
        else:
            max_onset_dev = self._max_beat_res * 2
            onset_devs = np.minimum(np.maximum(onset_devs, -max_onset_dev), max_onset_dev)
            onset_dev_tokens = onset_devs + max_onset_dev

        # Scale performed durations based on score durations and convert to tokens
        if additional_params["rel_perf_duration"]:
            rel_perf_durations = perf_durations / score_durations
            perf_duration_tokens = find_closest(self.rel_performed_durations, rel_perf_durations)
        else:
            perf_duration_tokens = find_closest(self._duration_values[1:] * self._max_beat_res, perf_durations) + 1

        # Append RelOnsetDev, and RelPerfDuration tokens
        tokens = np.concatenate([
            tokens,
            onset_dev_tokens[:, None] + self.zero_token,
            perf_duration_tokens[:, None] + self.zero_token
        ], axis=1).astype(int)

        tok_sequence = TokSequence(ids=tokens.tolist())
        self.complete_sequence(tok_sequence)

        return tok_sequence

    def _performance_track_to_tokens(
            self, track: Instrument
    ) -> List[List[Union[Event, int]]]:
        r"""Converts a track (miditoolkit.Instrument object) into a sequence of pre-performance tokens.

        Note: the actual performance tokens are computed later for the whole MIDI in `performance_midi_to_tokens`.

        A time step is a list of tokens where:
           (list index: token type)
           0: Bar (as MASK token, taken from score tokens afterwards)
           1: Position (as an Event object to compute onset deviation and substituted by score position afterwards)
           2: Pitch (as an Event object for sorting purpose afterwards)
           3: Velocity
           4: Duration (as an Event object to compute performed duration and substituted by score duration afterwards)
           (5: Tempo)
           (6: TimeSignature)
           (7: Program)

        :param track: track object to convert
        :return: sequence of corresponding performance tokens
        """
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / self._max_beat_res

        tokens = []
        current_tempo_idx = 0
        current_tempo = self._current_midi_metadata["tempo_changes"][current_tempo_idx].tempo

        for note in track.notes:
            # Note attributes
            pos = note.start / ticks_per_sample
            duration = (note.end - note.start) / ticks_per_sample
            token = [
                MASK_TOKEN,  # Copied later from score tokens
                Event(
                    type="Position",
                    time=note.start,
                    value=pos,
                    desc=f"{note.start} ticks"
                ),
                Event(
                    type="Pitch",
                    value=note.pitch,
                    time=note.start,
                    desc=-1 if track.is_drum else track.program,
                ),
                f"Velocity_{note.velocity}",
                Event(
                    type="Duration",
                    time=note.start,
                    value=duration,
                    desc=f"{duration} ticks"
                )
            ]

            # (Tempo)
            if self.config.use_tempos:
                # If the current tempo is not the last one
                if current_tempo_idx + 1 < len(self._current_midi_metadata["tempo_changes"]):
                    # Will loop over incoming tempo changes
                    for tempo_change in self._current_midi_metadata["tempo_changes"][current_tempo_idx + 1:]:
                        # If this tempo change happened before the current moment
                        if tempo_change.time <= note.start:
                            current_tempo = tempo_change.tempo
                            current_tempo_idx += 1  # update tempo value (might not change) and index
                        elif tempo_change.time > note.start:
                            break  # this tempo change is beyond the current time step, we break the loop
                token.append(f"Tempo_{current_tempo}")

            # (TimeSignature)
            if self.config.use_time_signatures:
                token.append(MASK_TOKEN)  # Copied later from score tokens

            # (Program)
            if self.config.use_programs:
                token.append(f"Program_{-1 if track.is_drum else track.program}")

            # (PositionShift)
            if self.config.additional_params["use_position_shifts"]:
                token.append(MASK_TOKEN)  # Copied later from score tokens

            # (NotesInOnset) and (PositionInOnset)
            if self.config.additional_params["use_onset_indices"]:
                token.append(MASK_TOKEN)
                token.append(MASK_TOKEN)

            tokens.append(token)

        return tokens

    @_in_as_seq()
    def performance_tokens_to_midi(
            self,
            tokens: Union[TokSequence, List, np.ndarray, Any],
            output_path: Optional[str] = None,
            time_division: int = TIME_DIVISION
    ) -> MidiFile:
        r"""Converts performance tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

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
        note_on_ticks = ticks_data["note_on"]

        # Compute position shifts
        if self.config.additional_params["use_position_shifts"]:
            pos_shifts = self.decode_token_type(tokens, "PositionShift")
        else:
            pos_shifts = self.compute_position_shifts(note_on_ticks / ticks_per_sample)

        # Onset Deviations to ticks
        if self.config.additional_params["rel_onset_dev"]:
            rel_onset_devs = self.decode_token_type(tokens, "RelOnsetDev")
            pos_shifts[pos_shifts == 0] = 1
            onset_devs = (rel_onset_devs * pos_shifts * ticks_per_sample).astype(int)
        else:
            onset_devs = self.decode_token_type(tokens, "OnsetDev")
            onset_devs *= ticks_per_sample

        # Shift onsets
        note_on_ticks += onset_devs
        note_on_ticks = np.maximum(0, note_on_ticks).astype(int)

        # Performed Durations to ticks and NoteOFF ticks
        if self.config.additional_params["rel_perf_duration"]:
            rel_perf_durations = self.decode_token_type(tokens, "RelPerfDuration")
            perf_durations = (rel_perf_durations * durations).astype(int)
        else:
            perf_durations = self.decode_token_type(tokens, "PerfDuration")
            perf_durations *= ticks_per_sample

        note_off_ticks = (note_on_ticks + perf_durations).astype(int)

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
            beat_ticks = ticks_data["bar"] if self.config.additional_params["bar_tempos"] else ticks_data["beat"]
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

        if self.config.additional_params["cut_overlapping_notes"]:
            for track in midi.instruments:
                cut_overlapping_notes(track.notes)

            # recompute `max_tick` and remove overlapping tempo changes
            midi.max_tick = max([max([note.end for note in track.notes[-100:]]) for track in midi.instruments])
            midi.tempo_changes = [tempo for tempo in midi.tempo_changes if tempo.time < midi.max_tick]

        # Write MIDI file
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            midi.dump(output_path)
        return midi

    @_in_as_seq()
    def score_tokens_as_performance(self, score_tokens: Union[TokSequence, List, np.ndarray, Any]) -> TokSequence:
        r"""Converts a sequence of score tokens into a sequence of performance tokens,
        the tokens corresponding to a deadpan performance with no variation from score notes.
        """
        tokens = np.array(score_tokens.ids)

        # Obtain and distribute zero onset deviation tokens
        if self.config.additional_params["rel_onset_dev"]:
            zero_onset_token = self[self.vocab_types_idx["RelOnsetDev"], "RelOnsetDev_0.0"]
        else:
            zero_onset_token = self[self.vocab_types_idx["OnsetDev"], "OnsetDev_0"]
        onset_dev_tokens = np.full_like(tokens[:, 0], fill_value=zero_onset_token)

        # Obtain and distribute no articulation performed duration tokens
        if self.config.additional_params["rel_perf_duration"]:
            perf_duration_token = self[self.vocab_types_idx["RelPerfDuration"], "RelPerfDuration_1.0"]
            perf_duration_tokens = np.full_like(tokens[:, 0], fill_value=perf_duration_token)
        else:
            perf_duration_tokens = tokens[:, self.vocab_types_idx["Duration"]]

        tokens = np.concatenate([
            tokens,
            onset_dev_tokens[:, None],
            perf_duration_tokens[:, None]
        ], axis=1).astype(int)

        return TokSequence(ids=tokens.tolist())

    def _quantize_notes(self, notes: List[Note], time_division: int, is_score: bool = True):
        r"""Quantize the notes attributes: their pitch, velocity, start and end values.
        It shifts the notes so that they start at times that match the time resolution
        (e.g. 16 samples per bar).
        Note durations will be clipped to the maximum duration that can be handled by the tokenizer. This is done
        to prevent having incorrect offset values when computing rests.
        Notes with pitches outside of self.pitch_range will be deleted.

        **NOTE:**: the note start/end quantizaton is applied only to score MIDI notes.

        :param notes: notes to quantize
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
        :param is_score: whether the notes are from score MIDI or not
        """
        ticks_per_sample = int(time_division / self._max_beat_res)
        max_duration_ticks = max(tu[1] for tu in self.config.beat_res) * time_division
        i = 0
        pitches = range(*self.config.pitch_range)
        while i < len(notes):
            if notes[i].pitch not in pitches:
                del notes[i]
                continue

            if is_score:
                start_offset = notes[i].start % ticks_per_sample
                notes[i].start += (
                    -start_offset
                    if start_offset <= ticks_per_sample / 2
                    else ticks_per_sample - start_offset
                )
                if notes[i].end - notes[i].start > max_duration_ticks:
                    notes[i].end = notes[i].start + max_duration_ticks
                else:
                    end_offset = notes[i].end % ticks_per_sample
                    notes[i].end += (
                        -end_offset
                        if end_offset <= ticks_per_sample / 2
                        else ticks_per_sample - end_offset
                    )

                    # if this happens to often, consider using a higher beat resolution
                    # like 8 samples per beat or 24 samples per bar
                    if notes[i].start == notes[i].end:
                        notes[i].end += ticks_per_sample

            if notes[i].velocity > 0:
                notes[i].velocity = self.velocities[1:][int(np.argmin(np.abs(self.velocities[1:] - notes[i].velocity)))]
            i += 1

    def _create_base_vocabulary(self) -> List[List[str]]:
        r"""Creates the vocabulary, as a list of string tokens.

        :return: the vocabulary as a list of string.
        """
        vocab = super()._create_base_vocabulary()

        # POSITION SHIFT
        if self.config.additional_params["use_position_shifts"]:
            self.position_shifts = self._create_position_shifts()
            vocab.append([f"PositionShift_{i}" for i in self.position_shifts])

        # ONSET INDICES
        if self.config.additional_params["use_onset_indices"]:
            max_notes_in_onset = self.config.additional_params["max_notes_in_onset"]
            vocab.append([f"NotesInOnset_{i + 1}" for i in range(max_notes_in_onset)])
            vocab.append([f"PositionInOnset_{i}" for i in range(max_notes_in_onset)])

        # (RELATIVE) ONSET (POSITION) DEVIATION
        if self.config.additional_params["rel_onset_dev"]:  # relative
            if self.rel_onset_deviations is None:
                self.rel_onset_deviations = self._create_relative_onset_deviations()
            vocab.append([f"RelOnsetDev_{i}" for i in self.rel_onset_deviations])
        else:  # absolute
            nb_positions = self._max_beat_res * 2  # up to two quarter notes
            vocab.append([f"OnsetDev_{i}" for i in range(-nb_positions, nb_positions + 1)])

        # (RELATIVE) PERFORMED DURATION
        if self.config.additional_params["rel_perf_duration"]:  # relative
            if self.rel_performed_durations is None:
                self.rel_performed_durations = self._create_relative_performed_durations()
            vocab.append([f"RelPerfDuration_{i}" for i in self.rel_performed_durations])
        else:
            vocab.append(vocab[self.vocab_types_idx["Duration"]])

        return vocab

    def _get_token_types(self) -> List[str]:
        r"""Creates an ordered list of available token types."""
        token_types = super()._get_token_types()

        # Score tokens
        if self.config.additional_params["use_position_shifts"]:
            token_types.append("PositionShift")

        if self.config.additional_params["use_onset_indices"]:
            token_types.append("NotesInOnset")
            token_types.append("PositionInOnset")

        # Performance tokens
        if self.config.additional_params["rel_onset_dev"]:
            token_types.append("RelOnsetDev")
        else:
            token_types.append("OnsetDev")

        if self.config.additional_params["rel_perf_duration"]:
            token_types.append("RelPerfDuration")
        else:
            token_types.append("PerfDuration")

        return token_types

    def _create_position_shifts(self) -> np.ndarray:
        r"""Creates the possible position shifts in `max_bet_res`, an array of integers.
        The more beats the position shift occupies, the smaller the resolution of position shift.

        :return: the position shift bins
        """
        position_shifts = np.concatenate([
            np.arange(0, 2 * self._max_beat_res, 1),  # 0-2 beats with precision 1
            np.arange(2 * self._max_beat_res, 4 * self._max_beat_res, 2),  # 2-4 beats with precision 2
            np.arange(4 * self._max_beat_res, 8 * self._max_beat_res, 8),  # 4-8 beats with precision 8
            np.arange(8 * self._max_beat_res, 16 * self._max_beat_res + 1, 16),  # 8-16 beats with precision 16
        ])

        return position_shifts

    def _create_relative_onset_deviations(self) -> np.ndarray:
        r"""Creates the relative onset deviation bins based on some heuristics.
        The larger the factor, the smaller the resolution.

        :return: the relative onset deviation bins
        """
        onset_dev_quant = (self.config.additional_params["nb_onset_devs"] - 1) // 8

        rel_onset_devs = np.concatenate([
            # 25% from 0 to 1/24
            np.linspace(0.0, 1 / 24, onset_dev_quant + 1),
            # 25% from 1/24 to 1/8
            np.linspace(1 / 24, 1 / 8, onset_dev_quant + 1)[1:],
            # 25% from 1/8 to 1/3
            np.linspace(1 / 8, 1 / 3, onset_dev_quant + 1)[1:],
            # 12.5% from 1/3 to 3/5
            np.linspace(1 / 3, 3 / 5, onset_dev_quant // 2 + 1)[1:],
            # 6.25% from 3/5 to 1.0
            np.linspace(3 / 5, 1.0, onset_dev_quant // 4 + 1)[1:],
            # 6.25% from 1.0 to 4.0
            (2 ** (8 * np.arange(onset_dev_quant // 4 + 1) / onset_dev_quant))[1:]
        ])
        rel_onset_devs = np.round(rel_onset_devs, 4)
        rel_onset_devs = np.sort(np.concatenate([-rel_onset_devs[1:], rel_onset_devs]))  # add negative deviations

        return rel_onset_devs

    def _create_relative_performed_durations(self) -> np.ndarray:
        r"""Creates the relative performed duration bins based on some heuristics.
        The larger the factor, the smaller the resolution.

        :return: the relative onset deviation bins
        """
        perf_dur_quant = (self.config.additional_params["nb_perf_durations"] - 1) // 4

        rel_performed_durations = np.concatenate([
            # 25% from 1/10 to 2/5
            np.linspace(1 / 10, 2 / 5, perf_dur_quant + 1),
            # 25% from 2/5 to 2/3
            np.linspace(2 / 5, 2 / 3, perf_dur_quant + 1)[1:],
            # 25% from 2/3 to 1.0
            np.linspace(2 / 3, 1.0, perf_dur_quant + 1)[1:],
            # 12.5% from 1.0 to 5/4
            np.linspace(1.0, 5 / 4, perf_dur_quant // 2 + 1)[1:],
            # 6.25% from 5/4 to 3/2
            np.linspace(5 / 4, 3 / 2, perf_dur_quant // 4 + 1)[1:],
            # 6.25% from 3/2 to 3.0
            (2 ** (4 * np.arange(perf_dur_quant // 4 + 1) / perf_dur_quant) * 3 / 2)[1:],
        ])
        rel_performed_durations = np.round(rel_performed_durations, 4)

        return rel_performed_durations

    def compute_position_shifts(self, score_positions, onset_shift: Optional[bool] = None):
        r"""Computes absolute position shifts between onsets from score positions.

        :param score_positions: score positions in ticks/beats
        :param onset_shift: if provided, overwrites tokenizer setting for onset_shift position shift
        :return: the position shifts
        """
        onset_shift = self.config.additional_params["onset_position_shifts"] if onset_shift is None else onset_shift
        if onset_shift:
            unique_score_pos, score_pos_counts = np.unique(score_positions, return_counts=True)
            score_pos_ids = np.arange(len(unique_score_pos)).repeat(score_pos_counts)
            pos_shifts = unique_score_pos[score_pos_ids] - unique_score_pos[score_pos_ids - 1]
            pos_shifts[pos_shifts < 0] = score_positions[pos_shifts < 0]
        else:
            pos_shifts = np.concatenate([score_positions[:1], np.diff(score_positions)])
        return pos_shifts

    def compute_onset_values(self, score_positions):
        r"""Computes number of notes and positions of notes in onsets.

        :param score_positions: score positions in ticks/beats
        :return: the number of notes and positions of notes in onsets
        """
        unique_score_pos, score_pos_counts = np.unique(score_positions, return_counts=True)
        score_pos_ids = np.arange(len(unique_score_pos)).repeat(score_pos_counts)

        notes_in_onset = score_pos_counts[score_pos_ids]
        notes_in_onset = np.minimum(notes_in_onset, self.config.additional_params["max_notes_in_onset"])

        pos_in_onset = np.repeat(np.cumsum(-score_pos_counts) + score_pos_counts, score_pos_counts)
        pos_in_onset = pos_in_onset + np.arange(len(pos_in_onset))
        pos_in_onset = np.minimum(pos_in_onset, self.config.additional_params["max_notes_in_onset"] - 1)

        return score_pos_ids, notes_in_onset, pos_in_onset

    def decode_token_type(self, tokens: np.ndarray, token_type: str) -> np.ndarray:
        r"""Decodes values from tokens for given token_type.

        :param tokens: a sequence of tokens
        :param token_type: tokens' dimension to compute values
        :return: values for a provided type
        """
        type_tokens_or_values = super().decode_token_type(tokens, token_type)
        if token_type == "PositionShift":
            return self.position_shifts[type_tokens_or_values]
        elif token_type == "OnsetDev":
            return type_tokens_or_values - self._max_beat_res * 2  # max_onset_dev
        elif token_type == "RelOnsetDev":
            return self.rel_onset_deviations[type_tokens_or_values]
        elif token_type == "PerfDuration":
            return self._duration_values[type_tokens_or_values] * self._max_beat_res
        elif token_type == "RelPerfDuration":
            return self.rel_performed_durations[type_tokens_or_values]
        else:
            return type_tokens_or_values

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
        values = super().token_type_values(token_type, normalize, special_tokens=False)
        if token_type == "PositionShift":
            values = self.position_shifts / self._max_beat_res
            if normalize:
                values = np.log2(values + 1)
        elif token_type in ("NotesInOnset", "PositionInOnset"):
            values = np.arange(1, self.config.additional_params["max_notes_in_onset"] + 1)
            if normalize:
                values = values / self.config.additional_params["max_notes_in_onset"]
        elif token_type == "OnsetDev":
            values = np.arange(-2 * self._max_beat_res, 2 * self._max_beat_res + 1) / self._max_beat_res
            if normalize:
                values = values / values[-1]
        elif token_type == "RelOnsetDev":
            values = self.rel_onset_deviations
            if normalize:
                values = np.sign(values) * np.log(np.abs(values) + 1)
        elif token_type == "RelPerfDuration":
            values = self.rel_performed_durations
            if normalize:
                values = np.log(np.abs(values) + 1)
        if special_tokens:
            values = np.concatenate([np.zeros(self.zero_token), values])
        return values
