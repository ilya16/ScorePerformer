"""
SPMuple2 (ScorePerformanceMusic-tuple) encoding method,
an improved SPMuple score-performance encoding with local window-based tempo calculation.
"""

from pathlib import Path
from typing import List, Optional, Union, Any

import numpy as np
from miditok.constants import TEMPO, MIDI_INSTRUMENTS
from miditok.midi_tokenizer import _in_as_seq
from miditok.utils import remove_duplicated_notes
from miditoolkit import MidiFile, Instrument, Note, TimeSignature

from scoreperformer.utils import find_closest
from .spmuple import SPMuple
from ..classes import TokSequence
from ..constants import TIME_DIVISION, MASK_TOKEN
from ...midi.sync import sync_performance_midi
from ...midi.utils import cut_overlapping_notes


class SPMuple2(SPMuple):
    r"""SPMuple2: ScorePerformanceMusic-tuple encoding.

    An improved SPMuple encoding with local window-based iterative tempo calculation
    (with an option to omit Tempo tokens at all), score-specific tokens
    (Bar, Position, Pitch, Velocity, Duration, TimeSig, Program, PositionShift)
    and performance-specific tokens (RelOnsetDeviation, RelPerformedDuration).

    Uses OctupleM encoding for score MIDIs and adds performance tokens for performance MIDIs.
    Supports local window and inter-onset tempos.
    """

    def _tweak_config_before_creating_voc(self):
        additional_params = self.config.additional_params

        # default parameters
        additional_params["rel_onset_dev"] = True
        additional_params["nb_onset_devs"] = additional_params.get("nb_onset_devs", 161)

        additional_params["rel_perf_duration"] = True
        additional_params["nb_perf_durations"] = additional_params.get("nb_perf_durations", 81)

        super()._tweak_config_before_creating_voc()

        # tempo encoding/decoding parameters
        additional_params["onset_tempos"] = additional_params.get("onset_tempos", False)
        additional_params["tempo_window"] = additional_params.get("tempo_window", 8.)
        additional_params["tempo_min_onset_dist"] = additional_params.get("tempo_min_onset_dist", 0.5)
        additional_params["tempo_min_onsets"] = additional_params.get("tempo_min_onsets", 8)

        additional_params["use_quantized_tempos"] = additional_params.get("use_quantized_tempos", True)
        additional_params["decode_recompute_tempos"] = additional_params.get("decode_recompute_tempos", False)

        # outlier detection and processing
        additional_params["limit_rel_onset_devs"] = additional_params.get("limit_rel_onset_devs", True)

    def preprocess_midi(self, midi: MidiFile, is_score: bool = True):
        r"""Preprocess a MIDI file to be used by SPMuple2 encoding.

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

        # Recalculate max_tick is this could have change after notes quantization
        if len(midi.instruments) > 0:
            midi.max_tick = max([max([note.end for note in track.notes]) for track in midi.instruments])
            midi.tempo_changes = [tempo for tempo in midi.tempo_changes if tempo.time < midi.max_tick]

        # Not needed for performance MIDIs (time sigs from scores, tempo recalculated and (quantized))
        if is_score:
            if self.config.use_tempos:
                self._quantize_tempos(midi.tempo_changes, midi.ticks_per_beat)

            if self.config.use_time_signatures:
                self._quantize_time_signatures(midi.time_signature_changes, midi.ticks_per_beat)

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
            *11: RelativeOnsetDeviation (performance token)
            *12: RelativePerformedDuration (performance token)

        :param midi: the MIDI object to convert.
        :param score_tokens: corresponding score tokens :class:`miditok.TokSequence`.
        :param alignment: optional alignment between performance and score tokens.
        :return: the performance token representation, i.e. tracks converted into sequences of tokens
        """
        additional_params = self.config.additional_params

        # Prepare constants used for calculations
        time_division = self._current_midi_metadata["time_division"]
        ticks_per_sample = time_division / self._max_beat_res
        tempo_scale = self._current_midi_metadata["tempo_scale"] = 60 / time_division

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
            if self.config.use_tempos:
                time_step[self.vocab_types_idx["Tempo"]] = MASK_TOKEN

        # Convert tokens to ids
        tokens = np.array(self._tokens_to_ids(tokens))

        # Process score tokens
        score_tokens = np.array(score_tokens.ids)

        # Apply alignment
        if alignment is not None:
            tokens, perf_positions, perf_durations = map(
                lambda x: x[alignment], (tokens, perf_positions, perf_durations)
            )

        # Copy score tokens to midi tokens
        token_types = ["Bar", "Position", "Duration", "TimeSig"]
        if additional_params["use_position_shifts"]:
            token_types.append("PositionShift")
        if additional_params["use_onset_indices"]:
            token_types.extend(["NotesInOnset", "PositionInOnset"])
        for token_type in token_types:
            idx = self.vocab_types_idx[token_type]
            tokens[:, idx] = score_tokens[:, idx]

        tokens = tokens.astype(int)

        # Compute NoteON, Time Signature and Bar ticks
        ticks_data = self.compute_ticks(score_tokens, time_division, compute_beat_ticks=False)

        # Get score position and duration ticks
        score_ticks = ticks_data["note_on"]
        duration_ticks = self.decode_token_type(score_tokens, "Duration") * ticks_per_sample

        # Compute performance note positions
        ttt_map = midi.get_tick_to_time_mapping()
        perf_times = ttt_map[(perf_positions * ticks_per_sample).astype(int)]
        perf_offset_times = ttt_map[((perf_positions + perf_durations) * ticks_per_sample).astype(int)]

        # Record performed notes
        is_performed = tokens[:, self.vocab_types_idx["Velocity"]] != self.zero_token

        # Get unique performed score onsets
        score_onsets = np.unique(score_ticks[is_performed])

        # Build onset pairs: a list of tuples (onset_score_pos, onset_perf_time)
        _offset = 0
        onset_pairs = [(0, 0)]
        for onset_tick in score_onsets:
            onset_mask = score_ticks[_offset:] == onset_tick
            onset_perf_times = perf_times[_offset:][onset_mask]
            onset_time = onset_perf_times[is_performed[_offset:][onset_mask]].mean()

            onset_pairs.append((onset_tick, onset_time))
            _offset += len(onset_perf_times)

        onset_pairs = np.array(onset_pairs)

        # Compute initial tempo using a subset of onset pairs
        start_pairs = onset_pairs[onset_pairs[:, 1] <= 4 * additional_params["tempo_window"]]
        if len(start_pairs) < additional_params["tempo_min_onsets"]:
            start_pairs = onset_pairs[:additional_params["tempo_min_onsets"]]

        # Compute weighted initial tempo
        initial_tempo = self.compute_local_tempo(distances=start_pairs[start_pairs[:, 1] > 0.] - start_pairs[0])
        self._current_midi_metadata["initial_tempo"] = initial_tempo

        # Process zero first onset
        if onset_pairs[1, 0] == 0:
            onset_pairs[0] = [-1, -1 / initial_tempo * tempo_scale]

        if additional_params["onset_tempos"]:
            initial_tempo = self.compute_onset_tempo(onset_pairs[1], prev_onset_pair=onset_pairs[0])

        # Iteratively compute weighted local tempos, assign them to notes
        _offset, num_tokens = 0, len(tokens)
        tempos = [initial_tempo]
        note_tempos, note_next_tempos = np.ones(num_tokens), np.ones(num_tokens)
        note_onsets, note_prev_onsets = np.zeros((num_tokens, 2)), np.zeros((num_tokens, 2))
        for i, onset_pair in enumerate(onset_pairs[1:]):
            onset_tick, onset_time = onset_pair
            prev_onset_tick, prev_onset_time = onset_pairs[i]

            # Compute onset deviations for current notes
            onset_mask = score_ticks == onset_tick
            onset_time_shift = (onset_tick - prev_onset_tick) / tempos[-1] * tempo_scale
            note_perf_times = perf_times[onset_mask][is_performed[onset_mask]]
            note_onset_devs = note_perf_times - (prev_onset_time + onset_time_shift)
            note_rel_onset_devs = note_onset_devs / onset_time_shift
            start_idx = np.where(onset_mask)[0][0]

            # Limit relative onset deviations to max relative onset deviation if required
            if additional_params["limit_rel_onset_devs"] \
                    and np.any(np.abs(note_rel_onset_devs) > self.rel_onset_deviations[-1]):
                # Compute time shift for the onset deviations
                _onset_shift = (1 - self.rel_onset_deviations[-1] / np.abs(note_rel_onset_devs).max())
                _onset_shift *= -note_onset_devs[np.abs(note_onset_devs).argmax()]

                onset_time += _onset_shift
                onset_pairs[i + 1:, 1] += _onset_shift
                perf_times[start_idx:] += _onset_shift
                perf_offset_times[start_idx:] += _onset_shift

            if additional_params["onset_tempos"]:
                tempo = self.compute_onset_tempo(onset_pairs[i + 1], prev_onset_pair=onset_pairs[i])
            else:
                if onset_time < 2 * additional_params["tempo_min_onset_dist"]:
                    tempo = initial_tempo  # not enough history, use initial tempo
                else:
                    # Cut onsets in a local window
                    pairs_in_window = self.filter_onsets_in_window(onset_pair, onset_pairs, index=i + 1)

                    # Compute local tempo
                    tempo = self.compute_local_tempo(distances=onset_pair - pairs_in_window)

            tempos.append(tempo)

            note_tempos[onset_mask] = tempos[i]
            note_next_tempos[onset_mask] = tempos[i + 1]
            note_prev_onsets[onset_mask] = onset_pairs[i]
            note_onsets[onset_mask] = onset_pairs[i + 1]

        # Save MIDI data for external use
        self._current_midi_metadata.update(**{
            "onset_pairs": onset_pairs,
            "tempos": np.array(tempos),
            "note_tempos": note_tempos,
            "note_next_tempos": note_next_tempos
        })

        # Assign neighbouring tempos for not performed notes
        for _tempos in [note_tempos, note_next_tempos]:
            for i in range(1, len(_tempos)):
                if _tempos[i] == 0.:
                    _tempos[i] = _tempos[i - 1]

        # Compute tempo tokens if they are present in the encoding
        if self.config.use_tempos:
            tempo_tokens = find_closest(self.tempos, note_tempos) + self.zero_token
            tokens[:, self.vocab_types_idx["Tempo"]] = tempo_tokens

        # Compute onset deviations and RelativeOnsetDeviation tokens
        note_time_shifts = (note_onsets[:, 0] - note_prev_onsets[:, 0]) / note_tempos * tempo_scale
        note_onset_devs = perf_times - (note_prev_onsets[:, 1] + note_time_shifts)
        note_onset_devs[~is_performed] = 0  # zero out onsets for not performed notes

        note_rel_onset_devs = np.zeros_like(note_onset_devs)
        note_rel_onset_devs[is_performed] = note_onset_devs[is_performed] / note_time_shifts[is_performed]

        rel_onset_dev_tokens = find_closest(self.rel_onset_deviations, note_rel_onset_devs) + self.zero_token

        # Compute performed durations RelativePerformedDuration tokens
        perf_time_durations = perf_offset_times - perf_times
        score_time_durations = duration_ticks / note_tempos * tempo_scale

        note_rel_perf_durations = perf_time_durations / score_time_durations
        note_rel_perf_durations[~is_performed] = 1  # "zero out" durations for not performed notes

        rel_perf_duration_tokens = find_closest(self.rel_performed_durations, note_rel_perf_durations) + self.zero_token

        self._current_midi_metadata.update(**{
            "note_time_shifts": note_time_shifts,
            "note_onset_devs": note_onset_devs,
            "score_time_durations": score_time_durations,
            "perf_time_durations": perf_time_durations
        })

        # Append RelOnsetDev, and RelPerfDuration tokens
        tokens = np.concatenate([
            tokens,
            rel_onset_dev_tokens[:, None],
            rel_perf_duration_tokens[:, None]
        ], axis=1)

        tok_sequence = TokSequence(ids=tokens.tolist(), meta={"initial_tempo": initial_tempo})
        self.complete_sequence(tok_sequence)

        return tok_sequence

    @_in_as_seq()
    def performance_tokens_to_midi(
            self,
            tokens: Union[TokSequence, List, np.ndarray, Any],
            output_path: Optional[str] = None,
            time_division: int = TIME_DIVISION,
            initial_tempo: Optional[int] = None
    ) -> MidiFile:
        r"""Converts performance tokens (:class:`miditok.TokSequence`) into a MIDI and saves it.

        :param tokens: tokens to convert. Can be either a list of :class:`miditok.TokSequence`,
        :param output_path: path to save the file. (default: None)
        :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI to create).
        :param initial_tempo: starting/average performance tempo
        :return: the midi object (:class:`miditoolkit.MidiFile`).
        """
        additional_params = self.config.additional_params

        assert time_division % self._max_beat_res == 0, \
            f"Invalid time division, please give one divisible by {self._max_beat_res}"
        midi = MidiFile(ticks_per_beat=time_division)
        ticks_per_sample = time_division // self._max_beat_res
        tempo_scale = 60 / time_division
        self._current_midi_metadata = {"tempo_scale": tempo_scale}

        if isinstance(tokens, TokSequence):
            initial_tempo = tokens.meta.get("initial_tempo", initial_tempo)
        tokens = np.array(tokens.ids)

        # Compute NoteON, Time Signature and Bar ticks
        ticks_data = self.compute_ticks(tokens, time_division, compute_beat_ticks=False)

        # Get score positions
        score_ticks = ticks_data["note_on"]
        score_positions = score_ticks / ticks_per_sample

        # Get score durations
        duration_ticks = self.decode_token_type(tokens, "Duration") * ticks_per_sample

        # Build Time Signature changes
        time_sigs, time_sig_ticks = ticks_data["time_sig"]
        midi.time_signature_changes = [
            TimeSignature(int(time_sigs[i][0]), int(time_sigs[i][1]), int(time_sig_ticks[i]))
            for i in range(len(time_sigs))
        ]

        # Record performed notes
        is_performed = tokens[:, self.vocab_types_idx["Velocity"]] != self.zero_token

        # Get unique performed score onsets
        score_onsets = np.unique(score_ticks[is_performed])

        # Get token tempos
        token_tempos = self.decode_token_type(tokens, "Tempo")

        # Create list of tempos
        if not additional_params["decode_recompute_tempos"] or additional_params["onset_tempos"]:
            tempo = token_tempos[score_ticks == score_onsets[0]].mean()
        else:
            tempo = initial_tempo or TEMPO

        # Decode RelativeOnsetDeviation and RelativePerformedDuration tokens
        note_rel_onset_devs = self.decode_token_type(tokens, "RelOnsetDev")
        note_rel_perf_durations = self.decode_token_type(tokens, "RelPerfDuration")

        # Build onset pairs, compute performance notes start and end times
        onset_pairs = np.array([(0, 0)]) if score_positions[0] > 0 else np.array([(-1, -1 / tempo * tempo_scale)])
        prev_onset_tick, prev_onset_time = onset_pairs[0]

        _offset, num_tokens = 0, len(score_positions)
        perf_times, perf_offset_times = np.zeros(num_tokens), np.zeros(num_tokens)

        for i, onset_tick in enumerate(score_onsets):
            onset_mask = score_ticks[_offset:] == onset_tick

            if not additional_params["decode_recompute_tempos"] or additional_params["onset_tempos"]:
                tempo = token_tempos[_offset:][onset_mask].mean()

            score_shift = onset_tick - prev_onset_tick

            # Compute time shift using tempo
            time_shift = score_shift / tempo * tempo_scale
            onset_time = prev_onset_time + time_shift

            # Compute onset deviations for each note
            onset_devs = note_rel_onset_devs[_offset:][onset_mask] * time_shift
            onset_perf_times = onset_time + onset_devs

            # Average across performed notes
            onset_time = onset_perf_times[is_performed[_offset:][onset_mask]].mean()

            # Add new onset pair
            onset_pairs = np.concatenate([onset_pairs, [(onset_tick, onset_time)]])
            onset_pair = onset_pairs[-1]

            # Process performed durations to compute note offset time
            onset_score_time_durations = duration_ticks[_offset:][onset_mask] / tempo * tempo_scale
            onset_perf_time_durations = note_rel_perf_durations[_offset:][onset_mask] * onset_score_time_durations

            # Save note attributes
            perf_times[_offset:][onset_mask] = onset_perf_times
            perf_offset_times[_offset:][onset_mask] = onset_perf_times + onset_perf_time_durations

            # Compute next tempo
            if additional_params["decode_recompute_tempos"] and not additional_params["onset_tempos"]:
                if onset_time < 2 * additional_params["tempo_min_onset_dist"]:
                    tempo = initial_tempo  # not enough history, use initial tempo
                else:
                    # Cut onsets in a local window
                    pairs_in_window = self.filter_onsets_in_window(onset_pair, onset_pairs[:-1], index=i + 1)

                    # Compute local tempo
                    tempo = self.compute_local_tempo(distances=onset_pair - pairs_in_window)

            _offset += len(onset_perf_times)
            prev_onset_tick, prev_onset_time = onset_tick, onset_time

        # Note attributes
        pitches = self.decode_token_type(tokens, "Pitch")
        velocities = self.decode_token_type(tokens, "Velocity")

        # Max tick and time
        max_tick = (score_ticks + duration_ticks)[is_performed].max()
        max_time = perf_offset_times.max()

        # Create notes using compute note attributes (NOTE: will not work with multi-track MIDI)
        notes = [
            Note(velocity=velocities[i], pitch=pitches[i], start=perf_times[i], end=perf_offset_times[i])
            for i in range(len(pitches)) if is_performed[i]
        ]

        # Appends created notes to MIDI object
        midi.instruments.append(Instrument(0, False, MIDI_INSTRUMENTS[0]["name"]))
        midi.instruments[-1].notes = notes
        midi.max_tick = max_tick

        # Synchronize created MIDI by beats
        midi = sync_performance_midi(
            score_midi=midi,
            perf_midi=midi,
            onset_pairs=onset_pairs,
            is_absolute_timing=True,
            max_time=max_time,
            bar_sync=False,
            inplace=True
        )

        # Cut overlapping notes if any
        if additional_params["cut_overlapping_notes"]:
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

    def _create_relative_onset_deviations(self) -> np.ndarray:
        r"""Creates the relative onset deviation bins.
        The larger the factor, the smaller the resolution.

        :return: the relative onset deviation bins
        """
        onset_dev_quant = (self.config.additional_params["nb_onset_devs"] - 1) // 10

        rel_onset_devs = np.concatenate([
            # 20% from 0 to 1/20
            np.linspace(0, 1 / 20, onset_dev_quant + 1),
            # 20% from 1/20 to 1/10
            np.linspace(1 / 20, 1 / 10, onset_dev_quant + 1)[1:],
            # 20% from 1/10 to 1/6
            np.linspace(1 / 10, 1 / 6, onset_dev_quant + 1)[1:],
            # 20% from 1/6 to 1/3
            (2 ** (np.arange(onset_dev_quant + 1) / onset_dev_quant) * 1 / 6)[1:],
            # 10% from 1/3 to 1/2
            (2 ** (np.log(3 / 2) / np.log(2) * np.arange(onset_dev_quant // 2 + 1) / onset_dev_quant * 2) * 1 / 3)[1:],
            # 5% from 1/2 to 3/4
            (2 ** (np.log(3 / 2) / np.log(2) * np.arange(onset_dev_quant // 4 + 1) / onset_dev_quant * 4) * 1 / 2)[1:],
            # 2.5% from 3/4 to 1
            (2 ** (np.log(4 / 3) / np.log(2) * np.arange(onset_dev_quant // 8 + 1) / onset_dev_quant * 8) * 3 / 4)[1:],
            # 2.5% from 1 to 2
            (2 ** (np.arange(onset_dev_quant // 8 + 1) / onset_dev_quant * 8))[1:]
        ])
        rel_onset_devs = np.round(rel_onset_devs, 4)
        rel_onset_devs = np.sort(np.concatenate([-rel_onset_devs[1:], rel_onset_devs]))  # add negative deviations

        return rel_onset_devs

    def _create_relative_performed_durations(self) -> np.ndarray:
        r"""Creates the relative performed duration bins based on some heuristics.
        The larger the factor, the smaller the resolution.

        :return: the relative onset deviation bins
        """
        perf_dur_quant = (self.config.additional_params["nb_perf_durations"] - 1) // 5

        rel_performed_durations = np.concatenate([
            # 20% from 1/10 to 1/3
            np.linspace(1 / 10, 1 / 3, perf_dur_quant + 1),
            # 40% from 1/3 to 4/5
            np.linspace(1 / 3, 4 / 5, 2 * perf_dur_quant + 1)[1:],
            # 20% from 4/5 to 1
            np.linspace(4 / 5, 1., perf_dur_quant + 1)[1:],
            # 10% from 1 to 5/4
            np.linspace(1.0, 5 / 4, perf_dur_quant // 2 + 1)[1:],
            # 5% from 5/4 to 3/2
            np.linspace(5 / 4, 3 / 2, perf_dur_quant // 4 + 1)[1:],
            # 5% from 3/2 to 3
            (2 ** (4 * np.arange(perf_dur_quant // 4 + 1) / perf_dur_quant) * 3 / 2)[1:],
        ])
        rel_performed_durations = np.round(rel_performed_durations, 4)

        return rel_performed_durations

    def filter_onsets_in_window(self, onset_pair: np.ndarray, onset_pairs: np.ndarray, index: int):
        r"""Selects onsets in the local window for the specified onset.

        :param onset_pair: current onset (tick, time) pair
        :param onset_pairs: all onset (tick, time) pairs
        :param index: index of the current onset in the list of pairs
        :return: the subset of onset pairs in the local window
        """
        _, onset_time = onset_pair
        additional_params = self.config.additional_params

        candidate_pairs = onset_pairs[:index][
            onset_pairs[:index, 1] <= onset_time - additional_params["tempo_min_onset_dist"]
            ]
        if len(candidate_pairs) == 0:
            candidate_pairs = onset_pairs[:index]

        pairs_in_window = candidate_pairs[candidate_pairs[:, 1] >= onset_time - additional_params["tempo_window"]]

        if len(pairs_in_window) < additional_params["tempo_min_onsets"]:  # collect minimum required number of onsets
            pairs_in_window = candidate_pairs[max(0, len(candidate_pairs) - additional_params["tempo_min_onsets"]):]
            pairs_in_window = pairs_in_window[
                pairs_in_window[:, 1] >= onset_time - 4 * additional_params["tempo_window"]
                ]

        if len(pairs_in_window) == 0:  # if suddenly no pairs found, take all previous and hope for the best
            pairs_in_window = candidate_pairs

        return pairs_in_window

    def compute_local_tempo(self, distances: np.ndarray):
        r"""Computes weighted local tempo from the tick and time distances.

        :param distances: all onset (tick, time) distance pairs
        :return: the computed local tempo
        """
        local_tempos = distances[:, 0] / distances[:, 1] * self._current_midi_metadata["tempo_scale"]
        weights = 1 - distances[:, 1] / (distances[:, 1].max() + 0.01)
        weights /= weights.sum()

        tempo = max(self.tempos[0], (weights * local_tempos).sum())

        if self.config.use_tempos and self.config.additional_params["use_quantized_tempos"]:
            tempo = self.tempos[find_closest(self.tempos, tempo)]

        return tempo

    def compute_onset_tempo(self, onset_pair: np.ndarray, prev_onset_pair: np.ndarray):
        r"""Computes onset tempo from the tick and time distance for current and previous onsets.

        :param onset_pair: current pair onset (tick, time)
        :param prev_onset_pair: previous onset pair (tick, time)
        :return: the computed local tempo
        """
        if onset_pair[1] <= prev_onset_pair[1]:
            tempo = self.tempos[-1]
        else:
            tempo = (onset_pair[0] - prev_onset_pair[0]) / (onset_pair[1] - prev_onset_pair[1])
            tempo *= self._current_midi_metadata["tempo_scale"]

        if self.config.use_tempos and self.config.additional_params["use_quantized_tempos"]:
            tempo = self.tempos[find_closest(self.tempos, tempo)]

        return tempo
