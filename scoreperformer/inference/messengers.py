""" Token-to-MIDIEvent messengers. """

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np
from miditok.constants import TEMPO

from scoreperformer.data.tokenizers import SPMuple, SPMuple2

NOTE_ON_MIDI_EVENT = 144


@dataclass
class IntermediateData:
    tempos: Optional[np.ndarray] = None


class SPMupleMessenger:
    def __init__(self, tokenizer: SPMuple):
        self.tokenizer = tokenizer
        self._init_variables()

    def _init_variables(self):
        self.beat_resolution = max(self.tokenizer.config.beat_res.values())

    def tokens_to_messages(
            self,
            tokens: np.ndarray,
            note_attributes: bool = True,
            note_on_events: bool = True,
            note_off_events: bool = True,
            intermediates: Optional[IntermediateData] = None,
            return_intermediates: bool = False,
            to_times: bool = True,
            sort: bool = True
    ):
        # Compute NoteON, Time Signature, Bar and Beat ticks
        ticks_data = self.tokenizer.compute_ticks(tokens, self.beat_resolution, compute_beat_ticks=True)

        # Note positions
        perf_durations = durations = self.tokenizer.decode_token_type(tokens, 'Duration')
        note_on_ticks = ticks_data['note_on'].astype(float)

        if isinstance(self.tokenizer, SPMuple):  # Process and transform relative deviations and scales
            # Compute position shifts
            if self.tokenizer.config.additional_params['use_position_shifts']:
                pos_shifts = self.tokenizer.decode_token_type(tokens, 'PositionShift')
            else:
                pos_shifts = self.tokenizer.compute_position_shifts(note_on_ticks)  # (?) messages starting not from 0

            # OnsetDevs to ticks
            if self.tokenizer.config.additional_params['rel_onset_dev']:
                rel_onset_devs = self.tokenizer.decode_token_type(tokens, 'RelOnsetDev')
                pos_shifts[pos_shifts == 0] = 1
                onset_devs = rel_onset_devs * pos_shifts
            else:
                onset_devs = self.tokenizer.decode_token_type(tokens, 'OnsetDev')

            # Shift onsets
            note_on_ticks += onset_devs
            note_on_ticks = np.maximum(0, note_on_ticks)

            # PerfDuration to ticks
            if self.tokenizer.config.additional_params['rel_perf_duration']:
                rel_perf_durations = self.tokenizer.decode_token_type(tokens, 'RelPerfDuration')
                perf_durations = rel_perf_durations * durations
            else:
                perf_durations = self.tokenizer.decode_token_type(tokens, 'PerfDuration')

        note_off_ticks = note_on_ticks + perf_durations

        # Note attributes
        assert note_on_events or note_off_events
        if note_attributes:
            pitches = self.tokenizer.decode_token_type(tokens, 'Pitch')
            velocities = self.tokenizer.decode_token_type(tokens, 'Velocity')

        # Process Tempo changes
        tempo_indices = np.concatenate(
            [[0], np.where(np.diff(tokens[:, self.tokenizer.vocab_types_idx['Tempo']]))[0] + 1]
        )
        tempos = self.tokenizer.decode_token_type(tokens[tempo_indices], 'Tempo')

        prev_tempos = intermediates.tempos if intermediates is not None else None
        start_tempo_change = prev_tempos is not None and prev_tempos[-1, 0] != tempos[0]
        if start_tempo_change:
            tempos = np.concatenate([[prev_tempos[-1, 0]], tempos])

        prev_tempo_tick = 0 if prev_tempos is None else prev_tempos[-1, 1]
        prev_tempo_time = 0. if prev_tempos is None else prev_tempos[-1, 2]

        # Get beat positions to tie Tempo change to them
        beat_ticks = ticks_data["bar"] if self.tokenizer.config.additional_params["bar_tempos"] else ticks_data["beat"]

        # Tempo ticks and Tempo changes
        tempo_ticks = note_on_ticks[tempo_indices]  # Note: position at the start of the beat
        tempo_ticks = beat_ticks[np.minimum(np.searchsorted(beat_ticks, tempo_ticks), beat_ticks.shape[0] - 1)]
        tempo_ticks[0] = prev_tempo_tick

        if start_tempo_change:
            tempo_ticks = np.concatenate([
                [tempo_ticks[0]],
                [beat_ticks[np.minimum(np.searchsorted(beat_ticks, note_on_ticks[0]), beat_ticks.shape[0] - 1)]],
                tempo_ticks[1:]
            ])

        tempo_times = np.cumsum(
            np.concatenate([[prev_tempo_time], np.diff(tempo_ticks) / self.beat_resolution * 60 / tempos[:-1]])
        )
        new_tempos = np.stack([tempos, tempo_ticks, tempo_times], axis=-1)

        messages = []
        if note_attributes:
            midi_msgs = np.full_like(pitches, NOTE_ON_MIDI_EVENT)
            if note_on_events:
                messages.append(np.stack([note_on_ticks, midi_msgs, pitches, velocities], axis=-1))
            if note_off_events:
                messages.append(np.stack([note_off_ticks, midi_msgs, pitches, np.zeros(velocities.shape[0])], axis=-1))
        else:
            if note_on_events:
                messages.append(note_on_ticks)
            if note_off_events:
                messages.append(note_off_ticks)
        messages = np.concatenate(messages, axis=0)

        if to_times:
            messages = self.messages_to_times(messages, new_tempos, sort=sort)
        elif sort:
            messages = self.sort_messages(messages)

        if return_intermediates:
            if prev_tempos is None:
                prev_tempos = new_tempos
            else:
                prev_tempos = np.concatenate([prev_tempos, new_tempos[1:]], axis=0)

            # remove duplicates by ticks and tempos
            tempo_ticks = np.concatenate([prev_tempos[:, 1], [-1]])
            prev_tempos = prev_tempos[(tempo_ticks[1:] - tempo_ticks[:-1]) != 0]
            tempos = np.concatenate([[-1], prev_tempos[:, 0]])
            prev_tempos = prev_tempos[(tempos[1:] - tempos[:-1]) != 0]

            return messages, IntermediateData(tempos=prev_tempos)
        else:
            return messages

    def messages_to_times(
            self,
            messages: np.ndarray,
            tempos: np.ndarray,
            sort: bool = True,
            inplace: bool = True
    ):
        tempos, tempo_ticks, tempo_times = tempos[:, 0], tempos[:, 1], tempos[:, 2]

        msg_ticks = messages[:, 0] if len(messages.shape) == 2 else messages
        msg_tempo_ids = np.searchsorted(tempo_ticks, msg_ticks, side='right') - 1
        tempos, tempo_ticks, tempo_times = map(lambda t: t[msg_tempo_ids], (tempos, tempo_ticks, tempo_times))
        msg_times = tempo_times + (msg_ticks - tempo_ticks) / self.beat_resolution * 60 / tempos

        messages = messages if inplace else copy.copy(messages)

        if len(messages.shape) == 2:
            messages[:, 0] = msg_times
        else:
            messages[:] = msg_times

        if sort:
            messages = self.sort_messages(messages)

        return messages

    @staticmethod
    def sort_messages(messages: np.ndarray):
        if len(messages.shape) == 2:
            return messages[np.lexsort((-messages[:, 3], messages[:, 2], messages[:, 0]))]
        else:
            return messages[np.lexsort((messages,))]

    @staticmethod
    def filter_messages(messages: np.ndarray, start: float = 0.):
        if len(messages.shape) == 2:
            return messages[messages[:, 0] >= start]
        else:
            return messages[messages >= start]


@dataclass
class SPMuple2IntermediateData(IntermediateData):
    initial_tempo: float = TEMPO
    onset_pairs: Optional[np.ndarray] = None


class SPMuple2Messenger(SPMupleMessenger):
    def __init__(self, tokenizer: SPMuple2):
        super().__init__(tokenizer=tokenizer)

    def tokens_to_messages(
            self,
            tokens: np.ndarray,
            note_attributes: bool = True,
            note_on_events: bool = True,
            note_off_events: bool = True,
            intermediates: Optional[SPMuple2IntermediateData] = None,
            return_intermediates: bool = False,
            to_times: bool = True,
            sort: bool = True
    ):
        assert to_times, "Tick messages are not supported with SPMuple2 encoding"
        self.tokenizer: SPMuple2

        tempo_scale = 60 / self.beat_resolution  # time_division
        self.tokenizer._current_midi_metadata = {'tempo_scale': tempo_scale}

        # Compute NoteON, Time Signature, Bar and Beat ticks
        ticks_data = self.tokenizer.compute_ticks(tokens, self.beat_resolution, compute_beat_ticks=True)

        # Note positions
        durations = self.tokenizer.decode_token_type(tokens, 'Duration')
        note_on_ticks = ticks_data['note_on'].astype(float)

        if intermediates is None:
            intermediates = SPMuple2IntermediateData()

        # Get token tempos
        token_tempos = self.tokenizer.decode_token_type(tokens, 'Tempo')

        # Create list of tempos
        tempos = intermediates.tempos
        if tempos is None:
            tempos = np.array([[intermediates.initial_tempo, 0, 0.]])
        tempo = tempos[-1, 0]

        # Record performed notes
        is_performed = tokens[:, self.tokenizer.vocab_types_idx['Velocity']] != self.tokenizer.zero_token

        # Get unique performed score onsets
        score_onsets = np.unique(note_on_ticks[is_performed])

        # Decode RelativeOnsetDeviation and RelativePerformedDuration tokens
        note_rel_onset_devs = self.tokenizer.decode_token_type(tokens, 'RelOnsetDev')
        note_rel_perf_durations = self.tokenizer.decode_token_type(tokens, 'RelPerfDuration')

        # Build onset pairs, compute performance notes start and end times
        onset_pairs = intermediates.onset_pairs
        if onset_pairs is None:
            if note_on_ticks[0] > 0:
                onset_pairs = np.array([(0, 0, 1)])
            else:
                onset_pairs = np.array([(-1, -1 / tempo * tempo_scale, 1)])
        prev_onset_tick, prev_onset_time, prev_num = onset_pairs[-1]

        num_tokens = len(note_on_ticks)
        perf_times, perf_offset_times = np.zeros(num_tokens), np.zeros(num_tokens)

        for i, onset_tick in enumerate(score_onsets):
            repeated_onset = onset_tick == tempos[-1, 1] and onset_tick > 0
            if repeated_onset:
                prev_onset_tick, prev_onset_time, prev_num = onset_pairs[-2]
                tempo = tempos[-2, 0]

            onset_mask = note_on_ticks == onset_tick
            num = onset_mask.sum()

            if not self.tokenizer.config.additional_params["decode_recompute_tempos"] \
                    or self.tokenizer.config.additional_params["onset_tempos"]:
                if repeated_onset:
                    tempo = (tempo * prev_num + token_tempos[onset_mask].sum()) / (prev_num + num)
                else:
                    tempo = token_tempos[onset_mask].mean()

            score_shift = onset_tick - prev_onset_tick

            # Compute time shift using tempo
            time_shift = score_shift / tempo * tempo_scale
            onset_time = prev_onset_time + time_shift

            # Compute onset deviations for each note
            onset_devs = note_rel_onset_devs[onset_mask] * time_shift
            onset_perf_times = onset_time + onset_devs

            # Average across performed notes
            if repeated_onset:
                onset_time = (onset_pairs[-1, 1] * prev_num + onset_perf_times[is_performed[onset_mask]].sum())
                onset_time /= (prev_num + num)
            else:
                onset_time = onset_perf_times[is_performed[onset_mask]].mean()

            # Add new onset pair
            if repeated_onset:
                onset_pairs[-1] = np.array([onset_tick, onset_time, prev_num + num])
            else:
                onset_pairs = np.concatenate([onset_pairs, [(onset_tick, onset_time, num)]])
            onset_pair = onset_pairs[-1]

            # Process performed durations to compute note offset time
            onset_score_time_durations = durations[onset_mask] / tempo * tempo_scale
            onset_perf_time_durations = note_rel_perf_durations[onset_mask] * onset_score_time_durations

            # Save note attributes
            perf_times[onset_mask] = onset_perf_times
            perf_offset_times[onset_mask] = onset_perf_times + onset_perf_time_durations

            # Compute next tempo
            if self.tokenizer.config.additional_params["decode_recompute_tempos"] \
                    and not self.tokenizer.config.additional_params["onset_tempos"]:
                if onset_time < 2 * self.tokenizer.config.additional_params["tempo_min_onset_dist"]:
                    tempo = intermediates.initial_tempo  # not enough history, use initial tempo
                else:
                    # Cut onsets in a local window
                    pairs_in_window = self.tokenizer.filter_onsets_in_window(
                        onset_pair[:2], onset_pairs[:-1, :2], index=len(onset_pairs) - 1
                    )

                    # Compute local tempo
                    tempo = self.tokenizer.compute_local_tempo(distances=onset_pair[:2] - pairs_in_window)

            if repeated_onset:
                tempos[-1] = np.array([[tempo, onset_tick, onset_time]])
            else:
                tempos = np.concatenate([tempos, np.array([[tempo, onset_tick, onset_time]])])

            if repeated_onset:
                prev_onset_tick, prev_onset_time, prev_num = onset_pairs[-1]
            else:
                prev_onset_tick, prev_onset_time, prev_num = onset_tick, onset_time, num

        # Note attributes
        assert note_on_events or note_off_events
        if note_attributes:
            pitches = self.tokenizer.decode_token_type(tokens, 'Pitch')
            velocities = self.tokenizer.decode_token_type(tokens, 'Velocity')

        messages = []
        if note_attributes:
            midi_msgs = np.full_like(pitches, NOTE_ON_MIDI_EVENT)
            if note_on_events:
                messages.append(np.stack([perf_times, midi_msgs, pitches, velocities], axis=-1))
            if note_off_events:
                messages.append(
                    np.stack([perf_offset_times, midi_msgs, pitches, np.zeros(velocities.shape[0])], axis=-1))
        else:
            if note_on_events:
                messages.append(perf_times)
            if note_off_events:
                messages.append(perf_offset_times)
        messages = np.concatenate(messages, axis=0)

        if sort:
            messages = self.sort_messages(messages)

        if return_intermediates:
            intermediates = SPMuple2IntermediateData(
                tempos=tempos,
                initial_tempo=intermediates.initial_tempo,
                onset_pairs=onset_pairs
            )

            return messages, intermediates
        else:
            return messages
