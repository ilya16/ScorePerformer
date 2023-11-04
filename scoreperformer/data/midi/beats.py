from typing import Optional, List

import numpy as np
from miditoolkit import MidiFile, TimeSignature

BEATS_IN_BARS = {
    6: 2,
    9: 3,
    18: 3,
    12: 4,
    24: 4
}


def get_ticks_per_bar(time_sig: TimeSignature, ticks_per_beat: int = 480):
    return ticks_per_beat * 4 * time_sig.numerator // time_sig.denominator


def get_inter_beat_interval(
        *,
        time_sig: Optional[TimeSignature],
        ticks_per_bar: Optional[int] = None,
        ticks_per_beat: int = 480
):
    if ticks_per_bar is None:
        ticks_per_bar = get_ticks_per_bar(time_sig, ticks_per_beat=ticks_per_beat)

    num_beat_in_bar = BEATS_IN_BARS.get(time_sig.numerator, time_sig.numerator)
    inter_beat_interval = int(ticks_per_bar / num_beat_in_bar)

    return inter_beat_interval


def get_bar_beat_ticks(
        midi: Optional[MidiFile] = None,
        *,
        time_sigs: Optional[List[TimeSignature]] = None,
        ticks_per_beat: Optional[int] = None,
        max_tick: Optional[int] = None
):
    assert midi is not None or all(map(lambda x: x is not None, (time_sigs, ticks_per_beat, max_tick)))

    if midi is not None:
        time_sigs = midi.time_signature_changes
        ticks_per_beat = midi.ticks_per_beat
        max_tick = midi.max_tick - 1

    bar_ticks, beat_ticks = [], []
    for i, time_sig in enumerate(time_sigs):
        last_tick = time_sigs[i + 1].time if i < len(time_sigs) - 1 else max_tick

        ticks_per_bar = get_ticks_per_bar(time_sig, ticks_per_beat=ticks_per_beat)
        bar_ticks.append(np.arange(time_sig.time, last_tick, ticks_per_bar))

        inter_beat_interval = get_inter_beat_interval(
            time_sig=time_sig, ticks_per_bar=ticks_per_bar, ticks_per_beat=ticks_per_beat
        )
        beat_ticks.append(np.arange(time_sig.time, last_tick, inter_beat_interval))

    if len(time_sigs) > 1:
        bar_ticks, beat_ticks = np.concatenate(bar_ticks), np.concatenate(beat_ticks)
    else:
        bar_ticks, beat_ticks = bar_ticks[0], beat_ticks[0]

    return bar_ticks, beat_ticks


def get_performance_beats(
        score_beats: np.ndarray,
        position_pairs: np.ndarray,
        max_tick: Optional[int] = None,
        max_time: Optional[float] = None,
        monotonic_times: bool = False,
        ticks_per_beat: int = 480
):
    if monotonic_times:
        mono_position_pairs = [position_pairs[0]]
        cur_pair = prev_pair = position_pairs[0]
        for pair in position_pairs[1:]:
            min_shift_time = (pair[0] - cur_pair[0]) / ticks_per_beat / 10  # tempo 600
            if pair[0] != prev_pair[0] and pair[1] > prev_pair[1] and pair[1] > cur_pair[1] + min_shift_time:
                mono_position_pairs.append(pair)
                cur_pair = pair
            prev_pair = pair
        position_pairs = np.array(mono_position_pairs)

    if max_tick is not None and max_time is not None:
        position_pairs = np.concatenate([position_pairs, [(max_tick, max_time)]])
        score_beats = np.concatenate([score_beats, [max_tick]])

    onset_ticks, perf_times = position_pairs[:, 0], position_pairs[:, 1]
    beat_onset_indices = np.minimum(len(onset_ticks) - 1, np.searchsorted(onset_ticks, score_beats))

    # fill known beats
    perf_beats = []
    for i, beat in enumerate(score_beats):
        onset_idx = beat_onset_indices[i]
        if onset_ticks[onset_idx] == beat:
            perf_beat = perf_times[onset_idx]
        else:
            # interpolate
            if i == 0 or onset_idx == 0:
                onset_idx += 1

            left_tick, right_tick = onset_ticks[onset_idx - 1], onset_ticks[onset_idx]
            left_time, right_time = perf_times[onset_idx - 1], perf_times[onset_idx]

            perf_beat = left_time + (right_time - left_time) * (beat - left_tick) / (right_tick - left_tick)

        perf_beats.append(perf_beat)

    if max_tick is not None and max_time is not None:
        if score_beats[-2] == score_beats[-1]:
            score_beats = score_beats[:-1]
            perf_beats = perf_beats[:-1]

    perf_beats = np.array(perf_beats)

    return score_beats, perf_beats
