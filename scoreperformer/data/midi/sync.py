import copy
from typing import Optional

import numpy as np
from miditoolkit import MidiFile, TempoChange, Marker

from scoreperformer.utils import find_closest
from .beats import get_inter_beat_interval, get_bar_beat_ticks, get_performance_beats
from .timing import (
    convert_symbolic_timing_to_absolute,
    convert_absolute_timing_to_symbolic
)
from .utils import filter_late_midi_events


def sync_performance_midi(
        score_midi: MidiFile,
        perf_midi: MidiFile,
        onset_pairs: np.ndarray,
        is_absolute_timing: bool = False,
        max_time: Optional[float] = None,
        ticks_per_beat: int = 480,
        bar_sync: bool = True,
        inplace: bool = True,
        verbose: bool = False
):
    """
    Synchronizes performance MIDI with score MIDI bars/beats through the onset pairs.

    Also adds silent (unperformed) performance notes as special markers based on the alignment.
    """
    perf_midi = copy.deepcopy(perf_midi) if not inplace else perf_midi

    # preprocess performance midi
    filter_late_midi_events(perf_midi)
    max_tick = score_midi.max_tick

    if not is_absolute_timing:
        tick_to_time = perf_midi.get_tick_to_time_mapping()
        max_time = tick_to_time[-1]
    else:
        assert max_time is not None, "`max_time` should be explicitly provided for MIDI with absolute timing"
        tick_to_time = None

    # compute score and performance onsets
    score_bars, score_beats = get_bar_beat_ticks(score_midi)
    score_onsets = score_bars if bar_sync else score_beats
    score_onsets, perf_onsets = get_performance_beats(
        score_onsets, onset_pairs,
        max_tick=max_tick - 1, max_time=max_time,
        monotonic_times=True, ticks_per_beat=ticks_per_beat
    )
    perf_shift = perf_onsets[0]
    perf_onsets -= perf_shift
    max_time -= perf_shift

    perf_score_tick_ratio = ticks_per_beat / score_midi.ticks_per_beat

    time_signatures = score_midi.time_signature_changes

    time_sig_ticks, quarter_note_factors, inter_onset_intervals = [], [], []
    for time_sig in time_signatures:
        time_sig_ticks.append(time_sig.time)
        quarter_note_factors.append(4 * time_sig.numerator / time_sig.denominator)
        inter_onset_intervals.append(
            get_inter_beat_interval(time_sig=time_sig, ticks_per_beat=score_midi.ticks_per_beat)
        )

    time_sig_ticks, quarter_note_factors, inter_onset_intervals = map(
        np.array, (time_sig_ticks, quarter_note_factors, inter_onset_intervals)
    )
    inter_beat_intervals = inter_onset_intervals

    ticks_per_bar = (score_midi.ticks_per_beat * quarter_note_factors).astype(int)
    beats_per_bar = ticks_per_bar / inter_beat_intervals
    ioi_in_quarters = ibi_in_quarters = quarter_note_factors / beats_per_bar

    if bar_sync:
        inter_onset_intervals = inter_onset_intervals * beats_per_bar
        ioi_in_quarters = ioi_in_quarters * beats_per_bar

    if verbose:
        print(f'score: time_sigs={time_signatures}\n'
              f'       ticks_per_beat={score_midi.ticks_per_beat}, ticks_per_bar={ticks_per_bar}\n'
              f'       inter_beat_intervals={inter_beat_intervals}, inter_onset_intervals={inter_onset_intervals}\n'
              f'       ibi_in_quarters={ibi_in_quarters}, ioi_in_quarters={ioi_in_quarters}')

    # compute tempos
    intervals = np.diff(perf_onsets)
    if np.any(intervals <= 0.):
        return None

    time_sig_indices = (np.searchsorted(time_sig_ticks, score_onsets, side='right') - 1)[:-1]
    inter_onset_ratios = np.diff(score_onsets) / inter_onset_intervals[time_sig_indices]
    tempos = 60 / intervals * ioi_in_quarters[time_sig_indices] * inter_onset_ratios

    if verbose:
        print(f'tempos: ({tempos.min():.3f}, {tempos.max():.3f}), {np.median(tempos):.3f}')

    # get absolute timing of instruments
    if is_absolute_timing:
        abs_instr = perf_midi.instruments
    else:
        abs_instr = convert_symbolic_timing_to_absolute(
            perf_midi.instruments, tick_to_time, inplace=inplace, time_shift=-perf_shift
        )

    # compute time to tick mapping
    inter_onset_intervals = inter_onset_intervals[time_sig_indices] * perf_score_tick_ratio * inter_onset_ratios
    resample_timing = []
    for i in range(len(perf_onsets) - 1):
        start_beat, end_beat = perf_onsets[i], perf_onsets[i + 1]
        resample_timing.append(np.linspace(start_beat, end_beat, int(inter_onset_intervals[i]) + 1)[:-1])

    resample_timing.append([max_time])
    resample_timing = np.round(np.concatenate(resample_timing), 6)

    # new a midifile obj
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    # convert abs to sym
    sym_instr = convert_absolute_timing_to_symbolic(abs_instr, resample_timing, inplace=inplace)

    # process timing of markers
    markers = perf_midi.markers if hasattr(perf_midi, 'markers') else []
    for marker in markers:
        marker.time = find_closest(resample_timing, float(tick_to_time[marker.time]) - perf_shift)
        if marker.text.startswith('NoteI'):
            pitch, start, end = map(int, marker.text.split('_')[1:])
            start, end = map(lambda x: find_closest(resample_timing, float(tick_to_time[x]) - perf_shift), (start, end))
            marker.text = f'NoteI_{pitch}_{start}_{end}'

    # tempo
    tempo_changes = []
    onset_ticks = find_closest(resample_timing, perf_onsets)
    for pos_tick, tempo in zip(onset_ticks[:-1], tempos):
        tempo_changes.append(TempoChange(tempo=float(tempo), time=int(pos_tick)))

    tempo_changes = [tempo for tempo in tempo_changes if tempo.time < resample_timing.shape[0]]

    # markers
    markers.insert(0, Marker(text=f'Shift_{perf_shift:.6f}', time=0))

    # set attributes
    midi.tempo_changes = tempo_changes
    midi.time_signature_changes = time_signatures
    midi.instruments = sym_instr
    midi.markers = markers
    midi.max_tick = resample_timing.shape[0]

    return midi
