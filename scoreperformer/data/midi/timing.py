import copy
from typing import List

import numpy as np
from miditoolkit import Instrument

from scoreperformer.utils import find_closest
from ..midi.containers import Note


def convert_symbolic_timing_to_absolute(
        tracks: List[Instrument],
        tick_to_time: np.ndarray,
        inplace: bool = True,
        time_shift: float = 0.
):
    tracks = tracks if inplace else copy.deepcopy(tracks)

    for track in tracks:
        track.notes = [
            Note(pitch=n.pitch, velocity=n.velocity,
                 start=time_shift + float(tick_to_time[n.start]),
                 end=time_shift + float(tick_to_time[n.end]))
            for n in track.notes
        ]
        for control_change in track.control_changes:
            control_change.time = time_shift + float(tick_to_time[control_change.time])
        for pedal in track.pedals:
            pedal.start = time_shift + float(tick_to_time[pedal.start])
            pedal.end = time_shift + float(tick_to_time[pedal.end])
        for pitch_bend in track.pitch_bends:
            pitch_bend.time = time_shift + float(tick_to_time[pitch_bend.time])

    return tracks


def convert_absolute_timing_to_symbolic(
        tracks: List[Instrument],
        time_to_tick: np.ndarray,
        inplace: bool = True
):
    tracks = tracks if inplace else copy.deepcopy(tracks)

    def process_interval_events(events):
        start_times = np.array(list(map(lambda x: x.start, events)))
        start_ticks = find_closest(time_to_tick, start_times)
        end_times = np.array(list(map(lambda x: x.end, events)))
        end_ticks = find_closest(time_to_tick, end_times)
        for event, start_t, end_t in zip(events, start_ticks, end_ticks):
            if start_t == end_t:
                end_t += 1
            event.start = start_t
            event.end = end_t

    def process_time_events(events):
        times = np.array(list(map(lambda x: x.time, events)))
        ticks = find_closest(time_to_tick, times)
        for event, t in zip(events, ticks):
            event.time = t

    for track in tracks:
        process_interval_events(track.notes)
        process_interval_events(track.pedals)
        process_time_events(track.control_changes)
        process_time_events(track.pitch_bends)

    return tracks
