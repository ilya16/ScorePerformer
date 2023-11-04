import copy
from collections import defaultdict
from typing import Optional, List

import numpy as np
from miditoolkit import Note, MidiFile

from scoreperformer.utils import find_closest


def sort_notes(
        notes: List[Note],
        compute_sort_indices: bool = False,
        order: str = 'time'
):
    assert order in ('time', 'pitch')

    sort_ids = None
    if order == 'time':
        if compute_sort_indices:
            sort_ids = np.lexsort([[n.end for n in notes], [n.pitch for n in notes], [n.start for n in notes]])
        notes.sort(key=lambda n: (n.start, n.pitch, n.end))
    elif order == 'pitch':
        if compute_sort_indices:
            sort_ids = np.lexsort([[n.end for n in notes], [n.start for n in notes], [n.pitch for n in notes]])
        notes.sort(key=lambda n: (n.pitch, n.start, n.end))

    return notes, sort_ids


def cut_overlapping_notes(notes: List[Note], return_sort_indices: bool = False):
    r"""Find and cut the first of the two overlapping notes, i.e. with the same pitch,
    and the second note starting before the ending of the first note.

    :param notes: notes to analyse
    :param return_sort_indices: return indices by which the original notes were sorted
    """
    # sort by pitch, then time
    notes, sort_ids = sort_notes(notes, compute_sort_indices=return_sort_indices, order='pitch')

    for i in range(1, len(notes)):
        prev_note, note = notes[i - 1], notes[i]
        if prev_note.pitch == note.pitch and prev_note.end >= note.start:
            if note.start <= 1:
                note.start = 2
            prev_note.end = note.start - 1
            if prev_note.start >= prev_note.end:  # resulted in an invalid note, fix it too
                prev_note.start = prev_note.end - 1

    # sort back by time, then pitch
    notes, sort_ids_back = sort_notes(notes, compute_sort_indices=return_sort_indices, order='time')

    if return_sort_indices:
        sort_ids = sort_ids[sort_ids_back]
        return notes, sort_ids
    return notes


def remove_duplicated_notes(notes: List[Note], return_sort_indices: bool = False):
    r"""Find and remove exactly similar notes, i.e. with the same pitch, start and end.

    :param notes: notes to analyse
    :param return_sort_indices: return indices by which the original notes were sorted
    """
    # sort by pitch, then time
    notes, sort_ids = sort_notes(notes, compute_sort_indices=return_sort_indices, order='pitch')

    for i in range(len(notes) - 1, 0, -1):  # removing possible duplicated notes
        if notes[i].pitch == notes[i - 1].pitch and notes[i].start == notes[i - 1].start and \
                notes[i].end >= notes[i - 1].end:
            del notes[i]

    # sort back by time, then pitch
    notes, sort_ids_back = sort_notes(notes, compute_sort_indices=return_sort_indices, order='time')

    if return_sort_indices:
        sort_ids = sort_ids[sort_ids_back]
        return notes, sort_ids
    return notes


def remove_short_notes(notes: List[Note], time_division: int, max_beat_res: int = 32):
    r"""Find and remove short notes, i.e. with the same pitch, start and end.

    :param notes: notes to analyse
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param max_beat_res: maximum beat resolution for one sample
    """
    ticks_per_sample = int(time_division / max_beat_res)

    for i in range(len(notes) - 1, 0, -1):
        note = notes[i]
        if note.end - note.start < ticks_per_sample // 2:
            del notes[i]

    return notes


def filter_late_midi_events(midi: MidiFile, max_tick: Optional[int] = None, sort: bool = False):
    max_tick = max_tick or midi.max_tick

    for track in midi.instruments:
        if sort:
            track.control_changes.sort(key=lambda c: c.time)
        for i, control_change in enumerate(track.control_changes):
            if control_change.time > max_tick:
                track.control_changes = track.control_changes[:i]
                break

        if sort:
            track.pedals.sort(key=lambda p: p.start)
        for i, pedal in enumerate(track.pedals):
            if pedal.end > max_tick:
                track.pedals = track.pedals[:i]
                break

        if sort:
            track.pitch_bends.sort(key=lambda p: p.time)
        for i, pitch_bend in enumerate(track.pitch_bends):
            if pitch_bend.time > max_tick:
                track.pitch_bends = track.pitch_bends[:i]
                break

    return midi


def shift_midi_notes(
        midi: MidiFile,
        time_shift: float = 0.,
        offset: float = 0.,
        inplace: bool = True,
        return_shifted_indices: bool = False
):
    midi = midi if inplace else copy.deepcopy(midi)

    midi.max_tick *= 4
    ttt = midi.get_tick_to_time_mapping()

    def process_continuous_events(elements):
        start_ticks = np.array(list(map(lambda x: x.start, elements)))
        end_ticks = np.array(list(map(lambda x: x.end, elements)))
        start_times, end_times = ttt[start_ticks], ttt[end_ticks]
        new_start_ticks = find_closest(ttt, start_times + time_shift)
        new_end_ticks = find_closest(ttt, end_times + time_shift)
        for el, time, start_t, end_t in zip(elements, start_times, new_start_ticks, new_end_ticks):
            if time >= offset:
                if start_t == end_t:
                    end_t += 1
                el.start = start_t
                el.end = end_t
        return np.where(start_times >= offset)[0]

    def process_instant_events(elements):
        ticks = np.array(list(map(lambda x: x.time, elements)))
        times = ttt[ticks]
        new_ticks = find_closest(ttt, times + time_shift)
        for el, time, tick in zip(elements, times, new_ticks):
            if time >= offset:
                el.time = tick
        return np.where(times >= offset)[0]

    # shift relevant notes in MIDI
    shifted_indices = defaultdict(list)
    for track_idx, track in enumerate(midi.instruments):
        shifted_indices['note'].append((track_idx, process_continuous_events(track.notes)))
        if track.pedals:
            shifted_indices['pedal'].append((track_idx, process_continuous_events(track.pedals)))
        if track.control_changes:
            shifted_indices['control_change'].append((track_idx, process_instant_events(track.control_changes)))
        if track.pitch_bends:
            shifted_indices['pitch_bend'].append((track_idx, process_instant_events(track.pitch_bends)))

    midi.max_tick = max([max([note.end for note in track.notes]) for track in midi.instruments]) + 1

    if return_shifted_indices:
        return midi, shifted_indices
    return midi


def resample_midi(midi: MidiFile, ticks_per_beat: int, inplace: bool = True):
    if midi.ticks_per_beat == ticks_per_beat:
        return midi

    midi = midi if inplace else copy.deepcopy(midi)

    scale = ticks_per_beat / midi.ticks_per_beat

    def process_continuous_events(elements):
        for el in elements:
            el.start = int(scale * el.start)
            el.end = int(scale * el.end)

    def process_instant_events(elements):
        for el in elements:
            el.time = int(scale * el.time)

    # resample MIDI events
    for track in midi.instruments:
        process_continuous_events(track.notes)
        if track.pedals:
            process_continuous_events(track.pedals)
        if track.control_changes:
            process_instant_events(track.control_changes)
        if track.pitch_bends:
            process_instant_events(track.pitch_bends)

    process_instant_events(midi.time_signature_changes)
    process_instant_events(midi.tempo_changes)
    process_instant_events(midi.key_signature_changes)

    midi.max_tick = max([max([note.end for note in track.notes]) for track in midi.instruments]) + 1
    return midi
