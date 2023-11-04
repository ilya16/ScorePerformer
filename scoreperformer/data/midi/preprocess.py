from typing import Optional, List

from miditok.utils import merge_tracks
from miditoolkit import MidiFile, Marker, Instrument

from ..midi import quantization as midi_quan
from ..midi import utils as midi_utl
from ..midi.containers import Note


def preprocess_midi(
        midi: MidiFile,
        to_single_track: bool = True,
        sort_events: bool = True,
        clean_duplicates: bool = True,
        cut_overlapped_notes: bool = False,
        clean_short_notes: bool = False,
        quantize_notes: bool = False,
        quantize_midi_changes: bool = False,
        filter_late_events: bool = True,
        target_ticks_per_beat: Optional[int] = None
):
    if len(midi.instruments) == 0:
        return midi

    if len(midi.instruments) > 1 and to_single_track:
        merge_tracks(midi.instruments, effects=True)

    # TODO: order of notes changes inside, handle the case `sort_events = False`
    for track in midi.instruments:
        if clean_duplicates:
            midi_utl.remove_duplicated_notes(track.notes)

        if cut_overlapped_notes:
            midi_utl.cut_overlapping_notes(track.notes)

        if clean_short_notes:
            midi_utl.remove_short_notes(track.notes, time_division=midi.ticks_per_beat)

        if quantize_notes:
            midi_quan.quantize_notes(track.notes, time_division=midi.ticks_per_beat)
            if clean_duplicates:
                midi_utl.remove_duplicated_notes(track.notes)

    if sort_events:
        for track in midi.instruments:
            track.notes.sort(key=lambda x: (x.start, x.pitch, x.end))  # sort notes
        midi.max_tick = max([max([note.end for note in track.notes[-100:]]) for track in midi.instruments])
    else:
        midi.max_tick = max([max([note.end for note in track.notes]) for track in midi.instruments]) + 1

    midi.instruments = [track for track in midi.instruments if len(track.notes) > 0]

    if filter_late_events:
        midi_utl.filter_late_midi_events(midi, sort=sort_events)

    if quantize_midi_changes:
        midi_quan.quantize_time_signatures(midi.time_signature_changes, time_division=midi.ticks_per_beat)
        midi_quan.quantize_tempos(midi.tempo_changes, time_division=midi.ticks_per_beat)
        midi_quan.quantize_key_signatures(midi.key_signature_changes, time_division=midi.ticks_per_beat)

    if target_ticks_per_beat is not None:
        midi_utl.resample_midi(midi, ticks_per_beat=target_ticks_per_beat)

    return midi


def insert_silent_notes(
        midi: MidiFile,
        markers: Optional[List[Marker]] = None,
        track_idx: Optional[int] = None
):
    markers = markers or midi.markers

    notes = []
    for m in markers:
        if m.text.startswith('NoteS'):
            pitch, start_tick, end_tick = map(int, m.text.split('_')[1:])
            notes.append(Note(pitch, 0, start_tick, end_tick))

    if track_idx is None:
        track = Instrument(0, False, 'Unperformed Notes')
        track.notes = notes
        midi.instruments.append(track)
    else:
        midi.instruments[track_idx].notes += notes

    if midi.instruments[-1].name != 'Unperformed Notes':
        midi.instruments.append(Instrument(0, False, 'Unperformed Notes'))

    return midi
