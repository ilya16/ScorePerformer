import copy

import IPython.display as ipd
import note_seq
from miditoolkit import MidiFile
from note_seq import midi_file_to_note_sequence


def cut_midi(
        midi: MidiFile,
        min_tick: int = 0,
        max_tick: int = 1e9,
        cut_end_tick: bool = True,
        save_path: str = '/tmp/tmp.mid'
):
    midi = copy.deepcopy(midi)

    for track in midi.instruments:
        track.notes = [n for n in track.notes if min_tick <= n.start <= max_tick]
        for n in track.notes:
            n.start -= min_tick
            if cut_end_tick:
                n.end = min(n.end, max_tick)
            n.end -= min_tick

        if hasattr(track, "control_changes"):
            track.control_changes = [c for c in track.control_changes if min_tick <= c.time <= max_tick]
            for c in track.control_changes:
                c.time -= min_tick
        if hasattr(track, "pedals"):
            track.pedals = [p for p in track.pedals if min_tick <= p.start <= max_tick]
            for p in track.pedals:
                p.start -= min_tick
                p.end -= min_tick

    midi.tempo_changes = [t for t in midi.tempo_changes if min_tick <= t.time <= max_tick]
    for t in midi.tempo_changes:
        t.time -= min_tick

    midi.max_tick = max([n.end for n in midi.instruments[0].notes])
    midi.max_tick = max(midi.max_tick, midi.tempo_changes[-1].time + 1)

    if save_path is not None:
        midi.dump(save_path)

    return midi


def midi_to_audio(
        path: str = '/tmp/tmp.mid',
        sample_rate: int = 22050,
        play: bool = True
):
    ns = midi_file_to_note_sequence(path)
    audio = note_seq.fluidsynth(ns, sample_rate=sample_rate)
    if play:
        ipd.display(ipd.Audio(audio, rate=sample_rate))
    return audio
