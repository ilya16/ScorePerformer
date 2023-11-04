from typing import List, Optional, Tuple

from miditoolkit import Note, TempoChange, TimeSignature, KeySignature


def quantize_notes(
        notes: List[Note],
        time_division: int,
        max_beat_res: int = 32,
        pitch_range: Optional[Tuple[int, int]] = (21, 109)
):
    """ Quantize notes, i.e. their pitch, start and end values.
    Shifts the notes' start and end times to match the quantization (e.g. 16 samples per bar)
    Notes with pitches outside of pitch_range are deleted.
    :param notes: notes to quantize
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param max_beat_res: maximum beat resolution for one sample
    :param pitch_range: pitch range from within notes should be
    """
    ticks_per_sample = int(time_division / max_beat_res)
    i = 0
    while i < len(notes):
        note = notes[i]
        if pitch_range is not None and (note.pitch < pitch_range[0] or note.pitch >= pitch_range[1]):
            del notes[i]
            continue
        start_offset = note.start % ticks_per_sample
        end_offset = note.end % ticks_per_sample
        note.start += -start_offset if start_offset <= ticks_per_sample / 2 else ticks_per_sample - start_offset
        note.end += -end_offset if end_offset <= ticks_per_sample / 2 else ticks_per_sample - end_offset

        if note.start == note.end:
            note.end += ticks_per_sample

        i += 1


def quantize_tempos(
        tempos: List[TempoChange],
        time_division: int,
        max_beat_res: int = 32
):
    r"""Quantize the times of tempo change events.
    Consecutive identical tempo changes will be removed.

    :param tempos: tempo changes to quantize
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param max_beat_res: maximum beat resolution for one sample
    """
    ticks_per_sample = int(time_division / max_beat_res)
    i, prev_tempo = 0, -1
    while i < len(tempos):
        # Quantize tempo value
        if tempos[i].tempo == prev_tempo:
            del tempos[i]
            continue
        rest = tempos[i].time % ticks_per_sample
        tempos[i].time += -rest if rest <= ticks_per_sample / 2 else ticks_per_sample - rest
        prev_tempo = tempos[i].tempo
        i += 1


def compute_ticks_per_bar(time_sig: TimeSignature, time_division: int):
    r"""Computes time resolution of one bar in ticks.

    :param time_sig: time signature object
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :return: MIDI bar resolution, in ticks/bar
    """
    return int(time_division * 4 * time_sig.numerator / time_sig.denominator)


def quantize_time_signatures(time_sigs: List[TimeSignature], time_division: int):
    r"""Quantize the time signature changes, delayed to the next bar.
    See MIDI 1.0 Detailed specifications, pages 54 - 56, for more information on
    delayed time signature messages.

    :param time_sigs: time signature changes to quantize
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    """
    all_different = False
    while not all_different:
        all_different = True

        # delete one of neighbouring time signatures with same values or time
        prev_time_sig = time_sigs[0]
        i = 1
        while i < len(time_sigs):
            time_sig = time_sigs[i]

            if (time_sig.numerator, time_sig.denominator) == (prev_time_sig.numerator, prev_time_sig.denominator) or \
                    time_sig.time == prev_time_sig.time:
                del time_sigs[i]
                all_different = False
                continue
            prev_time_sig = time_sig
            i += 1

        # quantize times
        ticks_per_bar = compute_ticks_per_bar(time_sigs[0], time_division)
        current_bar = 0
        previous_tick = 0  # first time signature change is always at tick 0
        i = 1
        while i < len(time_sigs):
            time_sig = time_sigs[i]

            # determine the current bar of time sig
            bar_offset, rest = divmod(time_sig.time - previous_tick, ticks_per_bar)
            if rest > 0:  # time sig doesn't happen on a new bar, we update it to the next bar
                bar_offset += 1
                time_sig.time = previous_tick + bar_offset * ticks_per_bar

            # Update values
            ticks_per_bar = compute_ticks_per_bar(time_sig, time_division)
            current_bar += bar_offset
            previous_tick = time_sig.time
            i += 1


def quantize_key_signatures(
        key_signatures: List[KeySignature],
        time_division: int,
        max_beat_res: int = 32
):
    r"""Quantize the times of key signature change events.
    Consecutive identical key signature changes will be removed.

    :param key_signatures: key signature changes to quantize
    :param time_division: MIDI time division / resolution, in ticks/beat (of the MIDI being parsed)
    :param max_beat_res: maximum beat resolution for one sample
    """
    ticks_per_sample = int(time_division / max_beat_res)
    i, prev_tempo = 0, ''
    while i < len(key_signatures):
        # Quantize tempo value
        if key_signatures[i].key_name == prev_tempo:
            del key_signatures[i]
            continue
        rest = key_signatures[i].time % ticks_per_sample
        key_signatures[i].time += -rest if rest <= ticks_per_sample / 2 else ticks_per_sample - rest
        prev_tempo = key_signatures[i].key_name
        i += 1
