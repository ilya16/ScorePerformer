""" Music constants. """

# notes and pitch-sitch maps
NOTES_WSHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_WFLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
NOTE_MAP = {sitch: i for i, sitch in enumerate(NOTES_WSHARP)}
NOTE_INV_MAP = {i: sitch for sitch, i in NOTE_MAP.items()}
NOTE_MAP.update(**{sitch: i for i, sitch in enumerate(NOTES_WFLAT)})


def pitch2sitch(pitch):
    return NOTE_INV_MAP[pitch % 12] + str(pitch // 12 - 1)


def sitch2pitch(sitch):
    note = sitch[:1 + int(sitch[1] in ("#", "b"))]
    octave = sitch[len(note):]
    return NOTE_MAP[note] + 12 * (int(octave) + 1)
