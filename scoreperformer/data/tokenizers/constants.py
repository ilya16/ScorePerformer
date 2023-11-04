""" Tokenizer related constants. """

# override MidiTok default special tokens for backward compatibility
SPECIAL_TOKENS = ["PAD", "MASK", "SOS", "EOS"]

PAD_TOKEN = "PAD_None"
MASK_TOKEN = "MASK_None"
SOS_TOKEN = "SOS_None"
EOS_TOKEN = "EOS_None"

TIME_DIVISION = 480

SCORE_KEYS = [
    "Bar",
    "Position",
    "Pitch",
    "Velocity",
    "Duration",
    "Tempo",
    "TimeSig",
    "Program",
    "PositionShift",
    "NotesInOnset",
    "PositionInOnset"
]
PERFORMANCE_KEYS = SCORE_KEYS + [
    "OnsetDev",
    "PerfDuration",
    "RelOnsetDev",
    "RelPerfDuration"
]
