""" Extended miditok classes. """

from dataclasses import dataclass
from typing import Optional, Dict, Sequence

from miditok.classes import (
    TokSequence as MidiTokTokSequence,
    TokenizerConfig as MidiTokTokenizerConfig
)

from .constants import SPECIAL_TOKENS


@dataclass
class TokSequence(MidiTokTokSequence):
    meta: Optional[Dict[str, object]] = None


class TokenizerConfig(MidiTokTokenizerConfig):
    r"""
    MIDI tokenizer base class, containing common methods and attributes for all tokenizers.
    :param special_tokens: list of special tokens. This must be given as a list of strings given
            only the names of the tokens. (default: ``["PAD", "SOS", "EOS", "MASK"]``\)
    :param **kwargs: additional parameters that will be saved in `config.additional_params`.
    """

    def __init__(
            self,
            special_tokens: Sequence[str] = SPECIAL_TOKENS,
            **kwargs
    ):
        super().__init__(special_tokens=special_tokens, **kwargs)
