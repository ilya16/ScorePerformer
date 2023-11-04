from scoreperformer.utils import ExplicitEnum
from .classes import TokSequence, TokenizerConfig
from .common import OctupleM
from .spmuple import (
    SPMupleBase,
    SPMuple,
    SPMuple2,

    SPMupleOnset,
    SPMupleBeat,
    SPMupleBar,
    SPMupleWindow,
    SPMupleWindowRecompute
)


class TokenizerTypes(ExplicitEnum):
    OctupleM = "OctupleM"
    SPMuple = "SPMuple"
    SPMuple2 = "SPMuple2"
    SPMupleOnset = "SPMupleOnset"
    SPMupleBeat = "SPMupleBeat"
    SPMupleBar = "SPMupleBar"
    SPMupleWindow = "SPMupleWindow"
    SPMupleWindowRecompute = "SPMupleWindowRecompute"


TOKENIZERS = {
    TokenizerTypes.OctupleM: OctupleM,
    TokenizerTypes.SPMuple: SPMuple,
    TokenizerTypes.SPMuple2: SPMuple2,
    TokenizerTypes.SPMupleOnset: SPMupleOnset,
    TokenizerTypes.SPMupleBeat: SPMupleBeat,
    TokenizerTypes.SPMupleBar: SPMupleBar,
    TokenizerTypes.SPMupleWindow: SPMupleWindow,
    TokenizerTypes.SPMupleWindowRecompute: SPMupleWindowRecompute
}
