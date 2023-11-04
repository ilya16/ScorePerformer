from .indexers import TupleTokenSequenceIndexer
from .processors import TokenSequenceAugmentations, TupleTokenSequenceProcessor
from ..tokenizers import TokenizerTypes

TOKEN_SEQUENCE_PROCESSORS = {
    TokenizerTypes.OctupleM: TupleTokenSequenceProcessor,
    TokenizerTypes.SPMuple: TupleTokenSequenceProcessor
}

TOKEN_SEQUENCE_INDEXERS = {
    TokenizerTypes.OctupleM: TupleTokenSequenceIndexer,
    TokenizerTypes.SPMuple: TupleTokenSequenceIndexer
}
