from .articulation import ARTICULATION_PREFIX, ARTICULATION_KEYS
from .dynamic import DYNAMIC_PREFIX, DYNAMIC_KEYS, ABS_DYNAMIC_KEYS, REL_DYNAMIC_KEYS
from .parser import parse_directions
from .tempo import TEMPO_PREFIX, TEMPO_KEYS, ABS_TEMPO_KEYS, REL_TEMPO_KEYS, RET_TEMPO_KEYS
from .words import extract_main_keyword


def build_prefixed_keys(keys, prefix):
    return list(map(lambda d: f'{prefix}/' + extract_main_keyword(d), keys))


DYNAMIC_DIRECTION_KEYS = build_prefixed_keys(DYNAMIC_KEYS, DYNAMIC_PREFIX)
TEMPO_DIRECTION_KEYS = build_prefixed_keys(TEMPO_KEYS, TEMPO_PREFIX)
ARTICULATION_DIRECTION_KEYS = build_prefixed_keys(ARTICULATION_KEYS, ARTICULATION_PREFIX)
