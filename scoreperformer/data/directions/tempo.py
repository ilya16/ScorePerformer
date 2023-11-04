TEMPO_PREFIX = 'tempo'

ABS_TEMPO_KEYS = [
    'grave', 'largo', 'larghetto', 'lento',
    'adagio', 'andante', 'andantino', 'moderato',
    'allegretto', 'allegro', 'vivace',
    'presto', 'prestissimo'
]

REL_TEMPO_KEYS = [
    ('accelerando', 'acc', 'accel'),
    ('ritardando', 'rit', 'ritard'),
    ('rallentando', 'rall'),
    ('stringendo', 'string'),
    'calando', 'più mosso', 'animato', 'stretto', 'smorzando', 'ritenuto'
]

RET_TEMPO_KEYS = [
    ('tempo primo', 'tempo i'),
    'a tempo',
]

TEMPO_KEYS = ABS_TEMPO_KEYS + REL_TEMPO_KEYS + RET_TEMPO_KEYS
