DYNAMIC_PREFIX = 'dynamic'

ABS_DYNAMIC_KEYS = [
    'pppp', 'ppp', 'pp',
    ('p', 'piano'),
    'mp', 'mf',
    ('f', 'forte'),
    'ff', 'fff', 'ffff',
    'fp', 'ffp'
]

REL_DYNAMIC_KEYS = [
    ('crescendo', 'cresc'),
    ('diminuendo', 'dim', 'decresc'),
    ('sf', 'fz', 'sfz', 'sffz'),
    ('rf', 'rfz')
]

DYNAMIC_KEYS = ABS_DYNAMIC_KEYS + REL_DYNAMIC_KEYS


def hairpin_word_regularization(word):
    if 'decresc' in word:
        word = 'diminuendo'
    elif 'cresc' in word:
        word = 'crescendo'
    elif 'dim' in word:
        word = 'diminuendo'
    return word
