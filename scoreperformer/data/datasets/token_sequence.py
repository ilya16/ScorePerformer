""" Token sequence datasets. """

import os
from pathlib import Path, PurePath

from torch.utils.data import Dataset

from scoreperformer.utils import apply, load_json


def load_token_sequence(path, load_fn, processing_funcs=None):
    seq = load_fn(path)
    if processing_funcs:
        for func in processing_funcs:
            seq = func(seq)
    return seq


class TokenSequenceDataset(Dataset):
    def __init__(self, sequences, names=None):
        self.seqs = sequences

        self.names = names
        if names is not None:
            self._name_to_idx = {name: idx for idx, name in enumerate(self.names)}

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        return seq[0] if isinstance(seq, tuple) else seq

    def __len__(self):
        return len(self.seqs)


class LocalTokenSequenceDataset(TokenSequenceDataset):
    def __init__(self, root, files=None, suffix='.json', load_fn=load_json, preload=False, cache=False):
        self.root = root
        self.load_fn = load_fn

        if files is None:
            if os.path.isfile(root) and root.lower().endswith(suffix):
                files = [Path(root)]
            else:
                files = list(Path(root).glob('**/*' + suffix))
            files = list(map(Path, files))
        else:
            files = list(map(lambda x: Path(x).with_suffix(suffix), files))

        paths = [PurePath(os.path.join(self.root, file)) for file in files]

        self.paths = paths

        self._cache = cache

        self.seqs = self.load_sequences(preload=preload)
        names = [str(file).replace(suffix, '') for file in files]

        super().__init__(sequences=self.seqs, names=names)

    def load_sequence(self, path):
        return self.load_fn(path)

    def load_sequences(self, preload):
        if preload:
            return apply(self.paths, func=self.load_sequence, desc='Loading token sequences...')
        else:
            return [None] * len(self.paths)

    def __getitem__(self, idx):
        if self.seqs[idx] is None:
            seq = self.load_sequence(self.paths[idx])
            if self._cache:
                self.seqs[idx] = seq
        else:
            seq = self.seqs[idx]
        return seq[0] if isinstance(seq, tuple) else seq

    def __len__(self):
        return len(self.seqs)
