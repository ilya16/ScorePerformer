""" Performance direction embeddings collators. """

import torch


class DirectionEmbeddingCollator:
    def __init__(
            self,
            num_embeddings: int = 1,
            embedding_dim: int = 64,
    ):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def init_data(self, batch):
        embeddings = torch.zeros(len(batch), self.num_embeddings, self.embedding_dim)
        labels = torch.zeros(len(batch), dtype=torch.long)
        return embeddings, labels

    def process_sample(self, i, sample, data):
        embeddings, labels = data
        _, emb, label = sample

        emb = emb.unsqueeze(0) if emb.ndim == 1 else emb
        embeddings[i, -emb.shape[0]:] = emb
        labels[i] = label

    def __call__(self, batch, inference=False, return_tensors=True):
        data = self.init_data(batch)
        for i, sample in enumerate(batch):
            self.process_sample(i, sample, data)

        return {'embeddings': data[0], 'labels': data[1]}
