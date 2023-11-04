""" Embedding Classifier evaluator. """

import torch


class EmbeddingClassifierEvaluator:
    def __init__(self, model):
        self.model = model

    def _accuracy(self, predictions, labels):
        return (predictions == labels).float().mean()

    @torch.no_grad()
    def __call__(self, inputs, outputs):
        labels = inputs["labels"]
        predictions = torch.argmax(outputs.logits, dim=-1)
        metrics = {"accuracy": self._accuracy(predictions, labels)}

        return metrics
