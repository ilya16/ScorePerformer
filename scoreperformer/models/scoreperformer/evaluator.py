""" ScorePerformer metric evaluator. """

from typing import Optional, Union, List

import torch
import torch.nn.functional as F

from scoreperformer.data.collators import LMScorePerformanceInputs
from scoreperformer.data.tokenizers import OctupleM
from .model import ScorePerformerOutputs
from .transformer import TupleTransformerOutput
from .wrappers import ScorePerformerLMModes


class ScorePerformerEvaluator:
    def __init__(
            self,
            model,
            tokenizer: Optional[OctupleM] = None,
            label_pad_token_id: int = -100,
            weighted_distance: bool = False,
            ignore_keys: Optional[List[str]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id
        self.weighted_distance = weighted_distance
        self.ignore_keys = ignore_keys

        self.token_values = None
        if self.tokenizer is not None:
            self.token_values = {
                key: torch.from_numpy(values)[:, None]
                for key, values in self.tokenizer.token_values(normalize=False).items()
            }

    def _accuracy(self, predictions, labels):
        label_mask = labels != self.label_pad_token_id
        return (predictions[label_mask] == labels[label_mask]).float().mean()

    def _distance(self, predictions, targets):
        return (predictions - targets).abs().float().mean()

    def _weighted_distance(self, probs, targets, token_values):
        return ((targets[:, None] - token_values[None, :]).abs() * probs[..., None]).sum(dim=1).mean()

    @torch.no_grad()
    def __call__(
            self,
            inputs: Union[dict, LMScorePerformanceInputs],
            outputs: Union[TupleTransformerOutput, ScorePerformerOutputs],
            ignore_keys: Optional[List[str]] = None
    ):
        metrics = {}
        ignore_keys = ignore_keys or self.ignore_keys

        if isinstance(inputs, LMScorePerformanceInputs):
            labels = inputs.labels.tokens.to(outputs.hidden_state.device)
        else:
            labels = inputs["labels"]

        if self.model.mode in (ScorePerformerLMModes.CLM, ScorePerformerLMModes.MixedLM):
            labels = labels[:, 1:]

        if isinstance(outputs, ScorePerformerOutputs):
            outputs = outputs.perf_decoder

        predictions = torch.cat(
            list(map(lambda l: torch.argmax(l, dim=-1, keepdim=True), outputs.logits.values())),
            dim=-1
        )

        metrics[f"accuracy"] = self._accuracy(predictions, labels)
        if ignore_keys:
            use_ids = torch.tensor(
                [i for i, key in enumerate(outputs.logits.keys()) if key not in ignore_keys],
                device=predictions.device, dtype=torch.long
            )
            metrics[f"accuracy/pred"] = self._accuracy(predictions[..., use_ids], labels[..., use_ids])

        for i, (key, logits) in enumerate(outputs.logits.items()):
            if ignore_keys and key in ignore_keys:
                continue

            if torch.any(labels[..., i] != self.label_pad_token_id):
                metrics[f"accuracy/{key}"] = self._accuracy(predictions[..., i], labels[..., i])

        if self.token_values is not None:
            for i, (key, logits) in enumerate(outputs.logits.items()):
                if ignore_keys and key in ignore_keys:
                    continue

                self.token_values[key] = self.token_values[key].to(predictions.device)

                label_mask = labels[..., i] != self.label_pad_token_id
                if torch.any(label_mask):
                    preds = F.embedding(predictions[..., i][label_mask], self.token_values[key])
                    targets = F.embedding(labels[..., i][label_mask], self.token_values[key])

                    if self.weighted_distance:
                        probs = outputs.logits[key].softmax(dim=-1)[label_mask]
                        metrics[f"distance/{key}"] = self._weighted_distance(probs, targets, self.token_values[key])
                    else:
                        metrics[f"distance/{key}"] = self._distance(preds, targets)

        return metrics
