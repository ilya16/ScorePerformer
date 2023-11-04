""" Embedding Classifier Models"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from torch import Tensor

from scoreperformer.models.base import Model
from scoreperformer.modules.constructor import Registry, VariableModuleConfig


@dataclass
class EmbeddingClassifierOutput:
    logits: Tensor = None
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


EmbeddingClassifiersRegistry = type("_EmbeddingClassifiersRegistry", (Registry,), {})()


@dataclass
class EmbeddingClassifierConfig(VariableModuleConfig):
    input_dim: int = MISSING
    num_classes: int = MISSING
    dropout: bool = 0.
    weight: Optional[List[float]] = None


@dataclass
class LinearEmbeddingClassifierConfig(EmbeddingClassifierConfig):
    _target_: str = "linear"
    hidden_dims: Optional[Sequence[int]] = field(default_factory=lambda: (32,))


@EmbeddingClassifiersRegistry.register("linear")
class LinearEmbeddingClassifier(Model):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dims: Optional[Sequence[int]] = (32,),
            dropout: bool = 0.,
            class_weights: Optional[List[float]] = None
    ):
        super().__init__()

        self.num_classes = num_classes

        class_weights = torch.ones(num_classes) if class_weights is None else torch.tensor(class_weights)
        self.register_buffer("class_weights", class_weights.float())

        hidden_dims = hidden_dims or []
        hidden_dims = [hidden_dims] if isinstance(hidden_dims, int) else hidden_dims
        hidden_dims = list(hidden_dims)

        in_dims = [input_dim] + hidden_dims
        out_dims = hidden_dims + [num_classes]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < len(in_dims) - 1:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, embeddings: Tensor, labels: Optional[Tensor] = None):
        x = embeddings.squeeze(1) if embeddings.ndim == 3 else embeddings
        for layer in self.layers:
            x = layer(self.dropout(x))
        logits = x

        loss = losses = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        return EmbeddingClassifierOutput(
            logits=logits,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(self, inputs) -> Dict[str, Tensor]:
        return inputs


@dataclass
class SequentialEmbeddingClassifierConfig(EmbeddingClassifierConfig):
    _target_: str = "sequential"
    hidden_dim: int = 32


@EmbeddingClassifiersRegistry.register("sequential")
class SequentialEmbeddingClassifier(Model):
    def __init__(
            self,
            input_dim: int,
            num_classes: int,
            hidden_dim: int = 32,
            dropout: bool = 0.,
            class_weights: Optional[List[float]] = None
    ):
        super().__init__()

        self.num_classes = num_classes

        class_weights = torch.ones(num_classes) if class_weights is None else torch.tensor(class_weights)
        self.register_buffer("class_weights", class_weights.float())

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.output = nn.Linear(hidden_dim, num_classes)

    def forward(self, embeddings: Tensor, labels: Optional[Tensor] = None):
        self.gru.flatten_parameters()
        _, out = self.gru(embeddings)  # (1, b, h)
        logits = self.output(out[0])

        loss = losses = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)

        return EmbeddingClassifierOutput(
            logits=logits,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(self, inputs) -> Dict[str, Tensor]:
        return inputs


@dataclass
class MultiHeadEmbeddingClassifierOutput:
    logits: Dict[str, Tensor] = None
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


@dataclass
class MultiHeadEmbeddingClassifierConfig(VariableModuleConfig):
    _target_: str = "multi-head"
    input_dim: int = MISSING
    num_classes: Dict[str, int] = MISSING
    classifier: LinearEmbeddingClassifierConfig = MISSING
    class_samples: Optional[Dict[str, List[int]]] = None
    weighted_classes: bool = False
    loss_weight: float = 1.
    detach_inputs: Union[bool, float] = False


@EmbeddingClassifiersRegistry.register("multi-head")
class MultiHeadEmbeddingClassifier(Model):
    def __init__(
            self,
            input_dim: int,
            num_classes: Dict[str, int],
            classifier: LinearEmbeddingClassifierConfig,
            class_samples: Optional[Dict[str, List[int]]] = None,
            loss_weight: float = 1.,
            weighted_classes: bool = False,
            detach_inputs: Union[bool, float] = False
    ):
        super().__init__()

        self.num_classes = num_classes

        self.heads = nn.ModuleDict({})
        for key, num in num_classes.items():
            num_samples = class_samples.get(key, None) if class_samples is not None else None
            class_weights = self._class_weights(num_samples) if weighted_classes and num_samples is not None else None
            self.heads[key] = LinearEmbeddingClassifier.init(
                config=classifier,
                input_dim=input_dim,
                num_classes=num,
                class_weights=class_weights
            )

        self.loss_weight = loss_weight
        self.detach_inputs = float(detach_inputs)

    @staticmethod
    def _class_weights(num_samples: List[int], beta: float = 0.999, mult: int = 1e4):
        num_samples = np.maximum(num_samples, 1e-6)
        effective_num = 1.0 - np.power(beta, np.array(num_samples) * mult)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(num_samples)
        return weights.tolist()

    def forward(self, embeddings: Tensor, labels: Optional[Tensor] = None):
        embeddings = self.detach_inputs * embeddings.detach() + (1 - self.detach_inputs) * embeddings

        logits = {}
        loss, losses = 0., {}
        for i, (key, head) in enumerate(self.heads.items()):
            out = head(embeddings, labels=labels[..., i] if labels is not None else None)
            logits[key] = out.logits

            if out.loss:
                key = 'clf/' + key
                loss += out.loss
                losses[key] = out.loss

        loss = self.loss_weight * loss / len(self.heads)
        losses['clf'] = loss

        return MultiHeadEmbeddingClassifierOutput(
            logits=logits,
            loss=loss if labels is not None else None,
            losses=losses if labels is not None else None
        )

    def prepare_inputs(self, inputs) -> Dict[str, Tensor]:
        return inputs
