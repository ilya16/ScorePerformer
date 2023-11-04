""" ScorePerformer token embeddings and language modeling heads. """

from dataclasses import dataclass
from typing import Union, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
from omegaconf import MISSING
from torch import Tensor

from scoreperformer.modules.constructor import Constructor, Registry, VariableModuleConfig
from scoreperformer.modules.transformer import DiscreteContinuousEmbedding, DiscreteDenseContinuousEmbedding

_emb_dims = {
    'Bar': 128,
    'Position': 128,
    'Pitch': 128,
    'Velocity': 64,
    'Duration': 64,
    'Tempo': 64,
    'TimeSig': 16,
    'Program': 64,
    'OnsetDev': 64,
    'PerfDuration': 64
}

TupleTokenEmbeddingsRegistry = type("_TupleTokenEmbeddingsRegistry", (Registry,), {})()


@dataclass
class TupleTokenEmbeddingsConfig(VariableModuleConfig):
    _target_: str = "simple"
    num_tokens: Dict[str, int] = MISSING
    emb_dims: Union[Dict[str, int], int] = MISSING
    mode: str = "cat"
    project_emb_dim: int = 512,
    emb_norm: bool = False
    discrete: bool = True
    continuous: Union[bool, List[str]] = False
    continuous_dense: bool = False
    token_values: Optional[Dict[str, list]] = None
    discrete_ids: Optional[List[int]] = None
    tie_keys: Optional[Dict[str, str]] = None


@TupleTokenEmbeddingsRegistry.register("simple")
class TupleTokenEmbeddings(nn.Module, Constructor):
    def __init__(
            self,
            num_tokens: Dict[str, int],
            emb_dims: Union[Dict[str, int], int],
            mode: str = "cat",
            project_emb_dim: int = 512,
            emb_norm: bool = False,
            discrete: bool = True,
            continuous: Union[bool, List[str]] = False,
            continuous_dense: bool = False,
            token_values: Optional[Dict[str, list]] = None,
            discrete_ids: Optional[List[int]] = None,
            tie_keys: Optional[Dict[str, str]] = None
    ):
        super().__init__()

        self.mode = mode

        if self.mode == "sum":
            assert isinstance(emb_dims, int) \
                   or all([emb_dim == list(emb_dims.values())[0] for emb_dim in emb_dims.values()]), \
                "`emb_dims` in TupleTokenEmbeddings' `sum` mode should be the same for all keys."

        continuous_keys = continuous
        if isinstance(continuous, bool):
            continuous_keys = [key for key in num_tokens] if continuous else []
        else:
            continuous = len(continuous) > 0

        total_emb_dim = 0
        embeddings = {}
        token_values = token_values or {}
        for key, num in num_tokens.items():
            emb_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[key]
            if tie_keys and key in tie_keys:
                embeddings[key] = embeddings[tie_keys[key]]
                emb_dim = emb_dims if isinstance(emb_dims, int) else emb_dims[tie_keys[key]]
            elif key in continuous_keys:
                cls = DiscreteDenseContinuousEmbedding if continuous_dense else DiscreteContinuousEmbedding
                embeddings[key] = cls(
                    num_embeddings=num,
                    embedding_dim=emb_dim,
                    discrete=discrete,
                    continuous=True,
                    discrete_ids=discrete_ids,
                    token_values=token_values.get(key, None),
                    padding_idx=0
                )
            else:
                embeddings[key] = nn.Embedding(num, emb_dim, padding_idx=0)
            total_emb_dim += emb_dim if self.mode == "cat" else emb_dim - total_emb_dim

        self.embs = nn.ModuleDict(embeddings)
        self.norm = nn.LayerNorm(total_emb_dim) if emb_norm else nn.Identity()

        if total_emb_dim != project_emb_dim:
            self.project_emb = nn.Linear(total_emb_dim, project_emb_dim)

        self.num_tokens = num_tokens
        self.emb_dims = emb_dims
        self.total_emb_dim = total_emb_dim

        self.continuous = continuous
        self.continuous_keys = continuous_keys
        self.token_values = token_values
        self.init_()

    def init_(self):
        if not self.continuous:
            for key, emb in self.embs.items():
                weight_attr = "index_weight" if key in self.continuous_keys else "weight"
                nn.init.kaiming_normal_(getattr(emb, weight_attr))

    def _forward_embeddings(
            self,
            x: Tensor,
            values: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        token_embs = {
            key: emb(x[..., i], values=values[..., i] if values is not None else values)
            if key in self.continuous_keys else emb(x[..., i])
            for i, (key, emb) in enumerate(self.embs.items())
        }

        return token_embs

    def _forward_project(
            self,
            token_embs: Dict[str, Tensor]
    ) -> Tensor:
        if self.mode == "cat":
            token_emb = self.project_emb(self.norm(torch.cat(list(token_embs.values()), dim=-1)))
        else:
            token_emb = self.norm(sum(list(token_embs.values())))

        return token_emb

    def forward(
            self,
            x: Tensor,
            values: Optional[Tensor] = None,
            cache: Optional[Tensor] = None,
            return_embeddings: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        if cache is not None:
            x = x[:, cache.shape[1]:]
            values = values[:, cache.shape[1]:] if values is not None else values

        token_embs = self._forward_embeddings(x, values)
        token_emb = self._forward_project(token_embs)

        if cache is not None:
            token_emb = torch.cat([cache, token_emb], dim=1)

        if return_embeddings:
            return token_emb, token_embs
        else:
            return token_emb


@dataclass
class MultiSeqTupleTokenEmbeddingsConfig(TupleTokenEmbeddingsConfig):
    _target_: str = "multi-seq"
    multiseq_mode: str = "pre-sum"
    num_sequences: int = 2


@TupleTokenEmbeddingsRegistry.register("multi-seq")
class MultiSeqTupleTokenEmbeddings(TupleTokenEmbeddings):
    def __init__(
            self,
            num_tokens: Dict[str, int],
            emb_dims: Union[Dict[str, int], int],
            mode: str = "cat",
            project_emb_dim: int = 512,
            emb_norm: bool = False,
            discrete: bool = True,
            continuous: Union[bool, List[str]] = False,
            continuous_dense: bool = False,
            token_values: Optional[Dict[str, list]] = None,
            discrete_ids: Optional[List[int]] = None,
            tie_keys: Optional[Dict[str, str]] = None,
            multiseq_mode: str = "pre-sum",
            num_sequences: int = 2
    ):
        super().__init__(
            num_tokens=num_tokens,
            emb_dims=emb_dims,
            mode=mode,
            project_emb_dim=project_emb_dim,
            emb_norm=emb_norm,
            discrete=discrete,
            continuous=continuous,
            continuous_dense=continuous_dense,
            token_values=token_values,
            discrete_ids=discrete_ids,
            tie_keys=tie_keys
        )

        self.multiseq_mode = multiseq_mode
        self.num_sequences = num_sequences

        if self.multiseq_mode == "post-cat":
            self.project_multiemb = nn.Linear(num_sequences * project_emb_dim, project_emb_dim)

    def forward(
            self,
            tokens: Union[Tensor, List[Tensor]],
            values: Optional[Union[Tensor, List[Tensor]]] = None,
            cache: Optional[Tensor] = None,
            return_embeddings: bool = False
    ):
        if isinstance(tokens, List) and len(tokens) == 1:
            tokens = tokens[0]
            values = values[0] if isinstance(values, List) else values

        if isinstance(tokens, Tensor):
            return super().forward(tokens, values=values, cache=cache, return_embeddings=return_embeddings)

        if cache is not None:
            tokens = [t[:, cache.shape[1]:] for t in tokens]
            values = [v[:, cache.shape[1]:] for v in values] if values is not None else values

        if self.multiseq_mode == "pre-sum":
            tokens = torch.stack(tokens, dim=0)
            values = torch.stack(values, dim=0) if values is not None else None

            token_embs = {
                key: [e.squeeze(dim=0) for e in token_emb.split(1, dim=0)]
                for key, token_emb in self._forward_embeddings(tokens, values=values).items()
            }

            total_token_embs = {
                key: sum(token_embs[key])
                for key in token_embs
            }
            token_emb = self._forward_project(total_token_embs)
        elif self.multiseq_mode.startswith("post"):
            values = [None] * len(tokens) if values is None else values
            token_embs = [
                self._forward_embeddings(x=t, values=v)
                for t, v in zip(tokens, values)
            ]
            token_embs_proj = [self._forward_project(te) for te in token_embs]

            if self.multiseq_mode == "post-cat":
                assert len(token_embs_proj) == self.num_sequences
                token_emb = self.project_multiemb(torch.cat(token_embs_proj, dim=-1))
            else:
                token_emb = sum(token_embs_proj)
        else:
            return None

        if cache is not None:
            token_emb = torch.cat([cache, token_emb], dim=1)

        if return_embeddings:
            return token_emb, token_embs
        else:
            return token_emb


TupleTokenHeadsRegistry = type("_TupleTokenHeadsRegistry", (Registry,), {})()


@dataclass
class TupleTokenHeadsConfig(VariableModuleConfig):
    dim: int = MISSING


@dataclass
class TupleTokenLMHeadConfig(TupleTokenHeadsConfig):
    _target_: str = "lm"
    num_tokens: Optional[Dict[str, int]] = None
    embeddings: Optional[TupleTokenEmbeddings] = None
    filter_keys: Optional[List[str]] = None


@TupleTokenHeadsRegistry.register("lm")
class TupleTokenLMHead(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            num_tokens: Optional[Dict[str, int]] = None,
            embeddings: Optional[TupleTokenEmbeddings] = None,
            filter_keys: Optional[List[str]] = None
    ):
        assert num_tokens is not None or embeddings is not None
        super().__init__()

        num_tokens = num_tokens or embeddings.num_tokens
        self.heads = nn.ModuleDict({
            key: nn.Linear(dim, num)
            for key, num in num_tokens.items()
            if not filter_keys or key in filter_keys
        })

    def forward(self, x: Tensor, keys: Optional[Union[List[str], List[int]]] = None):
        logits = {
            key: head(x)
            for i, (key, head) in enumerate(self.heads.items())
            if keys is None or i in keys or key in keys
        }
        return logits


@dataclass
class TupleTokenTiedLMHeadConfig(TupleTokenHeadsConfig):
    _target_: str = "lm-tied"
    embeddings: TupleTokenEmbeddings = MISSING
    reuse_projection: bool = True


@TupleTokenHeadsRegistry.register("lm-tied")
class TupleTokenTiedLMHead(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            embeddings: TupleTokenEmbeddings,
            reuse_projection: bool = True
    ):
        super().__init__()

        self.embs = embeddings.embs
        self.total_emb_dim = embeddings.total_emb_dim
        self.split_dims = [token_emb.embedding_dim for token_emb in embeddings.embs.values()]

        if reuse_projection:
            assert dim == embeddings.project_emb.out_features, \
                f"Projection layer could be reused only if last input tensor dimension " \
                f"is equal to projection layer's `out_features = {embeddings.project_emb.out_features}`"
            self.project_emb = embeddings.project_emb
        else:
            self.project_emb = nn.Linear(dim, self.total_emb_dim, bias=False)

        self.norm = nn.LayerNorm(self.total_emb_dim)

    def forward(self, x: Tensor, keys: Optional[Union[List[str], List[int]]] = None):
        token_embs = self.norm(x @ self.project_emb.weight).split(self.split_dims, dim=-1)

        logits = {
            key: token_embs[i] @ self.embs[key].weight.t()
            for i, key in enumerate(self.embs.keys())
            if keys is None or i in keys or key in keys
        }
        return logits


@dataclass
class TupleTokenTiedSplitLMHeadConfig(TupleTokenHeadsConfig):
    _target_: str = "lm-tied-split"
    embeddings: TupleTokenEmbeddings = MISSING
    filter_keys: Optional[List[str]] = None


@TupleTokenHeadsRegistry.register("lm-tied-split")
class TupleTokenTiedSplitLMHead(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            embeddings: TupleTokenEmbeddings,
            filter_keys: Optional[List[str]] = None
    ):
        super().__init__()

        to_embs, heads = {}, {}
        for key, token_emb in embeddings.embs.items():
            if not filter_keys or key in filter_keys:
                to_embs[key] = nn.Sequential(
                    nn.Linear(dim, token_emb.embedding_dim),
                    nn.LayerNorm(token_emb.embedding_dim),
                )

        self.to_embs = nn.ModuleDict(to_embs)
        self.embs = embeddings.embs

    def forward(self, x: Tensor, keys: Optional[Union[List[str], List[int]]] = None):
        logits = {
            key: self.to_embs[key](x) @ self.embs[key].weight.t()
            for i, key in enumerate(self.embs.keys())
            if keys is None or i in keys or key in keys
        }
        return logits


@dataclass
class TupleTokenRegressionHeadConfig(TupleTokenHeadsConfig):
    _target_: str = "regression"
    regression_keys: List[str] = MISSING


@TupleTokenHeadsRegistry.register("regression")
class TupleTokenRegressionHead(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            regression_keys: List[str]
    ):
        super().__init__()

        self.layers = nn.ModuleDict({
            key: nn.Linear(dim, 1)
            for key in regression_keys
        })

    def forward(self, x: Tensor, keys: Optional[Union[List[str], List[int]]] = None):
        values = {
            key: layer(x)
            for i, (key, layer) in enumerate(self.layers.items())
            if keys is None or i in keys or key in keys
        }

        return values


@dataclass
class TupleTokenEmbeddingHeadConfig(TupleTokenHeadsConfig):
    _target_: str = "embedding"
    emb_dim: int = MISSING
    hidden_dim: Optional[int] = None
    depth: int = 2
    detach_inputs: Union[bool, float] = True


@TupleTokenHeadsRegistry.register("embedding")
class TupleTokenEmbeddingHead(nn.Module, Constructor):
    def __init__(
            self,
            dim: int,
            emb_dim: int,
            hidden_dim: Optional[int] = None,
            depth: int = 2,
            detach_inputs: Union[bool, float] = True
    ):
        super().__init__()

        hidden_dim = hidden_dim or emb_dim

        input_dims = [dim] + [hidden_dim] * (depth - 1)
        output_dims = [hidden_dim] * (depth - 1) + [emb_dim]

        layers = []
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            layers.append(nn.Linear(in_dim, out_dim))
            if i < depth - 1:
                layers.append(nn.Mish())

        self.layers = nn.Sequential(*layers)

        self.detach_inputs = detach_inputs

    def forward(self, x: Tensor):
        x = self.detach_inputs * x.detach() + (1 - self.detach_inputs) * x
        embeddings = self.layers(x)
        return embeddings
