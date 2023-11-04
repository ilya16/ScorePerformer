""" TupleTransformer: Transformer with support for tuple token sequences. """

from dataclasses import dataclass, MISSING
from typing import Optional, Union, Dict, List

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from scoreperformer.modules.constructor import Constructor, ModuleConfig
from scoreperformer.modules.transformer import (
    TransformerConfig, TransformerRegistry, TransformerIntermediates,
    AbsolutePositionalEmbedding
)
from scoreperformer.utils import ExplicitEnum
from .embeddings import (
    TupleTokenEmbeddingsConfig, TupleTokenEmbeddingsRegistry,
    TupleTokenHeadsConfig, TupleTokenHeadsRegistry,
    TupleTokenRegressionHeadConfig, TupleTokenRegressionHead
)


class EmbeddingModes(ExplicitEnum):
    SUM = "mean"
    CONCAT = "cat"
    ATTENTION = "attention"
    ADANORM = "adanorm"


@dataclass
class TupleTransformerCaches:
    token_emb: Optional[Tensor] = None
    transformer: Optional[TransformerIntermediates] = None


@dataclass
class TupleTransformerOutput:
    hidden_state: Tensor
    logits: Optional[Dict[str, Tensor]] = None
    attentions: Optional[List[Tensor]] = None
    caches: Optional[TupleTransformerCaches] = None
    reg_values: Optional[Dict[str, Tensor]] = None


@dataclass
class TupleTransformerConfig(ModuleConfig):
    num_tokens: Dict[str, int] = MISSING
    dim: int = 512
    max_seq_len: int = 1024
    transformer: Union[DictConfig, TransformerConfig] = TransformerConfig(_target_="default")

    token_embeddings: Union[DictConfig, TupleTokenEmbeddingsConfig] = TupleTokenEmbeddingsConfig()
    use_abs_pos_emb: bool = True
    emb_norm: bool = False
    emb_dropout: float = 0.0

    context_emb_dim: Optional[int] = None
    context_emb_mode: str = EmbeddingModes.ATTENTION
    style_emb_dim: Optional[int] = None
    style_emb_mode: str = EmbeddingModes.CONCAT

    lm_head: Optional[Union[DictConfig, TupleTokenHeadsConfig]] = None
    regression_head: Optional[Union[DictConfig, TupleTokenRegressionHeadConfig]] = None


class TupleTransformer(nn.Module, Constructor):
    def __init__(
            self,
            num_tokens: Dict[str, int],
            dim: int = 512,
            max_seq_len: int = 1024,
            transformer: Union[DictConfig, TransformerConfig] = TransformerConfig(_target_="default"),
            token_embeddings: Union[DictConfig, TupleTokenEmbeddingsConfig] = TupleTokenEmbeddingsConfig(),
            use_abs_pos_emb: bool = True,
            emb_norm: bool = False,
            emb_dropout: float = 0.0,
            context_emb_dim: Optional[int] = None,
            context_emb_mode: str = EmbeddingModes.ATTENTION,
            style_emb_dim: Optional[int] = None,
            style_emb_mode: str = EmbeddingModes.CONCAT,
            lm_head: Optional[Union[DictConfig, TupleTokenHeadsConfig]] = None,
            regression_head: Optional[Union[DictConfig, TupleTokenRegressionHeadConfig]] = None
    ):
        super().__init__()

        self.dim = dim
        self.max_seq_len = max_seq_len
        emb_dim = dim  # default(emb_dim, dim)

        self.context_emb_dim = context_emb_dim or 0
        self.context_emb_mode = context_emb_mode

        self.style_emb_dim = style_emb_dim or 0
        self.style_emb_mode = style_emb_mode

        self.token_emb = TupleTokenEmbeddingsRegistry.instantiate(
            config=token_embeddings,
            num_tokens=num_tokens,
            emb_dims=token_embeddings.get("emb_dims", emb_dim),
            project_emb_dim=emb_dim
        )

        if self.context_emb_mode != EmbeddingModes.ATTENTION:
            transformer.cross_attend = False

        self.transformer = TransformerRegistry.instantiate(
            transformer,
            dim=dim,
            use_adanorm=self.style_emb_mode == EmbeddingModes.ADANORM,
            style_emb_dim=self.style_emb_dim
        )

        self.pos_emb = None
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, self.max_seq_len)
            nn.init.kaiming_normal_(self.pos_emb.emb.weight)

        self.emb_norm = nn.LayerNorm(emb_dim) if emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0. else nn.Identity()

        self.project_emb = nn.Identity()
        total_emb_dim = (
                emb_dim
                + int(context_emb_mode == EmbeddingModes.CONCAT) * self.context_emb_dim
                + int(style_emb_mode == EmbeddingModes.CONCAT) * self.style_emb_dim
        )
        if total_emb_dim != dim:
            self.project_emb = nn.Linear(total_emb_dim, dim)

        self.lm_head = None
        if lm_head is not None:
            self.lm_head = TupleTokenHeadsRegistry.instantiate(
                config=lm_head, dim=dim, embeddings=self.token_emb
            )

        self.regression_head = None
        if regression_head is not None:
            assert self.token_emb.continuous, "TupleTokenRegressionHead depends on `continuous` token embeddings."
            self.regression_head = TupleTokenRegressionHead.init(
                config=regression_head, dim=dim
            )

    def forward(
            self,
            x: Tensor,
            mask: Optional[Tensor] = None,
            x_extra: Optional[Union[Tensor, List[Tensor]]] = None,
            style_embeddings: Optional[Tensor] = None,
            context: Optional[Tensor] = None,
            context_mask: Optional[Tensor] = None,
            caches: Optional[TupleTransformerCaches] = None,
            logits_keys: Optional[List] = None,
            return_embeddings: bool = False,
            return_attn: bool = False,
            return_caches: bool = False,
            **kwargs
    ):
        token_emb_cache = caches.token_emb if caches is not None else None
        if hasattr(self.token_emb, "multiseq_mode") and x_extra is not None:
            x_extra = [x_extra] if isinstance(x_extra, Tensor) else x_extra
            token_emb = self.token_emb([x] + x_extra, cache=token_emb_cache)
        else:
            token_emb = self.token_emb(x, cache=token_emb_cache)

        x = token_emb
        if self.pos_emb is not None:
            x = x + self.pos_emb(x)
        x = self.emb_norm(x)

        if context is not None and self.context_emb_mode == EmbeddingModes.CONCAT:
            context = context[:, :x.shape[1]]
            x = torch.cat([x, context], dim=-1)
            context = None

        if style_embeddings is not None:
            style_embeddings = style_embeddings[:, :x.shape[1]]
            if self.style_emb_mode == EmbeddingModes.CONCAT:
                x = torch.cat([x, style_embeddings], dim=-1)
                style_embeddings = None

        x = self.emb_dropout(x)
        x = self.project_emb(x)

        out, intermediates = self.transformer(
            x,
            mask=mask,
            context=context,
            context_mask=context_mask,
            style_embeddings=style_embeddings,
            intermediates_cache=caches.transformer if caches is not None else None,
            return_hiddens=True
        )

        logits = None
        if not return_embeddings and self.lm_head is not None:
            logits = self.lm_head(out, keys=logits_keys)

        reg_values = None
        if not return_embeddings and self.regression_head is not None:
            reg_values = self.regression_head(out, keys=logits_keys)

        attn_maps = None
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attention))

        caches = None
        if return_caches:
            caches = TupleTransformerCaches(
                token_emb=token_emb,
                transformer=intermediates
            )

        return TupleTransformerOutput(
            hidden_state=out,
            logits=logits,
            attentions=attn_maps,
            caches=caches,
            reg_values=reg_values
        )
