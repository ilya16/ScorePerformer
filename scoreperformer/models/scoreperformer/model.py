"""
TupleTransformer-based models.

- Performer:
    Standalone TupleTransformer model with language modeling wrappers.

- ScorePerformer:
    Combines PerformanceDecoder with ScoreEncoder and PerformanceEncoder TupleTransformers.
"""

from dataclasses import dataclass, MISSING
from typing import Optional, Dict, Union

from omegaconf import DictConfig
from torch import Tensor

from scoreperformer.data.collators import (
    PerformanceInputs,
    LMPerformanceInputs,
    MixedLMPerformanceInputs,
    ScorePerformanceInputs,
    LMScorePerformanceInputs,
    MixedLMScorePerformanceInputs
)
from scoreperformer.data.datasets import ScorePerformanceDataset
from scoreperformer.modules.constructor import ModuleConfig
from scoreperformer.utils import default
from .embeddings import TupleTokenLMHeadConfig
from .mmd_transformer import MMDTupleTransformer, MMDTupleTransformerOutput
from .transformer import (
    TupleTransformerConfig,
    TupleTransformerOutput,
    TupleTransformer
)
from .wrappers import (
    LMWrapper,
    ScorePerformerLMModes,
    ScorePerformerLMWrappers
)
from ..base import Model
from ..classifiers.model import (
    MultiHeadEmbeddingClassifierConfig,
    MultiHeadEmbeddingClassifier,
    MultiHeadEmbeddingClassifierOutput
)


# Performer model (plain transformer decoder with no encoders)

@dataclass
class PerformerConfig(ModuleConfig):
    transformer: TupleTransformerConfig = MISSING
    mode: Optional[str] = None


@dataclass
class PerformerOutputs(TupleTransformerOutput):
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


class Performer(Model):
    def __init__(
            self,
            transformer: Union[DictConfig, TupleTransformerConfig],
            mode: Optional[str] = None
    ):
        super().__init__()

        self.transformer = TupleTransformer.init(
            transformer,
            lm_head=transformer.get("lm_head", TupleTokenLMHeadConfig(dim=transformer.dim))
        )

        self.mode = mode
        if self.mode == ScorePerformerLMModes.MLM:
            self.prepare_for_mlm()
        elif self.mode == ScorePerformerLMModes.CLM:
            self.prepare_for_clm()
        elif self.mode == ScorePerformerLMModes.MixedLM:
            self.prepare_for_mixlm()

    def _prepare_for_lm(self, mode: ScorePerformerLMModes):
        if isinstance(self.transformer, TupleTransformer):
            self.transformer = ScorePerformerLMWrappers[mode](self.transformer)
        elif isinstance(self.perf_decoder, LMWrapper):
            self.transformer = ScorePerformerLMWrappers[mode](self.transformer.model)
        self.mode = mode

    def prepare_for_mlm(self):
        self._prepare_for_lm(mode=ScorePerformerLMModes.MLM)

    def prepare_for_clm(self):
        self._prepare_for_lm(mode=ScorePerformerLMModes.CLM)

    def prepare_for_mixlm(self):
        self._prepare_for_lm(mode=ScorePerformerLMModes.MixedLM)

    def forward(
            self,
            perf: Tensor,
            mask: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            masked_perf: Optional[Tensor] = None
    ):
        return self.transformer(
            perf, mask=mask, labels=labels, seq_masked=masked_perf
        )

    def prepare_inputs(self, inputs: PerformanceInputs):
        inputs_dict = {
            "perf": inputs.performances.tokens,
            "mask": inputs.performances.mask,
        }

        if isinstance(inputs, LMPerformanceInputs):
            inputs_dict["labels"] = inputs.labels.tokens

        if isinstance(inputs, MixedLMPerformanceInputs):
            inputs_dict["masked_perf"] = inputs.masked_performances.tokens

        return inputs_dict


# ScorePerformer model

@dataclass
class ScorePerformerConfig(ModuleConfig):
    num_tokens: Dict[str, int] = MISSING
    dim: int = MISSING
    perf_decoder: TupleTransformerConfig = MISSING
    score_encoder: Optional[TupleTransformerConfig] = None
    perf_encoder: Optional[TupleTransformerConfig] = None
    classifiers: Optional[Union[DictConfig, MultiHeadEmbeddingClassifierConfig]] = None
    tie_token_emb: bool = False
    mode: Optional[str] = None
    num_score_tokens: Optional[Dict[str, int]] = None


@dataclass
class ScorePerformerEncoderOutputs:
    score_embeddings: Optional[Tensor] = None
    score_mask: Optional[Tensor] = None
    perf_embeddings: Optional[Tensor] = None
    score_encoder: Optional[TupleTransformerOutput] = None
    perf_encoder: Optional[MMDTupleTransformerOutput] = None


@dataclass
class ScorePerformerOutputs:
    perf_decoder: TupleTransformerOutput
    score_encoder: Optional[TupleTransformerOutput] = None
    perf_encoder: Optional[MMDTupleTransformerOutput] = None
    classifiers: Optional[MultiHeadEmbeddingClassifierOutput] = None
    loss: Optional[Tensor] = None
    losses: Optional[Dict[str, Tensor]] = None


class ScorePerformer(Model):
    def __init__(
            self,
            num_tokens: Dict[str, int],
            dim: int,
            perf_decoder: Union[DictConfig, TupleTransformerConfig],
            score_encoder: Optional[Union[DictConfig, TupleTransformerConfig]] = None,
            perf_encoder: Optional[Union[DictConfig, TupleTransformerConfig]] = None,
            classifiers: Optional[Union[DictConfig, MultiHeadEmbeddingClassifierConfig]] = None,
            tie_token_emb: bool = False,
            mode: Optional[str] = None,
            num_score_tokens: Optional[Dict[str, int]] = None
    ):
        super().__init__()

        self.score_encoder = None
        if score_encoder is not None:
            self.score_encoder = TupleTransformer.init(
                score_encoder,
                num_tokens=num_score_tokens or num_tokens,
                dim=dim,
                lm_head=None
            )

        self.perf_encoder = None
        if perf_encoder is not None:
            self.perf_encoder = MMDTupleTransformer.init(
                perf_encoder,
                num_tokens=num_tokens,
                dim=dim,
                lm_head=None
            )

        self.classifiers = None
        if classifiers is not None and classifiers.num_classes is not None:
            assert self.perf_encoder is not None
            self.classifiers = MultiHeadEmbeddingClassifier.init(
                classifiers,
                input_dim=self.perf_encoder.embedding_dim
            )

        perf_decoder.transformer.cross_attend = self.score_encoder is not None
        context_emb_dim = None if self.score_encoder is None else self.score_encoder.dim
        style_emb_dim = None if self.perf_encoder is None else self.perf_encoder.embedding_dim

        self.perf_decoder = TupleTransformer.init(
            perf_decoder,
            num_tokens=num_tokens,
            dim=dim,
            context_emb_dim=context_emb_dim,
            style_emb_dim=style_emb_dim,
            lm_head=perf_decoder.get("lm_head", TupleTokenLMHeadConfig(dim=dim))
        )

        if tie_token_emb:
            for key, emb in self.perf_decoder.token_emb.embs.items():
                if self.score_encoder is not None and key in self.score_encoder.token_emb.embs:
                    self.score_encoder.token_emb.embs[key] = self.perf_decoder.token_emb.embs[key]
                if self.perf_encoder is not None and key in self.perf_encoder.token_emb.embs:
                    self.perf_encoder.token_emb.embs[key] = self.perf_decoder.token_emb.embs[key]

        self.mode = mode
        if self.mode == ScorePerformerLMModes.MLM:
            self.prepare_for_mlm()
        elif self.mode == ScorePerformerLMModes.CLM:
            self.prepare_for_clm()
        elif self.mode == ScorePerformerLMModes.MixedLM:
            self.prepare_for_mixlm()

    def _prepare_for_lm(self, mode: ScorePerformerLMModes):
        if isinstance(self.perf_decoder, TupleTransformer):
            self.perf_decoder = ScorePerformerLMWrappers[mode](self.perf_decoder)
        elif isinstance(self.perf_decoder, LMWrapper):
            self.perf_decoder = ScorePerformerLMWrappers[mode](self.perf_decoder.model)
        self.mode = mode

    def prepare_for_mlm(self):
        self._prepare_for_lm(mode=ScorePerformerLMModes.MLM)

    def prepare_for_clm(self):
        self._prepare_for_lm(mode=ScorePerformerLMModes.CLM)

    def prepare_for_mixlm(self):
        self._prepare_for_lm(mode=ScorePerformerLMModes.MixedLM)

    def forward_encoders(
            self,
            perf: Optional[Tensor] = None,
            perf_mask: Optional[Tensor] = None,
            score: Optional[Tensor] = None,
            score_mask: Optional[Tensor] = None,
            bars: Optional[Tensor] = None,
            beats: Optional[Tensor] = None,
            onsets: Optional[Tensor] = None,
            deadpan_mask: Optional[Tensor] = None,
            compute_loss: bool = True
    ):
        score_emb = perf_emb = None
        score_enc_out = perf_enc_out = None

        if self.score_encoder is not None:
            score_enc_out = self.score_encoder(score, mask=score_mask)
            score_emb = score_enc_out.hidden_state

        if self.perf_encoder is not None:
            perf_enc_out = self.perf_encoder(
                perf, mask=perf_mask,
                bars=bars, beats=beats, onsets=onsets,
                deadpan_mask=deadpan_mask,
                compute_loss=compute_loss
            )
            perf_emb = perf_enc_out.embeddings

        return ScorePerformerEncoderOutputs(
            score_embeddings=score_emb,
            score_mask=score_mask,
            perf_embeddings=perf_emb,
            score_encoder=score_enc_out,
            perf_encoder=perf_enc_out
        )

    def forward(
            self,
            perf: Tensor,
            perf_mask: Optional[Tensor] = None,
            score: Optional[Tensor] = None,
            score_mask: Optional[Tensor] = None,
            noisy_perf: Optional[Tensor] = None,
            noisy_perf_mask: Optional[Tensor] = None,
            masked_perf: Optional[Tensor] = None,
            labels: Optional[Tensor] = None,
            bars: Optional[Tensor] = None,
            beats: Optional[Tensor] = None,
            onsets: Optional[Tensor] = None,
            directions: Optional[Tensor] = None,
            deadpan_mask: Optional[Tensor] = None
    ):
        enc_out = self.forward_encoders(
            perf=default(noisy_perf, perf),
            perf_mask=default(noisy_perf_mask, perf_mask),
            score=score,
            score_mask=score_mask,
            bars=bars,
            beats=beats,
            onsets=onsets,
            deadpan_mask=deadpan_mask
        )

        perf_dec_out = self.perf_decoder(
            perf,
            mask=perf_mask,
            style_embeddings=enc_out.perf_embeddings,
            context=enc_out.score_embeddings,
            context_mask=enc_out.score_mask,
            labels=labels,
            seq_masked=masked_perf
        )
        loss, losses = perf_dec_out.loss, perf_dec_out.losses

        if enc_out.perf_encoder is not None:
            if enc_out.perf_encoder.loss is not None:
                loss += enc_out.perf_encoder.loss
                losses.update(**enc_out.perf_encoder.losses)

        clf_out = None
        if self.classifiers is not None:
            clf_mask = perf_mask if deadpan_mask is None else perf_mask & (~deadpan_mask[:, None])
            clf_out = self.classifiers(
                embeddings=enc_out.perf_encoder.full_embeddings[clf_mask],
                labels=directions[clf_mask]
            )
            if clf_out.loss is not None:
                loss += clf_out.loss
                losses.update(**clf_out.losses)

        return ScorePerformerOutputs(
            perf_decoder=perf_dec_out,
            score_encoder=enc_out.score_encoder,
            perf_encoder=enc_out.perf_encoder,
            classifiers=clf_out,
            loss=loss,
            losses=losses
        )

    def prepare_inputs(self, inputs: ScorePerformanceInputs):
        inputs_dict = {
            "perf": inputs.performances.tokens,
            "perf_mask": inputs.performances.mask,
            "score": inputs.scores.tokens,
            "score_mask": inputs.scores.mask,
        }

        if isinstance(inputs, LMScorePerformanceInputs):
            inputs_dict["labels"] = inputs.labels.tokens

        if inputs.noisy_performances is not None:
            inputs_dict["noisy_perf"] = inputs.noisy_performances.tokens
            inputs_dict["noisy_perf_mask"] = inputs.noisy_performances.mask

        if isinstance(inputs, MixedLMScorePerformanceInputs):
            inputs_dict["masked_perf"] = inputs.masked_performances.tokens

        if inputs.segments is not None:
            inputs_dict["bars"] = inputs.segments.bar
            inputs_dict["beats"] = inputs.segments.beat
            inputs_dict["onsets"] = inputs.segments.onset

        if inputs.directions is not None:
            inputs_dict["directions"] = inputs.directions

        if inputs.deadpan_mask is not None:
            inputs_dict["deadpan_mask"] = inputs.deadpan_mask

        return inputs_dict

    @staticmethod
    def inject_data_config(
            config: Optional[Union[DictConfig, ScorePerformerConfig]],
            dataset: Optional[ScorePerformanceDataset]
    ) -> Optional[Union[DictConfig, ModuleConfig]]:
        assert isinstance(dataset, ScorePerformanceDataset)

        config["num_tokens"] = dataset.tokenizer.performance_sizes
        config["num_score_tokens"] = dataset.tokenizer.score_sizes

        for key in ["score_encoder", "perf_encoder", "perf_decoder"]:
            if config.get(key) is not None:
                config[key]["token_embeddings"]["token_values"] = {
                    key: value.tolist() for key, value in dataset.tokenizer.token_values(normalize=True).items()
                }

        if config.get("classifiers") is not None and dataset.performance_directions is not None:
            config["classifiers"]["num_classes"] = dict(dataset.performance_direction_sizes)
            config["classifiers"]["class_samples"] = dict(dataset.get_direction_class_weights()[1])

        return config

    @staticmethod
    def cleanup_config(
            config: Optional[Union[DictConfig, ScorePerformerConfig]],
    ) -> Optional[Union[DictConfig, ModuleConfig]]:
        for key in ["score_encoder", "perf_encoder", "perf_decoder"]:
            if config.get(key) is not None:
                del config[key]["token_embeddings"]["token_values"]

        if config.get("classifiers") is not None:
            del config["classifiers"]["class_samples"]

        return config
