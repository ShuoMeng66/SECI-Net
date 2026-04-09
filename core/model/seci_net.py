from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .components import (
    BidirectionalSequenceEncoder,
    EvidenceRouter,
    FactorizedSpatialTemporalEncoding,
    GatedFeatureFusion,
    TransformerEncoder,
    apply_sequence_mask,
    masked_mean_pooling,
)


@dataclass
class SECIOutput:
    logits: Tensor
    features: Tensor
    token_features: Tensor
    transformer_features: Tensor
    recurrent_features: Tensor
    fusion_gates: Tensor
    block_scores: Tensor
    block_mask: Tensor
    router_probabilities: Tensor
    evidence_scores: Tensor
    evidence_indices: Tensor
    evidence_summary: Tensor
    recoverability: Tensor
    recoverability_logits: Tensor
    attention_mask: Tensor
    attention_maps: list[Tensor] | None = None


class HybridTextClassifier(nn.Module):
    architecture_name = "seci-net-v2"

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        transformer_layers: int = 2,
        num_heads: int = 4,
        ffn_hidden_dim: int = 256,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_idx: int = 0,
        attention_type: str = "standard",
        use_temporal_encoding: bool = False,
        temporal_hidden_dim: int = 64,
        differential_lambda_init: float = 0.5,
        block_size: int = 16,
        local_window_size: int = 8,
        topk_global_blocks: int = 2,
        router_hidden_dim: int | None = None,
        positive_class_index: int | None = None,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.use_temporal_encoding = use_temporal_encoding
        self.num_classes = num_classes
        self.block_size = block_size
        self.positive_class_index = (
            num_classes - 1 if positive_class_index is None else positive_class_index
        )

        self.model_config = {
            "vocab_size": vocab_size,
            "num_classes": num_classes,
            "embed_dim": embed_dim,
            "transformer_layers": transformer_layers,
            "num_heads": num_heads,
            "ffn_hidden_dim": ffn_hidden_dim,
            "lstm_hidden_dim": lstm_hidden_dim,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            "max_len": max_len,
            "pad_idx": pad_idx,
            "attention_type": attention_type,
            "use_temporal_encoding": use_temporal_encoding,
            "temporal_hidden_dim": temporal_hidden_dim,
            "differential_lambda_init": differential_lambda_init,
            "block_size": block_size,
            "local_window_size": local_window_size,
            "topk_global_blocks": topk_global_blocks,
            "router_hidden_dim": router_hidden_dim,
            "positive_class_index": self.positive_class_index,
        }

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.position_encoding = FactorizedSpatialTemporalEncoding(
            embed_dim=embed_dim,
            max_len=max_len,
            temporal_hidden_dim=temporal_hidden_dim,
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(
            num_layers=transformer_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            dropout=dropout,
            attention_type=attention_type,
            differential_lambda_init=differential_lambda_init,
            block_size=block_size,
            local_window_size=local_window_size,
            topk_global_blocks=topk_global_blocks,
        )
        self.sequence_encoder = BidirectionalSequenceEncoder(
            input_dim=embed_dim,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
        )
        self.fusion = GatedFeatureFusion(
            transformer_dim=embed_dim,
            recurrent_dim=self.sequence_encoder.output_dim,
            dropout=dropout,
        )
        self.evidence_router = EvidenceRouter(
            input_dim=embed_dim,
            block_size=block_size,
            topk_blocks=topk_global_blocks,
            router_hidden_dim=router_hidden_dim or ffn_hidden_dim,
            dropout=dropout,
        )
        self.feature_projection = nn.Sequential(
            nn.Linear(embed_dim * 2, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm(ffn_hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(ffn_hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, num_classes),
        )
        self.recoverability_head = nn.Sequential(
            nn.Linear(ffn_hidden_dim, max(ffn_hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(ffn_hidden_dim // 2, 1), 1),
        )

    def get_model_config(self) -> dict[str, int | float | bool | None | str]:
        return dict(self.model_config)

    def _embed_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        time_values: Tensor | None = None,
    ) -> Tensor:
        embeddings = self.embedding(input_ids)
        embeddings = self.position_encoding(
            embeddings,
            time_values=time_values if self.use_temporal_encoding else None,
        )
        embeddings = self.embedding_dropout(embeddings)
        return apply_sequence_mask(embeddings, attention_mask)

    def encode(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        lengths: Tensor,
        time_values: Tensor | None = None,
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        embeddings = self._embed_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            time_values=time_values,
        )
        attention_maps: list[Tensor] | None = None
        if return_attention:
            transformer_features, attention_maps = self.transformer_encoder(
                embeddings,
                attention_mask=attention_mask,
                return_attention=True,
            )
        else:
            transformer_features = self.transformer_encoder(
                embeddings,
                attention_mask=attention_mask,
            )
        recurrent_features = self.sequence_encoder(embeddings, lengths=lengths)
        fused_features, fusion_gates = self.fusion(
            transformer_features=transformer_features,
            recurrent_features=recurrent_features,
            attention_mask=attention_mask,
        )

        document_summary = masked_mean_pooling(fused_features, attention_mask)
        evidence_state = self.evidence_router(
            token_features=fused_features,
            attention_mask=attention_mask,
        )
        features = self.feature_norm(
            self.feature_projection(
                torch.cat([document_summary, evidence_state["evidence_summary"]], dim=-1)
            )
        )

        return {
            "features": features,
            "token_features": fused_features,
            "transformer_features": transformer_features,
            "recurrent_features": recurrent_features,
            "fusion_gates": fusion_gates,
            "attention_maps": attention_maps,
            **evidence_state,
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        lengths: Tensor | None = None,
        time_values: Tensor | None = None,
        return_features: bool = False,
        return_dict: bool = False,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | SECIOutput:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_idx).long()
        if lengths is None:
            lengths = attention_mask.sum(dim=1)

        encoded = self.encode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            time_values=time_values,
            return_attention=return_attention,
        )
        logits = self.classifier(encoded["features"])
        recoverability_logits = self.recoverability_head(encoded["features"]).squeeze(-1)
        recoverability = torch.sigmoid(recoverability_logits)

        outputs = SECIOutput(
            logits=logits,
            features=encoded["features"],
            token_features=encoded["token_features"],
            transformer_features=encoded["transformer_features"],
            recurrent_features=encoded["recurrent_features"],
            fusion_gates=encoded["fusion_gates"],
            block_scores=encoded["block_scores"],
            block_mask=encoded["block_mask"],
            router_probabilities=encoded["router_probabilities"],
            evidence_scores=encoded["evidence_scores"],
            evidence_indices=encoded["evidence_indices"],
            evidence_summary=encoded["evidence_summary"],
            recoverability=recoverability,
            recoverability_logits=recoverability_logits,
            attention_mask=attention_mask,
            attention_maps=encoded["attention_maps"],
        )

        if return_dict:
            return outputs
        if return_features:
            return outputs.logits, outputs.features
        return outputs.logits


SECINetClassifier = HybridTextClassifier
