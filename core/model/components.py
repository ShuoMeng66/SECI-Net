from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def apply_sequence_mask(sequence: Tensor, attention_mask: Tensor | None) -> Tensor:
    if attention_mask is None:
        return sequence
    return sequence * attention_mask.unsqueeze(-1).float()


def masked_softmax(scores: Tensor, mask: Tensor) -> Tensor:
    mask = mask.bool()
    scores = scores.masked_fill(~mask, -1e4)
    attention = torch.softmax(scores, dim=-1) * mask.float()
    attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return attention


def masked_mean_pooling(sequence: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    masked_sequence = sequence * mask
    return masked_sequence.sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )

        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.max_len:
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding max_len={self.max_len}."
            )
        return x + self.pe[:, : x.size(1)]


class FactorizedSpatialTemporalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 512,
        temporal_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.spatial_encoding = SinusoidalPositionalEncoding(embed_dim, max_len=max_len)
        self.temporal_projection = nn.Sequential(
            nn.Linear(1, temporal_hidden_dim),
            nn.SiLU(),
            nn.Linear(temporal_hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def _prepare_time_values(self, time_values: Tensor, seq_len: int) -> Tensor:
        if time_values.dim() == 1:
            time_values = time_values.unsqueeze(1).expand(-1, seq_len)
        elif time_values.dim() == 2:
            if time_values.size(1) < seq_len:
                raise ValueError(
                    f"time_values second dimension {time_values.size(1)} is smaller than seq_len={seq_len}."
                )
            time_values = time_values[:, :seq_len]
        else:
            raise ValueError("time_values must have shape [batch] or [batch, seq_len].")

        time_values = time_values.float()
        time_values = torch.sign(time_values) * torch.log1p(time_values.abs())
        return time_values.unsqueeze(-1)

    def forward(self, x: Tensor, time_values: Tensor | None = None) -> Tensor:
        x = self.spatial_encoding(x)
        if time_values is None:
            return x

        temporal_embedding = self.temporal_projection(
            self._prepare_time_values(time_values, seq_len=x.size(1))
        )
        return self.output_norm(x + temporal_embedding)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def _apply_key_padding_mask(self, scores: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        if key_padding_mask is None:
            return scores
        return scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        queries = self._reshape_heads(self.query_projection(x))
        keys = self._reshape_heads(self.key_projection(x))
        values = self._reshape_heads(self.value_projection(x))

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = self._apply_key_padding_mask(scores, key_padding_mask)

        attention = torch.softmax(scores, dim=-1)
        attention_probs = attention
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, values)

        output = self._merge_heads(context)
        output = self.output_projection(output)
        output = self.output_dropout(output)
        if return_attention:
            return output, attention_probs
        return output


class DifferentialMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        lambda_init: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_projection_1 = nn.Linear(embed_dim, embed_dim)
        self.key_projection_1 = nn.Linear(embed_dim, embed_dim)
        self.query_projection_2 = nn.Linear(embed_dim, embed_dim)
        self.key_projection_2 = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.lambda_parameter = nn.Parameter(torch.tensor(float(lambda_init)))
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def _masked_scores(self, queries: Tensor, keys: Tensor, key_padding_mask: Tensor | None) -> Tensor:
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        return scores

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        q1 = self._reshape_heads(self.query_projection_1(x))
        k1 = self._reshape_heads(self.key_projection_1(x))
        q2 = self._reshape_heads(self.query_projection_2(x))
        k2 = self._reshape_heads(self.key_projection_2(x))
        values = self._reshape_heads(self.value_projection(x))

        scores_1 = self._masked_scores(q1, k1, key_padding_mask)
        scores_2 = self._masked_scores(q2, k2, key_padding_mask)

        attention_1 = torch.softmax(scores_1, dim=-1)
        attention_2 = torch.softmax(scores_2, dim=-1)
        attention_probs_1 = attention_1
        attention_probs_2 = attention_2
        attention_1 = self.attention_dropout(attention_1)
        attention_2 = self.attention_dropout(attention_2)

        lambda_weight = torch.sigmoid(self.lambda_parameter)
        context = torch.matmul(attention_1, values) - lambda_weight * torch.matmul(attention_2, values)
        output = self._merge_heads(context)
        output = self.output_projection(output)
        output = self.output_dropout(output)
        if return_attention:
            lambda_weight = torch.sigmoid(self.lambda_parameter)
            effective_attention = attention_probs_1 - lambda_weight * attention_probs_2
            effective_attention = effective_attention.clamp_min(0.0)
            effective_attention = effective_attention / effective_attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            return output, effective_attention
        return output


class BlockSparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        block_size: int = 16,
        local_window_size: int = 8,
        topk_global_blocks: int = 2,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if local_window_size < 0:
            raise ValueError("local_window_size must be non-negative.")
        if topk_global_blocks <= 0:
            raise ValueError("topk_global_blocks must be positive.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.local_window_size = local_window_size
        self.topk_global_blocks = topk_global_blocks

        self.query_projection = nn.Linear(embed_dim, embed_dim)
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.value_projection = nn.Linear(embed_dim, embed_dim)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        self.block_query_projection = nn.Linear(embed_dim, embed_dim)
        self.block_key_projection = nn.Linear(embed_dim, embed_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def _compute_salient_blocks(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        batch_size, seq_len, embed_dim = x.size()
        num_blocks = math.ceil(seq_len / self.block_size)
        padded_len = num_blocks * self.block_size
        pad_len = padded_len - seq_len

        if pad_len > 0:
            x = torch.cat([x, x.new_zeros(batch_size, pad_len, embed_dim)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_zeros(batch_size, pad_len)],
                dim=1,
            )

        x_blocks = x.view(batch_size, num_blocks, self.block_size, embed_dim)
        mask_blocks = attention_mask.view(batch_size, num_blocks, self.block_size).unsqueeze(-1).float()
        block_summaries = (x_blocks * mask_blocks).sum(dim=2) / mask_blocks.sum(dim=2).clamp_min(1.0)

        document_summary = (x[:, :seq_len] * attention_mask[:, :seq_len].unsqueeze(-1).float()).sum(dim=1)
        document_summary = document_summary / attention_mask[:, :seq_len].sum(dim=1, keepdim=True).clamp_min(1.0)

        query = self.block_query_projection(document_summary).unsqueeze(1)
        keys = self.block_key_projection(block_summaries)
        block_scores = torch.matmul(query, keys.transpose(-2, -1)).squeeze(1) / math.sqrt(embed_dim)

        valid_block_mask = mask_blocks.squeeze(-1).sum(dim=-1) > 0
        block_scores = block_scores.masked_fill(~valid_block_mask, -1e4)

        topk = min(self.topk_global_blocks, num_blocks)
        return torch.topk(block_scores, k=topk, dim=-1).indices

    def _build_sparse_mask(self, x: Tensor, attention_mask: Tensor) -> Tensor:
        _, seq_len, _ = x.size()
        device = x.device

        positions = torch.arange(seq_len, device=device)
        query_positions = positions.view(seq_len, 1)
        key_positions = positions.view(1, seq_len)

        local_mask = (query_positions - key_positions).abs() <= self.local_window_size
        block_ids = positions // self.block_size
        same_block_mask = block_ids.view(seq_len, 1) == block_ids.view(1, seq_len)

        salient_block_indices = self._compute_salient_blocks(x, attention_mask)
        salient_key_mask = (
            block_ids.view(1, 1, seq_len) == salient_block_indices.unsqueeze(-1)
        ).any(dim=1)
        global_mask = salient_key_mask.unsqueeze(1).expand(-1, seq_len, -1)

        sparse_mask = local_mask.unsqueeze(0) | same_block_mask.unsqueeze(0) | global_mask
        valid_query_mask = attention_mask.unsqueeze(-1).bool()
        valid_key_mask = attention_mask.unsqueeze(1).bool()
        return sparse_mask & valid_query_mask & valid_key_mask

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        attention_mask = (~key_padding_mask).long() if key_padding_mask is not None else x.new_ones(
            x.size(0), x.size(1), dtype=torch.long
        )

        queries = self._reshape_heads(self.query_projection(x))
        keys = self._reshape_heads(self.key_projection(x))
        values = self._reshape_heads(self.value_projection(x))

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        sparse_mask = self._build_sparse_mask(x, attention_mask).unsqueeze(1)
        attention = masked_softmax(scores, sparse_mask)
        attention_probs = attention
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, values)

        output = self._merge_heads(context)
        output = self.output_projection(output)
        output = self.output_dropout(output)
        if return_attention:
            return output, attention_probs
        return output


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
        attention_type: str = "standard",
        differential_lambda_init: float = 0.5,
        block_size: int = 16,
        local_window_size: int = 8,
        topk_global_blocks: int = 2,
    ) -> None:
        super().__init__()
        if attention_type == "differential":
            self.self_attention = DifferentialMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                lambda_init=differential_lambda_init,
            )
        elif attention_type == "standard":
            self.self_attention = MultiHeadSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        elif attention_type == "block_sparse":
            self.self_attention = BlockSparseMultiHeadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                block_size=block_size,
                local_window_size=local_window_size,
                topk_global_blocks=topk_global_blocks,
            )
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if return_attention:
            attn_output, attention_probs = self.self_attention(
                x, key_padding_mask=key_padding_mask, return_attention=True
            )
        else:
            attn_output = self.self_attention(x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_output))
        x = apply_sequence_mask(x, attention_mask)

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        x = apply_sequence_mask(x, attention_mask)
        if return_attention:
            attention_map = attention_probs.mean(dim=1)
            return x, attention_map
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
        attention_type: str = "standard",
        differential_lambda_init: float = 0.5,
        block_size: int = 16,
        local_window_size: int = 8,
        topk_global_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
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
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        key_padding_mask = attention_mask == 0
        if not return_attention:
            for layer in self.layers:
                x = layer(
                    x,
                    key_padding_mask=key_padding_mask,
                    attention_mask=attention_mask,
                )
            return x

        attention_maps: list[Tensor] = []
        for layer in self.layers:
            x, attention_map = layer(
                x,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
                return_attention=True,
            )
            attention_maps.append(attention_map)
        return x, attention_maps


class BidirectionalSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_dim = hidden_dim * 2
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        packed = pack_padded_sequence(
            x,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=x.size(1),
        )
        return self.output_dropout(output)


class GatedFeatureFusion(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        recurrent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.recurrent_projection = nn.Linear(recurrent_dim, transformer_dim)
        self.gate = nn.Linear(transformer_dim * 2, transformer_dim)
        self.output_norm = nn.LayerNorm(transformer_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        transformer_features: Tensor,
        recurrent_features: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        projected_recurrent = self.recurrent_projection(recurrent_features)
        gate_values = torch.sigmoid(
            self.gate(torch.cat([transformer_features, projected_recurrent], dim=-1))
        )
        fused = gate_values * transformer_features + (1.0 - gate_values) * projected_recurrent
        fused = self.output_norm(fused)
        fused = self.output_dropout(fused)
        fused = apply_sequence_mask(fused, attention_mask)
        return fused, gate_values


class EvidenceRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        block_size: int = 16,
        topk_blocks: int = 2,
        router_hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if block_size <= 0:
            raise ValueError("block_size must be positive.")
        if topk_blocks <= 0:
            raise ValueError("topk_blocks must be positive.")

        router_hidden_dim = router_hidden_dim or input_dim
        self.block_size = block_size
        self.topk_blocks = topk_blocks
        self.router = nn.Sequential(
            nn.Linear(input_dim, router_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, 1),
        )

    def _pad_to_full_blocks(self, token_features: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, hidden_dim = token_features.size()
        num_blocks = math.ceil(seq_len / self.block_size)
        padded_length = num_blocks * self.block_size
        pad_len = padded_length - seq_len
        if pad_len <= 0:
            return token_features, attention_mask

        padded_features = torch.cat(
            [token_features, token_features.new_zeros(batch_size, pad_len, hidden_dim)],
            dim=1,
        )
        padded_mask = torch.cat(
            [attention_mask, attention_mask.new_zeros(batch_size, pad_len)],
            dim=1,
        )
        return padded_features, padded_mask

    def forward(self, token_features: Tensor, attention_mask: Tensor) -> dict[str, Tensor]:
        padded_features, padded_mask = self._pad_to_full_blocks(token_features, attention_mask)
        batch_size, padded_length, hidden_dim = padded_features.size()
        num_blocks = padded_length // self.block_size

        feature_blocks = padded_features.view(batch_size, num_blocks, self.block_size, hidden_dim)
        mask_blocks = padded_mask.view(batch_size, num_blocks, self.block_size)
        block_mask = mask_blocks.sum(dim=-1) > 0

        pooled_blocks = (
            feature_blocks * mask_blocks.unsqueeze(-1).float()
        ).sum(dim=2) / mask_blocks.sum(dim=-1, keepdim=True).clamp_min(1.0)
        block_scores = self.router(pooled_blocks).squeeze(-1)
        block_scores = block_scores.masked_fill(~block_mask, -1e4)
        router_probabilities = masked_softmax(block_scores, block_mask)
        evidence_summary = (router_probabilities.unsqueeze(-1) * pooled_blocks).sum(dim=1)

        topk = min(self.topk_blocks, num_blocks)
        evidence_indices = torch.topk(block_scores, k=topk, dim=-1).indices
        evidence_scores = router_probabilities.gather(dim=1, index=evidence_indices)
        evidence_valid = block_mask.gather(dim=1, index=evidence_indices)
        evidence_scores = evidence_scores * evidence_valid.float()

        return {
            "block_scores": block_scores,
            "block_mask": block_mask,
            "router_probabilities": router_probabilities,
            "evidence_indices": evidence_indices,
            "evidence_scores": evidence_scores,
            "evidence_summary": evidence_summary,
        }
