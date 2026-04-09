from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from core.data import TextRecord, Vocabulary, default_tokenizer, detokenize_tokens


DEFAULT_ASPECTS = [
    "generic",
    "service",
    "logistics",
    "product",
    "price",
    "experience",
    "environment",
]

ASPECT_KEYWORDS = {
    "service": ["service", "staff", "support", "agent", "客服", "服务", "态度", "回复", "处理"],
    "logistics": ["delivery", "shipping", "courier", "dispatch", "物流", "配送", "快递", "发货", "骑手"],
    "product": ["product", "quality", "broken", "feature", "商品", "产品", "质量", "功能"],
    "price": ["price", "refund", "cost", "charge", "价格", "退款", "收费", "补偿"],
    "experience": ["experience", "app", "website", "system", "体验", "页面", "系统", "卡顿", "闪退"],
    "environment": ["clean", "noise", "room", "package", "环境", "卫生", "包装", "噪音", "房间"],
}

GENERATED_FIELDNAMES = [
    "record_index",
    "merge_strategy",
    "text",
    "label",
    "counterfactual_text",
    "counterfactual_label",
    "source_aspect",
    "target_aspect",
    "synthetic_source",
]


@dataclass
class CounterfactualReviewRecord:
    text: str
    label: str
    counterfactual_text: str = ""
    counterfactual_label: str = ""
    target_aspect: str = "generic"
    source_aspect: str = "generic"
    time_value: Optional[float] = None
    counterfactual_time_value: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class GANTrainingConfig:
    embed_dim: int = 128
    hidden_dim: int = 192
    noise_dim: int = 64
    label_embed_dim: int = 32
    aspect_embed_dim: int = 32
    time_embed_dim: int = 16
    max_source_len: int = 128
    max_target_len: int = 128
    batch_size: int = 16
    learning_rate_generator: float = 1e-4
    learning_rate_discriminator: float = 2e-4
    adversarial_weight: float = 0.5
    reconstruction_weight: float = 1.0
    edit_weight: float = 0.1
    label_weight: float = 0.25
    aspect_weight: float = 0.15
    semantic_weight: float = 0.2
    gradient_clip_norm: float = 1.0
    label_smoothing: float = 0.0
    use_wasserstein: bool = False
    dropout: float = 0.1
    gumbel_temperature: float = 1.0


@dataclass
class GANAugmentationResult:
    augmented_records: list[TextRecord]
    generated_rows: list[dict[str, Any]]
    training_history: list[dict[str, float | int]]
    summary: dict[str, Any]
    artifacts: dict[str, str]
    warning: str | None = None


@dataclass
class _GenerationRequest:
    record_index: int
    request_record: CounterfactualReviewRecord
    merge_strategy: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_review_aspect(text: str, default_aspect: str = "generic") -> str:
    lower = text.lower()
    best_aspect = default_aspect
    best_hits = 0
    for aspect, keywords in ASPECT_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword.lower() in lower)
        if hits > best_hits:
            best_hits = hits
            best_aspect = aspect
    return best_aspect


def infer_target_label(
    current_label: str,
    labels: Sequence[str],
    default_target_label: str | None = None,
) -> str | None:
    if default_target_label is not None:
        return default_target_label
    remaining_labels = [label for label in labels if label != current_label]
    if len(remaining_labels) == 1:
        return remaining_labels[0]
    return None


def build_gan_vocab(
    records: Iterable[CounterfactualReviewRecord],
    tokenizer=default_tokenizer,
) -> Vocabulary:
    vocab = Vocabulary(specials=["<pad>", "<unk>", "<bos>", "<eos>"])
    texts: list[str] = []
    for record in records:
        texts.append(record.text)
        if record.counterfactual_text:
            texts.append(record.counterfactual_text)
    vocab.build(texts, tokenizer=tokenizer)
    return vocab


def build_label_mapping(records: Iterable[CounterfactualReviewRecord]) -> dict[str, int]:
    labels = sorted(
        {record.label for record in records}
        | {record.counterfactual_label for record in records if record.counterfactual_label}
    )
    return {label: index for index, label in enumerate(labels)}


def build_aspect_mapping(records: Iterable[CounterfactualReviewRecord]) -> dict[str, int]:
    aspects = {record.target_aspect or "generic" for record in records}
    aspects |= {record.source_aspect or "generic" for record in records}
    aspects |= set(DEFAULT_ASPECTS)
    return {aspect: index for index, aspect in enumerate(sorted(aspects))}


def pad_sequences(sequences: list[list[int]], pad_idx: int) -> tuple[Tensor, Tensor, Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    input_ids = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        input_ids[row_index, : seq_tensor.size(0)] = seq_tensor
        attention_mask[row_index, : seq_tensor.size(0)] = 1
    return input_ids, attention_mask, lengths


def sanitize_generated_sequences(token_ids: Tensor, eos_idx: int, pad_idx: int) -> tuple[Tensor, Tensor]:
    sanitized = token_ids.clone()
    attention_mask = torch.ones_like(sanitized, dtype=torch.long)
    for row_index in range(sanitized.size(0)):
        eos_positions = (sanitized[row_index] == eos_idx).nonzero(as_tuple=False)
        if eos_positions.numel() == 0:
            continue
        eos_position = int(eos_positions[0].item())
        if eos_position + 1 < sanitized.size(1):
            sanitized[row_index, eos_position + 1 :] = pad_idx
            attention_mask[row_index, eos_position + 1 :] = 0
    attention_mask = attention_mask * (sanitized != pad_idx).long()
    if attention_mask.sum(dim=1).eq(0).any():
        attention_mask[:, 0] = 1
    return sanitized, attention_mask


def extract_training_pairs(records: Sequence[TextRecord]) -> list[CounterfactualReviewRecord]:
    paired_records: list[CounterfactualReviewRecord] = []
    for index, record in enumerate(records):
        if not record.counterfactual_text:
            continue
        source_aspect = infer_review_aspect(record.text)
        target_aspect = infer_review_aspect(record.counterfactual_text, default_aspect=source_aspect)
        paired_records.append(
            CounterfactualReviewRecord(
                text=record.text,
                label=record.label,
                counterfactual_text=record.counterfactual_text,
                counterfactual_label=record.counterfactual_label or record.label,
                source_aspect=source_aspect,
                target_aspect=target_aspect,
                time_value=record.timestamp,
                counterfactual_time_value=(
                    record.counterfactual_timestamp
                    if record.counterfactual_timestamp is not None
                    else record.timestamp
                ),
                metadata={"record_index": index, "source": "paired_training"},
            )
        )
    return paired_records


def build_generation_requests(
    records: Sequence[TextRecord],
    labels: Sequence[str],
    supported_target_labels: set[str],
    augment_missing_only: bool = True,
    default_target_label: str | None = None,
) -> tuple[list[_GenerationRequest], dict[str, int]]:
    requests: list[_GenerationRequest] = []
    skipped = {
        "existing_counterfactual": 0,
        "no_target_label": 0,
        "unsupported_target_label": 0,
    }

    for index, record in enumerate(records):
        has_counterfactual = bool(record.counterfactual_text)
        if has_counterfactual and augment_missing_only:
            skipped["existing_counterfactual"] += 1
            continue

        target_label = record.counterfactual_label or infer_target_label(
            record.label,
            labels=labels,
            default_target_label=default_target_label,
        )
        if target_label is None:
            skipped["no_target_label"] += 1
            continue
        if target_label not in supported_target_labels:
            skipped["unsupported_target_label"] += 1
            continue

        source_aspect = infer_review_aspect(record.text)
        request_record = CounterfactualReviewRecord(
            text=record.text,
            label=record.label,
            counterfactual_text=record.counterfactual_text or "",
            counterfactual_label=target_label,
            source_aspect=source_aspect,
            target_aspect=source_aspect,
            time_value=record.timestamp,
            counterfactual_time_value=(
                record.counterfactual_timestamp
                if record.counterfactual_timestamp is not None
                else record.timestamp
            ),
            metadata={
                "record_index": index,
                "merge_strategy": "append" if has_counterfactual and not augment_missing_only else "backfill",
                "source_aspect": source_aspect,
                "source_has_counterfactual": has_counterfactual,
            },
        )
        requests.append(
            _GenerationRequest(
                record_index=index,
                request_record=request_record,
                merge_strategy=request_record.metadata["merge_strategy"],
            )
        )
    return requests, skipped


class CounterfactualReviewDataset(Dataset):
    def __init__(
        self,
        records: list[CounterfactualReviewRecord],
        vocab: Vocabulary,
        label_to_index: dict[str, int],
        aspect_to_index: dict[str, int],
        tokenizer=default_tokenizer,
        max_source_len: int = 128,
        max_target_len: int = 128,
    ) -> None:
        if not records:
            raise ValueError("CounterfactualReviewDataset requires at least one paired record.")
        self.records = records
        self.vocab = vocab
        self.label_to_index = label_to_index
        self.aspect_to_index = aspect_to_index
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.bos_idx = vocab.stoi["<bos>"]
        self.eos_idx = vocab.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.records)

    def encode_text(self, text: str, max_len: int) -> list[int]:
        token_ids = self.vocab.encode(self.tokenizer(text))[:max_len]
        if not token_ids:
            token_ids = [self.vocab.unk_idx]
        return token_ids

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        source_ids = self.encode_text(record.text, self.max_source_len)
        target_ids = self.encode_text(record.counterfactual_text, self.max_target_len - 1)
        return {
            "source_input_ids": source_ids,
            "target_input_ids": target_ids + [self.eos_idx],
            "decoder_input_ids": [self.bos_idx] + target_ids,
            "source_label": self.label_to_index[record.label],
            "target_label": self.label_to_index[record.counterfactual_label],
            "source_aspect": self.aspect_to_index.get(record.source_aspect or "generic", 0),
            "target_aspect": self.aspect_to_index.get(record.target_aspect or "generic", 0),
            "time_value": 0.0 if record.time_value is None else float(record.time_value),
            "counterfactual_time_value": (
                0.0 if record.counterfactual_time_value is None else float(record.counterfactual_time_value)
            ),
            "source_text": record.text,
            "target_text": record.counterfactual_text,
            "metadata": record.metadata or {},
        }


def collate_counterfactual_reviews(
    batch: list[dict[str, Any]],
    pad_idx: int,
) -> dict[str, Tensor | list[str] | list[dict[str, Any]]]:
    source_input_ids, source_attention_mask, source_lengths = pad_sequences(
        [item["source_input_ids"] for item in batch],
        pad_idx=pad_idx,
    )
    target_input_ids, target_attention_mask, _ = pad_sequences(
        [item["target_input_ids"] for item in batch],
        pad_idx=pad_idx,
    )
    decoder_input_ids, _, _ = pad_sequences(
        [item["decoder_input_ids"] for item in batch],
        pad_idx=pad_idx,
    )
    return {
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "source_lengths": source_lengths,
        "target_input_ids": target_input_ids,
        "target_attention_mask": target_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "source_labels": torch.tensor([item["source_label"] for item in batch], dtype=torch.long),
        "target_labels": torch.tensor([item["target_label"] for item in batch], dtype=torch.long),
        "source_aspects": torch.tensor([item["source_aspect"] for item in batch], dtype=torch.long),
        "target_aspects": torch.tensor([item["target_aspect"] for item in batch], dtype=torch.long),
        "time_values": torch.tensor([item["time_value"] for item in batch], dtype=torch.float),
        "counterfactual_time_values": torch.tensor(
            [item["counterfactual_time_value"] for item in batch],
            dtype=torch.float,
        ),
        "source_texts": [item["source_text"] for item in batch],
        "target_texts": [item["target_text"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }


class ReviewConditionEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        label_embed_dim: int,
        aspect_embed_dim: int,
        time_embed_dim: int,
        noise_dim: int,
    ) -> None:
        super().__init__()
        self.time_projection = nn.Sequential(
            nn.Linear(2, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.condition_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + label_embed_dim + aspect_embed_dim + time_embed_dim + noise_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        source_summary: Tensor,
        label_embedding: Tensor,
        aspect_embedding: Tensor,
        time_values: Tensor,
        counterfactual_time_values: Tensor,
        noise: Tensor,
    ) -> Tensor:
        temporal_features = self.time_projection(
            torch.stack([time_values, counterfactual_time_values], dim=-1)
        )
        condition_input = torch.cat(
            [source_summary, label_embedding, aspect_embedding, temporal_features, noise],
            dim=-1,
        )
        return self.condition_projection(condition_input)


class ReviewCounterfactualGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        num_aspects: int,
        config: GANTrainingConfig,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.noise_dim = config.noise_dim
        self.gumbel_temperature = config.gumbel_temperature
        self.token_embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=pad_idx)
        self.label_embedding = nn.Embedding(num_labels, config.label_embed_dim)
        self.aspect_embedding = nn.Embedding(num_aspects, config.aspect_embed_dim)
        self.source_encoder = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.condition_encoder = ReviewConditionEncoder(
            hidden_dim=config.hidden_dim,
            label_embed_dim=config.label_embed_dim,
            aspect_embed_dim=config.aspect_embed_dim,
            time_embed_dim=config.time_embed_dim,
            noise_dim=config.noise_dim,
        )
        self.edit_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        decoder_input_dim = config.embed_dim + config.hidden_dim * 2 + config.label_embed_dim + config.aspect_embed_dim
        self.decoder = nn.GRU(
            input_size=decoder_input_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_projection = nn.Linear(config.hidden_dim, vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def sample_noise(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randn(batch_size, self.noise_dim, device=device)

    def encode_source(
        self,
        source_input_ids: Tensor,
        source_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source_embeddings = self.token_embedding(source_input_ids)
        packed = pack_padded_sequence(
            source_embeddings,
            lengths=source_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, hidden = self.source_encoder(packed)
        encoded_sequence, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=source_input_ids.size(1),
        )
        source_summary = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return encoded_sequence, source_summary

    def forward(
        self,
        source_input_ids: Tensor,
        source_attention_mask: Tensor,
        source_lengths: Tensor,
        decoder_input_ids: Tensor,
        target_labels: Tensor,
        target_aspects: Tensor,
        time_values: Tensor,
        counterfactual_time_values: Tensor,
        noise: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if noise is None:
            noise = self.sample_noise(source_input_ids.size(0), source_input_ids.device)

        source_sequence, source_summary = self.encode_source(
            source_input_ids=source_input_ids,
            source_lengths=source_lengths,
        )
        label_context = self.label_embedding(target_labels)
        aspect_context = self.aspect_embedding(target_aspects)
        condition_state = self.condition_encoder(
            source_summary=source_summary,
            label_embedding=label_context,
            aspect_embedding=aspect_context,
            time_values=time_values,
            counterfactual_time_values=counterfactual_time_values,
            noise=noise,
        )

        edit_features = torch.cat(
            [
                source_sequence,
                condition_state.unsqueeze(1).expand(-1, source_sequence.size(1), -1),
            ],
            dim=-1,
        )
        edit_logits = self.edit_gate(edit_features).squeeze(-1)
        edit_probabilities = torch.sigmoid(edit_logits) * source_attention_mask.float()

        decoder_embeddings = self.token_embedding(decoder_input_ids)
        pooled_source = (
            source_sequence * source_attention_mask.unsqueeze(-1).float()
        ).sum(dim=1) / source_attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        expanded_source = pooled_source.unsqueeze(1).expand(-1, decoder_input_ids.size(1), -1)
        expanded_label = label_context.unsqueeze(1).expand(-1, decoder_input_ids.size(1), -1)
        expanded_aspect = aspect_context.unsqueeze(1).expand(-1, decoder_input_ids.size(1), -1)
        decoder_inputs = torch.cat(
            [decoder_embeddings, expanded_source, expanded_label, expanded_aspect],
            dim=-1,
        )
        decoder_output, _ = self.decoder(decoder_inputs, condition_state.unsqueeze(0))
        decoder_output = self.dropout(decoder_output)
        token_logits = self.output_projection(decoder_output)
        sampled_token_distributions = nn.functional.gumbel_softmax(
            token_logits,
            tau=self.gumbel_temperature,
            hard=True,
            dim=-1,
        )
        sampled_token_ids = sampled_token_distributions.argmax(dim=-1)
        sampled_token_embeddings = sampled_token_distributions @ self.token_embedding.weight
        return {
            "token_logits": token_logits,
            "edit_logits": edit_logits,
            "edit_probabilities": edit_probabilities,
            "source_summary": source_summary,
            "condition_state": condition_state,
            "sampled_token_ids": sampled_token_ids,
            "sampled_token_embeddings": sampled_token_embeddings,
        }

    @torch.no_grad()
    def greedy_generate(
        self,
        source_input_ids: Tensor,
        source_attention_mask: Tensor,
        source_lengths: Tensor,
        target_labels: Tensor,
        target_aspects: Tensor,
        time_values: Tensor,
        counterfactual_time_values: Tensor,
        bos_idx: int,
        eos_idx: int,
        max_gen_len: int,
    ) -> Tensor:
        batch_size = source_input_ids.size(0)
        generated = torch.full(
            (batch_size, 1),
            bos_idx,
            dtype=torch.long,
            device=source_input_ids.device,
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=source_input_ids.device)
        for _ in range(max_gen_len):
            output = self.forward(
                source_input_ids=source_input_ids,
                source_attention_mask=source_attention_mask,
                source_lengths=source_lengths,
                decoder_input_ids=generated,
                target_labels=target_labels,
                target_aspects=target_aspects,
                time_values=time_values,
                counterfactual_time_values=counterfactual_time_values,
            )
            next_token = output["token_logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            if finished.all():
                break
        return generated[:, 1:]


class ReviewCounterfactualDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        num_aspects: int,
        config: GANTrainingConfig,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config.embed_dim, padding_idx=pad_idx)
        self.review_encoder = nn.GRU(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        feature_dim = config.hidden_dim * 4 + config.label_embed_dim + config.aspect_embed_dim
        self.label_embedding = nn.Embedding(num_labels, config.label_embed_dim)
        self.aspect_embedding = nn.Embedding(num_aspects, config.aspect_embed_dim)
        self.real_fake_head = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.label_alignment_head = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, num_labels),
        )
        self.aspect_alignment_head = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, num_aspects),
        )

    def encode_review(
        self,
        attention_mask: Tensor,
        input_ids: Tensor | None = None,
        token_embeddings: Tensor | None = None,
    ) -> Tensor:
        if token_embeddings is None:
            if input_ids is None:
                raise ValueError("Either input_ids or token_embeddings must be provided.")
            embeddings = self.token_embedding(input_ids)
        else:
            embeddings = token_embeddings

        lengths = attention_mask.sum(dim=1).clamp_min(1)
        packed = pack_padded_sequence(
            embeddings,
            lengths=lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.review_encoder(packed)
        return torch.cat([hidden[-2], hidden[-1]], dim=-1)

    def forward(
        self,
        source_input_ids: Tensor,
        source_attention_mask: Tensor,
        candidate_attention_mask: Tensor,
        target_labels: Tensor,
        target_aspects: Tensor,
        candidate_input_ids: Tensor | None = None,
        candidate_embeddings: Tensor | None = None,
    ) -> dict[str, Tensor]:
        source_summary = self.encode_review(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
        )
        candidate_summary = self.encode_review(
            input_ids=candidate_input_ids,
            token_embeddings=candidate_embeddings,
            attention_mask=candidate_attention_mask,
        )
        label_context = self.label_embedding(target_labels)
        aspect_context = self.aspect_embedding(target_aspects)
        features = torch.cat(
            [source_summary, candidate_summary, label_context, aspect_context],
            dim=-1,
        )
        return {
            "real_fake_logits": self.real_fake_head(features).squeeze(-1),
            "label_logits": self.label_alignment_head(features),
            "aspect_logits": self.aspect_alignment_head(features),
            "source_summary": source_summary,
            "candidate_summary": candidate_summary,
        }


class CounterfactualReviewGAN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        num_aspects: int,
        config: GANTrainingConfig,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.generator = ReviewCounterfactualGenerator(
            vocab_size=vocab_size,
            num_labels=num_labels,
            num_aspects=num_aspects,
            config=config,
            pad_idx=pad_idx,
        )
        self.discriminator = ReviewCounterfactualDiscriminator(
            vocab_size=vocab_size,
            num_labels=num_labels,
            num_aspects=num_aspects,
            config=config,
            pad_idx=pad_idx,
        )
        self.config = config
        self.pad_idx = pad_idx


class ReviewGANTrainer:
    def __init__(
        self,
        gan: CounterfactualReviewGAN,
        vocab: Vocabulary,
        config: GANTrainingConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.gan = gan
        self.vocab = vocab
        self.config = config
        self.device = torch.device(device)
        self.gan.to(self.device)

        self.generator_optimizer = torch.optim.AdamW(
            self.gan.generator.parameters(),
            lr=config.learning_rate_generator,
        )
        self.discriminator_optimizer = torch.optim.AdamW(
            self.gan.discriminator.parameters(),
            lr=config.learning_rate_discriminator,
        )
        self.reconstruction_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.aux_label_criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.aux_aspect_criterion = nn.CrossEntropyLoss()

    def move_batch_to_device(self, batch: dict[str, Tensor | list[str] | list[dict[str, Any]]]):
        output: dict[str, Tensor | list[str] | list[dict[str, Any]]] = {}
        for key, value in batch.items():
            output[key] = value.to(self.device) if isinstance(value, Tensor) else value
        return output

    def semantic_anchor_loss(self, source_summary: Tensor, candidate_summary: Tensor) -> Tensor:
        source_summary = nn.functional.normalize(source_summary, dim=-1)
        candidate_summary = nn.functional.normalize(candidate_summary, dim=-1)
        return 1.0 - (source_summary * candidate_summary).sum(dim=-1).mean()

    def edit_locality_loss(self, edit_probabilities: Tensor, source_attention_mask: Tensor) -> Tensor:
        valid_steps = source_attention_mask.sum(dim=-1).clamp_min(1).float()
        edit_mass = edit_probabilities.sum(dim=-1) / valid_steps
        return edit_mass.mean()

    def train_discriminator_step(self, batch: dict[str, Tensor | list[str] | list[dict[str, Any]]]) -> dict[str, Tensor]:
        batch = self.move_batch_to_device(batch)
        generator_output = self.gan.generator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_lengths=batch["source_lengths"],
            decoder_input_ids=batch["decoder_input_ids"],
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
            time_values=batch["time_values"],
            counterfactual_time_values=batch["counterfactual_time_values"],
        )
        _, fake_target_attention_mask = sanitize_generated_sequences(
            generator_output["sampled_token_ids"].detach(),
            eos_idx=self.vocab.stoi["<eos>"],
            pad_idx=self.vocab.pad_idx,
        )

        real_output = self.gan.discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            candidate_input_ids=batch["target_input_ids"],
            candidate_attention_mask=batch["target_attention_mask"],
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
        )
        fake_output = self.gan.discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            candidate_embeddings=generator_output["sampled_token_embeddings"].detach(),
            candidate_attention_mask=fake_target_attention_mask,
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
        )

        if self.config.use_wasserstein:
            adversarial_loss = fake_output["real_fake_logits"].mean() - real_output["real_fake_logits"].mean()
        else:
            real_targets = torch.ones_like(real_output["real_fake_logits"])
            fake_targets = torch.zeros_like(fake_output["real_fake_logits"])
            adversarial_loss = 0.5 * (
                self.adversarial_criterion(real_output["real_fake_logits"], real_targets)
                + self.adversarial_criterion(fake_output["real_fake_logits"], fake_targets)
            )

        label_loss = self.aux_label_criterion(real_output["label_logits"], batch["target_labels"])
        aspect_loss = self.aux_aspect_criterion(real_output["aspect_logits"], batch["target_aspects"])
        discriminator_loss = (
            adversarial_loss
            + self.config.label_weight * label_loss
            + self.config.aspect_weight * aspect_loss
        )

        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        nn.utils.clip_grad_norm_(self.gan.discriminator.parameters(), self.config.gradient_clip_norm)
        self.discriminator_optimizer.step()

        return {
            "discriminator_loss": discriminator_loss.detach(),
            "discriminator_adv_loss": adversarial_loss.detach(),
            "discriminator_label_loss": label_loss.detach(),
            "discriminator_aspect_loss": aspect_loss.detach(),
            "fake_token_density": fake_target_attention_mask.float().mean().detach(),
        }

    def train_generator_step(self, batch: dict[str, Tensor | list[str] | list[dict[str, Any]]]) -> dict[str, Tensor]:
        batch = self.move_batch_to_device(batch)
        generator_output = self.gan.generator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_lengths=batch["source_lengths"],
            decoder_input_ids=batch["decoder_input_ids"],
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
            time_values=batch["time_values"],
            counterfactual_time_values=batch["counterfactual_time_values"],
        )
        generated_ids, generated_attention_mask = sanitize_generated_sequences(
            generator_output["sampled_token_ids"],
            eos_idx=self.vocab.stoi["<eos>"],
            pad_idx=self.vocab.pad_idx,
        )
        discriminator_output = self.gan.discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            candidate_embeddings=generator_output["sampled_token_embeddings"],
            candidate_attention_mask=generated_attention_mask,
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
        )

        if self.config.use_wasserstein:
            adversarial_loss = -discriminator_output["real_fake_logits"].mean()
        else:
            adversarial_loss = self.adversarial_criterion(
                discriminator_output["real_fake_logits"],
                torch.ones_like(discriminator_output["real_fake_logits"]),
            )

        reconstruction_loss = self.reconstruction_criterion(
            generator_output["token_logits"].reshape(-1, generator_output["token_logits"].size(-1)),
            batch["target_input_ids"].reshape(-1),
        )
        label_loss = self.aux_label_criterion(
            discriminator_output["label_logits"],
            batch["target_labels"],
        )
        aspect_loss = self.aux_aspect_criterion(
            discriminator_output["aspect_logits"],
            batch["target_aspects"],
        )
        semantic_loss = self.semantic_anchor_loss(
            discriminator_output["source_summary"],
            discriminator_output["candidate_summary"],
        )
        edit_loss = self.edit_locality_loss(
            generator_output["edit_probabilities"],
            batch["source_attention_mask"],
        )
        generator_loss = (
            self.config.reconstruction_weight * reconstruction_loss
            + self.config.adversarial_weight * adversarial_loss
            + self.config.label_weight * label_loss
            + self.config.aspect_weight * aspect_loss
            + self.config.semantic_weight * semantic_loss
            + self.config.edit_weight * edit_loss
        )

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        nn.utils.clip_grad_norm_(self.gan.generator.parameters(), self.config.gradient_clip_norm)
        self.generator_optimizer.step()

        return {
            "generator_loss": generator_loss.detach(),
            "generator_adv_loss": adversarial_loss.detach(),
            "generator_reconstruction_loss": reconstruction_loss.detach(),
            "generator_label_loss": label_loss.detach(),
            "generator_aspect_loss": aspect_loss.detach(),
            "generator_semantic_loss": semantic_loss.detach(),
            "generator_edit_loss": edit_loss.detach(),
            "generated_length": generated_attention_mask.sum(dim=1).float().mean().detach(),
            "pad_ratio": (generated_ids == self.vocab.pad_idx).float().mean().detach(),
        }

    def train_epoch(self, data_loader: DataLoader) -> dict[str, float]:
        metric_names = [
            "discriminator_loss",
            "discriminator_adv_loss",
            "discriminator_label_loss",
            "discriminator_aspect_loss",
            "generator_loss",
            "generator_adv_loss",
            "generator_reconstruction_loss",
            "generator_label_loss",
            "generator_aspect_loss",
            "generator_semantic_loss",
            "generator_edit_loss",
            "generated_length",
            "fake_token_density",
            "pad_ratio",
        ]
        totals = {name: 0.0 for name in metric_names}

        self.gan.train()
        for batch in data_loader:
            discriminator_metrics = self.train_discriminator_step(batch)
            generator_metrics = self.train_generator_step(batch)
            for name, value in {**discriminator_metrics, **generator_metrics}.items():
                totals[name] += float(value.item())

        num_batches = max(len(data_loader), 1)
        return {name: value / num_batches for name, value in totals.items()}

    @torch.no_grad()
    def generate_counterfactuals(self, batch: dict[str, Tensor | list[str] | list[dict[str, Any]]]) -> Tensor:
        batch = self.move_batch_to_device(batch)
        return self.gan.generator.greedy_generate(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            source_lengths=batch["source_lengths"],
            target_labels=batch["target_labels"],
            target_aspects=batch["target_aspects"],
            time_values=batch["time_values"],
            counterfactual_time_values=batch["counterfactual_time_values"],
            bos_idx=self.vocab.stoi["<bos>"],
            eos_idx=self.vocab.stoi["<eos>"],
            max_gen_len=self.config.max_target_len,
        )

    @torch.no_grad()
    def export_stage2_pairs(
        self,
        records: list[CounterfactualReviewRecord],
        tokenizer=default_tokenizer,
    ) -> list[dict[str, Any]]:
        label_to_index = build_label_mapping(records)
        aspect_to_index = build_aspect_mapping(records)
        dataset = CounterfactualReviewDataset(
            records=records,
            vocab=self.vocab,
            label_to_index=label_to_index,
            aspect_to_index=aspect_to_index,
            tokenizer=tokenizer,
            max_source_len=self.config.max_source_len,
            max_target_len=self.config.max_target_len,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_counterfactual_reviews(batch, self.vocab.pad_idx),
        )
        index_to_label = {index: label for label, index in label_to_index.items()}
        index_to_aspect = {index: aspect for aspect, index in aspect_to_index.items()}

        exported: list[dict[str, Any]] = []
        for batch in loader:
            generated_ids = self.generate_counterfactuals(batch)
            generated_ids, _ = sanitize_generated_sequences(
                generated_ids,
                eos_idx=self.vocab.stoi["<eos>"],
                pad_idx=self.vocab.pad_idx,
            )
            for source_text, source_label, target_label, target_aspect, metadata, token_ids in zip(
                batch["source_texts"],
                batch["source_labels"].tolist(),
                batch["target_labels"].tolist(),
                batch["target_aspects"].tolist(),
                batch["metadata"],
                generated_ids.cpu().tolist(),
            ):
                if self.vocab.stoi["<eos>"] in token_ids:
                    token_ids = token_ids[: token_ids.index(self.vocab.stoi["<eos>"])]
                generated_tokens = self.vocab.decode(token_ids, skip_special_tokens=True)
                generated_text = detokenize_tokens(generated_tokens)
                if not generated_text.strip():
                    generated_text = source_text
                exported.append(
                    {
                        "record_index": metadata.get("record_index", -1),
                        "merge_strategy": metadata.get("merge_strategy", "backfill"),
                        "text": source_text,
                        "label": index_to_label[source_label],
                        "counterfactual_text": generated_text,
                        "counterfactual_label": index_to_label[target_label],
                        "source_aspect": metadata.get("source_aspect", infer_review_aspect(source_text)),
                        "target_aspect": index_to_aspect[target_aspect],
                        "synthetic_source": "review_gan_stage1",
                    }
                )
        return exported


def save_generated_rows_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=GENERATED_FIELDNAMES)
        writer.writeheader()
        if rows:
            writer.writerows(
                {field: row.get(field, "") for field in GENERATED_FIELDNAMES}
                for row in rows
            )


def save_gan_artifacts(
    output_dir: Path,
    trainer: ReviewGANTrainer,
    config: GANTrainingConfig,
    label_to_index: dict[str, int],
    aspect_to_index: dict[str, int],
    history: list[dict[str, float | int]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_path = output_dir / "counterfactual_generator.pt"
    discriminator_path = output_dir / "counterfactual_discriminator.pt"
    vocab_path = output_dir / "counterfactual_vocab.json"
    labels_path = output_dir / "counterfactual_labels.json"
    aspects_path = output_dir / "counterfactual_aspects.json"
    config_path = output_dir / "gan_config.json"
    history_path = output_dir / "history.json"

    torch.save(trainer.gan.generator.state_dict(), generator_path)
    torch.save(trainer.gan.discriminator.state_dict(), discriminator_path)
    with vocab_path.open("w", encoding="utf-8") as file:
        json.dump(trainer.vocab.to_dict(), file, ensure_ascii=False, indent=2)
    with labels_path.open("w", encoding="utf-8") as file:
        json.dump(label_to_index, file, ensure_ascii=False, indent=2)
    with aspects_path.open("w", encoding="utf-8") as file:
        json.dump(aspect_to_index, file, ensure_ascii=False, indent=2)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, ensure_ascii=False, indent=2)
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    return {
        "generator": str(generator_path),
        "discriminator": str(discriminator_path),
        "vocab": str(vocab_path),
        "labels": str(labels_path),
        "aspects": str(aspects_path),
        "config": str(config_path),
        "history": str(history_path),
    }


def merge_generated_counterfactuals(
    records: Sequence[TextRecord],
    generated_rows: Sequence[dict[str, Any]],
) -> tuple[list[TextRecord], dict[str, int]]:
    augmented_records = [replace(record) for record in records]
    appended_records: list[TextRecord] = []
    backfilled = 0
    appended = 0

    for row in generated_rows:
        record_index = int(row["record_index"])
        generated_text = str(row["counterfactual_text"]).strip()
        if not generated_text:
            continue

        target_record = augmented_records[record_index]
        if row.get("merge_strategy") == "append":
            appended_records.append(
                replace(
                    target_record,
                    counterfactual_text=generated_text,
                    counterfactual_label=str(row["counterfactual_label"]),
                    counterfactual_timestamp=target_record.counterfactual_timestamp or target_record.timestamp,
                )
            )
            appended += 1
            continue

        if not target_record.counterfactual_text:
            augmented_records[record_index] = replace(
                target_record,
                counterfactual_text=generated_text,
                counterfactual_label=str(row["counterfactual_label"]),
                counterfactual_timestamp=target_record.counterfactual_timestamp or target_record.timestamp,
            )
            backfilled += 1

    augmented_records.extend(appended_records)
    return augmented_records, {"backfilled": backfilled, "appended": appended}


def train_gan_and_augment_records(
    records: Sequence[TextRecord],
    output_dir: str | Path,
    config: GANTrainingConfig,
    epochs: int = 10,
    seed: int = 42,
    device: torch.device | str | None = None,
    augment_missing_only: bool = True,
    tokenizer=default_tokenizer,
    default_target_label: str | None = None,
    verbose: bool = True,
) -> GANAugmentationResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_labels = sorted(
        {record.label for record in records}
        | {record.counterfactual_label for record in records if record.counterfactual_label}
    )
    paired_records = extract_training_pairs(records)
    generated_csv_path = output_dir / "generated_counterfactuals.csv"
    metrics_json_path = output_dir / "metrics.json"
    training_history: list[dict[str, float | int]] = []
    warning: str | None = None

    if not paired_records:
        warning = "No paired counterfactual examples found. Skipping GAN augmentation."
        save_generated_rows_csv(generated_csv_path, [])
        summary = {
            "enabled": True,
            "trained": False,
            "paired_records": 0,
            "generated_rows": 0,
            "backfilled_records": 0,
            "appended_records": 0,
            "augment_missing_only": augment_missing_only,
            "warning": warning,
        }
        with metrics_json_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        return GANAugmentationResult(
            augmented_records=[replace(record) for record in records],
            generated_rows=[],
            training_history=[],
            summary=summary,
            artifacts={
                "generated_rows": str(generated_csv_path),
                "metrics": str(metrics_json_path),
            },
            warning=warning,
        )

    set_seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    vocab = build_gan_vocab(paired_records, tokenizer=tokenizer)
    label_to_index = {label: index for index, label in enumerate(all_labels)}
    aspect_to_index = build_aspect_mapping(paired_records)

    dataset = CounterfactualReviewDataset(
        records=paired_records,
        vocab=vocab,
        label_to_index=label_to_index,
        aspect_to_index=aspect_to_index,
        tokenizer=tokenizer,
        max_source_len=config.max_source_len,
        max_target_len=config.max_target_len,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_counterfactual_reviews(batch, vocab.pad_idx),
    )
    gan = CounterfactualReviewGAN(
        vocab_size=len(vocab),
        num_labels=len(label_to_index),
        num_aspects=len(aspect_to_index),
        config=config,
        pad_idx=vocab.pad_idx,
    )
    trainer = ReviewGANTrainer(
        gan=gan,
        vocab=vocab,
        config=config,
        device=device,
    )

    for epoch in range(1, epochs + 1):
        metrics = trainer.train_epoch(data_loader)
        history_row: dict[str, float | int] = {"epoch": epoch}
        history_row.update(metrics)
        training_history.append(history_row)
        if verbose:
            print(
                f"[GAN] Epoch {epoch:03d}/{epochs:03d} | "
                f"g_loss={metrics['generator_loss']:.4f} | "
                f"d_loss={metrics['discriminator_loss']:.4f} | "
                f"recon={metrics['generator_reconstruction_loss']:.4f} | "
                f"edit={metrics['generator_edit_loss']:.4f}"
            )

    supported_target_labels = {record.counterfactual_label for record in paired_records if record.counterfactual_label}
    generation_requests, skipped_counts = build_generation_requests(
        records=records,
        labels=all_labels,
        supported_target_labels=supported_target_labels,
        augment_missing_only=augment_missing_only,
        default_target_label=default_target_label,
    )
    request_records = [request.request_record for request in generation_requests]
    generated_rows = trainer.export_stage2_pairs(request_records, tokenizer=tokenizer) if request_records else []
    augmented_records, merge_counts = merge_generated_counterfactuals(records, generated_rows)

    artifacts = save_gan_artifacts(
        output_dir=output_dir,
        trainer=trainer,
        config=config,
        label_to_index=label_to_index,
        aspect_to_index=aspect_to_index,
        history=training_history,
    )
    save_generated_rows_csv(generated_csv_path, generated_rows)

    summary = {
        "enabled": True,
        "trained": True,
        "device": str(device),
        "paired_records": len(paired_records),
        "training_rows": len(dataset),
        "generated_requests": len(generation_requests),
        "generated_rows": len(generated_rows),
        "backfilled_records": merge_counts["backfilled"],
        "appended_records": merge_counts["appended"],
        "augment_missing_only": augment_missing_only,
        "skipped_counts": skipped_counts,
    }
    with metrics_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "history": training_history,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    artifacts.update(
        {
            "generated_rows": str(generated_csv_path),
            "metrics": str(metrics_json_path),
        }
    )
    return GANAugmentationResult(
        augmented_records=augmented_records,
        generated_rows=generated_rows,
        training_history=training_history,
        summary=summary,
        artifacts=artifacts,
        warning=warning,
    )


__all__ = [
    "CounterfactualReviewRecord",
    "GANTrainingConfig",
    "GANAugmentationResult",
    "CounterfactualReviewGAN",
    "ReviewCounterfactualGenerator",
    "ReviewCounterfactualDiscriminator",
    "ReviewGANTrainer",
    "build_gan_vocab",
    "build_label_mapping",
    "build_aspect_mapping",
    "collate_counterfactual_reviews",
    "extract_training_pairs",
    "infer_review_aspect",
    "train_gan_and_augment_records",
]
