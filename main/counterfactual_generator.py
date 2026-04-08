import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.data import Vocabulary, load_text_classification_records


def default_tokenizer(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if " " in text:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return list(text)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def join_tokens(tokens: list[str], reference_text: str) -> str:
    if " " in reference_text:
        return " ".join(tokens)
    return "".join(tokens)


def infer_target_label(current_label: str, labels: list[str], default_target_label: Optional[str]) -> Optional[str]:
    if default_target_label is not None:
        return default_target_label
    remaining_labels = [label for label in labels if label != current_label]
    if len(remaining_labels) == 1:
        return remaining_labels[0]
    return None


class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim * 4)
        self.hidden_linear = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.input_linear(x_t) + self.hidden_linear(h_prev)
        input_gate, forget_gate, candidate_gate, output_gate = gates.chunk(4, dim=-1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        candidate_gate = torch.tanh(candidate_gate)
        output_gate = torch.sigmoid(output_gate)
        c_t = forget_gate * c_prev + input_gate * candidate_gate
        h_t = output_gate * torch.tanh(c_t)
        return h_t, c_t


class ConditionalCounterfactualGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 192,
        noise_dim: int = 64,
        label_embed_dim: int = 32,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.noise_dim = noise_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.label_embedding = nn.Embedding(num_labels, label_embed_dim)
        self.encoder_projection = nn.Linear(embed_dim + label_embed_dim + noise_dim, hidden_dim)
        self.decoder_cell = CustomLSTMCell(embed_dim + label_embed_dim + hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def encode_source(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_labels: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_embeddings = self.token_embedding(source_input_ids)
        pooled = (token_embeddings * source_attention_mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / source_attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        label_context = self.label_embedding(target_labels)
        if noise is None:
            noise = torch.randn(source_input_ids.size(0), self.noise_dim, device=source_input_ids.device)
        context = torch.cat([pooled, label_context, noise], dim=-1)
        return torch.tanh(self.encoder_projection(context))

    def forward(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_labels: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        context = self.encode_source(
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            target_labels=target_labels,
            noise=noise,
        )
        batch_size, target_len = decoder_input_ids.size()
        label_context = self.label_embedding(target_labels)

        hidden = context
        cell = torch.zeros_like(hidden)
        logits = []
        for time_step in range(target_len):
            token_embedding = self.token_embedding(decoder_input_ids[:, time_step])
            decoder_input = torch.cat([token_embedding, label_context, context], dim=-1)
            hidden, cell = self.decoder_cell(decoder_input, hidden, cell)
            hidden = self.dropout(hidden)
            logits.append(self.output_projection(hidden).unsqueeze(1))
        return torch.cat(logits, dim=1)

    def generate(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_labels: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_gen_len: int,
    ) -> torch.Tensor:
        context = self.encode_source(
            source_input_ids=source_input_ids,
            source_attention_mask=source_attention_mask,
            target_labels=target_labels,
        )
        label_context = self.label_embedding(target_labels)
        hidden = context
        cell = torch.zeros_like(hidden)

        current_tokens = torch.full(
            (source_input_ids.size(0),),
            bos_idx,
            dtype=torch.long,
            device=source_input_ids.device,
        )
        generated_tokens = []
        finished = torch.zeros(source_input_ids.size(0), dtype=torch.bool, device=source_input_ids.device)

        for _ in range(max_gen_len):
            token_embedding = self.token_embedding(current_tokens)
            decoder_input = torch.cat([token_embedding, label_context, context], dim=-1)
            hidden, cell = self.decoder_cell(decoder_input, hidden, cell)
            logits = self.output_projection(hidden)
            next_tokens = torch.argmax(logits, dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, eos_idx),
                next_tokens,
            )
            generated_tokens.append(next_tokens.unsqueeze(1))
            finished = finished | (next_tokens == eos_idx)
            current_tokens = next_tokens

        return torch.cat(generated_tokens, dim=1)


class CounterfactualDiscriminator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embed_dim: int = 128,
        hidden_dim: int = 192,
        label_embed_dim: int = 32,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.label_embedding = nn.Embedding(num_labels, label_embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2 + label_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def encode_sequence(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(input_ids)
        pooled = (embeddings * attention_mask.unsqueeze(-1).float()).sum(dim=1)
        pooled = pooled / attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return pooled

    def forward(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> torch.Tensor:
        source_representation = self.encode_sequence(source_input_ids, source_attention_mask)
        target_representation = self.encode_sequence(target_input_ids, target_attention_mask)
        label_representation = self.label_embedding(target_labels)
        features = torch.cat([source_representation, target_representation, label_representation], dim=-1)
        return self.classifier(features).squeeze(-1)


class CounterfactualPairDataset(Dataset):
    def __init__(
        self,
        records,
        vocab: Vocabulary,
        label_to_index: dict[str, int],
        tokenizer,
        max_source_len: int,
        max_target_len: int,
        bos_idx: int,
        eos_idx: int,
    ) -> None:
        self.examples = [
            record
            for record in records
            if record.counterfactual_text and (record.counterfactual_label or record.label)
        ]
        self.vocab = vocab
        self.label_to_index = label_to_index
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __len__(self) -> int:
        return len(self.examples)

    def encode_text(self, text: str, max_len: int) -> list[int]:
        token_ids = self.vocab.encode(self.tokenizer(text))[:max_len]
        if not token_ids:
            token_ids = [self.vocab.unk_idx]
        return token_ids

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.examples[index]
        source_ids = self.encode_text(record.text, self.max_source_len)
        target_ids = self.encode_text(record.counterfactual_text or "", self.max_target_len - 1)
        decoder_input_ids = [self.bos_idx] + target_ids
        decoder_target_ids = target_ids + [self.eos_idx]

        return {
            "source_input_ids": source_ids,
            "source_label": self.label_to_index[record.label],
            "target_label": self.label_to_index[record.counterfactual_label or record.label],
            "decoder_input_ids": decoder_input_ids,
            "decoder_target_ids": decoder_target_ids,
            "target_text": record.counterfactual_text or "",
            "source_text": record.text,
        }


def pad_sequences(sequences: list[list[int]], pad_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(sequence) for sequence in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_idx, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row_index, sequence in enumerate(sequences):
        seq_tensor = torch.tensor(sequence, dtype=torch.long)
        input_ids[row_index, : seq_tensor.size(0)] = seq_tensor
        attention_mask[row_index, : seq_tensor.size(0)] = 1
    return input_ids, attention_mask


def collate_pairs(batch: list[dict[str, object]], pad_idx: int) -> dict[str, torch.Tensor]:
    source_input_ids, source_attention_mask = pad_sequences(
        [item["source_input_ids"] for item in batch],
        pad_idx=pad_idx,
    )
    decoder_input_ids, _ = pad_sequences(
        [item["decoder_input_ids"] for item in batch],
        pad_idx=pad_idx,
    )
    decoder_target_ids, target_attention_mask = pad_sequences(
        [item["decoder_target_ids"] for item in batch],
        pad_idx=pad_idx,
    )

    return {
        "source_input_ids": source_input_ids,
        "source_attention_mask": source_attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_target_ids": decoder_target_ids,
        "target_input_ids": decoder_target_ids,
        "target_attention_mask": target_attention_mask,
        "source_labels": torch.tensor([item["source_label"] for item in batch], dtype=torch.long),
        "target_labels": torch.tensor([item["target_label"] for item in batch], dtype=torch.long),
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    generator: ConditionalCounterfactualGenerator,
    discriminator: CounterfactualDiscriminator,
    data_loader: DataLoader,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    reconstruction_criterion: nn.Module,
    adversarial_criterion: nn.Module,
    device: torch.device,
    pad_idx: int,
    adversarial_weight: float,
) -> dict[str, float]:
    generator.train()
    discriminator.train()

    total_generator_loss = 0.0
    total_discriminator_loss = 0.0
    total_reconstruction_loss = 0.0

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)

        generator_logits = generator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_labels=batch["target_labels"],
            decoder_input_ids=batch["decoder_input_ids"],
        )
        generated_ids = torch.argmax(generator_logits.detach(), dim=-1)
        generated_attention_mask = (generated_ids != pad_idx).long()

        real_scores = discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_input_ids=batch["target_input_ids"],
            target_attention_mask=batch["target_attention_mask"],
            target_labels=batch["target_labels"],
        )
        fake_scores = discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_input_ids=generated_ids,
            target_attention_mask=generated_attention_mask,
            target_labels=batch["target_labels"],
        )

        real_targets = torch.ones_like(real_scores)
        fake_targets = torch.zeros_like(fake_scores)
        discriminator_loss = (
            adversarial_criterion(real_scores, real_targets)
            + adversarial_criterion(fake_scores, fake_targets)
        ) * 0.5

        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        generator_logits = generator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_labels=batch["target_labels"],
            decoder_input_ids=batch["decoder_input_ids"],
        )
        reconstruction_loss = reconstruction_criterion(
            generator_logits.view(-1, generator_logits.size(-1)),
            batch["decoder_target_ids"].view(-1),
        )

        generated_ids = torch.argmax(generator_logits, dim=-1)
        generated_attention_mask = (generated_ids != pad_idx).long()
        generator_scores = discriminator(
            source_input_ids=batch["source_input_ids"],
            source_attention_mask=batch["source_attention_mask"],
            target_input_ids=generated_ids,
            target_attention_mask=generated_attention_mask,
            target_labels=batch["target_labels"],
        )
        generator_adversarial_loss = adversarial_criterion(
            generator_scores,
            torch.ones_like(generator_scores),
        )
        generator_loss = reconstruction_loss + adversarial_weight * generator_adversarial_loss

        generator_optimizer.zero_grad()
        generator_loss.backward()
        nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        generator_optimizer.step()

        total_generator_loss += generator_loss.item()
        total_discriminator_loss += discriminator_loss.item()
        total_reconstruction_loss += reconstruction_loss.item()

    num_batches = max(len(data_loader), 1)
    return {
        "generator_loss": total_generator_loss / num_batches,
        "discriminator_loss": total_discriminator_loss / num_batches,
        "reconstruction_loss": total_reconstruction_loss / num_batches,
    }


def build_generator_vocab(records, tokenizer) -> Vocabulary:
    vocab = Vocabulary(specials=["<pad>", "<unk>", "<bos>", "<eos>"])
    texts = []
    for record in records:
        texts.append(record.text)
        if record.counterfactual_text:
            texts.append(record.counterfactual_text)
    vocab.build(texts, tokenizer=tokenizer)
    return vocab


def save_generator_artifacts(
    save_dir: Path,
    generator: ConditionalCounterfactualGenerator,
    discriminator: CounterfactualDiscriminator,
    vocab: Vocabulary,
    label_to_index: dict[str, int],
    args: argparse.Namespace,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(generator.state_dict(), save_dir / "counterfactual_generator.pt")
    torch.save(discriminator.state_dict(), save_dir / "counterfactual_discriminator.pt")
    with (save_dir / "counterfactual_vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab.to_dict(), file, ensure_ascii=False, indent=2)
    with (save_dir / "counterfactual_labels.json").open("w", encoding="utf-8") as file:
        json.dump(label_to_index, file, ensure_ascii=False, indent=2)
    with (save_dir / "counterfactual_args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype adversarial counterfactual review generator for offline data augmentation."
    )
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--counterfactual_text_column", type=str, default="counterfactual_text")
    parser.add_argument("--counterfactual_label_column", type=str, default="counterfactual_label")
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "counterfactual_ckpt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--noise_dim", type=int, default=64)
    parser.add_argument("--label_embed_dim", type=int, default=32)
    parser.add_argument("--max_source_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--max_gen_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adversarial_weight", type=float, default=0.2)
    parser.add_argument("--num_generate", type=int, default=0, help="0 means generate for all eligible records.")
    parser.add_argument("--default_target_label", type=str, default=None)
    parser.add_argument("--augment_missing_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    records = load_text_classification_records(
        args.train_path,
        text_column=args.text_column,
        label_column=args.label_column,
        delimiter=args.delimiter,
        encoding=args.encoding,
        counterfactual_text_column=args.counterfactual_text_column,
        counterfactual_label_column=args.counterfactual_label_column,
    )

    paired_records = [record for record in records if record.counterfactual_text]
    if not paired_records:
        raise ValueError(
            "No paired counterfactual examples found. The prototype generator requires counterfactual_text data."
        )

    labels = sorted(
        {
            record.label
            for record in records
        }
        | {
            record.counterfactual_label
            for record in records
            if record.counterfactual_label is not None
        }
    )
    label_to_index = {label: index for index, label in enumerate(labels)}

    vocab = build_generator_vocab(records, tokenizer=default_tokenizer)
    bos_idx = vocab.stoi["<bos>"]
    eos_idx = vocab.stoi["<eos>"]

    dataset = CounterfactualPairDataset(
        records=records,
        vocab=vocab,
        label_to_index=label_to_index,
        tokenizer=default_tokenizer,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
    )
    if len(dataset) == 0:
        raise ValueError("No valid paired records available for generator training.")

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pairs(batch, pad_idx=vocab.pad_idx),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = ConditionalCounterfactualGenerator(
        vocab_size=len(vocab),
        num_labels=len(label_to_index),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        noise_dim=args.noise_dim,
        label_embed_dim=args.label_embed_dim,
        pad_idx=vocab.pad_idx,
    ).to(device)
    discriminator = CounterfactualDiscriminator(
        vocab_size=len(vocab),
        num_labels=len(label_to_index),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        label_embed_dim=args.label_embed_dim,
        pad_idx=vocab.pad_idx,
    ).to(device)

    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=args.lr)
    reconstruction_criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    adversarial_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(
            generator=generator,
            discriminator=discriminator,
            data_loader=data_loader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            reconstruction_criterion=reconstruction_criterion,
            adversarial_criterion=adversarial_criterion,
            device=device,
            pad_idx=vocab.pad_idx,
            adversarial_weight=args.adversarial_weight,
        )
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"generator_loss={metrics['generator_loss']:.4f} | "
            f"reconstruction_loss={metrics['reconstruction_loss']:.4f} | "
            f"discriminator_loss={metrics['discriminator_loss']:.4f}"
        )

    save_generator_artifacts(
        save_dir=Path(args.save_dir),
        generator=generator,
        discriminator=discriminator,
        vocab=vocab,
        label_to_index=label_to_index,
        args=args,
    )

    generation_records = []
    for record in records:
        if args.augment_missing_only and record.counterfactual_text:
            continue
        target_label = infer_target_label(record.label, labels, args.default_target_label)
        if target_label is None:
            continue
        generation_records.append((record, target_label))

    if args.num_generate > 0:
        generation_records = generation_records[: args.num_generate]

    output_rows = []
    generator.eval()
    with torch.no_grad():
        for record, target_label in generation_records:
            source_ids = vocab.encode(default_tokenizer(record.text))[: args.max_source_len]
            if not source_ids:
                source_ids = [vocab.unk_idx]

            source_input_ids, source_attention_mask = pad_sequences([source_ids], pad_idx=vocab.pad_idx)
            generated_ids = generator.generate(
                source_input_ids=source_input_ids.to(device),
                source_attention_mask=source_attention_mask.to(device),
                target_labels=torch.tensor([label_to_index[target_label]], dtype=torch.long, device=device),
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                max_gen_len=args.max_gen_len,
            )[0].cpu().tolist()

            if eos_idx in generated_ids:
                generated_ids = generated_ids[: generated_ids.index(eos_idx)]

            generated_tokens = vocab.decode(generated_ids, skip_special_tokens=True)
            generated_text = join_tokens(generated_tokens, record.text)
            output_rows.append(
                {
                    "text": record.text,
                    "label": record.label,
                    "generated_counterfactual_text": generated_text,
                    "generated_counterfactual_label": target_label,
                    "existing_counterfactual_text": record.counterfactual_text or "",
                    "existing_counterfactual_label": record.counterfactual_label or "",
                }
            )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()) if output_rows else [
            "text",
            "label",
            "generated_counterfactual_text",
            "generated_counterfactual_label",
            "existing_counterfactual_text",
            "existing_counterfactual_label",
        ])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved generated counterfactual reviews to: {output_path}")


if __name__ == "__main__":
    main()
