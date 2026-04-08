import csv
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from torch.utils.data import DataLoader, Dataset


def default_tokenizer(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if " " in text:
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return list(text)


@dataclass
class TextRecord:
    text: str
    label: str
    counterfactual_text: Optional[str] = None
    counterfactual_label: Optional[str] = None
    timestamp: Optional[float] = None
    counterfactual_timestamp: Optional[float] = None


class Vocabulary:
    def __init__(
        self,
        min_freq: int = 1,
        max_size: Optional[int] = None,
        specials: Optional[list[str]] = None,
    ) -> None:
        self.min_freq = min_freq
        self.max_size = max_size
        self.specials = specials or ["<pad>", "<unk>"]
        self.stoi: dict[str, int] = {}
        self.itos: list[str] = []

        for token in self.specials:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    @property
    def pad_idx(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_idx(self) -> int:
        return self.stoi["<unk>"]

    def __len__(self) -> int:
        return len(self.itos)

    def build(self, texts: Iterable[str], tokenizer: Callable[[str], list[str]]) -> None:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenizer(text))

        valid_items = [
            (token, freq)
            for token, freq in counter.items()
            if freq >= self.min_freq and token not in self.stoi
        ]
        valid_items.sort(key=lambda item: (-item[1], item[0]))

        if self.max_size is not None:
            remaining = max(self.max_size - len(self.itos), 0)
            valid_items = valid_items[:remaining]

        for token, _ in valid_items:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, self.unk_idx) for token in tokens]

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        special_set = set(self.specials)
        tokens: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self.itos):
                continue
            token = self.itos[token_id]
            if skip_special_tokens and token in special_set:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "specials": self.specials,
            "itos": self.itos,
        }


def parse_optional_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None
    return float(text)


def normalize_row_keys(row: dict[object, object]) -> dict[str, object]:
    return {str(key).lstrip("\ufeff"): value for key, value in row.items() if key is not None}


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        records: list[TextRecord],
        vocab: Vocabulary,
        label_to_index: dict[str, int],
        tokenizer: Callable[[str], list[str]],
        max_length: Optional[int] = None,
    ) -> None:
        self.records = records
        self.vocab = vocab
        self.label_to_index = label_to_index
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def _encode_text(self, text: str) -> list[int]:
        tokens = self.tokenizer(text)
        if self.max_length is not None and self.max_length > 0:
            tokens = tokens[: self.max_length]
        token_ids = self.vocab.encode(tokens)
        if not token_ids:
            token_ids = [self.vocab.unk_idx]
        return token_ids

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        input_ids = self._encode_text(record.text)
        example: dict[str, object] = {
            "input_ids": input_ids,
            "length": len(input_ids),
            "label": self.label_to_index[record.label],
            "text": record.text,
            "time_value": record.timestamp,
        }

        has_counterfactual = bool(record.counterfactual_text)
        example["counterfactual_available"] = has_counterfactual
        if has_counterfactual:
            counterfactual_label = record.counterfactual_label or record.label
            counterfactual_ids = self._encode_text(record.counterfactual_text or "")
            example.update(
                {
                    "counterfactual_input_ids": counterfactual_ids,
                    "counterfactual_length": len(counterfactual_ids),
                    "counterfactual_label": self.label_to_index[counterfactual_label],
                    "counterfactual_time_value": (
                        record.counterfactual_timestamp
                        if record.counterfactual_timestamp is not None
                        else record.timestamp
                    ),
                }
            )

        return example


def load_text_classification_records(
    file_path: str,
    text_column: str = "text",
    label_column: str = "label",
    delimiter: Optional[str] = None,
    encoding: str = "utf-8-sig",
    counterfactual_text_column: Optional[str] = None,
    counterfactual_label_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    counterfactual_timestamp_column: Optional[str] = None,
) -> list[TextRecord]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        with path.open("r", encoding=encoding, newline="") as file:
            current_delimiter = delimiter or ("\t" if suffix == ".tsv" else ",")
            reader = csv.DictReader(file, delimiter=current_delimiter)
            records = []
            for row in reader:
                normalized_row = normalize_row_keys(row)
                if not normalized_row.get(text_column) or normalized_row.get(label_column) is None:
                    continue

                records.append(
                    TextRecord(
                        text=str(normalized_row[text_column]),
                        label=str(normalized_row[label_column]),
                        counterfactual_text=(
                            str(normalized_row[counterfactual_text_column]).strip()
                            if counterfactual_text_column and normalized_row.get(counterfactual_text_column)
                            else None
                        ),
                        counterfactual_label=(
                            str(normalized_row[counterfactual_label_column]).strip()
                            if counterfactual_label_column and normalized_row.get(counterfactual_label_column)
                            else None
                        ),
                        timestamp=(
                            parse_optional_float(normalized_row.get(timestamp_column))
                            if timestamp_column
                            else None
                        ),
                        counterfactual_timestamp=(
                            parse_optional_float(normalized_row.get(counterfactual_timestamp_column))
                            if counterfactual_timestamp_column
                            else None
                        ),
                    )
                )
    else:
        with path.open("r", encoding=encoding) as file:
            records = []
            current_delimiter = delimiter or "\t"
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split(current_delimiter, maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Line {line_number} in {file_path} must be `label{current_delimiter}text` format."
                    )
                label, text = parts
                records.append(TextRecord(text=text, label=label))

    if not records:
        raise ValueError(f"No valid records found in {file_path}")
    return records


def build_label_mapping(*record_groups: list[TextRecord]) -> dict[str, int]:
    labels = set()
    for records in record_groups:
        for record in records:
            labels.add(record.label)
            if record.counterfactual_label is not None:
                labels.add(record.counterfactual_label)
    return {label: index for index, label in enumerate(sorted(labels))}


def pad_sequence_batch(
    sequences: list[list[int]],
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    batch_size = len(sequences)

    input_ids = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for row_index, sequence in enumerate(sequences):
        token_ids = torch.tensor(sequence, dtype=torch.long)
        seq_len = token_ids.size(0)
        input_ids[row_index, :seq_len] = token_ids
        attention_mask[row_index, :seq_len] = 1

    return input_ids, attention_mask, lengths


def collate_batch(batch: list[dict[str, object]], pad_idx: int) -> dict[str, torch.Tensor]:
    input_ids, attention_mask, lengths = pad_sequence_batch(
        [item["input_ids"] for item in batch],
        pad_idx=pad_idx,
    )
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    collated_batch: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "labels": labels,
    }

    if any(item.get("time_value") is not None for item in batch):
        collated_batch["time_values"] = torch.tensor(
            [0.0 if item.get("time_value") is None else float(item["time_value"]) for item in batch],
            dtype=torch.float,
        )

    counterfactual_available = torch.tensor(
        [bool(item.get("counterfactual_available")) for item in batch],
        dtype=torch.bool,
    )
    collated_batch["counterfactual_available"] = counterfactual_available

    if counterfactual_available.any():
        counterfactual_sequences = [
            item["counterfactual_input_ids"] if item.get("counterfactual_available") else [pad_idx]
            for item in batch
        ]
        cf_input_ids, cf_attention_mask, cf_lengths = pad_sequence_batch(
            counterfactual_sequences,
            pad_idx=pad_idx,
        )
        cf_labels = torch.tensor(
            [
                item["counterfactual_label"] if item.get("counterfactual_available") else item["label"]
                for item in batch
            ],
            dtype=torch.long,
        )

        collated_batch.update(
            {
                "counterfactual_input_ids": cf_input_ids,
                "counterfactual_attention_mask": cf_attention_mask,
                "counterfactual_lengths": cf_lengths,
                "counterfactual_labels": cf_labels,
            }
        )

        if any(item.get("counterfactual_time_value") is not None for item in batch):
            collated_batch["counterfactual_time_values"] = torch.tensor(
                [
                    0.0
                    if item.get("counterfactual_time_value") is None
                    else float(item["counterfactual_time_value"])
                    for item in batch
                ],
                dtype=torch.float,
            )

    return collated_batch


def iter_training_texts(train_records: list[TextRecord]) -> Iterable[str]:
    for record in train_records:
        yield record.text
        if record.counterfactual_text:
            yield record.counterfactual_text


def build_dataloaders(
    train_records: list[TextRecord],
    valid_records: Optional[list[TextRecord]] = None,
    test_records: Optional[list[TextRecord]] = None,
    batch_size: int = 32,
    min_freq: int = 1,
    max_vocab_size: Optional[int] = None,
    max_length: Optional[int] = None,
    tokenizer: Callable[[str], list[str]] = default_tokenizer,
    num_workers: int = 0,
) -> dict[str, object]:
    label_to_index = build_label_mapping(
        train_records,
        valid_records or [],
        test_records or [],
    )

    vocab = Vocabulary(min_freq=min_freq, max_size=max_vocab_size)
    vocab.build(iter_training_texts(train_records), tokenizer=tokenizer)

    train_dataset = TextClassificationDataset(
        records=train_records,
        vocab=vocab,
        label_to_index=label_to_index,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    valid_dataset = (
        TextClassificationDataset(
            records=valid_records,
            vocab=vocab,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if valid_records
        else None
    )
    test_dataset = (
        TextClassificationDataset(
            records=test_records,
            vocab=vocab,
            label_to_index=label_to_index,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        if test_records
        else None
    )

    def make_loader(dataset: Dataset, shuffle: bool) -> DataLoader:
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": partial(collate_batch, pad_idx=vocab.pad_idx),
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        return DataLoader(**loader_kwargs)

    return {
        "vocab": vocab,
        "label_to_index": label_to_index,
        "index_to_label": {index: label for label, index in label_to_index.items()},
        "train_loader": make_loader(train_dataset, shuffle=True),
        "valid_loader": make_loader(valid_dataset, shuffle=False) if valid_dataset else None,
        "test_loader": make_loader(test_dataset, shuffle=False) if test_dataset else None,
    }
