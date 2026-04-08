import argparse
import json
import random
import sys
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from core.data import build_dataloaders, load_text_classification_records
from core.model import HybridTextClassifier
from core.utils import ClassificationLoss, CounterfactualInterventionLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def compute_metrics(predictions: list[int], references: list[int]) -> dict[str, float]:
    accuracy = accuracy_score(references, predictions) if references else 0.0
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        references,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
    }


def forward_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    time_values: torch.Tensor | None,
) -> torch.Tensor:
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        lengths=lengths,
        time_values=time_values,
    )


def compute_counterfactual_losses(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    classification_criterion: nn.Module,
    intervention_criterion: CounterfactualInterventionLoss,
    counterfactual_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    available_mask = batch["counterfactual_available"]
    if not available_mask.any():
        zero = batch["labels"].new_zeros((), dtype=torch.float)
        return zero, zero

    time_values = batch.get("time_values")
    counterfactual_time_values = batch.get("counterfactual_time_values")

    factual_logits = forward_batch(
        model=model,
        input_ids=batch["input_ids"][available_mask],
        attention_mask=batch["attention_mask"][available_mask],
        lengths=batch["lengths"][available_mask],
        time_values=time_values[available_mask] if time_values is not None else None,
    )
    counterfactual_logits = forward_batch(
        model=model,
        input_ids=batch["counterfactual_input_ids"][available_mask],
        attention_mask=batch["counterfactual_attention_mask"][available_mask],
        lengths=batch["counterfactual_lengths"][available_mask],
        time_values=(
            counterfactual_time_values[available_mask]
            if counterfactual_time_values is not None
            else None
        ),
    )

    counterfactual_classification_loss = classification_criterion(
        counterfactual_logits,
        batch["counterfactual_labels"][available_mask],
    )
    intervention_loss = intervention_criterion(
        factual_logits=factual_logits,
        counterfactual_logits=counterfactual_logits,
        factual_labels=batch["labels"][available_mask],
        counterfactual_labels=batch["counterfactual_labels"][available_mask],
    )

    return counterfactual_weight * counterfactual_classification_loss, intervention_loss


def run_epoch(
    model: nn.Module,
    data_loader,
    classification_criterion: nn.Module,
    intervention_criterion: CounterfactualInterventionLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    counterfactual_weight: float,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_counterfactual_loss = 0.0
    total_intervention_loss = 0.0
    predictions: list[int] = []
    references: list[int] = []

    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        logits = forward_batch(
            model=model,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            lengths=batch["lengths"],
            time_values=batch.get("time_values"),
        )
        classification_loss = classification_criterion(logits, batch["labels"])
        counterfactual_classification_loss, intervention_loss = compute_counterfactual_losses(
            model=model,
            batch=batch,
            classification_criterion=classification_criterion,
            intervention_criterion=intervention_criterion,
            counterfactual_weight=counterfactual_weight,
        )
        loss = classification_loss + counterfactual_classification_loss + intervention_loss

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_counterfactual_loss += counterfactual_classification_loss.item()
        total_intervention_loss += intervention_loss.item()
        predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        references.extend(batch["labels"].cpu().tolist())

    metrics = compute_metrics(predictions, references)
    num_batches = max(len(data_loader), 1)
    metrics["loss"] = total_loss / num_batches
    metrics["counterfactual_loss"] = total_counterfactual_loss / num_batches
    metrics["intervention_loss"] = total_intervention_loss / num_batches
    return metrics


def format_metrics(split: str, metrics: dict[str, float]) -> str:
    return (
        f"{split}_loss={metrics['loss']:.4f} | "
        f"{split}_cf_loss={metrics['counterfactual_loss']:.4f} | "
        f"{split}_int_loss={metrics['intervention_loss']:.4f} | "
        f"{split}_acc={metrics['accuracy']:.4f} | "
        f"{split}_precision={metrics['macro_precision']:.4f} | "
        f"{split}_recall={metrics['macro_recall']:.4f} | "
        f"{split}_macro_f1={metrics['macro_f1']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SECI-Net with optional temporal encoding, block-sparse attention, and counterfactual losses."
    )
    parser.add_argument("--train_path", type=str, required=True, help="Path to train csv/tsv/txt file.")
    parser.add_argument("--valid_path", type=str, default=None, help="Path to validation file.")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test file.")
    parser.add_argument("--text_column", type=str, default="text", help="Text column name for csv/tsv.")
    parser.add_argument("--label_column", type=str, default="label", help="Label column name for csv/tsv.")
    parser.add_argument("--counterfactual_text_column", type=str, default=None)
    parser.add_argument("--counterfactual_label_column", type=str, default=None)
    parser.add_argument("--time_column", type=str, default=None)
    parser.add_argument("--counterfactual_time_column", type=str, default=None)
    parser.add_argument("--delimiter", type=str, default=None)
    parser.add_argument("--encoding", type=str, default="utf-8-sig")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ffn_hidden_dim", type=int, default=256)
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--attention_type",
        type=str,
        default="differential",
        choices=["standard", "differential", "block_sparse"],
    )
    parser.add_argument("--use_temporal_encoding", action="store_true")
    parser.add_argument("--temporal_hidden_dim", type=int, default=64)
    parser.add_argument("--differential_lambda_init", type=float, default=0.5)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=8)
    parser.add_argument("--topk_global_blocks", type=int, default=2)
    parser.add_argument("--counterfactual_weight", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--intervention_weight", type=float, default=0.5)
    parser.add_argument("--intervention_margin", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    return parser.parse_args()


def save_artifacts(
    model: HybridTextClassifier,
    output_dir: Path,
    vocab_info: dict[str, object],
    label_to_index: dict[str, int],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "best_model.pt")

    with (output_dir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump(vocab_info, file, ensure_ascii=False, indent=2)
    with (output_dir / "labels.json").open("w", encoding="utf-8") as file:
        json.dump(label_to_index, file, ensure_ascii=False, indent=2)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)


def load_records(file_path: str | None, args: argparse.Namespace):
    if file_path is None:
        return None
    return load_text_classification_records(
        file_path,
        text_column=args.text_column,
        label_column=args.label_column,
        delimiter=args.delimiter,
        encoding=args.encoding,
        counterfactual_text_column=args.counterfactual_text_column,
        counterfactual_label_column=args.counterfactual_label_column,
        timestamp_column=args.time_column,
        counterfactual_timestamp_column=args.counterfactual_time_column,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_records = load_records(args.train_path, args)
    valid_records = load_records(args.valid_path, args)
    test_records = load_records(args.test_path, args)

    data_bundle = build_dataloaders(
        train_records=train_records,
        valid_records=valid_records,
        test_records=test_records,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        num_workers=args.num_workers,
    )

    vocab = data_bundle["vocab"]
    label_to_index = data_bundle["label_to_index"]
    train_loader = data_bundle["train_loader"]
    valid_loader = data_bundle["valid_loader"]
    test_loader = data_bundle["test_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = HybridTextClassifier(
        vocab_size=len(vocab.itos),
        num_classes=len(label_to_index),
        embed_dim=args.embed_dim,
        transformer_layers=args.transformer_layers,
        num_heads=args.num_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=vocab.pad_idx,
        attention_type=args.attention_type,
        use_temporal_encoding=args.use_temporal_encoding,
        temporal_hidden_dim=args.temporal_hidden_dim,
        differential_lambda_init=args.differential_lambda_init,
        block_size=args.block_size,
        local_window_size=args.local_window_size,
        topk_global_blocks=args.topk_global_blocks,
    ).to(device)

    classification_criterion = ClassificationLoss(label_smoothing=args.label_smoothing)
    intervention_criterion = CounterfactualInterventionLoss(
        consistency_weight=args.consistency_weight,
        intervention_weight=args.intervention_weight,
        margin=args.intervention_margin,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            classification_criterion=classification_criterion,
            intervention_criterion=intervention_criterion,
            optimizer=optimizer,
            device=device,
            counterfactual_weight=args.counterfactual_weight,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{format_metrics('train', train_metrics)}"
        )

        if valid_loader is None:
            if train_metrics["macro_f1"] > best_score:
                best_score = train_metrics["macro_f1"]
                save_artifacts(model, output_dir, vocab.to_dict(), label_to_index, args)
            continue

        with torch.no_grad():
            valid_metrics = run_epoch(
                model=model,
                data_loader=valid_loader,
                classification_criterion=classification_criterion,
                intervention_criterion=intervention_criterion,
                optimizer=None,
                device=device,
                counterfactual_weight=args.counterfactual_weight,
            )
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | {format_metrics('valid', valid_metrics)}")

        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            save_artifacts(model, output_dir, vocab.to_dict(), label_to_index, args)

    if test_loader is not None:
        best_model_path = output_dir / "best_model.pt"
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))

        with torch.no_grad():
            test_metrics = run_epoch(
                model=model,
                data_loader=test_loader,
                classification_criterion=classification_criterion,
                intervention_criterion=intervention_criterion,
                optimizer=None,
                device=device,
                counterfactual_weight=args.counterfactual_weight,
            )
        print(f"Test | {format_metrics('test', test_metrics)}")

    print(f"Training finished. Best checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
