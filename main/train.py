import argparse
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from tqdm.auto import tqdm

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


def suggest_num_workers() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(8, cpu_count - 1)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
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
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    counterfactual_weight: float,
    split: str,
    epoch: int,
    total_epochs: int,
    use_amp: bool,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_counterfactual_loss = 0.0
    total_intervention_loss = 0.0
    total_correct = 0
    total_examples = 0
    predictions: list[int] = []
    references: list[int] = []

    progress_bar = tqdm(
        data_loader,
        total=len(data_loader),
        desc=f"{split.capitalize()} {epoch:03d}/{total_epochs:03d}",
        dynamic_ncols=True,
        leave=False,
    )

    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
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
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        total_counterfactual_loss += counterfactual_classification_loss.item()
        total_intervention_loss += intervention_loss.item()
        predicted_labels = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        total_correct += (predicted_labels == labels).sum().item()
        total_examples += labels.size(0)
        predictions.extend(predicted_labels.cpu().tolist())
        references.extend(labels.cpu().tolist())
        progress_bar.set_postfix(
            loss=f"{total_loss / step:.4f}",
            cf_loss=f"{total_counterfactual_loss / step:.4f}",
            int_loss=f"{total_intervention_loss / step:.4f}",
            acc=f"{(total_correct / max(total_examples, 1)):.4f}",
        )

    progress_bar.close()

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


def flatten_metrics(prefix: str, metrics: dict[str, float] | None) -> dict[str, float | str]:
    if metrics is None:
        return {
            f"{prefix}_loss": "",
            f"{prefix}_counterfactual_loss": "",
            f"{prefix}_intervention_loss": "",
            f"{prefix}_accuracy": "",
            f"{prefix}_macro_precision": "",
            f"{prefix}_macro_recall": "",
            f"{prefix}_macro_f1": "",
        }
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def save_metrics_history(output_dir: Path, history_rows: list[dict[str, float | int | str]]) -> None:
    if not history_rows:
        return

    csv_path = output_dir / "metrics_history.csv"
    json_path = output_dir / "metrics_history.json"
    fieldnames = list(history_rows[0].keys())

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(history_rows, file, ensure_ascii=False, indent=2)


def save_experiment_summary(
    output_dir: Path,
    args: argparse.Namespace,
    best_epoch: int | None,
    best_score: float,
    test_metrics: dict[str, float] | None,
    history_rows: list[dict[str, float | int | str]],
) -> None:
    summary = {
        "selection_metric": "valid_macro_f1" if any(row.get("valid_macro_f1", "") != "" for row in history_rows) else "train_macro_f1",
        "best_epoch": best_epoch,
        "best_score": None if best_score == float("-inf") else best_score,
        "test_metrics": test_metrics,
        "save_dir": str(output_dir),
        "train_args_path": str(output_dir / "train_args.json"),
        "best_model_path": str(output_dir / "best_model.pt"),
        "metrics_history_csv": str(output_dir / "metrics_history.csv"),
        "metrics_history_json": str(output_dir / "metrics_history.json"),
        "seed": args.seed,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


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
    parser.add_argument("--num_workers", type=int, default=suggest_num_workers())
    parser.add_argument("--amp", dest="use_amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.set_defaults(use_amp=torch.cuda.is_available())
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

    print("Loading dataset records...")
    train_records = load_records(args.train_path, args)
    valid_records = load_records(args.valid_path, args)
    test_records = load_records(args.test_path, args)
    print(
        f"Loaded records | train={len(train_records)} | "
        f"valid={len(valid_records) if valid_records is not None else 0} | "
        f"test={len(test_records) if test_records is not None else 0}"
    )

    print("Building vocabulary and dataloaders...")
    data_bundle = build_dataloaders(
        train_records=train_records,
        valid_records=valid_records,
        test_records=test_records,
        batch_size=args.batch_size,
        min_freq=args.min_freq,
        max_vocab_size=args.max_vocab_size,
        max_length=args.max_len,
        num_workers=args.num_workers,
    )

    vocab = data_bundle["vocab"]
    label_to_index = data_bundle["label_to_index"]
    train_loader = data_bundle["train_loader"]
    valid_loader = data_bundle["valid_loader"]
    test_loader = data_bundle["test_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device={device} | vocab_size={len(vocab.itos)} | num_classes={len(label_to_index)} | "
        f"train_batches={len(train_loader)} | "
        f"valid_batches={len(valid_loader) if valid_loader is not None else 0} | "
        f"test_batches={len(test_loader) if test_loader is not None else 0} | "
        f"num_workers={args.num_workers} | amp={args.use_amp and device.type == 'cuda'}"
    )
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
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_score = float("-inf")
    best_epoch: int | None = None
    history_rows: list[dict[str, float | int | str]] = []
    test_metrics: dict[str, float] | None = None
    print("Start training...")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            classification_criterion=classification_criterion,
            intervention_criterion=intervention_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            counterfactual_weight=args.counterfactual_weight,
            split="train",
            epoch=epoch,
            total_epochs=args.epochs,
            use_amp=args.use_amp,
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"{format_metrics('train', train_metrics)}"
        )

        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update(flatten_metrics("train", train_metrics))

        if valid_loader is None:
            if train_metrics["macro_f1"] > best_score:
                best_score = train_metrics["macro_f1"]
                best_epoch = epoch
                save_artifacts(model, output_dir, vocab.to_dict(), label_to_index, args)
            row.update(flatten_metrics("valid", None))
            row["is_best"] = int(best_epoch == epoch)
            history_rows.append(row)
            continue

        with torch.no_grad():
            valid_metrics = run_epoch(
                model=model,
                data_loader=valid_loader,
                classification_criterion=classification_criterion,
                intervention_criterion=intervention_criterion,
                optimizer=None,
                scaler=None,
                device=device,
                counterfactual_weight=args.counterfactual_weight,
                split="valid",
                epoch=epoch,
                total_epochs=args.epochs,
                use_amp=args.use_amp,
            )
        print(f"Epoch {epoch:03d}/{args.epochs:03d} | {format_metrics('valid', valid_metrics)}")
        row.update(flatten_metrics("valid", valid_metrics))

        if valid_metrics["macro_f1"] > best_score:
            best_score = valid_metrics["macro_f1"]
            best_epoch = epoch
            save_artifacts(model, output_dir, vocab.to_dict(), label_to_index, args)
        row["is_best"] = int(best_epoch == epoch)
        history_rows.append(row)

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
                scaler=None,
                device=device,
                counterfactual_weight=args.counterfactual_weight,
                split="test",
                epoch=args.epochs,
                total_epochs=args.epochs,
                use_amp=args.use_amp,
            )
        print(f"Test | {format_metrics('test', test_metrics)}")

    save_metrics_history(output_dir, history_rows)
    save_experiment_summary(
        output_dir=output_dir,
        args=args,
        best_epoch=best_epoch,
        best_score=best_score,
        test_metrics=test_metrics,
        history_rows=history_rows,
    )
    print(
        "Training finished. "
        f"Best checkpoint saved to: {output_dir} | "
        f"history: {output_dir / 'metrics_history.csv'} | "
        f"summary: {output_dir / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
