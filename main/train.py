from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from GAN import GANTrainingConfig, train_gan_and_augment_records
from core.data import TextRecord, build_dataloaders, load_text_classification_records
from core.model import (
    HybridTextClassifier,
    SECIOutput,
    load_checkpoint_bundle,
    save_checkpoint_bundle,
)
from core.utils import (
    ClassificationLoss,
    CounterfactualInterventionLoss,
    EvidenceSparsityLoss,
    RecoverabilityLoss,
)


STEP_METRIC_FIELDNAMES = [
    "split",
    "epoch",
    "total_epochs",
    "step",
    "total_steps",
    "global_step",
    "lr",
    "loss",
    "classification_loss",
    "counterfactual_loss",
    "intervention_loss",
    "recoverability_loss",
    "sparsity_loss",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "mean_recoverability",
    "mean_evidence_score",
]


LOSS_KEYS = [
    "loss",
    "classification_loss",
    "counterfactual_loss",
    "intervention_loss",
    "recoverability_loss",
    "sparsity_loss",
]


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


def infer_positive_class_index(
    label_to_index: dict[str, int],
    explicit_value: int | None,
) -> int:
    if explicit_value is not None:
        return explicit_value

    normalized = {str(label).strip().lower(): index for label, index in label_to_index.items()}
    for candidate in ("positive", "pos", "1", "true", "yes"):
        if candidate in normalized:
            return normalized[candidate]
    return max(label_to_index.values())


def forward_batch(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
    time_values: torch.Tensor | None,
) -> SECIOutput:
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        lengths=lengths,
        time_values=time_values,
        return_dict=True,
    )


def build_recoverability_targets(
    factual_labels: torch.Tensor,
    counterfactual_labels: torch.Tensor,
    positive_class_index: int,
) -> torch.Tensor:
    factual_positive = (factual_labels == positive_class_index).float()
    counterfactual_positive = (counterfactual_labels == positive_class_index).float()
    return (counterfactual_positive - factual_positive).clamp(min=0.0, max=1.0)


def mean_evidence_score(output: SECIOutput) -> torch.Tensor:
    return output.evidence_scores.mean()


def compute_counterfactual_losses(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    factual_output: SECIOutput,
    classification_criterion: nn.Module,
    intervention_criterion: CounterfactualInterventionLoss,
    recoverability_criterion: RecoverabilityLoss,
    sparsity_criterion: EvidenceSparsityLoss,
    counterfactual_weight: float,
    recoverability_weight: float,
    sparsity_weight: float,
    positive_class_index: int,
) -> dict[str, torch.Tensor]:
    zero = factual_output.logits.new_zeros(())
    factual_sparsity = sparsity_criterion(
        factual_output.router_probabilities,
        factual_output.block_mask,
    )

    available_mask = batch["counterfactual_available"]
    if not available_mask.any():
        return {
            "counterfactual_loss": zero,
            "intervention_loss": zero,
            "recoverability_loss": zero,
            "sparsity_loss": sparsity_weight * factual_sparsity,
        }

    counterfactual_time_values = batch.get("counterfactual_time_values")
    counterfactual_output = forward_batch(
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
    counterfactual_classification_loss = counterfactual_weight * classification_criterion(
        counterfactual_output.logits,
        batch["counterfactual_labels"][available_mask],
    )
    intervention_loss = intervention_criterion(
        factual_features=factual_output.features[available_mask],
        counterfactual_features=counterfactual_output.features,
        factual_labels=batch["labels"][available_mask],
        counterfactual_labels=batch["counterfactual_labels"][available_mask],
    )
    recoverability_targets = build_recoverability_targets(
        factual_labels=batch["labels"][available_mask],
        counterfactual_labels=batch["counterfactual_labels"][available_mask],
        positive_class_index=positive_class_index,
    )
    recoverability_loss = recoverability_weight * recoverability_criterion(
        factual_output.recoverability_logits[available_mask],
        recoverability_targets,
    )
    counterfactual_sparsity = sparsity_criterion(
        counterfactual_output.router_probabilities,
        counterfactual_output.block_mask,
    )
    sparsity_loss = sparsity_weight * 0.5 * (factual_sparsity + counterfactual_sparsity)

    return {
        "counterfactual_loss": counterfactual_classification_loss,
        "intervention_loss": intervention_loss,
        "recoverability_loss": recoverability_loss,
        "sparsity_loss": sparsity_loss,
    }


def metric_template() -> dict[str, float]:
    return {
        "loss": 0.0,
        "classification_loss": 0.0,
        "counterfactual_loss": 0.0,
        "intervention_loss": 0.0,
        "recoverability_loss": 0.0,
        "sparsity_loss": 0.0,
        "mean_recoverability": 0.0,
        "mean_evidence_score": 0.0,
    }


def run_epoch(
    model: nn.Module,
    data_loader,
    classification_criterion: nn.Module,
    intervention_criterion: CounterfactualInterventionLoss,
    recoverability_criterion: RecoverabilityLoss,
    sparsity_criterion: EvidenceSparsityLoss,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    counterfactual_weight: float,
    recoverability_weight: float,
    sparsity_weight: float,
    positive_class_index: int,
    split: str,
    epoch: int,
    total_epochs: int,
    use_amp: bool,
    log_every_steps: int = 0,
    step_logger: "StreamingCSVLogger" | None = None,
    global_step_start: int = 0,
) -> tuple[dict[str, float], int]:
    is_training = optimizer is not None
    model.train(is_training)

    totals = metric_template()
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

    total_steps = len(data_loader)
    for step, batch in enumerate(progress_bar, start=1):
        batch = move_batch_to_device(batch, device)
        autocast_context = (
            torch.autocast(device_type=device.type, dtype=torch.float16)
            if use_amp and device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            factual_output = forward_batch(
                model=model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                lengths=batch["lengths"],
                time_values=batch.get("time_values"),
            )
            classification_loss = classification_criterion(
                factual_output.logits,
                batch["labels"],
            )
            counterfactual_losses = compute_counterfactual_losses(
                model=model,
                batch=batch,
                factual_output=factual_output,
                classification_criterion=classification_criterion,
                intervention_criterion=intervention_criterion,
                recoverability_criterion=recoverability_criterion,
                sparsity_criterion=sparsity_criterion,
                counterfactual_weight=counterfactual_weight,
                recoverability_weight=recoverability_weight,
                sparsity_weight=sparsity_weight,
                positive_class_index=positive_class_index,
            )
            loss = classification_loss
            for key in ("counterfactual_loss", "intervention_loss", "recoverability_loss", "sparsity_loss"):
                loss = loss + counterfactual_losses[key]

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

        totals["loss"] += loss.item()
        totals["classification_loss"] += classification_loss.item()
        totals["counterfactual_loss"] += counterfactual_losses["counterfactual_loss"].item()
        totals["intervention_loss"] += counterfactual_losses["intervention_loss"].item()
        totals["recoverability_loss"] += counterfactual_losses["recoverability_loss"].item()
        totals["sparsity_loss"] += counterfactual_losses["sparsity_loss"].item()
        totals["mean_recoverability"] += factual_output.recoverability.mean().item()
        totals["mean_evidence_score"] += mean_evidence_score(factual_output).item()

        predicted_labels = torch.argmax(factual_output.logits, dim=-1)
        labels = batch["labels"]
        total_correct += (predicted_labels == labels).sum().item()
        total_examples += labels.size(0)
        predictions.extend(predicted_labels.cpu().tolist())
        references.extend(labels.cpu().tolist())
        progress_bar.set_postfix(
            loss=f"{totals['loss'] / step:.4f}",
            rec=f"{totals['mean_recoverability'] / step:.4f}",
            sparse=f"{totals['sparsity_loss'] / step:.4f}",
            acc=f"{(total_correct / max(total_examples, 1)):.4f}",
        )

        should_log_step = (
            step_logger is not None
            and log_every_steps > 0
            and (step % log_every_steps == 0 or step == total_steps)
        )
        if should_log_step:
            step_metrics = compute_metrics(predictions, references)
            for key in metric_template():
                step_metrics[key] = totals[key] / step
            step_logger.log(
                {
                    "split": split,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "step": step,
                    "total_steps": total_steps,
                    "global_step": global_step_start + step if is_training else "",
                    "lr": optimizer.param_groups[0]["lr"] if is_training else "",
                    **step_metrics,
                }
            )

    progress_bar.close()

    metrics = compute_metrics(predictions, references)
    num_batches = max(total_steps, 1)
    for key in metric_template():
        metrics[key] = totals[key] / num_batches
    return metrics, num_batches


def format_metrics(split: str, metrics: dict[str, float]) -> str:
    return (
        f"{split}_loss={metrics['loss']:.4f} | "
        f"{split}_cf_loss={metrics['counterfactual_loss']:.4f} | "
        f"{split}_repr_loss={metrics['intervention_loss']:.4f} | "
        f"{split}_rec_loss={metrics['recoverability_loss']:.4f} | "
        f"{split}_sparse_loss={metrics['sparsity_loss']:.4f} | "
        f"{split}_acc={metrics['accuracy']:.4f} | "
        f"{split}_macro_f1={metrics['macro_f1']:.4f}"
    )


def flatten_metrics(prefix: str, metrics: dict[str, float] | None) -> dict[str, float | str]:
    if metrics is None:
        return {
            f"{prefix}_{field}": ""
            for field in (
                *LOSS_KEYS,
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "mean_recoverability",
                "mean_evidence_score",
            )
        }
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


class StreamingCSVLogger:
    def __init__(self, csv_path: Path, fieldnames: list[str]) -> None:
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.csv_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, row: dict[str, float | int | str]) -> None:
        normalized_row = {field: row.get(field, "") for field in self.fieldnames}
        self.writer.writerow(normalized_row)
        self.file.flush()
        os.fsync(self.file.fileno())

    def close(self) -> None:
        if not self.file.closed:
            self.file.flush()
            self.file.close()


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
    positive_class_index: int,
    test_metrics: dict[str, float] | None,
    history_rows: list[dict[str, float | int | str]],
) -> None:
    summary = {
        "architecture": "seci-net-v2",
        "selection_metric": (
            "valid_macro_f1"
            if any(row.get("valid_macro_f1", "") != "" for row in history_rows)
            else "train_macro_f1"
        ),
        "best_epoch": best_epoch,
        "best_score": None if best_score == float("-inf") else best_score,
        "positive_class_index": positive_class_index,
        "test_metrics": test_metrics,
        "save_dir": str(output_dir),
        "train_args_path": str(output_dir / "train_args.json"),
        "best_model_path": str(output_dir / "best_model.pt"),
        "metrics_history_csv": str(output_dir / "metrics_history.csv"),
        "metrics_history_json": str(output_dir / "metrics_history.json"),
        "step_metrics_csv": str(output_dir / "step_metrics.csv"),
        "seed": args.seed,
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SECI-Net v2 with parallel encoders, explicit evidence routing, and recoverability learning."
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
    parser.add_argument("--router_hidden_dim", type=int, default=None)
    parser.add_argument("--counterfactual_weight", type=float, default=0.5)
    parser.add_argument("--consistency_weight", type=float, default=0.5)
    parser.add_argument("--intervention_weight", type=float, default=0.5)
    parser.add_argument("--intervention_margin", type=float, default=0.2)
    parser.add_argument("--recoverability_weight", type=float, default=0.5)
    parser.add_argument("--sparsity_weight", type=float, default=0.05)
    parser.add_argument("--positive_class_index", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=suggest_num_workers())
    parser.add_argument("--amp", dest="use_amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false", help="Disable automatic mixed precision.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=100,
        help="Append running metrics to step_metrics.csv every N steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--enable_gan_augmentation",
        action="store_true",
        help="Train the standalone review GAN first and use it to augment train records before classification.",
    )
    parser.add_argument(
        "--gan_output_dir",
        type=str,
        default=None,
        help="Directory to store GAN artifacts. Defaults to <save_dir>/gan.",
    )
    parser.add_argument("--gan_epochs", type=int, default=5)
    parser.add_argument("--gan_batch_size", type=int, default=16)
    parser.add_argument("--gan_lr_generator", type=float, default=1e-4)
    parser.add_argument("--gan_lr_discriminator", type=float, default=2e-4)
    parser.add_argument("--gan_max_source_len", type=int, default=128)
    parser.add_argument("--gan_max_target_len", type=int, default=128)
    parser.add_argument(
        "--gan_augment_missing_only",
        dest="gan_augment_missing_only",
        action="store_true",
        help="Only backfill missing counterfactual fields with GAN outputs.",
    )
    parser.add_argument(
        "--gan_augment_all",
        dest="gan_augment_missing_only",
        action="store_false",
        help="Generate extra synthetic counterfactual pairs even when annotated pairs already exist.",
    )
    parser.set_defaults(use_amp=torch.cuda.is_available())
    parser.set_defaults(gan_augment_missing_only=True)
    return parser.parse_args(argv)


def save_artifacts(
    model: HybridTextClassifier,
    output_dir: Path,
    vocab_info: dict[str, Any],
    label_to_index: dict[str, int],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint_bundle(
        output_dir / "best_model.pt",
        model=model,
        vocab=vocab_info,
        label_to_index=label_to_index,
        extra={"train_args": vars(args)},
    )

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


def build_gan_config_from_args(args: argparse.Namespace) -> GANTrainingConfig:
    return GANTrainingConfig(
        batch_size=args.gan_batch_size,
        learning_rate_generator=args.gan_lr_generator,
        learning_rate_discriminator=args.gan_lr_discriminator,
        max_source_len=args.gan_max_source_len,
        max_target_len=args.gan_max_target_len,
        label_smoothing=args.label_smoothing,
    )


def maybe_augment_train_records(
    train_records: list[TextRecord],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[TextRecord]:
    if not args.enable_gan_augmentation:
        return train_records

    gan_output_dir = Path(args.gan_output_dir) if args.gan_output_dir else output_dir / "gan"
    print(f"[GAN] Preparing offline augmentation in: {gan_output_dir}")
    augmentation_result = train_gan_and_augment_records(
        records=train_records,
        output_dir=gan_output_dir,
        config=build_gan_config_from_args(args),
        epochs=args.gan_epochs,
        seed=args.seed,
        augment_missing_only=args.gan_augment_missing_only,
        verbose=True,
    )

    if augmentation_result.warning:
        print(f"[GAN] Warning: {augmentation_result.warning}")
    print(
        f"[GAN] paired={augmentation_result.summary.get('paired_records', 0)} | "
        f"generated={augmentation_result.summary.get('generated_rows', 0)} | "
        f"backfilled={augmentation_result.summary.get('backfilled_records', 0)} | "
        f"appended={augmentation_result.summary.get('appended_records', 0)}"
    )
    print(
        f"[GAN] rows={augmentation_result.artifacts.get('generated_rows', '')} | "
        f"metrics={augmentation_result.artifacts.get('metrics', '')}"
    )
    return augmentation_result.augmented_records


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
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

    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = maybe_augment_train_records(train_records, args=args, output_dir=output_dir)
    print(f"Train records after augmentation: {len(train_records)}")

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
    positive_class_index = infer_positive_class_index(
        label_to_index=label_to_index,
        explicit_value=args.positive_class_index,
    )
    train_loader = data_bundle["train_loader"]
    valid_loader = data_bundle["valid_loader"]
    test_loader = data_bundle["test_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device={device} | vocab_size={len(vocab.itos)} | num_classes={len(label_to_index)} | "
        f"positive_class_index={positive_class_index} | "
        f"train_batches={len(train_loader)} | "
        f"valid_batches={len(valid_loader) if valid_loader is not None else 0} | "
        f"test_batches={len(test_loader) if test_loader is not None else 0} | "
        f"num_workers={args.num_workers} | amp={args.use_amp and device.type == 'cuda'}"
    )
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
        router_hidden_dim=args.router_hidden_dim,
        positive_class_index=positive_class_index,
    ).to(device)

    classification_criterion = ClassificationLoss(label_smoothing=args.label_smoothing)
    intervention_criterion = CounterfactualInterventionLoss(
        consistency_weight=args.consistency_weight,
        intervention_weight=args.intervention_weight,
        margin=args.intervention_margin,
    )
    recoverability_criterion = RecoverabilityLoss()
    sparsity_criterion = EvidenceSparsityLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=args.use_amp and device.type == "cuda")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type == "cuda")

    best_score = float("-inf")
    best_epoch: int | None = None
    history_rows: list[dict[str, float | int | str]] = []
    test_metrics: dict[str, float] | None = None
    global_train_step = 0
    step_logger = StreamingCSVLogger(output_dir / "step_metrics.csv", STEP_METRIC_FIELDNAMES)
    print("Start training...")

    try:
        for epoch in range(1, args.epochs + 1):
            train_metrics, train_batches = run_epoch(
                model=model,
                data_loader=train_loader,
                classification_criterion=classification_criterion,
                intervention_criterion=intervention_criterion,
                recoverability_criterion=recoverability_criterion,
                sparsity_criterion=sparsity_criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=device,
                counterfactual_weight=args.counterfactual_weight,
                recoverability_weight=args.recoverability_weight,
                sparsity_weight=args.sparsity_weight,
                positive_class_index=positive_class_index,
                split="train",
                epoch=epoch,
                total_epochs=args.epochs,
                use_amp=args.use_amp,
                log_every_steps=args.log_every_steps,
                step_logger=step_logger,
                global_step_start=global_train_step,
            )
            global_train_step += train_batches
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
                save_metrics_history(output_dir, history_rows)
                continue

            with torch.no_grad():
                valid_metrics, _ = run_epoch(
                    model=model,
                    data_loader=valid_loader,
                    classification_criterion=classification_criterion,
                    intervention_criterion=intervention_criterion,
                    recoverability_criterion=recoverability_criterion,
                    sparsity_criterion=sparsity_criterion,
                    optimizer=None,
                    scaler=None,
                    device=device,
                    counterfactual_weight=args.counterfactual_weight,
                    recoverability_weight=args.recoverability_weight,
                    sparsity_weight=args.sparsity_weight,
                    positive_class_index=positive_class_index,
                    split="valid",
                    epoch=epoch,
                    total_epochs=args.epochs,
                    use_amp=args.use_amp,
                    log_every_steps=args.log_every_steps,
                    step_logger=step_logger,
                    global_step_start=global_train_step,
                )
            print(f"Epoch {epoch:03d}/{args.epochs:03d} | {format_metrics('valid', valid_metrics)}")
            row.update(flatten_metrics("valid", valid_metrics))

            if valid_metrics["macro_f1"] > best_score:
                best_score = valid_metrics["macro_f1"]
                best_epoch = epoch
                save_artifacts(model, output_dir, vocab.to_dict(), label_to_index, args)
            row["is_best"] = int(best_epoch == epoch)
            history_rows.append(row)
            save_metrics_history(output_dir, history_rows)

        if test_loader is not None:
            best_model_path = output_dir / "best_model.pt"
            if best_model_path.exists():
                checkpoint_bundle = load_checkpoint_bundle(best_model_path, map_location=device)
                model.load_state_dict(checkpoint_bundle["state_dict"], strict=True)

            with torch.no_grad():
                test_metrics, _ = run_epoch(
                    model=model,
                    data_loader=test_loader,
                    classification_criterion=classification_criterion,
                    intervention_criterion=intervention_criterion,
                    recoverability_criterion=recoverability_criterion,
                    sparsity_criterion=sparsity_criterion,
                    optimizer=None,
                    scaler=None,
                    device=device,
                    counterfactual_weight=args.counterfactual_weight,
                    recoverability_weight=args.recoverability_weight,
                    sparsity_weight=args.sparsity_weight,
                    positive_class_index=positive_class_index,
                    split="test",
                    epoch=args.epochs,
                    total_epochs=args.epochs,
                    use_amp=args.use_amp,
                    log_every_steps=args.log_every_steps,
                    step_logger=step_logger,
                    global_step_start=global_train_step,
                )
            print(f"Test | {format_metrics('test', test_metrics)}")

        save_metrics_history(output_dir, history_rows)
        save_experiment_summary(
            output_dir=output_dir,
            args=args,
            best_epoch=best_epoch,
            best_score=best_score,
            positive_class_index=positive_class_index,
            test_metrics=test_metrics,
            history_rows=history_rows,
        )
        print(
            "Training finished. "
            f"Best checkpoint saved to: {output_dir} | "
            f"history: {output_dir / 'metrics_history.csv'} | "
            f"step history: {output_dir / 'step_metrics.csv'} | "
            f"summary: {output_dir / 'summary.json'}"
        )
    finally:
        step_logger.close()


if __name__ == "__main__":
    main()
