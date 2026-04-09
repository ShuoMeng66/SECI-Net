from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from GAN import GANTrainingConfig, train_gan_and_augment_records
from core.data import load_text_classification_records


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thin CLI wrapper around GAN.py for counterfactual review augmentation."
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
    parser.add_argument("--gan_lr_generator", type=float, default=1e-4)
    parser.add_argument("--gan_lr_discriminator", type=float, default=2e-4)
    parser.add_argument("--max_source_len", type=int, default=128)
    parser.add_argument("--max_target_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment_missing_only", dest="augment_missing_only", action="store_true")
    parser.add_argument("--augment_all", dest="augment_missing_only", action="store_false")
    parser.set_defaults(augment_missing_only=True)
    return parser.parse_args(argv)


def write_output_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
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
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows({field: row.get(field, "") for field in fieldnames} for row in rows)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    records = load_text_classification_records(
        args.train_path,
        text_column=args.text_column,
        label_column=args.label_column,
        delimiter=args.delimiter,
        encoding=args.encoding,
        counterfactual_text_column=args.counterfactual_text_column,
        counterfactual_label_column=args.counterfactual_label_column,
    )

    result = train_gan_and_augment_records(
        records=records,
        output_dir=args.save_dir,
        config=GANTrainingConfig(
            batch_size=args.batch_size,
            learning_rate_generator=args.gan_lr_generator,
            learning_rate_discriminator=args.gan_lr_discriminator,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
        ),
        epochs=args.epochs,
        seed=args.seed,
        augment_missing_only=args.augment_missing_only,
        verbose=True,
    )

    output_path = Path(args.output_path)
    write_output_rows(output_path, result.generated_rows)
    if result.warning:
        print(f"[GAN] Warning: {result.warning}")
    print(
        f"[GAN] generated_rows={len(result.generated_rows)} | "
        f"backfilled={result.summary.get('backfilled_records', 0)} | "
        f"appended={result.summary.get('appended_records', 0)}"
    )
    print(f"Saved generated counterfactual reviews to: {output_path}")


if __name__ == "__main__":
    main()
