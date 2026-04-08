import argparse
import csv
from pathlib import Path

from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a CSV dataset into train/valid/test files.")
    parser.add_argument("--train_path", type=str, required=True, help="Source CSV used for training split.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for train/valid/test CSV files.")
    parser.add_argument("--label_column", type=str, default="label")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Optional existing test split. If provided, only the train file is split into train/valid.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test ratio when --test_path is not provided.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_rows(file_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
        if not rows:
            raise ValueError(f"No rows found in {file_path}")
        return rows, reader.fieldnames or []


def write_rows(rows: list[dict[str, str]], fieldnames: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_stratify_labels(rows: list[dict[str, str]], label_column: str):
    labels = [row[label_column] for row in rows]
    label_counts = {label: labels.count(label) for label in set(labels)}
    return labels if len(label_counts) > 1 and min(label_counts.values()) > 1 else None


def main() -> None:
    args = parse_args()
    train_rows, fieldnames = read_rows(Path(args.train_path))

    if args.test_path:
        test_rows, _ = read_rows(Path(args.test_path))
        train_rows, valid_rows = train_test_split(
            train_rows,
            test_size=args.valid_size,
            random_state=args.seed,
            stratify=get_stratify_labels(train_rows, args.label_column),
        )
    else:
        train_valid_rows, test_rows = train_test_split(
            train_rows,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=get_stratify_labels(train_rows, args.label_column),
        )
        valid_ratio = args.valid_size / (1.0 - args.test_size)
        train_rows, valid_rows = train_test_split(
            train_valid_rows,
            test_size=valid_ratio,
            random_state=args.seed,
            stratify=get_stratify_labels(train_valid_rows, args.label_column),
        )

    output_dir = Path(args.output_dir)
    write_rows(train_rows, fieldnames, output_dir / "train.csv")
    write_rows(valid_rows, fieldnames, output_dir / "valid.csv")
    write_rows(test_rows, fieldnames, output_dir / "test.csv")

    print(f"train={len(train_rows)} | valid={len(valid_rows)} | test={len(test_rows)}")
    print(f"Saved split files to {output_dir}")


if __name__ == "__main__":
    main()
