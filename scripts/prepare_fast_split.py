"""Create a smaller stratified subset for faster benchmark runs."""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def sample_rows(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    per_label = max(1, limit // len(grouped))
    sampled: list[dict[str, str]] = []
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        sampled.extend(label_rows[:per_label])
    rng.shuffle(sampled)
    return sampled[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_limit", type=int, default=4000)
    parser.add_argument("--valid_limit", type=int, default=500)
    parser.add_argument("--test_limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source_dir)
    output = Path(args.output_dir)
    write_rows(output / "train.csv", sample_rows(read_rows(source / "train.csv"), args.train_limit, args.seed))
    write_rows(output / "valid.csv", sample_rows(read_rows(source / "valid.csv"), args.valid_limit, args.seed + 1))
    write_rows(output / "test.csv", sample_rows(read_rows(source / "test.csv"), args.test_limit, args.seed + 2))
    print(f"Saved fast split to {output}")


if __name__ == "__main__":
    main()
