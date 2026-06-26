"""Compute descriptive statistics for downloaded CSV splits."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def summarize_split(rows: list[dict[str, str]], text_column: str = "text", label_column: str = "label") -> dict:
    lengths = [len(row[text_column].split()) for row in rows]
    labels = Counter(row[label_column] for row in rows)
    return {
        "count": len(rows),
        "label_distribution": dict(sorted(labels.items())),
        "length_tokens_mean": round(statistics.mean(lengths), 2),
        "length_tokens_median": round(statistics.median(lengths), 2),
        "length_tokens_p90": round(sorted(lengths)[int(0.9 * max(len(lengths) - 1, 0))], 2),
        "length_tokens_max": max(lengths) if lengths else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="tables/data_stats.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    stats = {}
    for split in ("train", "valid", "test"):
        path = dataset_dir / f"{split}.csv"
        if path.exists():
            stats[split] = summarize_split(read_rows(path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
