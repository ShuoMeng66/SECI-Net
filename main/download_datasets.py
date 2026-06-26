import argparse
import csv
import os
import tarfile
import tempfile
from pathlib import Path

import gdown


DATASET_SPECS = {
    "yelp_polarity": {
        "hf_name": "yelp_polarity",
        "direct_type": "torchtext_archive",
        "direct_url": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbNUpYQ2N3SGlFaDg",
        "archive_root": "yelp_review_polarity_csv",
    },
    "amazon_polarity": {
        "hf_name": "amazon_polarity",
        "direct_type": "torchtext_archive",
        "direct_url": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUGN1N25TOEpJNHE",
        "archive_root": "amazon_review_polarity_csv",
    },
    "ag_news": {
        "hf_name": "ag_news",
        "direct_type": "torchtext_archive",
        "direct_url": "https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbUDNpeUdjb0wxRms",
        "archive_root": "ag_news_csv",
    },
    "imdb": {
        "hf_name": "imdb",
        "direct_type": "acl_imdb_archive",
        "direct_url": "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        "archive_root": "aclImdb",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a public text classification dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp_polarity",
        choices=sorted(DATASET_SPECS.keys()),
        help="Built-in dataset shortcut.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory used to store raw CSV splits.")
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "hf", "hf-mirror", "direct"],
        help="Dataset source backend.",
    )
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="Mirror endpoint used when --source hf-mirror or auto fallback is enabled.",
    )
    parser.add_argument("--max_rows_per_split", type=int, default=0, help="0 means keep the full split.")
    return parser.parse_args()


def write_split(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)


def clamp_rows(rows: list[dict[str, str]], max_rows_per_split: int) -> list[dict[str, str]]:
    if max_rows_per_split <= 0 or len(rows) <= max_rows_per_split:
        return rows
    import random

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    rng = random.Random(42)
    per_label = max_rows_per_split // len(grouped)
    sampled: list[dict[str, str]] = []
    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        sampled.extend(label_rows[:per_label])
    rng.shuffle(sampled)
    return sampled[:max_rows_per_split]


def export_hf_dataset(dataset_name: str, output_dir: Path, max_rows_per_split: int, endpoint: str | None = None) -> None:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    for split_name, split in dataset.items():
        rows = []
        for row_index, sample in enumerate(split):
            if max_rows_per_split > 0 and row_index >= max_rows_per_split:
                break
            rows.append(
                {
                    "text": str(sample["text"]).strip(),
                    "label": str(sample["label"]),
                }
            )

        output_path = output_dir / f"{split_name}.csv"
        write_split(rows, output_path)
        print(f"[hf] saved {len(rows)} rows to {output_path}")


def maybe_zero_based(label_text: str) -> str:
    try:
        return str(int(label_text) - 1)
    except ValueError:
        return label_text


def read_torchtext_csv(file_path: Path) -> list[dict[str, str]]:
    rows = []
    with file_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            label = maybe_zero_based(row[0].strip())
            text = " ".join(part.strip() for part in row[1:] if part.strip())
            if text:
                rows.append({"text": text, "label": label})
    return rows


def read_acl_imdb_split(split_dir: Path, positive_label: str = "1", negative_label: str = "0") -> list[dict[str, str]]:
    rows = []
    for label_name, label_value in [("pos", positive_label), ("neg", negative_label)]:
        for file_path in sorted((split_dir / label_name).glob("*.txt")):
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                rows.append({"text": text, "label": label_value})
    return rows


def export_direct_dataset(spec: dict[str, str], output_dir: Path, max_rows_per_split: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        archive_path = temp_dir_path / "dataset.tar.gz"
        try:
            gdown.download(spec["direct_url"], str(archive_path), quiet=False, fuzzy=True)
        except TypeError:
            gdown.download(spec["direct_url"], str(archive_path), quiet=False)

        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temp_dir_path)

        if spec["direct_type"] == "torchtext_archive":
            root_dir = temp_dir_path / spec["archive_root"]
            train_rows = clamp_rows(read_torchtext_csv(root_dir / "train.csv"), max_rows_per_split)
            test_rows = clamp_rows(read_torchtext_csv(root_dir / "test.csv"), max_rows_per_split)
            write_split(train_rows, output_dir / "train.csv")
            write_split(test_rows, output_dir / "test.csv")
            print(f"[direct] saved {len(train_rows)} rows to {output_dir / 'train.csv'}")
            print(f"[direct] saved {len(test_rows)} rows to {output_dir / 'test.csv'}")
            return

        if spec["direct_type"] == "acl_imdb_archive":
            root_dir = temp_dir_path / spec["archive_root"]
            train_rows = clamp_rows(read_acl_imdb_split(root_dir / "train"), max_rows_per_split)
            test_rows = clamp_rows(read_acl_imdb_split(root_dir / "test"), max_rows_per_split)
            write_split(train_rows, output_dir / "train.csv")
            write_split(test_rows, output_dir / "test.csv")
            print(f"[direct] saved {len(train_rows)} rows to {output_dir / 'train.csv'}")
            print(f"[direct] saved {len(test_rows)} rows to {output_dir / 'test.csv'}")
            return

        raise ValueError(f"Unsupported direct_type: {spec['direct_type']}")


def try_download(args: argparse.Namespace, spec: dict[str, str]) -> None:
    output_dir = Path(args.output_dir)
    errors: list[str] = []

    backends = {
        "hf": lambda: export_hf_dataset(spec["hf_name"], output_dir, args.max_rows_per_split, endpoint=None),
        "hf-mirror": lambda: export_hf_dataset(
            spec["hf_name"],
            output_dir,
            args.max_rows_per_split,
            endpoint=args.hf_endpoint,
        ),
        "direct": lambda: export_direct_dataset(spec, output_dir, args.max_rows_per_split),
    }

    if args.source == "auto":
        order = ["hf-mirror", "hf", "direct"]
    else:
        order = [args.source]

    for source_name in order:
        try:
            if source_name == "direct" and spec.get("direct_type") is None:
                raise ValueError(f"{args.dataset} has no direct download backend")
            backends[source_name]()
            print(f"Download finished with source={source_name}")
            return
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")
            print(f"Source {source_name} failed: {exc}")

    joined_errors = "\n".join(errors)
    raise RuntimeError(
        "All dataset download backends failed.\n"
        f"Tried:\n{joined_errors}\n"
        "If you are on AutoDL or a mainland network, try `--source hf-mirror` first, then `--source direct`."
    )


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    try_download(args, spec)


if __name__ == "__main__":
    main()
