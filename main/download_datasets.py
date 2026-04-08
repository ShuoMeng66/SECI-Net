import argparse
import csv
from pathlib import Path

from datasets import load_dataset


DATASET_SPECS = {
    "yelp_polarity": {
        "hf_name": "yelp_polarity",
        "text_column": "text",
        "label_column": "label",
    },
    "ag_news": {
        "hf_name": "ag_news",
        "text_column": "text",
        "label_column": "label",
    },
    "imdb": {
        "hf_name": "imdb",
        "text_column": "text",
        "label_column": "label",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a public text classification dataset from Hugging Face.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="yelp_polarity",
        choices=sorted(DATASET_SPECS.keys()),
        help="Built-in dataset shortcut.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory that will store raw CSV splits.")
    parser.add_argument("--max_rows_per_split", type=int, default=0, help="0 means keep the full split.")
    return parser.parse_args()


def write_split(rows, fieldnames: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    spec = DATASET_SPECS[args.dataset]
    dataset = load_dataset(spec["hf_name"])
    output_dir = Path(args.output_dir)

    for split_name, split in dataset.items():
        rows = []
        for row_index, sample in enumerate(split):
            if args.max_rows_per_split > 0 and row_index >= args.max_rows_per_split:
                break
            rows.append(
                {
                    "text": str(sample[spec["text_column"]]).strip(),
                    "label": str(sample[spec["label_column"]]),
                }
            )

        output_path = output_dir / f"{split_name}.csv"
        write_split(rows, ["text", "label"], output_path)
        print(f"Saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
