"""Run the minimal benchmark suite and write summary.json for each experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "main" / "train.py"


def run_train(name: str, save_dir: Path, extra_args: list[str], common: list[str]) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        *common,
        "--save_dir",
        str(save_dir),
        *extra_args,
    ]
    print(f"\n=== Running {name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default=str(PROJECT_ROOT / "data" / "yelp_fast"))
    parser.add_argument("--amazon_dir", type=str, default=str(PROJECT_ROOT / "data" / "imdb_fast"))
    parser.add_argument("--checkpoints_root", type=str, default=str(PROJECT_ROOT / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="42,43")
    parser.add_argument("--skip_amazon", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    amazon_dir = Path(args.amazon_dir)
    root = Path(args.checkpoints_root)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    common = [
        "--train_path",
        str(dataset_dir / "train.csv"),
        "--valid_path",
        str(dataset_dir / "valid.csv"),
        "--test_path",
        str(dataset_dir / "test.csv"),
        "--epochs",
        str(args.epochs),
        "--max_len",
        str(args.max_len),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        "0",
    ]

    main_models = {
        "bilstm": ["--transformer_layers", "0", "--attention_type", "standard"],
        "transformer": ["--lstm_layers", "0", "--attention_type", "standard"],
        "hybrid": ["--attention_type", "differential"],
        "seci_net_full": ["--attention_type", "block_sparse"],
    }

    ablations = {
        "seci_wo_cf": [
            "--attention_type",
            "block_sparse",
            "--counterfactual_weight",
            "0",
            "--recoverability_weight",
            "0",
        ],
        "seci_wo_gan": ["--attention_type", "block_sparse"],
        "seci_wo_sparsity": ["--attention_type", "block_sparse", "--sparsity_weight", "0"],
        "seci_wo_recurrent": ["--attention_type", "block_sparse", "--lstm_layers", "0"],
    }

    manifest: dict[str, list[str]] = {"main": [], "ablation": [], "amazon": []}

    for seed in seeds:
        for name, extra in main_models.items():
            save_dir = root / "yelp" / name / f"seed_{seed}"
            run_train(name, save_dir, ["--seed", str(seed), *extra], common)
            manifest["main"].append(str(save_dir))

    seed = seeds[0]
    for name, extra in ablations.items():
        save_dir = root / "yelp" / "ablation" / name / f"seed_{seed}"
        extra_args = list(extra)
        if name == "seci_wo_gan":
            pass
        run_train(name, save_dir, ["--seed", str(seed), *extra_args], common)
        manifest["ablation"].append(str(save_dir))

    if not args.skip_amazon and (amazon_dir / "train.csv").exists():
        amazon_common = [
            "--train_path",
            str(amazon_dir / "train.csv"),
            "--valid_path",
            str(amazon_dir / "valid.csv"),
            "--test_path",
            str(amazon_dir / "test.csv"),
            "--epochs",
            str(args.epochs),
            "--max_len",
            str(args.max_len),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            "0",
            "--seed",
            str(seed),
        ]
        for name, extra in {
            "amazon_transformer": ["--lstm_layers", "0", "--attention_type", "standard"],
            "amazon_hybrid": ["--attention_type", "differential"],
            "amazon_seci_net_full": ["--attention_type", "block_sparse"],
        }.items():
            save_dir = root / "amazon" / name / f"seed_{seed}"
            run_train(name, save_dir, extra, amazon_common)
            manifest["amazon"].append(str(save_dir))

    manifest_path = root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
