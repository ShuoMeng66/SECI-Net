#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

if [[ -f /etc/network_turbo ]]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity \
  --source direct || python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity \
  --source hf-mirror

python main/split_dataset.py \
  --train_path data/raw/yelp_polarity/train.csv \
  --test_path data/raw/yelp_polarity/test.csv \
  --output_dir data/yelp_polarity \
  --valid_size 0.1

python main/download_datasets.py \
  --dataset imdb \
  --output_dir data/raw/imdb \
  --source direct \
  --max_rows_per_split 8000

python main/split_dataset.py \
  --train_path data/raw/imdb/train.csv \
  --test_path data/raw/imdb/test.csv \
  --output_dir data/imdb \
  --valid_size 0.1

python scripts/prepare_fast_split.py \
  --source_dir data/imdb \
  --output_dir data/imdb_fast \
  --train_limit 4000 --valid_limit 500 --test_limit 1000

python -c "import csv; from collections import Counter
for p in ['data/imdb_fast/train.csv','data/imdb_fast/test.csv']:
    c = Counter(r['label'] for r in csv.DictReader(open(p, encoding='utf-8-sig')))
    print(p, dict(c))
    assert len(c) >= 2, f'IMDB labels not balanced in {p}: {c}'"

python scripts/data_stats.py --dataset_dir data/yelp_polarity --output tables/data_stats_yelp_full.json

python scripts/run_minimal_experiments.py \
  --dataset_dir data/yelp_polarity \
  --amazon_dir data/imdb_fast \
  --epochs 8 \
  --max_len 256 \
  --batch_size 64 \
  --seeds 42,43,44 \
  2>&1 | tee logs/run_full.log

python scripts/export_results.py --checkpoints_root checkpoints --output_dir tables
cat tables/results_rollup.json
