# SECI-Net

SECI-Net is a PyTorch implementation of a hybrid text classifier that combines a custom Transformer encoder, a custom stacked LSTM, block-sparse evidence routing, and optional counterfactual intervention learning.

The repository is organized as code-first project for developers. It includes:

- a readable hybrid model implementation in `core/model/hybrid_text_model.py`
- data loading utilities in `core/data/text_dataset.py`
- a training entrypoint in `main/train.py`
- a public dataset downloader in `main/download_datasets.py`
- a dataset splitting utility in `main/split_dataset.py`
- a standalone counterfactual generation prototype in `main/counterfactual_generator.py`

## Project Structure

```text
SECI-Net/
├── core/
│   ├── data/
│   ├── model/
│   └── utils/
├── main/
│   ├── counterfactual_generator.py
│   ├── download_datasets.py
│   ├── split_dataset.py
│   └── train.py
├── .gitignore
├── README.md
├── README-zh.md
└── requirements.txt
```

## Quick Start

### 1. Clone the project

```bash
git clone git@github.com:ShuoMeng66/SECI-Net.git
cd SECI-Net
```

### 2. Create the virtual environment

```bash
python -m venv SECINet
```

Windows PowerShell:

```powershell
.\SECINet\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source SECINet/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download a public dataset

The default quick-start dataset is `yelp_polarity`.

```bash
python main/download_datasets.py --dataset yelp_polarity --output_dir data/raw/yelp_polarity
```

If you are on AutoDL, a campus server, or a mainland network where Hugging Face is unstable, use the mirror mode first:

```bash
python main/download_datasets.py --dataset yelp_polarity --output_dir data/raw/yelp_polarity --source hf-mirror
```

If the mirror still fails, fall back to direct archive download:

```bash
python main/download_datasets.py --dataset yelp_polarity --output_dir data/raw/yelp_polarity --source direct
```

Supported built-in shortcuts:

- `yelp_polarity`
- `ag_news`
- `imdb`

### 5. Split the dataset

If the downloaded dataset already has a test split, keep it and split only the training file into `train` and `valid`:

```bash
python main/split_dataset.py \
  --train_path data/raw/yelp_polarity/train.csv \
  --test_path data/raw/yelp_polarity/test.csv \
  --output_dir data/yelp_polarity
```

### 6. Start training

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity
```

## Training Output

`main/train.py` prints a full metric summary at every epoch and shows a live per-batch progress bar during training:

- total loss
- counterfactual classification loss
- intervention loss
- accuracy
- macro precision
- macro recall
- macro F1

When a validation set is provided, the best checkpoint is selected by validation macro F1.

Each run also writes experiment artifacts to `--save_dir`:

- `best_model.pt`
- `train_args.json`
- `vocab.json`
- `labels.json`
- `metrics_history.csv`
- `metrics_history.json`
- `summary.json`

## Data Format

SECI-Net accepts `csv`, `tsv`, and `txt` files.

For `csv` and `tsv`, the default columns are:

- `text`
- `label`

Optional columns:

- `counterfactual_text`
- `counterfactual_label`
- `time_column`
- `counterfactual_time_column`

For plain text files, each line must be:

```text
label<TAB>text
```

## Main Training Arguments

- `--batch_size`
- `--epochs`
- `--lr`
- `--weight_decay`
- `--embed_dim`
- `--transformer_layers`
- `--num_heads`
- `--ffn_hidden_dim`
- `--lstm_hidden_dim`
- `--lstm_layers`
- `--dropout`
- `--attention_type`
- `--block_size`
- `--local_window_size`
- `--topk_global_blocks`
- `--counterfactual_weight`
- `--consistency_weight`
- `--intervention_weight`
- `--save_dir`

## Counterfactual Generator Prototype

If you already have paired counterfactual examples, you can run the standalone generator:

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

## Design Notes

This repository favors direct, inspectable code over heavy abstraction. The training loop, metrics, and data processing are intentionally kept close to the underlying PyTorch flow so that other developers can trace the implementation quickly.

## License

No license file has been added yet.
