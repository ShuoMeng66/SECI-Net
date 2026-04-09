# SECI-Net

SECI-Net is a PyTorch project for evidence-aware text classification, counterfactual supervision, and counterfactual review augmentation. The repository is organized as a code-first open-source project: the focus is on readable training code, reproducible experiments, and components that can be reused in downstream NLP work.

This public code snapshot intentionally centers on the research code. Local-only materials such as paper writing folders and the frontend review console are not part of the published workflow.

## Highlights

- Hybrid classifier with Transformer-style contextual modeling, stacked recurrent modeling, and explicit evidence routing
- Counterfactual training support for paired factual / counterfactual samples
- Recoverability and intervention losses for controlled representation learning
- Standalone `GAN.py` module for offline counterfactual review augmentation
- Training, checkpointing, dataset preparation, and smoke-test coverage in one repository
- Optional local inference API for quick model inspection

## Repository Layout

```text
SECI-Net/
├── core/
│   ├── data/
│   │   ├── __init__.py
│   │   └── text_dataset.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── checkpointing.py
│   │   ├── components.py
│   │   ├── hybrid_text_model.py
│   │   └── seci_net.py
│   └── utils/
│       ├── __init__.py
│       └── losses.py
├── main/
│   ├── counterfactual_generator.py
│   ├── download_datasets.py
│   ├── split_dataset.py
│   └── train.py
├── tests/
│   ├── test_gan.py
│   └── test_seci_net_v2.py
├── api_server.py
├── GAN.py
├── GAN.md
├── README.md
├── README-zh.md
└── requirements.txt
```

## Installation

### 1. Clone the repository

```bash
git clone git@github.com:ShuoMeng66/SECI-Net.git
cd SECI-Net
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Data Format

SECI-Net accepts `csv`, `tsv`, and `txt`.

Default structured columns:

- `text`
- `label`

Optional counterfactual columns:

- `counterfactual_text`
- `counterfactual_label`
- `time_column`
- `counterfactual_time_column`

Plain-text files must use:

```text
label<TAB>text
```

## Quick Start

### 1. Download a dataset

The built-in downloader supports:

- `yelp_polarity`
- `ag_news`
- `imdb`

Example:

```bash
python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity
```

If you are on a mainland network or a server where Hugging Face access is unstable, try:

```bash
python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity \
  --source hf-mirror
```

### 2. Split the dataset

```bash
python main/split_dataset.py \
  --train_path data/raw/yelp_polarity/train.csv \
  --test_path data/raw/yelp_polarity/test.csv \
  --output_dir data/yelp_polarity
```

### 3. Train the classifier

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity
```

## GAN-Based Counterfactual Augmentation

SECI-Net now supports an offline augmentation stage for review-style counterfactuals. The workflow is:

1. train the standalone GAN on paired counterfactual examples
2. generate missing counterfactual reviews for the training split
3. merge the generated pairs back into the training records
4. continue with the regular SECI-Net classification training

Enable it directly from the main training script:

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity_gan \
  --enable_gan_augmentation \
  --gan_epochs 5 \
  --gan_batch_size 16 \
  --gan_max_source_len 128 \
  --gan_max_target_len 128
```

Important defaults:

- GAN augmentation only touches the training split
- human-annotated counterfactual pairs are preserved by default
- when no paired counterfactual data is available, the GAN stage is skipped cleanly

You can also run the standalone wrapper:

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

See [`GAN.md`](./GAN.md) for the design rationale.

## Training Outputs

Each training run writes experiment artifacts under `--save_dir`, including:

- `best_model.pt`
- `train_args.json`
- `vocab.json`
- `labels.json`
- `metrics_history.csv`
- `metrics_history.json`
- `step_metrics.csv`
- `summary.json`

When GAN augmentation is enabled, an additional `gan/` directory is created under `--save_dir` by default:

- `generated_counterfactuals.csv`
- `metrics.json`
- `counterfactual_generator.pt`
- `counterfactual_discriminator.pt`
- `counterfactual_vocab.json`
- `counterfactual_labels.json`

## Inference API

For lightweight local inspection, you can run:

```bash
python api_server.py
```

The local API serves a simple prediction endpoint around a saved checkpoint and is intended for local experimentation rather than production deployment.

## Testing

Run the current unit and smoke tests with:

```bash
python -m unittest tests.test_seci_net_v2 tests.test_gan
```

The test suite covers:

- tensor-shape and checkpoint smoke tests for SECI-Net
- GAN data collation and fallback behavior
- GAN forward passes and one-epoch CPU training
- train-time integration with and without GAN augmentation

## Design Philosophy

This repository prefers inspectable code over heavy abstraction. The model, data flow, loss composition, and training loop are kept close to plain PyTorch so that researchers and engineers can trace what is happening without unpacking a large framework layer.

## Roadmap

Planned improvements include:

- stronger aspect modeling for counterfactual reviews
- better filtering or reranking for GAN-generated samples
- broader benchmark coverage
- clearer reproducibility presets for ablation studies

## License

A license file has not been added yet.
