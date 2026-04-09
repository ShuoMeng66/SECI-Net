[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SECI-Net</h3>

  <p align="center">
    Evidence-aware text classification with counterfactual supervision and offline GAN augmentation
    <br />
    <a href="https://github.com/ShuoMeng66/SECI-Net"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ShuoMeng66/SECI-Net">View Demo</a>
    ·
    <a href="https://github.com/ShuoMeng66/SECI-Net/issues">Report Bug</a>
    ·
    <a href="https://github.com/ShuoMeng66/SECI-Net/issues">Request Feature</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

SECI-Net is a PyTorch project for evidence-aware text classification, counterfactual supervision, and offline counterfactual augmentation. It is designed as a code-first, research-friendly repository that keeps training, data processing, and evaluation close to plain PyTorch for clarity and extensibility.

The repository includes a standalone `GAN.py` module for counterfactual review augmentation, plus a training pipeline that can optionally use GAN-generated samples as an offline augmentation stage.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Built With

* [PyTorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

This section explains how to get SECI-Net running locally.

### Prerequisites

* Python 3.10+
* pip

### Installation

1. Clone the repo
   ```bash
   git clone git@github.com:ShuoMeng66/SECI-Net.git
   ```
2. Create a virtual environment
   ```bash
   python -m venv .venv
   ```
3. Activate the environment
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

### 1. Download a public dataset

```bash
python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity
```

If Hugging Face access is unstable, use:

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

### 3. Train SECI-Net

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity
```

### 4. Enable GAN-based counterfactual augmentation

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

For a standalone augmentation run:

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

For API-based inspection:

```bash
python api_server.py
```

See `GAN.md` for the design rationale behind the offline augmentation stage.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ROADMAP -->
## Roadmap

- [ ] Stronger aspect modeling for counterfactual reviews
- [ ] Filtering or reranking for GAN-generated samples
- [ ] Broader benchmark coverage and reproducibility presets

See the [open issues](https://github.com/ShuoMeng66/SECI-Net/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTRIBUTING -->
## Contributing

Contributions are welcome. If you have a suggestion that would make this project better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m "Add AmazingFeature"`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

This repository does not include a license file yet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Project Link: [https://github.com/ShuoMeng66/SECI-Net](https://github.com/ShuoMeng66/SECI-Net)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [PyTorch](https://pytorch.org/)
* [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/ShuoMeng66/SECI-Net.svg?style=for-the-badge
[contributors-url]: https://github.com/ShuoMeng66/SECI-Net/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/ShuoMeng66/SECI-Net.svg?style=for-the-badge
[forks-url]: https://github.com/ShuoMeng66/SECI-Net/network/members
[stars-shield]: https://img.shields.io/github/stars/ShuoMeng66/SECI-Net.svg?style=for-the-badge
[stars-url]: https://github.com/ShuoMeng66/SECI-Net/stargazers
[issues-shield]: https://img.shields.io/github/issues/ShuoMeng66/SECI-Net.svg?style=for-the-badge
[issues-url]: https://github.com/ShuoMeng66/SECI-Net/issues
[license-shield]: https://img.shields.io/github/license/ShuoMeng66/SECI-Net.svg?style=for-the-badge
[license-url]: https://github.com/ShuoMeng66/SECI-Net/blob/main/LICENSE
