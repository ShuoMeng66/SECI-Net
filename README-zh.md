[![贡献者][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stars][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]

<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SECI-Net</h3>

  <p align="center">
    证据感知文本分类与反事实学习，支持离线 GAN 增强
    <br />
    <a href="https://github.com/ShuoMeng66/SECI-Net"><strong>查看文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/ShuoMeng66/SECI-Net">演示</a>
    ·
    <a href="https://github.com/ShuoMeng66/SECI-Net/issues">报告 Bug</a>
    ·
    <a href="https://github.com/ShuoMeng66/SECI-Net/issues">提出需求</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>目录</summary>
  <ol>
    <li>
      <a href="#关于项目">关于项目</a>
      <ul>
        <li><a href="#技术栈">技术栈</a></li>
      </ul>
    </li>
    <li>
      <a href="#开始使用">开始使用</a>
      <ul>
        <li><a href="#准备工作">准备工作</a></li>
        <li><a href="#安装">安装</a></li>
      </ul>
    </li>
    <li><a href="#使用方式">使用方式</a></li>
    <li><a href="#路线图">路线图</a></li>
    <li><a href="#参与贡献">参与贡献</a></li>
    <li><a href="#许可证">许可证</a></li>
    <li><a href="#联系">联系</a></li>
    <li><a href="#致谢">致谢</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## 关于项目

SECI-Net 是一个基于 PyTorch 的证据感知文本分类项目，支持反事实监督和离线 GAN 增强。仓库面向研究与工程协作场景，尽量保持训练流程和数据处理的可读性与可追踪性。

当前公开版本主要包含核心代码与训练流程；论文写作目录、课程论文目录以及本地前端页面等内容不会进入开源主线。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


### 技术栈

* [PyTorch](https://pytorch.org/)
* [scikit-learn](https://scikit-learn.org/)
* [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- GETTING STARTED -->
## 开始使用

### 准备工作

* Python 3.10+
* pip

### 安装

1. 克隆仓库
   ```bash
   git clone git@github.com:ShuoMeng66/SECI-Net.git
   ```
2. 创建虚拟环境
   ```bash
   python -m venv .venv
   ```
3. 激活虚拟环境
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
4. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- USAGE EXAMPLES -->
## 使用方式

### 1. 下载数据集

```bash
python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity
```

如果网络环境不稳定，可尝试：

```bash
python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity \
  --source hf-mirror
```

### 2. 划分数据集

```bash
python main/split_dataset.py \
  --train_path data/raw/yelp_polarity/train.csv \
  --test_path data/raw/yelp_polarity/test.csv \
  --output_dir data/yelp_polarity
```

### 3. 训练 SECI-Net

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity
```

### 4. 启用 GAN 反事实增强

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

如需单独运行增强：

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

更多设计说明请参考 `GAN.md`。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- ROADMAP -->
## 路线图

- [ ] 更强的 aspect 建模与标注
- [ ] GAN 生成样本过滤或 rerank
- [ ] 扩展更多公开基准和复现实验配置

查看完整计划与已知问题请参考 [issues](https://github.com/ShuoMeng66/SECI-Net/issues)。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- CONTRIBUTING -->
## 参与贡献

欢迎提交改进。如果你有更好的想法，可以 fork 仓库并提交 PR，或者直接新开 issue。

1. Fork 本项目
2. 创建分支 (`git checkout -b feature/AmazingFeature`)
3. 提交修改 (`git commit -m "Add AmazingFeature"`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- LICENSE -->
## 许可证

当前仓库尚未补充 License 文件。

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- CONTACT -->
## 联系

项目主页: [https://github.com/ShuoMeng66/SECI-Net](https://github.com/ShuoMeng66/SECI-Net)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## 致谢

* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)
* [PyTorch](https://pytorch.org/)
* [Hugging Face Datasets](https://huggingface.co/docs/datasets/)

<p align="right">(<a href="#readme-top">返回顶部</a>)</p>


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
