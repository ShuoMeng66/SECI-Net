# SECI-Net

SECI-Net 是一个基于 PyTorch 的证据感知文本分类与反事实学习项目，面向“可读代码、可复现实验、可复用模块”来组织。这个公开仓库聚焦研究代码本身：训练、数据、增强、测试和推理接口都集中在同一个代码库中，方便继续做实验和开源协作。

当前公开版本只保留代码主线。论文写作目录、课程论文目录和本地前端页面等内容不纳入开源工作流。

## 项目亮点

- 混合式文本分类器：结合上下文建模、序列建模和显式证据路由
- 支持事实样本 / 反事实样本成对监督
- 提供干预损失与 recoverability 学习目标
- 独立的 `GAN.py` 反事实评论离线增强模块
- 包含训练脚本、数据准备、checkpoint 工具和单元测试
- 提供可选的本地推理 API，便于快速查看模型输出

## 仓库结构

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

## 安装

### 1. 克隆仓库

```bash
git clone git@github.com:ShuoMeng66/SECI-Net.git
cd SECI-Net
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

建议使用 Python 3.10 及以上版本。

## 数据格式

SECI-Net 支持 `csv`、`tsv` 和 `txt`。

结构化文件默认字段：

- `text`
- `label`

可选的反事实字段：

- `counterfactual_text`
- `counterfactual_label`
- `time_column`
- `counterfactual_time_column`

纯文本格式要求每行是：

```text
label<TAB>text
```

## 快速开始

### 1. 下载数据集

内置快捷数据集包括：

- `yelp_polarity`
- `ag_news`
- `imdb`

示例：

```bash
python main/download_datasets.py \
  --dataset yelp_polarity \
  --output_dir data/raw/yelp_polarity
```

如果在校园服务器、AutoDL 或访问 Hugging Face 不稳定的网络环境，建议优先尝试：

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

### 3. 训练分类模型

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity
```

## 基于 GAN 的反事实增强

当前版本支持“先离线增强、再分类训练”的反事实评论工作流：

1. 使用已有成对反事实样本训练独立 GAN
2. 为训练集里缺失反事实字段的样本生成候选反事实评论
3. 把生成结果回填到训练记录
4. 继续执行常规的 SECI-Net 分类训练

直接在主训练脚本里开启：

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

几个重要默认行为：

- GAN 只增强训练集
- 默认保留人工标注反事实，不直接覆盖
- 如果训练集里没有可用的成对反事实样本，GAN 阶段会自动跳过，不中断分类训练

也可以单独运行包装脚本：

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

设计说明见 [`GAN.md`](./GAN.md)。

## 训练产物

每次训练会在 `--save_dir` 下写出实验产物，包括：

- `best_model.pt`
- `train_args.json`
- `vocab.json`
- `labels.json`
- `metrics_history.csv`
- `metrics_history.json`
- `step_metrics.csv`
- `summary.json`

如果开启 GAN 增强，默认还会在 `--save_dir/gan` 下额外生成：

- `generated_counterfactuals.csv`
- `metrics.json`
- `counterfactual_generator.pt`
- `counterfactual_discriminator.pt`
- `counterfactual_vocab.json`
- `counterfactual_labels.json`

## 本地推理 API

如果想快速用本地 checkpoint 做推理，可以运行：

```bash
python api_server.py
```

这个接口主要用于本地实验和结果查看，不是生产部署方案。

## 测试

当前单元测试与 smoke test 可以通过下面的命令运行：

```bash
python -m unittest tests.test_seci_net_v2 tests.test_gan
```

测试覆盖了：

- SECI-Net 的张量 shape 和 checkpoint 流程
- GAN 数据整理和缺省字段回退
- GAN 前向计算与 CPU 单轮训练
- 启用 / 不启用 GAN 增强时的训练集成流程

## 设计理念

这个项目尽量避免过度抽象。模型结构、数据流、损失函数和训练循环都尽量贴近原始 PyTorch 写法，让研究者和工程同学都能快速顺着代码理解具体实现。

## 后续计划

后续可继续推进的方向包括：

- 更强的 aspect 建模与标注方式
- 对 GAN 生成样本增加过滤或 rerank
- 扩展更多基准数据集
- 增加更标准的消融实验配置

## License

当前仓库还没有补充 License 文件。
