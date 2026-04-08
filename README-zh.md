# SECI-Net

SECI-Net 是一个基于 PyTorch 的混合文本分类项目，实现了自定义 Transformer 编码器、自定义堆叠 LSTM、块状稀疏证据路由，以及可选的反事实干预学习。

这个仓库按“代码优先”的开源项目方式整理，重点是让程序员能快速读懂、快速上手。仓库中包含：

- `core/model/hybrid_text_model.py`：核心模型实现
- `core/data/text_dataset.py`：数据读取与 DataLoader 构建
- `main/train.py`：训练入口
- `main/download_datasets.py`：公开数据集下载脚本
- `main/split_dataset.py`：数据集划分脚本
- `main/counterfactual_generator.py`：反事实生成原型

## 项目结构

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

## 快速启动

### 1. 克隆项目

```bash
git clone git@github.com:ShuoMeng66/SECI-Net.git
cd SECI-Net
```

### 2. 创建虚拟环境

```bash
python -m venv SECINet
```

Windows PowerShell：

```powershell
.\SECINet\Scripts\Activate.ps1
```

Linux / macOS：

```bash
source SECINet/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载数据集

快速启动默认使用 `yelp_polarity`：

```bash
python main/download_datasets.py --dataset yelp_polarity --output_dir data/raw/yelp_polarity
```

如果是在 AutoDL、校园服务器或者访问 Hugging Face 不稳定的网络环境下，建议优先使用镜像模式：

```bash
python main/download_datasets.py --dataset yelp_polarity --output_dir data/raw/yelp_polarity --source hf-mirror
```

如果镜像模式仍然失败，再切到直接下载模式：

```bash
python main/download_datasets.py --dataset yelp_polarity --output_dir data/raw/yelp_polarity --source direct
```

当前内置的数据集快捷选项包括：

- `yelp_polarity`
- `ag_news`
- `imdb`

### 5. 划分数据集

如果下载后的数据集已经自带 `test` 划分，可以只把训练集切成 `train` 和 `valid`：

```bash
python main/split_dataset.py \
  --train_path data/raw/yelp_polarity/train.csv \
  --test_path data/raw/yelp_polarity/test.csv \
  --output_dir data/yelp_polarity
```

### 6. 开始训练

```bash
python main/train.py \
  --train_path data/yelp_polarity/train.csv \
  --valid_path data/yelp_polarity/valid.csv \
  --test_path data/yelp_polarity/test.csv \
  --save_dir checkpoints/yelp_polarity
```

## 训练输出

`main/train.py` 会在训练过程中显示按 batch 刷新的进度条，并在每个 epoch 打印完整指标，包括：

- 总损失 `loss`
- 反事实分类损失 `counterfactual_loss`
- 干预损失 `intervention_loss`
- 准确率 `accuracy`
- 宏平均精确率 `macro_precision`
- 宏平均召回率 `macro_recall`
- 宏平均 F1 `macro_f1`

如果提供验证集，最佳模型按验证集的 `macro_f1` 保存。

每次训练还会在 `--save_dir` 下产出一组便于做消融实验和结果复现的文件：

- `best_model.pt`
- `train_args.json`
- `vocab.json`
- `labels.json`
- `metrics_history.csv`
- `metrics_history.json`
- `summary.json`

## 数据格式

SECI-Net 支持 `csv`、`tsv` 和 `txt`。

对于 `csv` 和 `tsv`，默认列名是：

- `text`
- `label`

可选列包括：

- `counterfactual_text`
- `counterfactual_label`
- `time_column`
- `counterfactual_time_column`

对于纯文本文件，每行格式必须是：

```text
label<TAB>text
```

## 常用训练参数

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

## 反事实生成原型

如果已经有成对的反事实样本，可以运行独立生成器：

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

## 设计说明

这个仓库尽量避免把训练、指标和数据处理封装得过深。核心目标是让熟悉 PyTorch 的开发者可以顺着代码快速定位训练逻辑、模型结构和数据流，而不是先花时间拆抽象层。

## License

当前还没有补充 License 文件。
