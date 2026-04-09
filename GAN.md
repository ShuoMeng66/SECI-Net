# 反事实评论 GAN 模块说明

## 简介

`GAN.py` 提供一个面向反事实评论的离线增强模块，目标是从已标注的事实评论与反事实评论配对样本中学习生成器，并补齐训练集中缺失的 `counterfactual_*` 字段。模块以“条件式生成 + 多头判别 + 局部改写约束”为核心设计，面向长文本与可控编辑场景。

## 代码组成

主要对象与职责如下：

- `CounterfactualReviewRecord`：反事实评论训练样本的结构化表示
- `GANTrainingConfig`：GAN 训练超参数配置
- `CounterfactualReviewGAN`：生成器与判别器封装
- `ReviewCounterfactualGenerator`：条件生成器（带编辑门控）
- `ReviewCounterfactualDiscriminator`：判别器（真伪、标签、aspect 三头）
- `ReviewGANTrainer`：GAN 训练与生成导出
- `train_gan_and_augment_records(...)`：高层入口，完成训练、生成与回填

## 核心机制

### 1. 条件生成

生成器输入包含：

- 原评论编码摘要
- 目标标签 embedding
- 目标 aspect embedding
- 时间差特征
- 噪声向量

生成过程为“原评论 + 目标条件”的定向改写，而非无条件生成。

### 2. 编辑门控

`edit_gate` 约束改写范围，鼓励局部编辑，降低整段重写概率。

### 3. 多头判别

判别器同时输出：

- `real / fake` 概率
- 目标标签对齐
- 目标 aspect 对齐

### 4. 训练损失

训练过程中固定组合以下损失：

- 重构损失
- 对抗损失
- 标签对齐损失
- aspect 对齐损失
- 语义锚定损失
- 编辑稀疏损失

## 数据字段约定

GAN 训练依赖成对反事实样本，字段要求与主训练流程一致：

- `text`
- `label`
- `counterfactual_text`
- `counterfactual_label`

可选字段：

- `time_column`
- `counterfactual_time_column`

aspect 默认通过关键词规则推断，覆盖：

- `service`
- `logistics`
- `product`
- `price`
- `experience`
- `environment`
- `generic`

若缺失时间信息，则事实时间复制到反事实时间；若无 aspect 命中，则回退为 `generic`。

## 使用方式

### 1. 主训练脚本中启用

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

默认仅补齐缺失反事实样本，验证集与测试集不参与增强。

### 2. 独立增强脚本

```bash
python main/counterfactual_generator.py \
  --train_path data/your_dataset/train.csv \
  --counterfactual_text_column counterfactual_text \
  --counterfactual_label_column counterfactual_label \
  --output_path outputs/generated_counterfactuals.csv
```

## 产物与输出

启用 GAN 增强后，默认生成以下文件（`--save_dir/gan`）：

- `generated_counterfactuals.csv`：合成反事实样本
- `metrics.json`：训练摘要与统计
- `counterfactual_generator.pt`
- `counterfactual_discriminator.pt`
- `counterfactual_vocab.json`
- `counterfactual_labels.json`
- `counterfactual_aspects.json`
- `gan_config.json`
- `history.json`

## 与训练链路的关系

GAN 增强为离线阶段，整体训练顺序为：

1. 从训练集筛出成对反事实样本训练 GAN
2. 对缺失反事实字段的样本生成候选反事实评论
3. 合并生成样本回训练集
4. 进入常规 SECI-Net 训练流程

验证集与测试集不参与增强，确保评估纯净性。
