# DVC 版本管理文档

## 1. 概述

### 1.1 DVC 版本管理系统的目的和重要性

DVC（Data Version Control）是一个用于机器学习项目的版本控制系统，它扩展了 Git 的功能，使得大型数据文件、模型文件和实验结果能够被有效地版本控制。

**主要目的：**

- **数据版本管理**：跟踪数据集的变化，包括原始数据、标注数据、处理后的数据等
- **模型版本管理**：保存训练过程中的模型检查点和最终模型权重
- **实验可复现性**：确保实验结果可以完全复现，包括数据版本、代码版本和配置参数
- **存储效率**：通过缓存机制避免重复存储相同的数据文件
- **团队协作**：在团队中共享数据版本，同时保持数据安全性

**重要性：**

- 在机器学习项目中，数据和模型文件通常很大，无法直接放入 Git 仓库
- 数据的微小变化可能导致模型性能的显著差异，需要精确跟踪
- 实验的可复现性是科学研究的基本要求，也是工业应用的基础
- 在内网服务器环境中，数据需要本地化存储，不能上传到公共云存储

### 1.2 Git + DVC 的分工说明

**Git 负责：**

- 源代码版本控制（Python 脚本、配置文件等）
- `.dvc` 文件（记录数据文件的元信息和哈希值）
- 项目配置文件（[`requirements.txt`](requirements.txt:1)、[`README.md`](README.md:1) 等）
- 文档文件（本文档、API 文档等）
- 小型文本文件和脚本文件

**DVC 负责：**

- 大型数据文件（数据集、图像、标注文件等）
- 模型权重文件（`.pt`、`.pth`、`.onnx` 等）
- 训练日志和运行日志
- 实验结果文件（图表、评估指标等）
- 任何超过 Git 管理能力的大型文件

**工作流程：**

```
用户修改数据 → DVC 跟踪数据变化 → 更新 .dvc 文件 → Git 提交 .dvc 文件
```

## 2. 项目结构

### 2.1 DVC 管理的目录和文件

以下目录已添加到 DVC 版本控制：

| 目录 | .dvc 文件 | 说明 |
|------|-----------|------|
| [`data/`](data.dvc:6) | [`data.dvc`](data.dvc:1) | 原始数据集、标注图像、处理后的数据集（train/val/test 划分） |
| [`logs/`](logs.dvc:1) | [`logs.dvc`](logs.dvc:1) | 日志文件 |
| [`training/test_results/`](training/test_results.dvc:1) | [`training/test_results.dvc`](training/test_results.dvc:1) | 测试结果、图表等实验数据 |
| [`checkpoints/`](checkpoints.dvc:6) | [`checkpoints.dvc`](checkpoints.dvc:1) | 模型检查点（.pt、.pth 等） |
| [`runs/`](runs.dvc:1) | [`runs.dvc`](runs.dvc:1) | YOLO 训练运行目录 |
| [`training/model/`](training/model.dvc:1) | [`training/model.dvc`](training/model.dvc:1) | 模型权重文件 |

**DVC 配置文件：**

- [`.dvc/config`](.dvc/config:1) - DVC 配置文件（远程存储配置等）
- [`.dvcignore`](.dvcignore:1) - DVC 忽略规则文件
- `.dvc/cache/` - DVC 缓存目录（本地存储，不提交到 Git）

### 2.2 Git 管理的文件

**源代码：**

- [`training/scripts/`](training/scripts:1) - 训练脚本
- [`training/src/`](training/src:1) - 源代码
- [`training/configs/`](training/configs:1) - 配置文件
- [`app/`](app:1) - 应用程序代码

**配置文件：**

- [`requirements.txt`](requirements.txt:1) - Python 依赖
- [`.gitignore`](.gitignore:1) - Git 忽略规则
- [`.dvcignore`](.dvcignore:1) - DVC 忽略规则

**文档：**

- [`README.md`](README.md:1) - 项目说明文档
- [`DVC_VERSION_CONTROL.md`](DVC_VERSION_CONTROL.md:1) - 本文档
- [`training/docs/`](training/docs:1) - 训练相关文档

**DVC 元数据文件：**

- [`data.dvc`](data.dvc:1)
- [`logs.dvc`](logs.dvc:1)
- [`checkpoints.dvc`](checkpoints.dvc:1)
- [`runs.dvc`](runs.dvc:1)
- [`training/model.dvc`](training/model.dvc:1)
- [`training/test_results.dvc`](training/test_results.dvc:1)

### 2.3 项目目录结构图

```
Aerovision-V1/
├── .dvc/                    # DVC 配置目录（不提交到 Git）
│   ├── cache/              # DVC 缓存（本地存储）
│   ├── config              # DVC 配置文件
│   └── tmp/                # DVC 临时文件
├── .dvcignore              # DVC 忽略规则（Git 管理）
├── .gitignore              # Git 忽略规则
├── data/                   # 数据集（DVC 管理）
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   └── splits/             # train/val/test 划分
├── logs/                   # 日志文件（DVC 管理）
├── checkpoints/            # 模型检查点（DVC 管理）
├── runs/                   # YOLO 训练运行目录（DVC 管理）
├── training/
│   ├── scripts/            # 训练脚本（Git 管理）
│   ├── src/                # 源代码（Git 管理）
│   ├── configs/            # 配置文件（Git 管理）
│   ├── model/              # 模型权重（DVC 管理）
│   ├── test_results/       # 测试结果（DVC 管理）
│   └── docs/               # 文档（Git 管理）
├── app/                    # 应用程序（Git 管理）
├── requirements.txt        # Python 依赖（Git 管理）
├── README.md               # 项目说明（Git 管理）
└── DVC_VERSION_CONTROL.md  # 本文档（Git 管理）
```

## 3. 快速开始

### 3.1 克隆项目并获取数据

**步骤 1：克隆 Git 仓库**

```bash
# 克隆项目
git clone https://github.com/your-username/Aerovision-V1.git
cd Aerovision-V1
```

**步骤 2：安装依赖**

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 确保 DVC 已安装
pip install dvc
```

**步骤 3：拉取 DVC 管理的数据**

```bash
# 拉取所有 DVC 管理的数据
dvc pull

# 或者只拉取特定的数据
dvc pull data.dvc
dvc pull checkpoints.dvc
dvc pull training/model.dvc
```

**步骤 4：验证数据**

```bash
# 检查数据状态
dvc status

# 查看数据文件信息
dvc data
```

### 3.2 提交数据更改

**步骤 1：修改数据**

```bash
# 添加新的数据文件到 data/ 目录
# 或者修改现有的数据文件
```

**步骤 2：更新 DVC 跟踪**

```bash
# 检查哪些数据发生了变化
dvc status

# 更新 DVC 跟踪（如果添加了新文件）
dvc add data/new_files/

# 或者如果只是修改了已跟踪的文件
# DVC 会自动检测变化
```

**步骤 3：提交到 Git**

```bash
# 查看 Git 状态
git status

# 添加 .dvc 文件到 Git
git add data.dvc
git add training/model.dvc
# ... 添加其他相关的 .dvc 文件

# 提交更改
git commit -m "更新数据集版本 v2.0

- 添加了 100 个新的训练样本
- 更新了数据划分比例
- 修复了标注错误"

# 推送到远程仓库
git push origin main
```

**步骤 4：推送 DVC 数据（如果使用远程存储）**

```bash
# 注意：本项目使用本地存储，此步骤仅在使用远程存储时需要
# dvc push
```

## 4. 日常使用

### 4.1 添加新数据到 DVC

**场景 1：添加新的数据目录**

```bash
# 假设你有一个新的数据目录 data/experiment_1/
dvc add data/experiment_1/

# 这会创建一个 data/experiment_1.dvc 文件
# 将 .dvc 文件添加到 Git
git add data/experiment_1.dvc
git commit -m "添加实验数据集 experiment_1"
git push
```

**场景 2：向现有 DVC 目录添加新文件**

```bash
# 向已跟踪的 data/ 目录添加新文件
# DVC 会自动检测变化

# 检查状态
dvc status

# 更新 .dvc 文件
dvc add data/

# 提交到 Git
git add data.dvc
git commit -m "向数据集添加新样本"
git push
```

**场景 3：添加大型模型文件**

```bash
# 添加训练好的模型
dvc add training/model/best_model.pt

# 提交到 Git
git add training/model/best_model.pt.dvc
git commit -m "添加最佳模型权重"
git push
```

### 4.2 提交数据版本

**完整的数据版本提交流程：**

```bash
# 1. 检查当前状态
dvc status
git status

# 2. 更新所有变化的 DVC 文件
dvc add data/
dvc add checkpoints/
dvc add training/model/
dvc add training/test_results/

# 3. 查看变化
git diff data.dvc

# 4. 添加到 Git
git add *.dvc
git add training/*.dvc

# 5. 提交（使用详细的提交信息）
git commit -m "版本 v2.1.0: 模型性能提升

数据变化:
- 更新训练集：新增 500 个样本
- 优化数据划分：train/val/test = 0.8/0.1/0.1

模型变化:
- 新增检查点：epoch_50.pt (mAP: 0.85)
- 最佳模型：best_model.pt (mAP: 0.87)

实验结果:
- 测试准确率：92.3%
- 推理速度：15ms/image

配置:
- 训练轮数：50 epochs
- 批次大小：32
- 学习率：0.001

关联代码版本: commit abc1234"

# 6. 推送到远程
git push origin main
```

### 4.3 切换数据版本

**场景 1：切换到历史数据版本**

```bash
# 查看历史提交
git log --oneline

# 切换到特定的 Git 提交
git checkout <commit-hash>

# 拉取对应版本的数据
dvc checkout

# 验证数据版本
dvc data
```

**场景 2：比较不同版本的数据**

```bash
# 查看数据变化
git diff HEAD~1 data.dvc

# 查看特定文件的变化
dvc diff data.dvc
```

**场景 3：恢复到之前的数据版本**

```bash
# 恢复 data/ 目录到上一个版本
git checkout HEAD~1 -- data.dvc
dvc checkout data.dvc

# 提交恢复操作
git add data.dvc
git commit -m "回退数据版本到 v1.0"
```

### 4.4 查看数据历史

**查看数据版本历史：**

```bash
# 查看 data/ 目录的版本历史
git log --oneline data.dvc

# 查看详细的版本信息
git log -p data.dvc

# 查看所有 DVC 文件的历史
git log --oneline *.dvc training/*.dvc
```

**查看数据统计信息：**

```bash
# 查看当前数据集大小
dvc data

# 查看特定目录的数据信息
dvc data data/

# 查看所有 DVC 管理的数据
dvc data --all
```

**查看数据变化摘要：**

```bash
# 查看当前版本与上一个版本的差异
dvc diff

# 查看特定版本的差异
dvc diff HEAD~1
```

## 5. 实验可复现性

### 5.1 如何记录实验配置

**方法 1：使用配置文件**

```yaml
# training/configs/experiments/exp_001.yaml
experiment_name: "baseline_model_v1"
data_version: "v2.0"
code_version: "commit abc1234"

data:
  train_path: "data/train/"
  val_path: "data/val/"
  test_path: "data/test/"
  num_classes: 10

model:
  architecture: "YOLOv8"
  pretrained: true
  input_size: [640, 640]

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  optimizer: "Adam"
  weight_decay: 0.0005

augmentation:
  horizontal_flip: true
  rotation: 15
  brightness: 0.2
  contrast: 0.2

logging:
  save_dir: "runs/exp_001/"
  log_interval: 10
  save_checkpoint_interval: 5
```

**方法 2：在提交信息中记录**

```bash
git commit -m "实验 exp_001: 基线模型训练

实验配置:
- 模型: YOLOv8-n
- 数据版本: v2.0 (data.dvc @ commit def5678)
- 训练轮数: 50 epochs
- 批次大小: 32
- 学习率: 0.001

数据增强:
- 水平翻转: 是
- 旋转: ±15°
- 亮度: ±0.2
- 对比度: ±0.2

超参数:
- 优化器: Adam
- 权重衰减: 0.0005
- 学习率调度: CosineAnnealing

结果:
- 训练损失: 0.123
- 验证 mAP: 0.85
- 测试准确率: 87.5%

代码版本: commit abc1234
配置文件: training/configs/experiments/exp_001.yaml"
```

### 5.2 如何保存实验结果

**步骤 1：保存模型和检查点**

```bash
# 训练完成后，将模型添加到 DVC
dvc add checkpoints/exp_001/
dvc add training/model/exp_001_best.pt
```

**步骤 2：保存实验结果**

```bash
# 保存测试结果和图表
dvc add training/test_results/exp_001/
```

**步骤 3：保存训练日志**

```bash
# 保存训练日志
dvc add runs/exp_001/
dvc add logs/exp_001.log
```

**步骤 4：创建实验记录文件**

```bash
# 创建实验记录文件
cat > experiments/exp_001.md << EOF
# 实验 exp_001: 基线模型训练

## 基本信息
- 实验日期: 2024-01-15
- 实验人员: 张三
- 实验目的: 建立基线模型性能

## 配置信息
- 配置文件: training/configs/experiments/exp_001.yaml
- 数据版本: v2.0 (data.dvc @ commit def5678)
- 代码版本: commit abc1234

## 超参数
- 模型架构: YOLOv8-n
- 训练轮数: 50
- 批次大小: 32
- 学习率: 0.001
- 优化器: Adam

## 数据增强
- 水平翻转: 是
- 旋转: ±15°
- 亮度: ±0.2
- 对比度: ±0.2

## 结果
- 训练损失: 0.123
- 验证 mAP: 0.85
- 测试准确率: 87.5%
- 推理速度: 15ms/image

## 模型文件
- 最佳模型: training/model/exp_001_best.pt
- 检查点: checkpoints/exp_001/

## 结果文件
- 测试结果: training/test_results/exp_001/
- 训练日志: runs/exp_001/
- 日志文件: logs/exp_001.log

## 结论
基线模型性能良好，可以作为后续优化的参考。

## 备注
无
EOF

# 提交到 Git
git add experiments/exp_001.md
```

**步骤 5：提交所有更改**

```bash
# 添加所有 .dvc 文件
git add *.dvc training/*.dvc

# 添加实验记录
git add experiments/exp_001.md

# 提交
git commit -m "实验 exp_001 完成

- 训练基线模型
- 验证 mAP: 0.85
- 测试准确率: 87.5%
- 详细记录见: experiments/exp_001.md"

git push origin main
```

### 5.3 如何复现历史实验

**步骤 1：找到实验的 Git 提交**

```bash
# 查看实验历史
git log --oneline --grep="exp_001"

# 或者查看实验记录文件
git log --oneline experiments/exp_001.md
```

**步骤 2：切换到对应的代码版本**

```bash
# 切换到实验的代码版本
git checkout <commit-hash>
```

**步骤 3：恢复数据版本**

```bash
# 查看实验使用的数据版本
git show <commit-hash>:experiments/exp_001.md | grep "数据版本"

# 切换到对应的数据版本
git checkout <data-commit-hash> -- data.dvc
dvc checkout data.dvc
```

**步骤 4：恢复模型和结果**

```bash
# 恢复模型文件
git checkout <commit-hash> -- training/model.dvc
dvc checkout training/model.dvc

# 恢复实验结果
git checkout <commit-hash> -- training/test_results.dvc
dvc checkout training/test_results.dvc
```

**步骤 5：验证复现**

```bash
# 加载模型并运行测试
python training/scripts/test.py \
    --model training/model/exp_001_best.pt \
    --data data/test/ \
    --config training/configs/experiments/exp_001.yaml

# 比较结果是否一致
# 应该得到与原始实验相同的结果
```

**步骤 6：重新训练（如果需要）**

```bash
# 使用相同的配置重新训练
python training/scripts/train.py \
    --config training/configs/experiments/exp_001.yaml \
    --output runs/exp_001_reproduce/
```

## 6. 版本信息

### 6.1 如何记录详细的版本信息

**版本信息记录最佳实践：**

```bash
# 使用结构化的提交信息
git commit -m "[版本 v2.1.0] 模型性能提升

数据变更:
- 版本: v2.0 → v2.1
- 变更类型: 数据增强
- 详细说明:
  * 新增 500 个训练样本
  * 优化数据划分比例
  * 修复 10 个标注错误
- 影响范围: data/train/, data/val/, data/test/

模型变更:
- 版本: v1.5 → v2.1
- 变更类型: 架构优化
- 详细说明:
  * 改进骨干网络
  * 优化注意力机制
  * 调整损失函数权重
- 影响范围: training/model/, checkpoints/

代码变更:
- 版本: commit abc1234 → def5678
- 主要修改:
  * training/scripts/train_detection.py: 改进数据加载
  * training/src/models/yolo.py: 优化模型架构
  * training/configs/training.yaml: 更新超参数

性能对比:
- mAP: 0.85 → 0.87 (+2.4%)
- 准确率: 87.5% → 89.2% (+1.7%)
- 推理速度: 15ms → 14ms (+7.1%)

测试结果:
- 测试集准确率: 89.2%
- 验证集 mAP: 0.87
- 训练时间: 3.5 小时

配置信息:
- 训练轮数: 50 epochs
- 批次大小: 32
- 学习率: 0.001
- 数据增强: 水平翻转, 旋转, 亮度调整

依赖变更:
- 新增依赖: 无
- 更新依赖: torch==2.0.0, torchvision==0.15.0
- 移除依赖: 无

兼容性:
- Python 版本: 3.8+
- PyTorch 版本: 2.0.0+
- CUDA 版本: 11.7+

已知问题:
- 无

后续计划:
- 尝试更大的模型 (YOLOv8-m)
- 优化推理速度
- 添加更多数据增强策略

关联 Issue: #123
关联 PR: #456"
```

### 6.2 如何关联代码版本和数据版本

**方法 1：在提交信息中明确关联**

```bash
git commit -m "更新数据版本 v2.1

数据版本: v2.1 (data.dvc @ commit abc1234)
代码版本: def5678 (当前提交)

关联说明:
- 此数据版本与代码版本 def5678 兼容
- 使用此数据版本需要代码版本 def5678 或更高
- 数据格式变更需要对应的代码更新"
```

**方法 2：使用标签标记版本**

```bash
# 为数据版本打标签
git tag -a data-v2.1 -m "数据版本 v2.1

- 新增 500 个样本
- 优化数据划分
- 修复标注错误

关联代码版本: def5678
关联模型版本: model-v2.1"

# 为模型版本打标签
git tag -a model-v2.1 -m "模型版本 v2.1

- mAP: 0.87
- 准确率: 89.2%
- 推理速度: 14ms

关联数据版本: data-v2.1
关联代码版本: def5678"

# 推送标签
git push origin --tags
```

**方法 3：创建版本映射文件**

```bash
# 创建 VERSION_MAPPING.md
cat > VERSION_MAPPING.md << EOF
# 版本映射表

## 版本 v2.1.0

| 组件 | 版本 | Git 提交 | 说明 |
|------|------|----------|------|
| 数据 | v2.1 | abc1234 | 新增 500 个样本 |
| 模型 | v2.1 | def5678 | mAP: 0.87 |
| 代码 | def5678 | def5678 | 改进数据加载 |
| 配置 | v2.1 | ghi7890 | 更新超参数 |

## 版本 v2.0.0

| 组件 | 版本 | Git 提交 | 说明 |
|------|------|----------|------|
| 数据 | v2.0 | xyz9999 | 初始数据集 |
| 模型 | v1.5 | uvw8888 | mAP: 0.85 |
| 代码 | uvw8888 | uvw8888 | 基线实现 |
| 配置 | v2.0 | rst7777 | 基础配置 |

## 兼容性说明

- v2.1.0 与 v2.0.0 数据格式兼容
- v2.1.0 模型需要 v2.1.0 数据版本
- v2.1.0 代码向后兼容 v2.0.0 模型
EOF

# 提交到 Git
git add VERSION_MAPPING.md
git commit -m "添加版本映射表"
git push origin main
```

**方法 4：使用 DVC 的实验跟踪功能**

```bash
# 创建实验记录
dvc exp run -n exp_001 \
    -S model.architecture=YOLOv8-n \
    -S training.epochs=50 \
    -S training.batch_size=32 \
    -S training.learning_rate=0.001

# 查看实验列表
dvc exp show

# 查看实验详情
dvc exp show exp_001

# 提交实验
dvc exp commit exp_001

# 将实验分支合并到主分支
git merge exp_001
```

## 7. 常见问题

### 7.1 DVC 基础问题

**Q1: DVC 和 Git 有什么区别？**

A: Git 用于管理小型文件（源代码、配置文件等），而 DVC 用于管理大型文件（数据集、模型文件等）。DVC 通过 `.dvc` 文件记录大型文件的元信息，这些 `.dvc` 文件由 Git 管理。实际的大型文件存储在 DVC 缓存中。

**Q2: 为什么我的数据文件没有被 DVC 跟踪？**

A: 可能的原因：
1. 文件在 [`.dvcignore`](.dvcignore:1) 中被忽略
2. 文件没有被 `dvc add` 命令添加
3. 文件在 `.gitignore` 中被忽略（但 DVC 不应该受此影响）

解决方法：
```bash
# 检查 .dvcignore
cat .dvcignore

# 添加文件到 DVC
dvc add path/to/file

# 检查状态
dvc status
```

**Q3: 如何查看 DVC 缓存使用了多少空间？**

A: 使用以下命令：
```bash
# 查看缓存大小
dvc cache dir

# 查看缓存统计信息
dvc cache dir -v
```

**Q4: 如何清理 DVC 缓存？**

A: 谨慎清理，因为缓存中的文件可能被多个版本使用：
```bash
# 清理未使用的缓存文件
dvc gc

# 清理所有缓存（危险操作！）
dvc cache dir --force
```

### 7.2 数据同步问题

**Q5: 为什么 `dvc pull` 很慢？**

A: 可能的原因：
1. 数据文件很大
2. 网络速度慢
3. 本地缓存空间不足

解决方法：
```bash
# 只拉取需要的文件
dvc pull data.dvc

# 使用压缩传输
dvc pull --compress

# 查看进度
dvc pull -v
```

**Q6: 如何解决 DVC 文件冲突？**

A: DVC 文件冲突通常发生在多人同时修改同一个 `.dvc` 文件时：

```bash
# 查看冲突
git status

# 解决冲突（选择一个版本或手动合并）
git checkout --theirs data.dvc  # 使用远程版本
git checkout --ours data.dvc    # 使用本地版本

# 或者手动编辑 data.dvc 文件

# 标记冲突已解决
git add data.dvc

# 提交
git commit -m "解决 DVC 文件冲突"
```

**Q7: 如何恢复误删的数据文件？**

A: 如果数据文件被误删，可以从 DVC 缓存恢复：

```bash
# 恢复所有 DVC 管理的文件
dvc checkout

# 恢复特定的 .dvc 文件
dvc checkout data.dvc

# 如果缓存也被删除了，需要从远程拉取
dvc pull
```

### 7.3 实验复现问题

**Q8: 为什么复现实验时结果不一致？**

A: 可能的原因：
1. 数据版本不匹配
2. 代码版本不匹配
3. 随机种子不同
4. 硬件环境不同（GPU/CPU）
5. 依赖库版本不同

解决方法：
```bash
# 1. 确保数据版本正确
git checkout <data-commit> -- data.dvc
dvc checkout data.dvc

# 2. 确保代码版本正确
git checkout <code-commit>

# 3. 设置相同的随机种子
# 在代码中添加：
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# 4. 使用相同的硬件配置
# 检查 GPU 信息
nvidia-smi

# 5. 确保依赖版本一致
pip freeze > requirements.txt
pip install -r requirements.txt
```

**Q9: 如何记录实验的随机种子？**

A: 在配置文件中记录随机种子：

```yaml
# training/configs/experiments/exp_001.yaml
experiment:
  name: "exp_001"
  random_seed: 42  # 固定随机种子

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
```

在代码中使用：
```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config['experiment']['random_seed'])
```

### 7.4 性能优化问题

**Q10: 如何加快 DVC 操作速度？**

A: 优化方法：

```bash
# 1. 使用并行操作
dvc pull --jobs 8

# 2. 增加缓存大小
dvc config cache.dir .dvc/cache
dvc config cache.type symlink,hardlink,copy

# 3. 使用硬链接（在支持的文件系统上）
dvc config cache.type hardlink

# 4. 禁用不必要的检查
dvc checkout --no-exec
```

**Q11: 如何减少 DVC 缓存占用空间？**

A: 优化方法：

```bash
# 1. 清理未使用的缓存
dvc gc

# 2. 使用压缩
dvc config cache.compression true

# 3. 定期清理旧的实验数据
# 手动删除不需要的实验目录
dvc add training/test_results/
git add training/test_results.dvc
git commit -m "清理旧实验数据"
```

### 7.5 内网环境问题

**Q12: 在内网环境中如何共享 DVC 数据？**

A: 由于本项目使用本地存储，有以下几种方法：

```bash
# 方法 1: 使用共享存储（如果有）
# 将 .dvc/cache 目录挂载到共享存储
dvc config cache.dir /shared/path/.dvc/cache

# 方法 2: 手动复制数据
# 在源机器上打包数据
tar -czf data.tar.gz data/

# 复制到目标机器
scp data.tar.gz user@target:/path/to/project/

# 在目标机器上解压
tar -xzf data.tar.gz

# 方法 3: 使用移动存储
# 将数据复制到移动硬盘
# 然后在内网服务器上复制数据
```

**Q13: 如何在内网服务器上部署项目？**

A: 详细步骤见第 8 节"内网服务器部署"。

## 8. 内网服务器部署

### 8.1 部署准备

**环境要求：**

- 操作系统：Linux（推荐 Ubuntu 20.04+）
- Python 版本：3.8+
- GPU：NVIDIA GPU（可选，用于训练）
- 存储空间：至少 100GB 可用空间（根据数据集大小调整）
- 内存：至少 16GB RAM

**软件依赖：**

```bash
# 安装 Python
sudo apt update
sudo apt install python3.8 python3-pip python3-venv

# 安装 Git
sudo apt install git

# 安装 CUDA（如果需要 GPU）
# 参考 NVIDIA 官方文档
```

### 8.2 部署步骤

**步骤 1：克隆项目**

```bash
# 克隆 Git 仓库
git clone https://github.com/your-username/Aerovision-V1.git
cd Aerovision-V1
```

**步骤 2：创建虚拟环境**

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

**步骤 3：安装依赖**

```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt

# 确保 DVC 已安装
pip install dvc
```

**步骤 4：传输数据**

由于项目使用本地存储，需要手动传输数据文件：

**方法 A：使用移动存储（推荐）**

```bash
# 在开发机器上打包数据
tar -czf data_package.tar.gz \
    data/ \
    logs/ \
    checkpoints/ \
    runs/ \
    training/model/ \
    training/test_results/ \
    .dvc/cache/

# 复制到移动硬盘
# 然后在内网服务器上复制数据

# 在内网服务器上解压
tar -xzf data_package.tar.gz
```

**方法 B：使用网络传输（如果内网有访问权限）**

```bash
# 使用 rsync 同步数据
rsync -avz --progress \
    user@dev-machine:/path/to/Aerovision-V1/data/ \
    /path/to/Aerovision-V1/data/

rsync -avz --progress \
    user@dev-machine:/path/to/Aerovision-V1/.dvc/cache/ \
    /path/to/Aerovision-V1/.dvc/cache/

# 同步其他 DVC 管理的目录
```

**方法 C：使用共享存储（如果有）**

```bash
# 将数据复制到共享存储
cp -r data/ /shared/path/Aerovision-V1/
cp -r .dvc/cache/ /shared/path/Aerovision-V1/.dvc/

# 在内网服务器上创建符号链接
ln -s /shared/path/Aerovision-V1/data data
ln -s /shared/path/Aerovision-V1/.dvc/cache .dvc/cache
```

**步骤 5：验证数据**

```bash
# 检查 DVC 状态
dvc status

# 查看数据信息
dvc data

# 验证数据完整性
dvc verify
```

**步骤 6：测试环境**

```bash
# 测试 GPU（如果可用）
python training/test_script/check_gpu.py

# 测试环境配置
python training/test_script/verify_env.py

# 运行简单的前向传播测试
python training/test_script/simple_forward.py
```

### 8.3 数据同步说明

**从开发机器同步到内网服务器：**

```bash
# 1. 在开发机器上打包数据
tar -czf data_update.tar.gz \
    data/ \
    .dvc/cache/

# 2. 传输到内网服务器
scp data_update.tar.gz user@server:/path/to/Aerovision-V1/

# 3. 在内网服务器上解压
tar -xzf data_update.tar.gz

# 4. 验证数据
dvc status
```

**从内网服务器同步到开发机器：**

```bash
# 1. 在内网服务器上打包数据
tar -czf results_package.tar.gz \
    training/model/ \
    training/test_results/ \
    checkpoints/ \
    logs/

# 2. 传输到开发机器
scp user@server:/path/to/Aerovision-V1/results_package.tar.gz ./

# 3. 在开发机器上解压
tar -xzf results_package.tar.gz

# 4. 提交到 Git
dvc add training/model/
dvc add training/test_results/
git add training/model.dvc training/test_results.dvc
git commit -m "从内网服务器同步实验结果"
```

**增量同步（只传输变化的文件）：**

```bash
# 使用 rsync 进行增量同步
rsync -avz --progress --delete \
    user@dev-machine:/path/to/Aerovision-V1/data/ \
    /path/to/Aerovision-V1/data/

rsync -avz --progress --delete \
    user@dev-machine:/path/to/Aerovision-V1/.dvc/cache/ \
    /path/to/Aerovision-V1/.dvc/cache/
```

### 8.4 内网服务器上的工作流程

**训练模型：**

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 确保数据是最新的
dvc status

# 3. 运行训练
python training/scripts/train_detection.py \
    --config training/configs/experiments/exp_001.yaml \
    --output runs/exp_001/

# 4. 保存模型和结果
dvc add training/model/best_model.pt
dvc add training/test_results/exp_001/

# 5. 记录实验信息
cat > experiments/exp_001_server.md << EOF
# 实验 exp_001 (内网服务器)

## 基本信息
- 训练日期: $(date +%Y-%m-%d)
- 训练机器: 内网服务器
- GPU型号: $(nvidia-smi --query-gpu=name --format=csv,noheader)
- 配置文件: training/configs/experiments/exp_001.yaml

## 训练结果
- 训练损失: 0.123
- 验证 mAP: 0.87
- 训练时间: 3.5 小时

## 模型文件
- 最佳模型: training/model/best_model.pt
- 检查点: checkpoints/exp_001/

## 备注
- 在内网服务器上完成训练
- 数据版本: data.dvc @ $(git log -1 --format=%H data.dvc)
EOF

# 6. 打包结果
tar -czf exp_001_results.tar.gz \
    training/model/ \
    training/test_results/ \
    checkpoints/ \
    logs/ \
    experiments/exp_001_server.md
```

**同步结果回开发机器：**

```bash
# 1. 传输结果文件
scp exp_001_results.tar.gz user@dev-machine:/path/to/Aerovision-V1/

# 2. 在开发机器上解压
tar -xzf exp_001_results.tar.gz

# 3. 更新 DVC
dvc add training/model/
dvc add training/test_results/
git add training/model.dvc training/test_results.dvc experiments/exp_001_server.md

# 4. 提交到 Git
git commit -m "内网服务器训练结果: exp_001

- 训练机器: 内网服务器
- 验证 mAP: 0.87
- 训练时间: 3.5 小时
- 详细记录见: experiments/exp_001_server.md"

git push origin main
```

### 8.5 内网环境注意事项

**安全注意事项：**

1. **数据安全**：确保敏感数据不会泄露到公共网络
2. **访问控制**：限制对内网服务器的访问权限
3. **备份策略**：定期备份重要数据和模型文件
4. **日志审计**：记录所有数据传输和操作

**性能优化：**

1. **存储优化**：使用 SSD 存储 DVC 缓存以提高访问速度
2. **网络优化**：如果可能，使用千兆网络进行数据传输
3. **并行处理**：使用多 GPU 并行训练以提高效率

**维护建议：**

1. **定期清理**：定期清理不再使用的实验数据和缓存
2. **版本管理**：保持清晰的版本记录和映射关系
3. **文档更新**：及时更新部署文档和操作手册
4. **监控告警**：设置系统监控和告警机制

## 附录

### A. DVC 常用命令速查

```bash
# 初始化 DVC
dvc init

# 添加文件/目录到 DVC
dvc add <path>

# 检查状态
dvc status

# 拉取数据
dvc pull

# 推送数据
dvc push

# 检出数据
dvc checkout

# 查看数据信息
dvc data

# 查看数据差异
dvc diff

# 清理缓存
dvc gc

# 运行实验
dvc exp run

# 查看实验
dvc exp show

# 提交实验
dvc exp commit
```

### B. Git + DVC 工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│                    开发流程                                  │
└─────────────────────────────────────────────────────────────┘

1. 修改代码和数据
   ↓
2. 更新 DVC 跟踪
   dvc add data/ model/
   ↓
3. 提交到 Git
   git add *.dvc *.py
   git commit -m "版本信息"
   ↓
4. 推送到远程
   git push origin main

┌─────────────────────────────────────────────────────────────┐
│                    部署流程                                  │
└─────────────────────────────────────────────────────────────┘

1. 克隆项目
   git clone <repo>
   ↓
2. 安装依赖
   pip install -r requirements.txt
   ↓
3. 传输数据
   (手动传输或使用共享存储)
   ↓
4. 验证数据
   dvc status
   ↓
5. 运行实验
   python train.py
   ↓
6. 保存结果
   dvc add model/ results/
   ↓
7. 同步结果
   (传输回开发机器并提交到 Git)

┌─────────────────────────────────────────────────────────────┐
│                    复现流程                                  │
└─────────────────────────────────────────────────────────────┘

1. 找到实验提交
   git log --oneline
   ↓
2. 切换代码版本
   git checkout <commit>
   ↓
3. 恢复数据版本
   git checkout <commit> -- data.dvc
   dvc checkout data.dvc
   ↓
4. 恢复模型和结果
   git checkout <commit> -- model.dvc results.dvc
   dvc checkout model.dvc results.dvc
   ↓
5. 运行测试
   python test.py
   ↓
6. 验证结果
   (与原始结果对比)
```

### C. 参考资源

- [DVC 官方文档](https://dvc.org/doc)
- [DVC 用户指南](https://dvc.org/doc/user-guide)
- [DVC 实验跟踪](https://dvc.org/doc/user-guide/experiment-management)
- [Git 官方文档](https://git-scm.com/doc)
- [YOLO 训练文档](training/docs/workflow.md)

---

**文档版本**: v1.0

**最后更新**: 2024-01-15

**维护者**: 项目团队

**联系方式**: 见项目 README.md
