# AeroVision 开发流程规范

## 1. 项目概述

### 1.1 项目信息
- **项目名称**：AeroVision 飞机识别系统
- **项目目标**：使用 YOLOv8 最大版本完成机型分类、航司检测和 OCR 识别
- **数据位置**：服务器 `mnt/disk/Aerovision-V1-data`

### 1.2 任务说明
本系统旨在实现对机场场景中飞机的自动识别，包括：
- **机型分类**：识别飞机的具体型号（如 A320、B737-800 等）
- **航司检测**：识别飞机所属航空公司
- **OCR 识别**：识别飞机的注册号

### 1.3 技术栈
- **深度学习框架**：PyTorch
- **目标检测模型**：YOLOv8x (Ultralytics)
- **OCR 引擎**：PaddleOCR
- **数据标注工具**：内部工具

---

## 2. 目录结构

```
training/
├── configs/                           # 配置文件
│   ├── data_paths.yaml               # 数据路径配置
│   └── aircraft_classify.yaml        # 训练配置
├── data/                             # 数据目录
│   ├── raw/                          # 原始图片
│   ├── processed/aircraft_crop/      # 裁剪后的飞机图片
│   │   ├── unsorted/                 # 待标注的裁剪图片
│   │   ├── train/                    # 训练集（70%）
│   │   ├── val/                      # 验证集（15%）
│   │   └── test/                     # 测试集（15%）
│   └── labels/                       # 标注文件
│       ├── aircraft_labels.csv       # 主标注文件
│       ├── type_classes.json         # 机型类别映射
│       ├── airline_classes.json      # 航司类别映射
│       ├── train.csv                 # 训练集标注
│       ├── val.csv                   # 验证集标注
│       ├── test.csv                  # 测试集标注
│       └── registration/            # 注册号区域标注（YOLO格式）
│           ├── IMG_0001.txt
│           └── ...
├── scripts/                          # 脚本目录
│   ├── verify_env.py                 # 环境验证
│   ├── crop_aircraft.py              # 飞机裁剪
│   ├── split_dataset.py              # 数据集划分
│   ├── verify_data.py                # 数据验证
│   ├── train_stage2.py               # Stage 2 训练
│   ├── evaluate_stage2.py            # Stage 2 评估
│   └── ocr_pipeline.py               # OCR Pipeline
├── src/                              # 源代码
│   └── ocr/                          # OCR 模块
├── checkpoints/                      # 模型检查点
│   └── stage2/                       # Stage 2 模型
│       └── best.pt                   # 最佳模型权重
├── logs/                             # 日志目录
└── docs/                             # 文档目录
    ├── workflow.md                   # 开发流程规范（本文档）
    └── data_format.md                # 数据格式规范
```

---

## 3. 开发流程

### Stage 0: 环境配置

#### 目标
验证开发环境是否满足训练要求。

#### 执行步骤
```bash
# 验证环境
python training/scripts/verify_env.py
```

#### 验证内容
- Python 版本（建议 >= 3.8）
- PyTorch 版本
- CUDA 可用性
- YOLOv8 安装状态
- PaddleOCR 安装状态
- 必要的依赖库

#### 输出
- 环境验证报告（控制台输出）
- 如有问题，需根据提示修复环境

---

### Stage 1: 数据准备

#### 目标
从原始图片中裁剪飞机，进行标注，并划分数据集。

#### 1.1 飞机裁剪
使用预训练的 YOLOv8x 模型从原始图片中检测并裁剪飞机。

```bash
python training/scripts/crop_aircraft.py \
    --input training/data/raw \
    --output training/data/processed/aircraft_crop/unsorted \
    --model yolov8x.pt
```

**参数说明**：
- `--input`: 原始图片目录
- `--output`: 裁剪后的飞机图片输出目录
- `--model`: 用于检测飞机的 YOLOv8 模型路径

**输出**：
- 裁剪后的飞机图片保存在 `training/data/processed/aircraft_crop/unsorted/`

#### 1.2 数据标注
使用 Label Studio 对裁剪后的飞机图片进行标注。

**标注内容**：
- 机型名称（typename）
- 航司名称（airlinename）
- 清晰度评分（clarity）
- 遮挡程度评分（block）
- 注册号文字（registration，可选）

**标注工具**：
- Label Studio
- 标注结果导出为 CSV 格式

**输出**：
- 标注文件：`training/data/labels/aircraft_labels.csv`

#### 1.3 数据集划分
将标注好的数据划分为训练集、验证集和测试集。

```bash
python training/scripts/split_dataset.py \
    --labels training/data/labels/aircraft_labels.csv
```

**参数说明**：
- `--labels`: 主标注文件路径

**划分比例**：
- 训练集：70%
- 验证集：15%
- 测试集：15%

**输出**：
- `training/data/labels/train.csv` - 训练集标注
- `training/data/labels/val.csv` - 验证集标注
- `training/data/labels/test.csv` - 测试集标注
- `training/data/labels/type_classes.json` - 机型类别映射
- `training/data/labels/airline_classes.json` - 航司类别映射

#### 1.4 数据验证
验证数据集的完整性和格式正确性。

```bash
python training/scripts/verify_data.py
```

**验证内容**：
- 图片文件是否存在
- 标注文件格式是否正确
- 数据集划分是否合理
- 类别映射是否完整

**输出**：
- 数据验证报告（控制台输出）
- 如有问题，需根据提示修复数据

---

### Stage 2: 机型分类训练

#### 目标
训练 YOLOv8x-cls 模型进行机型分类。

#### 2.1 训练
```bash
python training/scripts/train_stage2.py \
    --config training/configs/aircraft_classify.yaml \
    --epochs 30 \
    --batch-size 32
```

**参数说明**：
- `--config`: 训练配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小

**配置文件内容** ([`aircraft_classify.yaml`](training/configs/aircraft_classify.yaml))：
- 数据路径
- 模型参数
- 训练超参数
- 增强策略

**输出**：
- 模型检查点：`training/checkpoints/stage2/best.pt`
- 训练日志：`training/logs/stage2/`
- 训练指标：准确率、损失曲线等

#### 2.2 评估
```bash
python training/scripts/evaluate_stage2.py \
    --model training/checkpoints/stage2/best.pt
```

**参数说明**：
- `--model`: 模型权重文件路径

**评估指标**：
- Top-1 准确率
- Top-5 准确率
- 混淆矩阵
- 各类别准确率

**输出**：
- 评估报告（控制台输出）
- 混淆矩阵可视化（可选）

---

### Stage 6: OCR 注册号识别

#### 目标
实现飞机注册号的检测和识别。

#### 6.1 注册号检测模型训练
使用 YOLOv8x 训练注册号区域检测模型。

**数据准备**：
- 标注注册号区域（YOLO 格式）
- 标注文件位置：`training/data/labels/registration/`

**训练命令**：
```bash
yolo detect train \
    data=registration_data.yaml \
    model=yolov8x.pt \
    epochs=50 \
    batch=16
```

#### 6.2 注册号识别
使用训练好的检测模型和 PaddleOCR 进行注册号识别。

```bash
python training/scripts/ocr_pipeline.py \
    --detector yolov8x.pt \
    --image input.jpg \
    --output result.json
```

**参数说明**：
- `--detector`: 注册号检测模型路径
- `--image`: 输入图片路径
- `--output`: 输出结果文件路径

**输出**：
- JSON 格式的识别结果
- 包含注册号文本和置信度

---

## 4. 数据格式规范

详细的数据格式规范请参考 [`data_format.md`](training/docs/data_format.md)。

### 主要内容
- 数据集组织结构
- 主标注文件格式 (CSV)
- 注册号区域标注格式 (YOLO)
- 类别映射文件格式 (JSON)
- 数据质量标准
- 数据集划分比例

---

## 5. 模型配置

### 5.1 机型分类模型
- **模型类型**：YOLOv8x-cls
- **模型文件**：`yolov8x-cls.pt`
- **输入尺寸**：224x224
- **类别数**：根据数据集动态确定

### 5.2 注册号检测模型
- **模型类型**：YOLOv8x
- **模型文件**：`yolov8x.pt`
- **输入尺寸**：640x640
- **类别数**：1（registration）

### 5.3 OCR 引擎
- **引擎**：PaddleOCR
- **检测模型**：PP-OCRv3 检测模型
- **识别模型**：PP-OCRv3 识别模型
- **方向分类器**：可选

---

## 6. 过关标准

### 6.1 Stage 2: 机型分类
- **Top-1 准确率**：> 80%
- **Top-5 准确率**：> 95%
- **验证集损失**：稳定收敛

### 6.2 Stage 6: 注册号识别
- **完全正确率**：> 75%（注册号完全正确）
- **字符准确率**：> 90%（单个字符正确率）
- **检测 mAP**：> 85%

### 6.3 数据质量要求
- **清晰度**：训练集中清晰度 > 0.7 的样本占比 > 80%
- **遮挡程度**：训练集中遮挡程度 < 0.5 的样本占比 > 80%
- **类别平衡**：每个类别至少包含 50 个样本

---

## 7. 注意事项

### 7.1 数据安全
- 原始数据备份：在进行任何数据处理前，请先备份原始数据
- 标注数据版本控制：使用 Git 管理标注文件的变更

### 7.2 训练资源
- GPU 内存：建议 >= 16GB
- 训练时间：根据数据集大小，Stage 2 训练预计需要 2-4 小时
- 磁盘空间：确保有足够的磁盘空间存储模型和日志

### 7.3 调试建议
- 从小数据集开始：先使用少量数据验证流程
- 监控训练过程：使用 TensorBoard 监控训练指标
- 定期保存检查点：避免训练中断导致进度丢失

---

## 8. 常见问题

### Q1: 环境验证失败怎么办？
A: 请检查 Python 版本、CUDA 版本和依赖库安装情况，参考错误提示进行修复。

### Q2: 数据划分后类别不平衡怎么办？
A: 可以使用分层抽样方法重新划分，或对少数类别进行数据增强。

### Q3: 训练过程中显存不足怎么办？
A: 减小 `batch-size`，或使用梯度累积。

### Q4: OCR 识别准确率低怎么办？
A: 检查注册号区域的标注质量，确保注册号清晰可见，可以尝试调整 OCR 模型参数。

---

## 9. 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2025-12-31 | 初始版本，定义基本开发流程 |

---

## 10. 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)
