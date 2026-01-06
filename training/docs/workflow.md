# AeroVision 开发流程规范

## 1. 项目概述

### 1.1 项目信息
- **项目名称**：AeroVision 飞机识别系统
- **项目目标**：使用 YOLOv8x 完成飞机检测、注册号检测和 OCR 识别
- **数据位置**：服务器 `mnt/disk/Aerovision-V1-data`

### 1.2 任务说明
本系统旨在实现对机场场景中飞机的自动识别，包括：
- **飞机检测**：检测图片中的飞机位置
- **注册号检测**：检测飞机注册号区域
- **OCR 识别**：识别飞机的注册号文字

### 1.3 技术栈
- **深度学习框架**：PyTorch
- **目标检测模型**：YOLOv8x (Ultralytics)
- **OCR 引擎**：PaddleOCR
- **数据标注工具**：内部工具

---

## 2. 目录结构

```
training/
├── config/                           # 配置文件
│   ├── default.yaml               # 默认配置
│   └── config_loader.py           # 配置加载器
├── configs/                           # 配置文件
│   └── data_paths.yaml               # 数据路径配置
├── data/                             # 数据目录
│   ├── raw/                          # 原始图片
│   ├── processed/detection/          # 处理后的检测数据
│   │   ├── train/                    # 训练集（70%）
│   │   ├── val/                      # 验证集（15%）
│   │   ├── test/                     # 测试集（15%）
│   │   └── dataset.yaml              # YOLOv8数据集配置
│   └── labels/                       # 标注文件
│       ├── aircraft_labels.csv       # 主标注文件
│       ├── train.csv                 # 训练集标注
│       ├── val.csv                   # 验证集标注
│       ├── test.csv                  # 测试集标注
│       └── registration/            # 注册号区域标注（YOLO格式）
│           ├── IMG_0001.txt
│           └── ...
├── scripts/                          # 脚本目录
│   ├── verify_env.py                 # 环境验证
│   ├── crop_aircraft.py              # 飞机裁剪
│   ├── crop_airplane.py              # 飞机裁剪（备选）
│   ├── split_dataset.py              # 数据集划分
│   ├── verify_data.py                # 数据验证
│   ├── prepare_dataset.py            # 数据准备
│   ├── prepare_detection_dataset.py  # 检测数据集准备
│   ├── prepare_aerovision_dataset.py # AeroVision数据集准备
│   ├── train_detection_model.py      # 检测模型训练
│   ├── ocr_pipeline.py               # OCR Pipeline
│   └── review_crops.py             # 裁剪结果审核
├── src/                              # 源代码
│   └── ocr/                          # OCR 模块
│       └── paddle_ocr.py             # PaddleOCR 实现
├── checkpoints/                      # 模型检查点
│   └── registration_detection/       # 检测模型
│       └── best.pt                  # 最佳模型权重
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
准备检测数据集，包括数据标注和数据集划分。

#### 1.1 数据标注
使用标注工具对原始图片进行标注。

**标注内容**：
- 飞机边界框（bbox）
- 注册号区域边界框（可选）
- 机型名称（typename，可选）
- 航司名称（airlinename，可选）
- 清晰度评分（clarity，可选）
- 遮挡程度评分（block，可选）
- 注册号文字（registration，可选）

**标注工具**：
- LabelImg
- Label Studio
- 标注结果导出为 YOLO 格式

**输出**：
- 标注文件：`training/data/labels/aircraft_labels.csv`
- YOLO 格式标注：`training/data/labels/registration/*.txt`

#### 1.2 数据集准备
使用脚本准备 YOLOv8 检测数据集。

```bash
python training/scripts/prepare_detection_dataset.py \
    --labels-csv data/labels.csv \
    --images-dir data/labeled \
    --output-dir data/processed/detection \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**参数说明**：
- `--labels-csv`: 主标注文件路径
- `--images-dir`: 原始图片目录
- `--output-dir`: 输出目录
- `--train-ratio`: 训练集比例
- `--val-ratio`: 验证集比例
- `--test-ratio`: 测试集比例

**划分比例**：
- 训练集：70%
- 验证集：15%
- 测试集：15%

**输出**：
- `data/processed/detection/train/` - 训练集图片和标签
- `data/processed/detection/val/` - 验证集图片和标签
- `data/processed/detection/test/` - 测试集图片和标签
- `data/processed/detection/dataset.yaml` - YOLOv8 数据集配置

#### 1.3 数据验证
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

### Stage 2: 检测模型训练

#### 目标
训练 YOLOv8x 模型进行飞机和注册号检测。

#### 2.1 训练
```bash
python training/scripts/train_detection.py \
    --data data/processed/detection/dataset.yaml \
    --model x \
    --epochs 50 \
    --batch 8 \
    --imgsz 640
```

**参数说明**：
- `--data`: YOLO 数据集配置文件路径
- `--model`: 模型大小（n, s, m, l, x），推荐使用 x
- `--epochs`: 训练轮数
- `--batch`: 批次大小
- `--imgsz`: 输入图像大小
- `--device`: 设备（0=GPU, cpu=CPU）

**配置文件内容**：
- 数据路径
- 模型参数
- 训练超参数
- 增强策略

**输出**：
- 模型检查点：`training/checkpoints/registration_detection/weights/best.pt`
- 训练日志：`training/logs/`
- 训练指标：mAP、损失曲线等

#### 2.2 推理测试
使用训练好的模型进行推理测试。

```bash
python app/test_inference.py \
    --model training/checkpoints/registration_detection/weights/best.pt \
    --image data/labeled/example.jpg \
    --config training/configs/config/inference.yaml
```

---

### Stage 3: OCR 注册号识别

#### 目标
实现飞机注册号的检测和识别。

#### 3.1 注册号识别
使用训练好的检测模型和 PaddleOCR 进行注册号识别。

```bash
python training/scripts/ocr_pipeline.py \
    --detector training/checkpoints/registration_detection/weights/best.pt \
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

### 5.1 检测模型
- **模型类型**：YOLOv8x
- **模型文件**：`yolov8x.pt`
- **输入尺寸**：640x640
- **类别数**：根据数据集动态确定

### 5.2 OCR 引擎
- **引擎**：PaddleOCR
- **检测模型**：PP-OCRv3 检测模型
- **识别模型**：PP-OCRv3 识别模型
- **方向分类器**：可选

---

## 6. 过关标准

### 6.1 检测模型
- **mAP@0.5**：> 85%
- **mAP@0.5:0.95**：> 70%
- **验证集损失**：稳定收敛

### 6.2 注册号识别
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
- 训练时间：根据数据集大小，检测模型训练预计需要 2-4 小时
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
A: 减小 `batch`，或使用梯度累积。

### Q4: OCR 识别准确率低怎么办？
A: 检查注册号区域的标注质量，确保注册号清晰可见，可以尝试调整 OCR 模型参数。

---

## 9. 更新日志

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v2.0 | 2026-01-02 | 重构为专注于检测任务，删除分类相关内容 |
| v1.0 | 2025-12-31 | 初始版本，定义基本开发流程 |

---

## 10. 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [PaddleOCR 文档](https://github.com/PaddlePaddle/PaddleOCR)
