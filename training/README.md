# AeroVision Training Module

航空摄影智能审核系统 - 模型训练模块

## 目录

- [概述](#概述)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [快速开始](#快速开始)
- [数据流程](#数据流程)
- [训练脚本](#训练脚本)
- [配置系统](#配置系统)
- [模型说明](#模型说明)
- [OCR 模块](#ocr-模块)
- [常见问题](#常见问题)

## 概述

本模块用于训练 AeroVision 审核系统所需的深度学习模型，包括：

| 任务 | 模型 | 用途 |
|------|------|------|
| 飞机机型分类 | YOLOv8-cls | 识别飞机具体型号（A320、B737-800 等） |
| 航空公司识别 | YOLOv8-cls | 识别航司涂装（国航、东航等） |
| 注册号检测 | YOLOv8 | 检测图片中注册号区域位置 |
| 注册号识别 | PaddleOCR | OCR 识别注册号文字内容 |

## 目录结构

```
training/
├── configs/                    # 配置文件
│   ├── base.yaml              # 基础配置（项目信息、设备、种子）
│   ├── config/                # 模块配置
│   │   ├── paths.yaml         # 路径配置
│   │   ├── training.yaml      # 训练参数
│   │   ├── airline.yaml       # 航司训练配置
│   │   ├── yolo.yaml          # YOLO 检测配置
│   │   ├── crop.yaml          # 裁剪配置
│   │   ├── augmentation.yaml  # 数据增强配置
│   │   ├── ocr.yaml           # OCR 配置
│   │   └── logging.yaml       # 日志配置
│   ├── config_loader.py       # 配置加载器
│   └── __init__.py
│
├── scripts/                    # 训练脚本
│   ├── prepare_dataset.py     # 数据准备（验证、清洗）
│   ├── split_dataset.py       # 数据集划分（train/val/test）
│   ├── crop_airplane.py       # YOLO 飞机检测与裁剪
│   ├── train_classify.py      # 机型分类训练
│   ├── train_airline.py       # 航司识别训练
│   ├── train_detection.py     # 注册号检测训练
│   ├── verify_data.py         # 数据验证工具
│   └── review_crops.py        # 裁剪结果审查
│
├── ocr/                        # OCR 模块
│   ├── paddle_ocr.py          # PaddleOCR 封装
│   └── demo_bbox_ocr.py       # OCR 演示脚本
│
├── data/                       # 数据目录
│   ├── raw/                   # 原始图片
│   ├── processed/             # 处理后数据
│   │   └── labeled/           # 标注数据
│   ├── prepared/              # 准备好的数据（prepare_dataset 输出）
│   └── splits/                # 划分后数据（split_dataset 输出）
│
├── model/                      # 预训练模型存放目录
├── ckpt/                       # 检查点保存目录
├── logs/                       # 训练日志
├── output/                     # 训练输出（YOLO 结果）
├── test_script/                # 测试脚本
└── docs/                       # 文档
```

## 环境配置

### 系统要求

- Python 3.11+
- CUDA 11.8+ (GPU 训练)
- 8GB+ GPU 显存

### 依赖安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖

```
# 深度学习框架
torch>=2.0.0
ultralytics>=8.0.0       # YOLOv8
timm>=0.9.0              # 预训练模型库

# OCR
paddlepaddle>=2.5.0
paddleocr>=2.7.0

# 数据处理
pandas>=2.0.0
pillow>=10.0.0
opencv-python>=4.8.0
albumentations>=1.3.0

# 工具
pyyaml>=6.0
tqdm>=4.65.0
tensorboard>=2.14.0
```

### GPU 环境验证

```bash
cd training/test_script
python check_gpu.py
python verify_env.py
```

## 快速开始

### 1. 数据准备

```bash
cd training/scripts

# 步骤1: 验证并清洗原始标注数据
python prepare_dataset.py --labels ../data/processed/labeled/labels.csv \
                          --images ../data/processed/labeled/images

# 步骤2: 划分数据集
python split_dataset.py --prepare-dir ../data/prepared/latest
```

### 2. 训练模型

```bash
# 机型分类
python train_classify.py --epochs 100 --batch-size 32

# 航司识别
python train_airline.py --epochs 100 --batch-size 32

# 注册号检测
python train_detection.py --epochs 100 --batch-size 16 --imgsz 640
```

### 3. 使用 OCR

```bash
cd training/ocr

# 从边界框文件识别
python paddle_ocr.py <image_path> <bbox_txt>
```

## 数据流程

### 完整训练流程

```
原始图片 (data/raw/)
        ↓
   [crop_airplane.py]  ← 使用 YOLO 检测裁剪飞机
        ↓
裁剪后图片 (data/processed/aircraft_crop/)
        ↓
   [人工标注]  ← 标注机型、航司、质量等
        ↓
标注文件 (labels.csv)
        ↓
   [prepare_dataset.py]  ← 验证、清洗、去重
        ↓
清洗后数据 (data/prepared/<timestamp>/)
        ↓
   [split_dataset.py]  ← 划分 train/val/test
        ↓
划分后数据 (data/splits/<timestamp>/)
    ├── aerovision/     ← 分类数据集
    │   ├── aircraft/   ← 机型分类
    │   │   ├── train/<class_name>/
    │   │   ├── val/<class_name>/
    │   │   └── test/<class_name>/
    │   └── airline/    ← 航司分类
    └── detection/      ← 检测数据集 (YOLO 格式)
        ├── images/
        ├── labels/
        └── dataset.yaml
```

### 标注文件格式

主标注文件 `labels.csv`:

```csv
filename,typename,airlinename,registration,clarity,block
IMG_0001.jpg,A320,China Eastern,B-1234,0.95,0
IMG_0002.jpg,B737-800,Air China,B-5678,0.85,0
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| filename | string | ✓ | 图片文件名 |
| typename | string | ✓ | 机型名称 |
| airlinename | string | | 航司名称 |
| registration | string | | 注册号 |
| clarity | float | ✓ | 清晰度 (0-1) |
| block | int | ✓ | 遮挡标记 (0/1) |

## 训练脚本

### train_classify.py - 机型分类

基于 YOLOv8-cls 微调的机型分类模型。

```bash
# 使用配置文件默认参数
python train_classify.py

# 自定义参数
python train_classify.py \
    --data ../data/splits/latest/aerovision/aircraft \
    --model yolov8m-cls.pt \
    --epochs 100 \
    --batch-size 32 \
    --imgsz 224 \
    --lr0 0.001 \
    --optimizer AdamW \
    --device 0
```

**关键参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | yolov8n-cls.pt | 预训练模型 |
| `--epochs` | 100 | 训练轮数 |
| `--batch-size` | 32 | 批次大小 |
| `--imgsz` | 224 | 输入图片尺寸 |
| `--lr0` | 0.001 | 初始学习率 |
| `--patience` | 50 | 早停耐心值 |
| `--device` | 0 | GPU ID |

### train_airline.py - 航司识别

与机型分类类似，针对航空公司涂装识别。

```bash
python train_airline.py \
    --data ../data/splits/latest/aerovision/airline \
    --epochs 100 \
    --batch-size 32
```

### train_detection.py - 注册号检测

基于 YOLOv8 的目标检测模型，用于定位注册号区域。

```bash
python train_detection.py \
    --data ../data/splits/latest/detection/dataset.yaml \
    --model yolov8n.pt \
    --epochs 100 \
    --batch-size 16 \
    --imgsz 640
```

**特殊配置**:
- 关闭水平翻转 (文字方向敏感)
- 减小旋转角度 (±5°)
- 适度马赛克增强

## 配置系统

### 模块化配置

配置采用模块化设计，所有相对路径相对于 `training/configs` 目录。

```python
from configs import load_config

# 加载所有配置模块
config = load_config()

# 只加载特定模块
config = load_config(modules=['training', 'paths'], load_all_modules=False)

# 访问配置
epochs = config.get('training.epochs')
data_path = config.get_path('data.splits.root')  # 自动转为绝对路径

# 运行时覆盖
config = load_config(device={'default': 'cpu'})
```

### 配置文件说明

**base.yaml** - 基础配置:
```yaml
project:
  name: "AeroVision-V1"
  version: "0.1.0"

device:
  default: "cuda"
  gpu_ids: [0]

seed:
  random: 42
  numpy: 42
  torch: 42
```

**training.yaml** - 训练参数:
```yaml
training:
  epochs: 100
  batch_size: 32
  image_size: 224
  workers: 8
  amp: true  # 混合精度

  optimizer:
    type: "AdamW"
    lr0: 0.001
    momentum: 0.937
    weight_decay: 0.0005

  scheduler:
    cosine: true
    lrf: 0.01

  early_stopping:
    patience: 50
```

**paths.yaml** - 路径配置:
```yaml
data:
  splits:
    root: "../data/splits"
    latest: "../data/splits/latest"

models:
  root: "../model"
  pretrained:
    yolov8n_cls: "../model/yolov8n-cls.pt"

checkpoints:
  classify: "../ckpt/classify"
  detection: "../ckpt/detection"

logs:
  classify: "../logs/classify"
  tensorboard: "../logs/tensorboard"
```

## 模型说明

### YOLOv8 Classification

用于机型分类和航司识别。

| 模型 | 参数量 | 推荐场景 |
|------|--------|----------|
| yolov8n-cls | 2.7M | 快速原型 |
| yolov8s-cls | 6.4M | 轻量部署 |
| yolov8m-cls | 12.9M | 平衡选择 |
| yolov8l-cls | 37.5M | 高精度 |
| yolov8x-cls | 57.4M | 最佳精度 |

### YOLOv8 Detection

用于注册号区域检测。

| 模型 | 参数量 | mAP | 推荐场景 |
|------|--------|-----|----------|
| yolov8n | 3.2M | 37.3 | 边缘设备 |
| yolov8s | 11.2M | 44.9 | 轻量部署 |
| yolov8m | 25.9M | 50.2 | 平衡选择 |

### 模型输出位置

训练完成后，模型保存在:

```
output/
├── classify/
│   └── aircraft_classifier_<timestamp>/
│       └── weights/
│           ├── best.pt   # 最佳模型
│           └── last.pt   # 最后模型
├── airline/
│   └── airline_classifier_<timestamp>/
│       └── weights/
└── detection/
    └── registration_detector_<timestamp>/
        └── weights/
```

## OCR 模块

### PaddleOCR 封装

`ocr/paddle_ocr.py` 封装了 PaddleOCR，专门用于注册号识别。

```python
from ocr.paddle_ocr import RegistrationOCR, create_ocr

# 创建实例
ocr = create_ocr(lang='en', rec_model_name='PP-OCRv4_server_rec_doc')

# 从边界框识别
result = ocr.recognize_from_bbox(
    image='aircraft.jpg',
    bbox=[0.85, 0.65, 0.12, 0.04],  # YOLO 格式 [x_center, y_center, w, h]
    padding=0.1
)

print(result)
# {'text': 'B-1234', 'confidence': 0.95, 'valid': True, 'bbox': (x1, y1, x2, y2)}

# 从标注文件批量识别
results = ocr.recognize_from_txt('aircraft.jpg', 'aircraft.txt')
```

### OCR 后处理

- 转大写，移除空格
- 仅保留 `A-Z`, `0-9`, `-`
- 长度验证: 4-10 字符
- 置信度阈值: 0.5

## TensorBoard 监控

```bash
# 启动 TensorBoard
tensorboard --logdir training/logs/tensorboard

# 或指定特定日志
tensorboard --logdir training/logs/classify/<timestamp>
```

## 常见问题

### Q: CUDA out of memory

减小批次大小或图片尺寸:
```bash
python train_classify.py --batch-size 16 --imgsz 160
```

### Q: 找不到数据集

确保按顺序执行数据准备流程:
```bash
python prepare_dataset.py  # 先准备
python split_dataset.py    # 再划分
python train_classify.py   # 最后训练
```

### Q: 模型下载失败

手动下载模型到 `training/model/` 目录:
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-cls.pt \
     -O training/model/yolov8n-cls.pt
```

### Q: PaddleOCR 初始化慢

首次运行会下载模型，后续会使用缓存。可预先下载:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(lang='en', text_recognition_model_name='PP-OCRv4_server_rec_doc')
```

### Q: 配置路径问题

所有相对路径都相对于 `training/configs/` 目录:
- `../data` = `training/data`
- `../model` = `training/model`

使用 `config.get_path()` 自动解析为绝对路径。

## 参考资源

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/)
- [timm Models](https://huggingface.co/docs/timm/)
