# FGVC_Aircraft 数据集快速上手指南

快速开始使用 FGVC_Aircraft 数据集训练机型和航司分类模型。

## 快速开始（5 步）

```bash
# 1. 进入训练脚本目录
cd Aerovision-V1/training/scripts

# 2. 运行完整工作流（转换 + 准备 + 划分 + 训练）
python fgvc_workflow.py \
  --fgvc-dir "D:\Users\...\Aerovision-V1\FGVC_Aircraft\raw" \
  --output-base "D:\Users\...\Aerovision-V1\data" \
  --task aircraft \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 224 \
  --device 0
```

## 分步操作

### 步骤 1: 转换数据集

```bash
cd Aerovision-V1/training/scripts

python convert_fgvc_aerocraft.py \
  --fgvc-dir "D:\...\FGVC_Aircraft\raw" \
  --output "D:\...\data\fgvc_converted\train" \
  --split train

python convert_fgvc_aerocraft.py \
  --fgvc-dir "D:\...\FGVC_Aircraft\raw" \
  --output "D:\...\data\fgvc_converted\val" \
  --split val

python convert_fgvc_aerocraft.py \
  --fgvc-dir "D:\...\FGVC_Aircraft\raw" \
  --output "D:\...\data\fgvc_converted\test" \
  --split test
```

### 步骤 2: 合并分割集

```bash
cd Aerovision-V1/training/scripts

python merge_fgvc_splits.py \
  --train "D:\...\data\fgvc_converted\train" \
  --val "D:\...\data\fgvc_converted\val" \
  --test "D:\...\data\fgvc_converted\test" \
  --output "D:\...\data\fgvc_converted\combined"
```

### 步骤 3: 准备数据集

```bash
cd Aerovision-V1/training/scripts

python prepare_dataset.py \
  --labels "D:\...\data\fgvc_converted\combined\labels.csv" \
  --images "D:\...\data\fgvc_converted\combined\images" \
  --output "D:\...\data\fgvc_prepared"
```

### 步骤 4: 划分数据集

```bash
cd Aerovision-V1/training/scripts

# 方法 1: 直接指定 prepared 目录
python split_dataset.py \
  --prepare-dir "D:\...\data\fgvc_prepared\<timestamp>" \
  --output "D:\...\data\fgvc_splits" \
  --mode all

# 方法 2: 从 latest.txt 读取
LATEST=$(cat data/fgvc_prepared/latest.txt)
python split_dataset.py \
  --prepare-dir "$LATEST" \
  --output "D:\...\data\fgvc_splits" \
  --mode all
```

### 步骤 5: 训练模型

#### 机型分类

```bash
cd Aerovision-V1/training/scripts

python train_classify.py \
  --data "D:\...\data\fgvc_splits\<timestamp>/aerovision/aircraft" \
  --model yolov8n-cls.pt \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 224 \
  --device 0
```

#### 航司分类

```bash
cd Aerovision-V1/training/scripts

python train_classify.py \
  --data "D:\...\data\fgvc_splits\<timestamp>/aerovision/airline" \
  --model yolov8n-cls.pt \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 224 \
  --device 0
```

## 脚本说明

| 脚本 | 说明 |
|--------|------|
| `convert_fgvc_aerocraft.py` | 转换 FGVC 格式为项目格式 |
| `merge_fgvc_splits.py` | 合并 train/val/test 为完整数据集 |
| `prepare_dataset.py` | 验证和清理数据（项目已有） |
| `split_dataset.py` | 划分为训练/验证/测试集（项目已有） |
| `train_classify.py` | 训练分类模型（项目已有） |
| `fgvc_workflow.py` | 自动化完整工作流 |
| `test_convert_fgvc_aircraft.py` | TDD 测试套件 |

## 数据结构

### FGVC_Aircraft 原始格式

```
FGVC_Aircraft/raw/
├── data/
│   ├── images/                 # 10,000 张图片 (7位数字.jpg)
│   ├── images_variant_train.txt  # 3334 行: image_id variant_name
│   ├── images_family_train.txt   # 3334 行: image_id family_name
│   ├── images_manufacturer_train.txt  # 3334 行: image_id manufacturer_name
│   ├── images_box.txt        # 10,000 行: image_id xmin ymin xmax ymax
│   ├── variants.txt           # 100 个变体列表
│   ├── families.txt           # 70 个族列表
│   └── manufacturers.txt      # 30 个制造商列表
```

### 项目目标格式

```
labels.csv:
- filename: 图片文件名 (如 1025794.jpg)
- typename: 机型变体 (如 707-320)
- airline: 制造商 (如 Boeing)
- clarity: 清晰度 (默认 0.9)
- block: 遮挡度 (默认 0.0)
- airplanearea: 边界框 (如 "3 144 998 431")

images/:
- 1025794.jpg
- 1340192.jpg
- ...
```

### 划分后格式

```
fgvc_splits/<timestamp>/aerovision/aircraft/
├── train/
│   ├── 707-320/      # 每个机型一个目录
│   ├── A320-200/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## 数据集统计

| 指标 | 数值 |
|--------|------|
| 总图片数 | 10,000 |
| 机型变体 | 100 类 |
| 机型族 | 70 类 |
| 制造商 | 30 类 |
| 训练集 | 7,000 张 |
| 验证集 | 1,500 张 |
| 测试集 | 1,500 张 |

## 测试驱动开发 (TDD)

```bash
# 运行所有测试
cd Aerovision-V1/training/scripts
python test_convert_fgvc_aircraft.py

# 测试包括：
# - parse_annotation_file
# - convert_to_project_format
# - box_file_parsing
# - merge_annotations
# - filter_missing_images
```

## 常用命令

### 监控训练

```bash
# TensorBoard
tensorboard --logdir training/logs/classify/<timestamp>/

# 查看日志
tail -f training/logs/classify/<timestamp>/train_<timestamp>.log

# 查看输出目录
ls output/classify/<experiment_name>/weights/
```

### 恢复训练

```bash
python train_classify.py \
  --data <data_path> \
  --resume output/classify/<exp_name>/weights/last.pt \
  --epochs 200
```

## 输出位置

```
Aerovision-V1/
├── data/
│   ├── fgvc_converted/     # 转换后的数据
│   ├── fgvc_prepared/      # 准备后的数据
│   └── fgvc_splits/        # 划分后的数据
├── training/
│   ├── output/
│   │   └── classify/
│   │       └── <experiment_name>/
│   │           ├── weights/
│   │           │   ├── best.pt
│   │           │   └── last.pt
│   │           └── results.csv
│   ├── ckpt/
│   │   └── classify/
│   │       └── <timestamp>/
│   │           ├── best.pt
│   │           └── last.pt
│   └── logs/
│       └── classify/
│           └── <timestamp>/
│               ├── train_<timestamp>.log
│               └── events.out.tfevents...
└── docs/
    └── FGVC_WORKFLOW.md     # 完整工作流文档
```

## 常见问题

**Q: 训练报错找不到数据集？**
A: 使用绝对路径。YOLOv8 分类需要目录结构，不是 YAML 文件。

**Q: 如何使用 GPU？**
A: 设置 `--device 0`（GPU 0）或 `--device cpu`（强制 CPU）。

**Q: 内存不足（OOM）？**
A: 减小 `--batch-size`（16 或 8）或 `--imgsz`（160 或 128）。

**Q: FGVC 的变体和族用哪个？**
A: 
- Variant: 最细粒度，100 类，适合细粒度分类
- Family: 中等粒度，70 类，适合粗粒度分类
推荐从 Variant 开始。

**Q: 数据划分比例？**
A: 项目默认 70%/15%/15%，可以通过 `split_dataset.py` 的 `--train-ratio` 参数调整。

## 更多信息

- `FGVC_WORKFLOW.md` - 完整工作流文档
- `AGENTS.md` - 项目开发指南
- `CLAUDE.md` - 训练模块详细说明
