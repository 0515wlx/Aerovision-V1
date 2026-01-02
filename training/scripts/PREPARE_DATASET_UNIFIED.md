# 统一数据集准备脚本说明

## 概述

`prepare_dataset.py` 是整合了 Aerovision 分类数据集和检测数据集准备流程的统一脚本。所有准备好的数据集都会输出到 `data/prepared/{timestamp}/` 目录下。

## 目录结构

```
data/
└── prepared/
    └── {timestamp}/              # 时间戳格式: YYYYMMDD_HHMMSS
        ├── aerovision/           # 分类数据集
        │   ├── aircraft/
        │   │   ├── train/
        │   │   │   ├── A320/
        │   │   │   ├── B737/
        │   │   │   └── ...
        │   │   ├── val/
        │   │   └── test/
        │   ├── labels/
        │   │   ├── type_classes.json
        │   │   └── dataset_statistics.json
        │   └── dataset_config.yaml
        └── detection/            # 检测数据集
            ├── images/
            │   ├── train/
            │   └── val/
            ├── labels/
            │   ├── train/
            │   └── val/
            └── dataset.yaml
```

## 功能特点

### 1. 统一管理
- 所有数据集准备在一个脚本中完成
- 统一输出到 `data/prepared/` 目录
- 使用时间戳确保每次运行的输出目录唯一

### 2. 灵活模式
- `all`: 同时准备分类和检测数据集（默认）
- `aerovision`: 只准备分类数据集
- `detection`: 只准备检测数据集

### 3. 配置集成
- 使用模块化配置系统
- 自动加载 `paths.yaml` 和 `base.yaml`
- 支持命令行参数覆盖

### 4. 数据集特性

#### Aerovision 分类数据集
- 按机型分类的图片目录结构
- train/val/test 三个数据集
- 类别映射文件（JSON）
- 数据集统计信息
- YOLOv8 配置文件

#### Detection 检测数据集
- YOLO 格式的检测数据集
- train/val 两个数据集
- 注册号检测标注
- YOLOv8 配置文件

## 使用方法

### 基本用法

#### 1. 准备所有数据集
```bash
python prepare_dataset.py
```

#### 2. 只准备分类数据集
```bash
python prepare_dataset.py --mode aerovision
```

#### 3. 只准备检测数据集
```bash
python prepare_dataset.py --mode detection
```

### 高级用法

#### 指定划分比例
```bash
# 分类数据集比例
python prepare_dataset.py \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --test-ratio 0.15

# 检测数据集比例
python prepare_dataset.py \
  --detection-train-ratio 0.8
```

#### 指定自定义路径
```bash
python prepare_dataset.py \
  --labels-csv path/to/labels.csv \
  --images-dir path/to/images \
  --registration-dir path/to/registration \
  --output-dir path/to/output
```

#### 使用自定义配置文件
```bash
python prepare_dataset.py --config my_config.yaml
```

#### 指定随机种子
```bash
python prepare_dataset.py --random-seed 123
```

## 配置说明

### paths.yaml

```yaml
# 数据路径
data:
  # 统一准备数据集输出根目录
  prepared_root: "../data/prepared"

  processed:
    labeled:
      images: "../data/processed/labeled/images"
      registration: "../data/processed/labeled/registration/registration_area"

# 标注文件
labels:
  main: "../data/processed/labeled/labels.csv"
```

### base.yaml

```yaml
# 随机种子
seed:
  random: 42
  numpy: 42
  torch: 42
```

## 配置优先级

```
命令行参数 > 配置文件 > 默认值
```

## 输出示例

### 完整输出（mode=all）

运行命令：
```bash
python prepare_dataset.py
```

输出目录：
```
data/prepared/20260102_221236/
├── aerovision/
│   ├── aircraft/
│   │   ├── train/
│   │   │   ├── A320/
│   │   │   │   ├── img001.jpg
│   │   │   │   └── img002.jpg
│   │   │   ├── B737/
│   │   │   └── ...
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── type_classes.json
│   │   └── dataset_statistics.json
│   └── dataset_config.yaml
└── detection/
    ├── images/
    │   ├── train/
    │   │   ├── img001.jpg
    │   │   └── img002.jpg
    │   └── val/
    ├── labels/
    │   ├── train/
    │   │   ├── img001.txt
    │   │   └── img002.txt
    │   └── val/
    └── dataset.yaml
```

### 只准备分类数据集（mode=aerovision）

运行命令：
```bash
python prepare_dataset.py --mode aerovision
```

输出目录：
```
data/prepared/20260102_221236/
└── aerovision/
    ├── aircraft/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── type_classes.json
    │   └── dataset_statistics.json
    └── dataset_config.yaml
```

### 只准备检测数据集（mode=detection）

运行命令：
```bash
python prepare_dataset.py --mode detection
```

输出目录：
```
data/prepared/20260102_221236/
└── detection/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml
```

## 优势

1. **统一管理**: 所有数据集准备在一个脚本中完成，避免重复代码
2. **目录规范**: 统一输出到 `data/prepared/` 目录，结构清晰
3. **时间追溯**: 使用时间戳可以追溯数据集的创建时间
4. **避免覆盖**: 每次运行生成独立的目录，不会覆盖之前的数据集
5. **灵活性高**: 支持三种模式，可以按需准备数据集
6. **配置集成**: 与项目配置系统无缝集成
7. **易于维护**: 单一脚本，维护成本低

## 与旧脚本的对比

| 特性 | 旧方式 | 新方式 |
|------|--------|--------|
| 脚本数量 | 2个独立脚本 | 1个统一脚本 |
| 输出位置 | 分散在不同目录 | 统一在 `data/prepared/` |
| 目录结构 | `data/processed/aerovision_{timestamp}/`<br>`data/detection_{timestamp}/` | `data/prepared/{timestamp}/aerovision/`<br>`data/prepared/{timestamp}/detection/` |
| 时间戳位置 | 每个数据集独立时间戳 | 共享同一时间戳 |
| 使用便捷性 | 需要分别运行 | 一次运行完成 |

## 依赖

### 必需
- Python 3.7+
- PyYAML
- pandas（用于读取CSV）

### 可选
- scikit-learn（用于检测数据集划分，未安装时使用简单随机划分）

安装依赖：
```bash
pip install pyyaml pandas scikit-learn
```

## 注意事项

1. **检测数据集**: 如果注册号标注目录不存在或为空，检测数据集准备会被跳过
2. **时间戳**: 同一时间戳下的 aerovision 和 detection 数据集使用相同的随机种子
3. **路径解析**: 所有相对路径都相对于 `/training/configs` 目录
4. **数据验证**: 运行前确保标注文件和图片目录存在且可访问

## 迁移指南

### 从旧脚本迁移

旧方式：
```bash
# 准备分类数据集
python prepare_aerovision_dataset.py

# 准备检测数据集
python prepare_detection_dataset.py
```

新方式：
```bash
# 一次性准备所有数据集
python prepare_dataset.py

# 或分别准备
python prepare_dataset.py --mode aerovision
python prepare_dataset.py --mode detection
```

### 旧数据集位置

旧脚本生成的数据集位置：
- `data/processed/aerovision_{timestamp}/`
- `data/detection_{timestamp}/`

新脚本生成的数据集位置：
- `data/prepared/{timestamp}/aerovision/`
- `data/prepared/{timestamp}/detection/`

可以手动移动旧数据集到新位置，或保留旧数据集继续使用。

## 相关文件

```
training/
├── scripts/
│   ├── prepare_dataset.py              # ✨ 新增：统一数据集准备脚本
│   ├── prepare_aerovision_dataset.py   # 🔄 保留：独立分类数据集准备
│   ├── prepare_detection_dataset.py    # 🔄 保留：独立检测数据集准备
│   └── PREPARE_DATASET_UNIFIED.md      # ✨ 新增：统一脚本说明文档
└── configs/
    └── config/
        └── paths.yaml                  # 配置文件
```

## 相关文档

- 配置系统使用指南: `training/configs/README.md`
- 配置系统总结: `training/configs/CONFIG_SUMMARY.md`
- 独立脚本更新说明: `training/scripts/PREPARE_SCRIPTS_UPDATE.md`

## 更新历史

- **2026-01-02**: 创建统一数据集准备脚本
  - 整合 Aerovision 和 Detection 数据集准备流程
  - 统一输出到 `data/prepared/{timestamp}/` 目录
  - 支持三种准备模式（all/aerovision/detection）
