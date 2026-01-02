# AeroVision-V1 配置系统使用指南

## 📁 配置文件结构

```
training/
├── config/                      # 配置加载器代码
│   ├── __init__.py
│   ├── config_loader.py         # 配置加载器
│   └── README.md
│
└── configs/                     # 配置文件目录
    ├── base.yaml                # ⭐ 全局基础配置
    └── config/                  # 模块配置目录
        ├── paths.yaml           # 路径配置
        ├── yolo.yaml            # YOLO检测配置
        ├── crop.yaml            # 图片裁剪配置
        ├── review.yaml          # 结果审查配置
        ├── training.yaml        # 模型训练配置
        ├── augmentation.yaml    # 数据增强配置
        ├── ocr.yaml             # OCR识别配置
        └── logging.yaml         # 日志配置
```

## 🎯 核心概念

### 1. 配置层次结构

- **base.yaml**: 全局基础配置，包含项目信息、设备设置、通用路径等
- **config/*.yaml**: 模块化配置，每个模块负责特定功能的配置

### 2. ⚠️ 重要：路径解析规则

**所有yaml文件中的相对路径都相对于 `/training/configs` 目录**

无论你在哪里运行python脚本，路径解析规则都是固定的：

```yaml
# 在任何yaml文件中
paths:
  data_root: "../data"      # → /training/data
  model: "../model/yolo.pt"  # → /training/model/yolo.pt
  logs: "../logs"           # → /training/logs
```

这样设计的好处：
- ✅ 在任何位置运行脚本都能正确找到文件
- ✅ 不同团队成员使用相同配置不会有路径问题
- ✅ 配置文件更加可移植

## 🚀 快速开始

### 1. 基本使用

```python
from configs import load_config

# 加载默认配置（base.yaml + 所有模块配置）
config = load_config()

# 访问配置
print(config.get('project.name'))           # AeroVision-V1
print(config.get('device.default'))         # cuda
print(config.get('detection.conf_threshold')) # 0.5
```

### 2. 只加载特定模块

```python
# 只加载base和yolo模块（提高加载速度）
config = load_config(modules=['yolo'], load_all_modules=False)

# 加载多个特定模块
config = load_config(modules=['yolo', 'crop', 'paths'], load_all_modules=False)
```

### 3. 运行时覆盖配置

```python
# 在加载时覆盖配置
config = load_config(
    device={'default': 'cpu'},           # 修改设备为CPU
    detection={'conf_threshold': 0.8}    # 修改YOLO置信度
)

# 或者使用update方法
config = load_config()
config.update({
    'basic': {'batch_size': 64},
    'crop': {'padding': 0.15}
})
```

### 4. 获取路径配置

```python
# ⚠️ 重要：使用get_path()自动将相对路径转换为绝对路径
data_root = config.get_path('paths.data_root')
# 返回: F:\bian\pyproject\Aerovision-V1\training\data (绝对路径)

# 如果目录不存在则创建
output_dir = config.get_path('paths.logs_root', create=True)

# 普通的get()返回字符串
data_root_str = config.get('paths.data_root')
# 返回: "../data" (相对路径字符串)
```

## 📖 配置访问方式

### 方式1: get方法（推荐）

```python
# 支持点号分隔的嵌套键
value = config.get('yolo.model.size')
value = config.get('detection.conf_threshold')
value = config.get('crop.padding')

# 提供默认值
value = config.get('non_existent_key', default=0.5)
```

### 方式2: 字典式访问

```python
value = config['device']['default']
value = config['paths']['data_root']
```

### 方式3: 属性式访问

```python
# 只能访问第一层
device_config = config.device
print(device_config)  # {'default': 'cuda', 'gpu_ids': [0], ...}
```

## 📝 配置文件说明

### base.yaml - 全局基础配置

包含：
- `project`: 项目信息（名称、版本）
- `device`: 设备配置（GPU/CPU）
- `paths`: 全局路径配置
- `image`: 图像处理基础配置
- `seed`: 随机种子
- `experiment`: 实验跟踪配置（WandB、TensorBoard）

### config/paths.yaml - 路径配置

详细的路径配置，包括：
- 数据路径（原始、处理后）
- 标注文件路径
- 模型文件路径
- 检查点路径
- 日志路径

### config/yolo.yaml - YOLO配置

YOLO检测相关配置：
- 模型选择和权重
- 检测参数（置信度、IoU阈值）
- 推理配置（设备、批次）
- 训练配置（如需微调）

### config/crop.yaml - 裁剪配置

图片裁剪相关：
- 裁剪参数（padding、最小/最大尺寸）
- 输出配置（质量、格式）
- 批处理配置
- 错误处理

### config/training.yaml - 训练配置

模型训练相关：
- 基础训练参数（批次、学习率、轮数）
- 优化器配置（Adam、AdamW、SGD）
- 学习率调度器
- 正则化方法
- 早停和检查点
- 多任务学习配置

### config/augmentation.yaml - 数据增强

数据增强配置：
- 几何变换（翻转、旋转、缩放）
- 颜色变换（亮度、对比度、饱和度）
- 质量变换（模糊、噪声）
- 高级增强（Mixup、CutMix）

### config/ocr.yaml - OCR配置

OCR识别配置（Stage 6）：
- OCR引擎选择（PaddleOCR、EasyOCR、Tesseract）
- 注册号识别特定配置
- 预处理和后处理
- 批处理配置

### config/logging.yaml - 日志配置

日志系统配置：
- 基础日志配置（级别、格式）
- 文件日志和轮转
- TensorBoard配置
- WandB配置
- 性能分析

## 💡 实际使用示例

### 示例1: 裁剪飞机图片

```python
from configs import load_config
from pathlib import Path

# 加载配置（只加载需要的模块）
config = load_config(modules=['yolo', 'crop', 'paths'])

# 获取配置
input_dir = config.get_path('data.raw')
output_dir = config.get_path('data.processed.aircraft_crop.unsorted', create=True)
yolo_model = config.get_path('models.pretrained.yolov8m')
conf_threshold = config.get('detection.conf_threshold')
padding = config.get('crop.padding')

print(f"输入目录: {input_dir}")
print(f"输出目录: {output_dir}")
print(f"YOLO模型: {yolo_model}")
print(f"置信度阈值: {conf_threshold}")
```

### 示例2: 训练模型

```python
from configs import load_config

# 加载训练相关配置
config = load_config(modules=['training', 'augmentation', 'paths', 'logging'])

# 训练参数
batch_size = config.get('basic.batch_size')
learning_rate = config.get('basic.learning_rate')
num_epochs = config.get('basic.num_epochs')
image_size = config.get('basic.image_size')

# 优化器配置
optimizer_type = config.get('optimizer.type')
if optimizer_type == 'adamw':
    optimizer_params = config.get('optimizer.adamw')

# 数据增强
aug_enabled = config.get('augmentation.enabled')
if aug_enabled:
    h_flip_prob = config.get('geometric.horizontal_flip.prob')
    rotation_limit = config.get('geometric.rotation.limit')

# 路径
checkpoint_dir = config.get_path('checkpoints.stage2', create=True)
log_dir = config.get_path('logs.training', create=True)
```

### 示例3: 运行时修改配置

```python
from configs import load_config
import argparse

# 命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--device', type=str, default=None)
args = parser.parse_args()

# 加载配置
config = load_config()

# 根据命令行参数覆盖配置
overrides = {}
if args.batch_size:
    overrides['basic'] = {'batch_size': args.batch_size}
if args.device:
    overrides['device'] = {'default': args.device}

if overrides:
    config.update(overrides)

# 使用配置
batch_size = config.get('basic.batch_size')
device = config.get('device.default')
print(f"使用批次大小: {batch_size}")
print(f"使用设备: {device}")
```

## 🔧 向后兼容

如果你有旧的配置文件（如 `config/default.yaml`），仍然可以加载：

```python
# 加载旧的配置文件
config = load_config('config/default.yaml')

# 这会直接加载该文件，不会加载模块化配置
```

## 📋 配置检查清单

在使用配置系统时，请确保：

- [ ] 所有相对路径都是相对于 `/training/configs` 目录
- [ ] 使用 `config.get_path()` 获取路径（自动转换为绝对路径）
- [ ] 只加载需要的模块以提高性能
- [ ] 在不同环境运行前检查路径是否正确
- [ ] 使用配置覆盖而不是直接修改yaml文件

## 🐛 常见问题

### Q1: 为什么路径找不到？

A: 确保你使用的是 `config.get_path()` 而不是 `config.get()`：

```python
# ❌ 错误：返回相对路径字符串 "../data"
path = config.get('paths.data_root')

# ✅ 正确：返回绝对路径 Path对象
path = config.get_path('paths.data_root')
```

### Q2: 如何添加新的配置项？

A: 在对应的模块yaml文件中添加：

```yaml
# configs/config/yolo.yaml
detection:
  conf_threshold: 0.5
  new_parameter: value  # 添加新参数
```

### Q3: 如何在不修改yaml的情况下临时改变配置？

A: 使用运行时覆盖：

```python
config = load_config(
    detection={'conf_threshold': 0.8},
    device={'default': 'cpu'}
)
```

### Q4: 路径解析是相对于哪里的？

A: **永远相对于 `/training/configs` 目录**，无论你在哪里运行脚本：

```yaml
paths:
  data: "../data"  # → /training/data（不是相对于运行脚本的位置）
```

## 📚 更多资源

- 配置加载器源码: `training/config/config_loader.py`
- 配置加载器文档: `training/config/README.md`
- 项目整体文档: `training/README.md`

## 🎉 总结

新的配置系统提供了：

1. **模块化**: 每个功能模块有独立的配置文件
2. **一致性**: 所有路径相对于固定的基准目录
3. **灵活性**: 支持按需加载、运行时覆盖
4. **可维护性**: 配置清晰、易于修改
5. **向后兼容**: 支持旧的配置文件格式

开始使用吧！🚀
