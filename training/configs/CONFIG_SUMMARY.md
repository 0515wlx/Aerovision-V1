# 配置系统重构总结

## 新的配置结构

```
training/
├── config/                          # 配置加载器代码（保持不变）
│   ├── __init__.py
│   ├── config_loader.py             # 更新：支持模块化配置
│   ├── default.yaml                 # 保留：向后兼容
│   └── README.md
│
└── configs/                         # 配置文件目录（新）
    ├── base.yaml                    # ⭐ 全局基础配置
    ├── config/                      # 模块配置目录
    │   ├── paths.yaml               # 路径配置
    │   ├── yolo.yaml                # YOLO检测配置
    │   ├── crop.yaml                # 图片裁剪配置
    │   ├── review.yaml              # 结果审查配置
    │   ├── training.yaml            # 模型训练配置
    │   ├── augmentation.yaml        # 数据增强配置
    │   ├── ocr.yaml                 # OCR识别配置（Stage 6）
    │   └── logging.yaml             # 日志配置
    ├── README.md                    # 📖 详细使用文档
    ├── config_usage_examples.py     # 使用示例
    └── data_paths.yaml              # 保留：原有配置
```

## 核心特性

### 1. 模块化配置

每个功能模块有独立的配置文件，方便管理和维护：
- `base.yaml`: 项目全局配置（项目信息、设备、通用路径、实验跟踪）
- `config/*.yaml`: 各个功能模块的详细配置

### 2. ⚠️ 重要：统一的路径解析规则

**所有yaml文件中的相对路径都相对于 `/training/configs` 目录**

```yaml
# 无论在哪里运行脚本，路径解析都是固定的
paths:
  data_root: "../data"      # → /training/data
  model: "../model/yolo.pt"  # → /training/model/yolo.pt
  logs: "../logs"           # → /training/logs
```

这确保了：
✅ 在任何位置运行脚本都能正确找到文件
✅ 不同团队成员使用相同配置不会有路径问题
✅ 配置文件更加可移植

### 3. 灵活的加载方式

```python
# 加载所有配置
config = load_config()

# 只加载特定模块（提高性能）
config = load_config(modules=['yolo', 'crop'], load_all_modules=False)

# 运行时覆盖配置
config = load_config(device={'default': 'cpu'})

# 向后兼容：加载旧配置
config = load_config('config/default.yaml')
```

## 快速开始

### 基本使用

```python
from configs import load_config

# 1. 加载配置
config = load_config()

# 2. 访问配置（支持点号分隔）
project_name = config.get('project.name')
device = config.get('device.default')
yolo_conf = config.get('detection.conf_threshold')

# 3. 获取路径（自动转换为绝对路径）
data_dir = config.get_path('paths.data_root')
model_path = config.get_path('paths.yolo_model')

# 4. 创建目录（如果不存在）
output_dir = config.get_path('paths.aircraft_crop_train', create=True)
```

### 实际使用示例

#### 场景1: 裁剪飞机图片

```python
from configs import load_config

# 加载需要的模块
config = load_config(modules=['yolo', 'crop', 'paths'])

# 获取配置
input_dir = config.get_path('data.raw')
output_dir = config.get_path('data.processed.aircraft_crop.unsorted', create=True)
yolo_model = config.get('model.weights')
conf_threshold = config.get('detection.conf_threshold')
padding = config.get('crop.padding')

# 使用配置进行裁剪...
```

#### 场景2: 训练模型

```python
from configs import load_config

# 加载训练相关配置
config = load_config(modules=['training', 'augmentation', 'paths'])

# 训练参数
batch_size = config.get('basic.batch_size')
learning_rate = config.get('basic.learning_rate')
num_epochs = config.get('basic.num_epochs')

# 优化器配置
optimizer_type = config.get('optimizer.type')
if optimizer_type == 'adamw':
    optimizer_params = config.get('optimizer.adamw')

# 数据增强
if config.get('augmentation.enabled'):
    h_flip_prob = config.get('geometric.horizontal_flip.prob')
    rotation_limit = config.get('geometric.rotation.limit')

# 路径
checkpoint_dir = config.get_path('checkpoints.stage2', create=True)
log_dir = config.get_path('logs.training', create=True)
```

## 配置模块说明

| 模块 | 文件 | 用途 |
|------|------|------|
| 全局基础 | `base.yaml` | 项目信息、设备、通用路径、实验跟踪 |
| 路径 | `config/paths.yaml` | 数据、模型、检查点、日志等详细路径 |
| YOLO | `config/yolo.yaml` | YOLO检测模型配置、推理参数 |
| 裁剪 | `config/crop.yaml` | 图片裁剪参数、输出配置 |
| 审查 | `config/review.yaml` | 裁剪结果可视化审查配置 |
| 训练 | `config/training.yaml` | 模型训练、优化器、学习率调度 |
| 数据增强 | `config/augmentation.yaml` | 几何、颜色、质量变换等 |
| OCR | `config/ocr.yaml` | 注册号OCR识别配置（Stage 6） |
| 日志 | `config/logging.yaml` | 日志系统、TensorBoard、WandB |

## 重要提示

### ✅ 推荐做法

1. 使用 `config.get_path()` 获取路径：
   ```python
   # ✅ 正确：返回绝对路径
   path = config.get_path('paths.data_root')
   ```

2. 只加载需要的模块：
   ```python
   # ✅ 提高性能
   config = load_config(modules=['yolo', 'crop'], load_all_modules=False)
   ```

3. 运行时覆盖而不是修改yaml：
   ```python
   # ✅ 临时修改配置
   config = load_config(device={'default': 'cpu'})
   ```

### ❌ 避免的做法

1. 不要用 `config.get()` 获取路径：
   ```python
   # ❌ 错误：返回相对路径字符串 "../data"
   path = config.get('paths.data_root')
   ```

2. 不要直接修改yaml文件进行临时测试

3. 不要假设路径相对于当前工作目录

## 向后兼容

旧的配置文件 `configs/default.yaml` 仍然保留（从 training/config/ 迁移），可以继续使用：

```python
# 加载旧配置（不使用模块化结构）
config = load_config('configs/default.yaml')
```

但建议逐步迁移到新的配置系统。

## 文档资源

- **详细使用文档**: `training/configs/README.md`
- **使用示例**: `training/configs/config_usage_examples.py`
- **配置加载器源码**: `training/configs/config_loader.py`

## 测试运行

```bash
# 测试配置加载器
cd training
python -m configs.config_loader

# 运行使用示例
cd training/configs
python config_usage_examples.py
```

## 总结

新的配置系统提供了：

1. ✅ **模块化**: 每个功能有独立配置文件，易于管理
2. ✅ **一致性**: 统一的路径解析规则，避免路径混乱
3. ✅ **灵活性**: 按需加载、运行时覆盖
4. ✅ **可维护性**: 配置清晰、结构化
5. ✅ **向后兼容**: 支持旧的配置文件格式

开始使用吧！🚀
