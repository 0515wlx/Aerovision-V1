# training/config/README.md
# 配置文件使用说明

## 文件结构

```
training/config/
├── __init__.py           # 模块初始化
├── config_loader.py      # 配置加载器
├── default.yaml          # 默认配置文件
└── README.md            # 本文件
```

## 快速开始

### 1. 基本使用

```python
from config import load_config

# 加载默认配置
config = load_config()

# 访问配置
print(config.get('yolo.conf_threshold'))  # 0.5
print(config['paths']['data_root'])       # ../data
```

### 2. 使用自定义配置文件

```python
# 加载自定义配置
config = load_config('my_config.yaml')
```

### 3. 动态覆盖配置

```python
# 在加载时覆盖某些配置
config = load_config(
    yolo={'conf_threshold': 0.8},
    crop={'padding': 0.15}
)
```

### 4. 获取路径配置

```python
# 获取路径并自动转换为Path对象
# ⚠️ 重要：相对路径会自动相对于yaml文件所在目录解析
data_root = config.get_path('paths.data_root')

# 如果目录不存在则创建
output_dir = config.get_path('paths.logs_root', create=True)
```

## ⭐ 路径解析规则

**重要提示：** yaml文件中的相对路径始终相对于**yaml配置文件所在的目录**进行解析，而不是相对于当前工作目录。

### 示例说明

假设你的配置文件位于：`F:\project\training\config\default.yaml`

配置内容：
```yaml
paths:
  data_root: "../data"
  model_root: "../model"
```

无论你在哪里运行脚本，`config.get_path('paths.data_root')` 都会解析为：
- `F:\project\training\data` （相对于config文件夹的上级目录）

这样可以确保：
- ✅ 在任何位置运行脚本都能正确找到文件
- ✅ 不同团队成员使用相同配置不会有路径问题
- ✅ 配置文件更加可移植

### 使用绝对路径

如果需要使用绝对路径，直接在yaml中写绝对路径即可：
```yaml
paths:
  data_root: "F:/my_data"  # 绝对路径
```

## 配置项说明

### paths - 路径配置
- `data_root`: 数据根目录
- `raw_images`: 原始图片目录
- `processed_root`: 处理后数据根目录
- `aircraft_crop`: 裁剪后飞机图片目录
- `model_root`: 模型文件根目录
- `yolo_model`: YOLO模型文件路径
- `logs_root`: 日志目录

### yolo - YOLO检测配置
- `model_size`: 模型大小（yolov8n/s/m/l/x）
- `conf_threshold`: 检测置信度阈值（0-1）
- `airplane_class_id`: 飞机类别ID（COCO数据集中为4）
- `device`: 使用的设备（cuda/cpu）

### crop - 裁剪配置
- `padding`: 边界框扩展比例
- `min_size`: 最小输出尺寸
- `output_quality`: JPEG输出质量
- `image_extensions`: 支持的图片格式

### review - 图片审查配置
- `n_samples`: 随机查看的样本数量
- `grid_cols`: 网格列数
- `fig_width`: 图形宽度
- `dpi`: 输出分辨率

### training - 训练配置（预留）
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `image_size`: 图片尺寸

## 在脚本中使用

### 示例1: crop_airplane.py

```python
from config import load_config

config = load_config()

def crop_aircraft():
    # 使用配置
    model_path = config.get_path('paths.yolo_model')
    conf_threshold = config.get('yolo.conf_threshold')
    padding = config.get('crop.padding')
    
    # ... 你的代码
```

### 示例2: review_crops.py

```python
from config import load_config

config = load_config()

def review_random_samples():
    n_samples = config.get('review.n_samples', 20)
    cols = config.get('review.grid_cols', 5)
    
    # ... 你的代码
```

## 创建自定义配置文件

1. 复制 `default.yaml` 为新文件
2. 修改需要的配置项
3. 在代码中加载自定义配置

```bash
cp default.yaml experiment1.yaml
# 编辑 experiment1.yaml
```

```python
config = load_config('config/experiment1.yaml')
```

## 注意事项

1. ⭐ **yaml文件中的相对路径相对于yaml文件所在目录**，不是当前工作目录
2. 使用 `get_path()` 方法会自动转换为绝对路径
3. 建议不要直接修改 `default.yaml`，而是创建自定义配置文件
4. 配置支持嵌套结构，使用点号访问（如 `yolo.conf_threshold`）
5. `get_path()` 返回的是Path对象，可以直接用于文件操作
