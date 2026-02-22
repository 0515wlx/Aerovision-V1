# Auto-Annotation Pipeline for Aircraft Dataset

自动化航空图片数据集标注流水线，使用训练好的YOLOv8分类模型对原始图片进行智能标注。

## 功能特性

1. **双模型预测**
   - 机型分类：使用 `yolo26x-cls-aircraft.pt`
   - 航司分类：使用 `yolo26x-cls-airline.pt`

2. **智能分类**
   - 高置信度 (≥95%)：自动标注
   - 中等置信度 (80-95%)：人工审核
   - 低置信度 (<80%)：人工审核

3. **新类别检测**
   - 使用HDBSCAN聚类算法检测可能的新类别
   - 新类别候选样本单独存放供人工审核

4. **文件自动组织**
   - 自动按照项目命名规范组织文件
   - 保存预测详情供人工审核

## 目录结构

```
scripts/auto_annotate/
├── __init__.py              # 包初始化文件
├── model_predictor.py       # 模型预测器
├── hdbscan_detector.py      # 新类别检测器
├── confidence_filter.py     # 置信度过滤器
├── file_organizer.py       # 文件组织器
├── pipeline.py             # 主流水线
├── auto_annotate.py       # 运行脚本
├── config.yaml            # 配置文件
├── test_simple.py         # 简单测试脚本
└── README.md              # 本文档
```

## 安装依赖

```bash
# 核心依赖
pip install ultralytics numpy

# 可选依赖（用于新类别检测）
pip install hdbscan

# 可选依赖（用于配置文件）
pip install pyyaml
```

## 使用方法

### 1. 准备模型文件

将训练好的模型文件放置到指定位置：

```bash
# 机型分类模型
/home/wlx/yolo26x-cls-aircraft.pt

# 航司分类模型
/home/wlx/yolo26x-cls-airline.pt
```

### 2. 准备原始图片

将需要标注的原始图片放置到：

```bash
/mnt/disk/AeroVision/images/
```

### 3. 配置文件

编辑 `config.yaml` 文件，修改以下配置：

```yaml
# 数据路径
raw_images_dir: /mnt/disk/AeroVision/images
labeled_dir: /mnt/disk/AeroVision/labeled
filtered_new_class_dir: /mnt/disk/AeroVision/filtered_new_class
filtered_95_dir: /mnt/disk/AeroVision/filtered_95

# 模型路径
aircraft_model_path: /home/wlx/yolo26x-cls-aircraft.pt
airline_model_path: /home/wlx/yolo26x-cls-airline.pt

# 置信度阈值
high_confidence_threshold: 0.95  # 自动标注阈值
low_confidence_threshold: 0.80   # 中等/低置信度分界

# HDBSCAN参数
hdbscan:
  min_cluster_size: 5
  min_samples: 3
  metric: euclidean
  cluster_selection_method: eom

# 推理参数
device: cpu  # 或 "cuda:0" 使用GPU
batch_size: 32
imgsz: 640
```

### 4. 运行流水线

```bash
# 使用默认配置
python scripts/auto_annotate/auto_annotate.py

# 使用自定义配置文件
python scripts/auto_annotate/auto_annotate.py --config path/to/config.yaml

# 指定模型路径
python scripts/auto_annotate/auto_annotate.py \
    --aircraft-model /path/to/aircraft.pt \
    --airline-model /path/to/airline.pt

# 使用GPU
python scripts/auto_annotate/auto_annotate.py --device cuda:0

# 自定义置信度阈值
python scripts/auto_annotate/auto_annotate.py \
    --high-threshold 0.90 \
    --low-threshold 0.75
```

### 5. 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `config.yaml` |
| `--raw-images` | 原始图片目录 | `/mnt/disk/AeroVision/images` |
| `--labeled-dir` | 自动标注输出目录 | `/mnt/disk/AeroVision/labeled` |
| `--filtered-new-class-dir` | 新类别候选目录 | `/mnt/disk/AeroVision/filtered_new_class` |
| `--filtered-95-dir` | 低置信度目录 | `/mnt/disk/AeroVision/filtered_95` |
| `--aircraft-model` | 机型模型路径 | `/home/wlx/yolo26x-cls-aircraft.pt` |
| `--airline-model` | 航司模型路径 | `/home/wlx/yolo26x-cls-airline.pt` |
| `--high-threshold` | 高置信度阈值 | 0.95 |
| `--low-threshold` | 低置信度阈值 | 0.80 |
| `--device` | 推理设备 | `cpu` |
| `--batch-size` | 批量大小 | 32 |
| `--imgsz` | 图片尺寸 | 640 |
| `--log-dir` | 日志目录 | labeled目录 |

## 输出结果

### 目录结构

```
/mnt/disk/AeroVision/
├── images/                    # 原始图片
├── labeled/                   # 自动标注的结果
│   ├── Boeing/               # 按机型分类
│   │   ├── img_001.jpg
│   │   └── img_002.jpg
│   ├── Airbus/
│   └── ...
│   ├── pipeline_statistics.json   # 流水线统计信息
│   └── auto_annotate_YYYYMMDD_HHMMSS.log
├── filtered_new_class/        # 新类别候选
│   ├── img_003.jpg
│   ├── img_004.jpg
│   └── prediction_details.json    # 预测详情
└── filtered_95/              # 低置信度（人工审核）
    ├── img_005.jpg
    ├── img_006.jpg
    └── prediction_details.json    # 预测详情
```

### 预测详情格式

`prediction_details.json` 包含每张图片的详细预测信息：

```json
{
  "total_predictions": 5,
  "predictions": [
    {
      "filename": "img_001.jpg",
      "aircraft": {
        "class_id": 0,
        "class_name": "Boeing",
        "confidence": 0.98,
        "top5": [
          {"id": 0, "name": "Boeing", "prob": 0.98},
          {"id": 1, "name": "Airbus", "prob": 0.01},
          ...
        ]
      },
      "airline": {
        "class_id": 2,
        "class_name": "China Eastern",
        "confidence": 0.96,
        "top5": [...]
      }
    },
    ...
  ],
  "new_class_indices": [3, 4],
  "timestamp": "2025-01-23T17:30:00"
}
```

### 统计信息格式

`pipeline_statistics.json` 包含流水线执行的统计信息：

```json
{
  "success": true,
  "statistics": {
    "total": 1000,
    "high_confidence_count": 750,
    "medium_confidence_count": 150,
    "low_confidence_count": 50,
    "filtered_95_count": 200,
    "new_class_count": 20,
    "high_confidence_ratio": 0.75,
    "filtered_95_ratio": 0.20,
    "new_class_ratio": 0.02
  },
  "duration_seconds": 123.45,
  "start_time": "2025-01-23T17:30:00",
  "end_time": "2025-01-23T17:32:03",
  "organizer_stats": {
    "labeled_count": 750,
    "filtered_95_count": 200,
    "new_class_count": 20,
    "skipped_count": 5,
    "error_count": 0,
    "total_processed": 970
  }
}
```

## 工作流程

```
1. 收集原始图片
   ↓
2. 加载模型
   ↓
3. 批量预测（机型 + 航司）
   ↓
4. 提取特征嵌入
   ↓
5. HDBSCAN聚类检测新类别
   ↓
6. 置信度过滤
   ├── 高置信度 (≥95%)  → 自动标注
   ├── 中等置信度 (80-95%) → 人工审核
   └── 低置信度 (<80%) → 人工审核
   ↓
7. 文件组织
   ├── 自动标注 → /labeled/<class_name>/
   ├── 新类别候选 → /filtered_new_class/
   └── 人工审核 → /filtered_95/
   ↓
8. 保存预测详情和统计信息
```

## 测试

运行简单测试脚本验证核心功能：

```bash
python scripts/auto_annotate/test_simple.py
```

输出示例：

```
============================================================
Testing Auto-Annotation Pipeline Components
============================================================

1. Testing ConfidenceFilter...
   ✓ ConfidenceFilter works correctly
     High: 1, Medium: 1, Low: 1

2. Testing FileOrganizer...
   ✓ Safe class name creation works
     'Boeing 737-800' -> 'Boeing_737-800'
     'Airbus A320/A321' -> 'Airbus_A320_A321'
   ✓ File organization works correctly
   ✓ Statistics: labeled=1, skipped=0

3. Testing Pipeline Initialization...
   ✓ Pipeline initialized successfully
   ✓ Statistics calculation works correctly
     Total: 18
     High conf: 10
     Filtered 95: 8
     New class: 2
   ✓ Config retrieval works correctly

============================================================
TEST SUMMARY
============================================================
✓ Core components are working correctly
```

## 文件命名规范

原始文件名会被保留，例如：

- `img_681db01413c897.99336087.jpg` → `img_681db01413c897.99336087.jpg`
- `img_681db94e55a568.62549327.png` → `img_681db94e55a568.62549327.png`

目录名会自动转换为安全格式：

- `Boeing 737-800` → `Boeing_737-800`
- `Airbus A320/A321` → `Airbus_A320_A321`
- `China Eastern` → `China_Eastern`

## 注意事项

1. **模型要求**：必须使用训练好的YOLOv8分类模型
2. **图片格式**：支持 `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
3. **HDBSCAN可选**：如果没有安装hdbscan，新类别检测将被禁用
4. **磁盘空间**：确保输出目录有足够的磁盘空间
5. **处理时间**：处理速度取决于图片数量、GPU/CPU性能和批量大小

## 故障排查

### 问题：模型文件未找到

**错误信息**：`FileNotFoundError: Aircraft model not found`

**解决方法**：检查模型文件路径是否正确，确保模型文件存在。

### 问题：没有图片处理

**错误信息**：`No images found in raw directory`

**解决方法**：
1. 检查 `raw_images_dir` 路径是否正确
2. 检查图片文件格式是否支持
3. 检查文件读取权限

### 问题：hdbscan未安装

**警告信息**：`hdbscan package not found`

**解决方法**：
```bash
pip install hdbscan
```

如果不安装hdbscan，新类别检测将被禁用，但其他功能正常。

### 问题：内存不足

**错误信息**：`CUDA out of memory` 或 `MemoryError`

**解决方法**：
1. 减小 `batch_size`
2. 减小 `imgsz`
3. 使用CPU（`--device cpu`）

## 性能优化建议

1. **使用GPU**：如果有可用的GPU，可以显著提升处理速度
   ```bash
   python scripts/auto_annotate/auto_annotate.py --device cuda:0
   ```

2. **调整批量大小**：根据GPU内存大小调整
   ```bash
   python scripts/auto_annotate/auto_annotate.py --batch-size 64
   ```

3. **多GPU**：使用多GPU并行处理（需要修改代码）

## 扩展功能

如果需要添加更多功能，可以：

1. **添加新的过滤规则**：修改 `confidence_filter.py`
2. **更改文件命名规范**：修改 `file_organizer.py`
3. **使用其他聚类算法**：替换 `hdbscan_detector.py`
4. **添加后处理步骤**：在 `pipeline.py` 中添加

## 许可证

本代码是 AeroVision-V1 项目的一部分。
