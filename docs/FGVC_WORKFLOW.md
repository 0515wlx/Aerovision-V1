# FGVC_Aircraft 数据集训练工作流指南

本文档说明如何在 Aerovision-V1 项目中使用 FGVC_Aircraft 数据集训练机型和航司分类模型。

## 数据集概述

FGVC_Aircraft 是一个细粒度飞机识别数据集，包含：
- **10,000 张图片**
- **100 个机型变体** (variant)
- **70 个机型族** (family)
- **30 个制造商** (manufacturer)
- 三等分：训练集 3334 张，验证集 3333 张，测试集 3333 张

## 完整工作流

### 1. 转换 FGVC 数据集

FGVC_Aircraft 的数据格式与项目不同，需要先转换：

```bash
cd Aerovision-V1/training/scripts

# 转换单个分割集
python convert_fgvc_aerocraft.py \
  --fgvc-dir "D:\Users\...\FGVC_Aircraft\raw" \
  --output "D:\Users\...\data\fgvc_converted\train" \
  --split train

# 或者使用工作流脚本一次性转换所有分割集
python fgvc_workflow.py \
  --fgvc-dir "D:\Users\...\FGVC_Aircraft\raw" \
  --output-base "D:\Users\...\data"
```

转换输出：
- `data/fgvc_converted/train/` - 训练集
- `data/fgvc_converted/val/` - 验证集
- `data/fgvc_converted/test/` - 测试集
- `data/fgvc_converted/combined/` - 合并后的完整数据集

转换后的格式：
- `filename`: 图片文件名 (如 1025794.jpg)
- `typename`: 机型变体名称 (如 707-320)
- `airline`: 制造商名称 (如 Boeing)
- `clarity`: 默认值 0.9
- `block`: 默认值 0.0
- `airplanearea`: 边界框坐标 (从 FGVC 提取)

### 2. 准备数据集

```bash
cd Aerovision-V1/training/scripts

python prepare_dataset.py \
  --labels "D:\Users\...\data\fgvc_converted\combined\labels.csv" \
  --images "D:\Users\...\data\fgvc_converted\combined\images" \
  --output "D:\Users\...\data\fgvc_prepared"
```

准备输出：
- `data/fgvc_prepared/<timestamp>/labels.csv` - 清洗后的标注
- `data/fgvc_prepared/<timestamp>/images/` - 验证后的图片
- `data/fgvc_prepared/latest.txt` - 最新目录引用

### 3. 划分数据集

```bash
cd Aerovision-V1/training/scripts

# 读取 prepare 输出目录
LATEST=$(cat data/fgvc_prepared/latest.txt)

python split_dataset.py \
  --prepare-dir "$LATEST" \
  --output "D:\Users\...\data\fgvc_splits" \
  --mode all
```

划分输出：
```
fgvc_splits/<timestamp>/
├── aerovision/
│   ├── aircraft/        # 机型分类数据集
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── airline/         # 航司分类数据集
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/          # 类别映射和统计
│       ├── type_classes.json
│       ├── airline_classes.json
│       └── dataset_statistics.json
├── train.csv
├── val.csv
└── test.csv
```

### 4. 训练机型分类模型

```bash
cd Aerovision-V1/training/scripts

python train_classify.py \
  --data "D:\Users\...\data\fgvc_splits/<timestamp>/aerovision/aircraft" \
  --model yolov8n-cls.pt \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 224 \
  --device 0 \
  --project output/classify \
  --name fgvc_aircraft_classifier
```

参数说明：
- `--data`: 数据集路径（指向 aircraft 目录）
- `--model`: 预训练模型（yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt）
- `--epochs`: 训练轮数（建议 100-200）
- `--batch-size`: 批次大小（根据 GPU 内存调整，32 或 64）
- `--imgsz`: 图像大小（224, 256, 384）
- `--device`: 设备（0=GPU 0, cpu=CPU）

### 5. 训练航司分类模型

```bash
cd Aerovision-V1/training/scripts

python train_classify.py \
  --data "D:\Users\...\data\fgvc_splits/<timestamp>/aerovision/airline" \
  --model yolov8n-cls.pt \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 224 \
  --device 0 \
  --project output/classify \
  --name fgvc_airline_classifier
```

## 使用工作流脚本

项目提供了自动化工作流脚本 `fgvc_workflow.py`，可以一键完成所有步骤：

```bash
cd Aerovision-V1/training/scripts

# 运行完整工作流（转换 + 准备 + 划分 + 训练）
python fgvc_workflow.py \
  --fgvc-dir "D:\Users\...\FGVC_Aircraft\raw" \
  --output-base "D:\Users\...\data" \
  --task all \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 224 \
  --device 0

# 只训练机型分类
python fgvc_workflow.py \
  --fgvc-dir "..." \
  --output-base "..." \
  --task aircraft \
  --epochs 100

# 只训练航司分类
python fgvc_workflow.py \
  --fgvc-dir "..." \
  --output-base "..." \
  --task airline \
  --epochs 100

# 跳过数据准备阶段（如果已准备好）
python fgvc_workflow.py \
  --fgvc-dir "..." \
  --output-base "..." \
  --task all \
  --skip-prepare \
  --skip-split
```

## 训练输出

训练完成后，输出位于：

```
output/classify/<experiment_name>/
├── weights/
│   ├── best.pt           # 最佳模型（验证集上表现最好）
│   └── last.pt           # 最新检查点
├── results.csv            # 训练结果统计
├── confusion_matrix.png   # 混淆矩阵
├── results.png           # 指标曲线
└── PR_curve.png           # PR 曲线
```

检查点也复制到：
```
ckpt/classify/<timestamp>/
├── best.pt
├── last.pt
└── train_<timestamp>.log
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir training/logs/classify/<timestamp>/
```

访问 http://localhost:6006/

### 查看日志

```bash
tail -f training/logs/classify/<timestamp>/train_<timestamp>.log
```

## 测试驱动开发 (TDD)

项目包含完整的测试套件：

```bash
cd Aerovision-V1/training/scripts

# 运行 FGVC 转换测试
python test_convert_fgvc_aircraft.py
```

测试覆盖：
- 标注文件解析
- 边界框文件解析
- 格式转换
- 数据合并
- 图片过滤
- 完整转换流程

## 数据集统计

转换后的数据集统计：
- **总图片数**: 10,000
- **机型变体**: 100 类
- **制造商**: 30 类
- **训练集**: 7,000 张 (70%)
- **验证集**: 1,500 张 (15%)
- **测试集**: 1,500 张 (15%)

## 注意事项

1. **路径问题**: 使用绝对路径避免相对路径解析错误
2. **显存不足**: 如果 OOM，减少 batch-size 或 imgsz
3. **模型选择**: 
   - yolov8n-cls.pt: 最小模型，训练快，精度稍低
   - yolov8s-cls.pt: 小型模型
   - yolov8m-cls.pt: 中型模型，推荐
4. **设备选择**: 使用 GPU 加速训练 (--device 0, 1, ...)
5. **数据划分**: FGVC 已有 train/val/test 划分，但项目会重新划分以适配项目流程

## 常见问题

### Q: 训练时找不到数据集目录？
A: 使用绝对路径，不要用相对路径。YOLOv8 分类需要目录，不是 YAML 文件。

### Q: 如何查看训练结果？
A: 查看训练输出目录的 weights/ 和 results.csv，或使用 TensorBoard。

### Q: 如何评估模型？
A: 使用 YOLO val() 方法或在训练脚本中设置 --mode val。

### Q: FGVC 的变体和族有什么区别？
A: 
- **Variant (变体)**: 最细粒度，如 "Boeing 737-700"，100 类
- **Family (族)**: 中等粒度，如 "Boeing 737"，70 类
- **Manufacturer (制造商)**: 粗粒度，如 "Boeing"，30 类

推荐使用 Variant 进行细粒度分类。

## 参考文档

- `AGENTS.md` - 项目总体指南
- `CLAUDE.md` - 训练模块详细指南
- `training/README.md` - 训练模块说明
- `conductor.md` - 训练阶段路线图

## 脚本列表

- `convert_fgvc_aerocraft.py` - FGVC 数据集转换
- `merge_fgvc_splits.py` - 合并分割集
- `fgvc_workflow.py` - 完整工作流脚本
- `test_convert_fgvc_aircraft.py` - TDD 测试套件
- `prepare_dataset.py` - 数据准备（项目已有）
- `split_dataset.py` - 数据划分（项目已有）
- `train_classify.py` - 分类训练（项目已有）
