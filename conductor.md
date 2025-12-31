# QuanPhotos AI 模型训练路线图

> 本文档是自训练航空照片识别系统的完整实施指南
> 详细的分步骤文档请查看 `training/docs/` 目录

---

## 目录

1. [项目目标](#项目目标)
2. [环境配置](#环境配置)
3. [目录结构](#目录结构)
4. [数据规范](#数据规范)
5. [训练阶段总览](#训练阶段总览)
6. [评估指标](#评估指标)
7. [常见问题](#常见问题)

---

## 项目目标

构建一个航空照片视觉识别系统，实现以下能力：

| 任务 | 输入 | 输出 | 优先级 | 说明 |
|------|------|------|--------|------|
| 机型分类 | 飞机图片 | Boeing 737-800 等 | P0 | 核心任务 |
| 航司识别 | 飞机图片 | China Eastern 等 | P1 | 多任务学习 |
| 清晰度评估 | 飞机图片 | 0-1 分数 | P1 | 质量评估 |
| 遮挡检测 | 飞机图片 | 0-1 分数 | P1 | 新增：判断飞机是否被遮挡 |
| 注册号识别 | 飞机图片 | B-1234 等字符串 | P2 | OCR 独立模块 |
| 置信度输出 | 所有预测 | 可信度分数 | P2 | 最终校准 |

---

## 环境配置

### 硬件要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| GPU | RTX 3060 12GB | RTX 4090 24GB |
| 内存 | 16GB | 32GB+ |
| 硬盘 | 100GB SSD | 500GB NVMe |
| CUDA | 11.8+ | 12.1+ |

### 软件环境

```bash
# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows

# 2. 安装核心依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安装训练相关包
pip install timm==1.0.3           # 预训练模型库
pip install ultralytics==8.1.0    # YOLOv8
pip install albumentations==1.4.0 # 数据增强
pip install wandb==0.16.0         # 实验追踪
pip install tensorboard==2.15.0   # 可视化
pip install pandas scikit-learn   # 数据处理

# 4. 安装 OCR 相关（阶段 6 使用）
pip install paddlepaddle-gpu paddleocr
```

### 验证安装

```python
# verify_env.py
import torch
import timm

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"timm: {timm.__version__}")

# 测试模型加载
model = timm.create_model("convnext_base", pretrained=True)
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(f"ConvNeXt output shape: {y.shape}")
print("✅ Environment OK!")
```

---

## 目录结构

```
Aerovision-V1/
├── conductor.md                # 本文档（总览）
├── requirements.txt            # 依赖
│
├── training/
│   ├── docs/                   # 📚 详细分步骤文档
│   │   ├── stage0_environment.md
│   │   ├── stage1_data_preparation.md
│   │   ├── stage2_single_task.md
│   │   ├── stage3_multi_head.md
│   │   ├── stage4_quality_block.md
│   │   ├── stage5_hybrid.md
│   │   ├── stage6_ocr.md
│   │   └── stage7_integration.md
│   │
│   ├── configs/                # 配置文件
│   │   ├── stage2_type.yaml
│   │   ├── stage3_multi.yaml
│   │   └── stage5_hybrid.yaml
│   │
│   ├── data/                   # 数据目录
│   │   ├── raw/                # 原始图片
│   │   ├── processed/          # 处理后数据
│   │   │   └── aircraft_crop/
│   │   │       ├── unsorted/   # 裁剪后待标注
│   │   │       ├── train/      # 训练集（按类别）
│   │   │       ├── val/        # 验证集
│   │   │       └── test/       # 测试集
│   │   └── labels/             # 标注文件
│   │       ├── aircraft_labels.csv
│   │       ├── type_classes.json
│   │       ├── airline_classes.json
│   │       └── registration/   # 注册号 bbox
│   │
│   ├── src/                    # 源代码
│   │   ├── data/               # 数据处理
│   │   ├── models/             # 模型定义
│   │   ├── trainers/           # 训练器
│   │   └── utils/              # 工具函数
│   │
│   ├── scripts/                # 运行脚本
│   ├── checkpoints/            # 模型检查点
│   └── logs/                   # 日志
│
└── annotation_server/          # 标注工具（可选）
```

---

## 数据规范

### 标注字段说明

你的标注文件包含以下字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | ✅ | 图片文件名 |
| `typeid` | int | ❌ | 机型ID（自动生成） |
| `typename` | string | ✅ | 机型名称，如 `A320`、`B737-800` |
| `airlineid` | int | ❌ | 航司ID（自动生成） |
| `airlinename` | string | ❌ | 航司名称，如 `China Eastern` |
| `clarity` | float | ✅ | 清晰度 0.0-1.0（1.0=清晰，0.0=模糊） |
| `block` | float | ✅ | **遮挡程度 0.0-1.0**（0.0=无遮挡，1.0=完全遮挡） |
| `registration` | string | ❌ | 注册号文字，如 `B-1234`，不可见则留空 |
| `airplanearea` | float | ❌ | 飞机占画面比例 0.0-1.0 |
| `registrationarea` | string | ❌ | 注册号区域 bbox（YOLO 格式或留空） |

### 标注文件示例

```csv
filename,typeid,typename,airlineid,airlinename,clarity,block,registration,airplanearea,registrationarea
IMG_0001.jpg,0,A320,1,China Eastern,0.95,0.0,B-1234,0.45,0.85 0.65 0.12 0.04
IMG_0002.jpg,1,B737-800,0,Air China,0.80,0.15,B-5678,0.38,
IMG_0003.jpg,7,A380,8,Emirates,0.70,0.40,,0.52,
IMG_0004.jpg,4,B787-9,3,Hainan Airlines,0.50,0.60,,0.30,
```

### 类别映射文件

```json
// labels/type_classes.json
{
  "classes": ["A320", "B737-800", "B747-400", "B777-300ER", "B787-9",
              "A330-300", "A350-900", "A380", "ARJ21", "C919"],
  "num_classes": 10
}
```

---

## 训练阶段总览

```
阶段 0 ──→ 阶段 1 ──→ 阶段 2 ──→ 阶段 3 ──→ 阶段 4 ──→ 阶段 5 ──→ 阶段 6 ──→ 阶段 7
  │          │          │          │          │          │          │          │
 环境       数据       单任务     多Head    清晰度     Hybrid     OCR      联合
 配置       准备      机型分类   +航司     +遮挡      模型融合   注册号    集成
  │          │          │          │          │          │          │          │
 1天       3-5天      2-3天      2天       2天       2-3天      3-4天     2天
```

### ⚠️ 铁律

1. **任何阶段没"过关"，不准跳到下一阶段**
2. **任何新东西，只加一个** —— 不要同时改多个变量
3. **每个阶段都要能单独跑** —— 保持模块独立

### 各阶段概览

| 阶段 | 名称 | 核心目标 | 详细文档 |
|------|------|----------|----------|
| 0 | 环境配置 | 跑通环境，理解基础概念 | [stage0_environment.md](training/docs/stage0_environment.md) |
| 1 | 数据准备 | 获得干净的飞机裁剪图 + 完成标注 | [stage1_data_preparation.md](training/docs/stage1_data_preparation.md) |
| 2 | 单任务训练 | ConvNeXt 机型分类跑通 | [stage2_single_task.md](training/docs/stage2_single_task.md) |
| 3 | 多 Head | 同时输出机型 + 航司 | [stage3_multi_head.md](training/docs/stage3_multi_head.md) |
| 4 | 清晰度+遮挡 | 添加 clarity 和 block 两个回归 Head | [stage4_quality_block.md](training/docs/stage4_quality_block.md) |
| 5 | Hybrid 融合 | ConvNeXt + Swin 特征融合 | [stage5_hybrid.md](training/docs/stage5_hybrid.md) |
| 6 | OCR | 注册号检测 + 识别 | [stage6_ocr.md](training/docs/stage6_ocr.md) |
| 7 | 联合集成 | 置信度校准 + 完整 Pipeline | [stage7_integration.md](training/docs/stage7_integration.md) |

---

## 评估指标

### 分类任务（机型 / 航司）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Top-1 Accuracy | > 85% | 主要指标 |
| Top-5 Accuracy | > 95% | 容错指标 |
| Macro F1 | > 0.80 | 各类平衡 |
| Per-class Accuracy | 各类 > 70% | 无严重短板 |

### 回归任务（清晰度 / 遮挡）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| MAE | < 0.10 | 平均绝对误差 |
| RMSE | < 0.15 | 均方根误差 |
| Correlation | > 0.85 | 预测值与真实值相关性 |

### OCR 任务

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 检测率 | > 90% | 能找到注册号区域 |
| 完全正确率 | > 80% | 整个字符串正确 |
| 字符准确率 | > 95% | 单字符准确 |

---

## 常见问题

### Q1: loss 不降？

检查顺序：
1. 学习率是否太大/太小？（先试 1e-4）
2. 数据加载是否正确？（打印几个样本看看）
3. 标签是否正确？（检查 CSV 和图片对应）
4. 模型是否在 `train()` 模式下？

### Q2: 显存不够？

解决方案（按优先级）：
1. 减小 `batch_size`（32 → 16 → 8）
2. 使用混合精度训练 `torch.cuda.amp`
3. 使用 gradient accumulation
4. 用更小的模型（convnext_base → convnext_small）

### Q3: 过拟合？

解决方案：
1. 增加数据量 / 数据增强
2. 增加 Dropout
3. 使用 Early Stopping
4. 减小模型复杂度

### Q4: 多任务效果变差？

解决方案：
1. 检查各任务 loss 的量级是否一致
2. 使用 loss 加权（小 loss 任务权重调大）
3. 分阶段训练：先训主任务，再加辅助任务

---

## 开始训练

准备好后，请按顺序阅读 `training/docs/` 中的文档：

1. 📘 [阶段 0：环境配置](training/docs/stage0_environment.md) ← **从这里开始**
2. 📗 [阶段 1：数据准备](training/docs/stage1_data_preparation.md)
3. 📙 [阶段 2：单任务训练](training/docs/stage2_single_task.md)
4. ...依此类推

祝训练顺利！🚀
