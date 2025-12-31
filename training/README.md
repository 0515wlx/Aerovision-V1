# AeroVision 训练模块

AeroVision 是一个基于 YOLOv8 最大版本的飞机识别系统，支持机型分类、航司检测和 OCR 识别。

## 📋 目录

- [项目简介](#项目简介)
- [快速开始](#快速开始)
- [目录结构](#目录结构)
- [核心功能](#核心功能)
- [技术栈](#技术栈)
- [文档链接](#文档链接)
- [注意事项](#注意事项)

## 项目简介

AeroVision 训练模块是一个完整的飞机识别系统训练框架，主要特性包括：

- **基于 YOLOv8x**：使用 YOLOv8 最大版本作为基座模型，提供最强大的检测能力
- **多任务支持**：支持机型分类、航司检测和 OCR 识别等多个任务
- **完整训练流程**：从数据准备到模型评估的端到端训练流程
- **模块化设计**：清晰的代码结构，便于维护和扩展

## 快速开始

### 1. 验证环境

```bash
python training/scripts/verify_env.py
```

### 2. 数据准备

```bash
# 裁剪飞机区域
python training/scripts/crop_aircraft.py

# 划分数据集
python training/scripts/split_dataset.py
```

### 3. 训练模型

```bash
# 训练 Stage 2（机型分类）
python training/scripts/train_stage2.py
```

### 4. 评估模型

```bash
# 评估 Stage 2 模型
python training/scripts/evaluate_stage2.py
```

## 目录结构

```
training/
├── configs/              # 配置文件
│   ├── aircraft_classify.yaml    # 机型分类配置
│   └── data_paths.yaml           # 数据路径配置
├── data/                 # 数据目录
├── scripts/             # 脚本目录
│   ├── crop_aircraft.py          # 飞机区域裁剪
│   ├── split_dataset.py          # 数据集划分
│   ├── train_stage2.py           # Stage 2 训练
│   ├── evaluate_stage2.py        # Stage 2 评估
│   ├── verify_env.py             # 环境验证
│   ├── verify_data.py            # 数据验证
│   ├── prepare_dataset.py        # 数据准备
│   ├── ocr_pipeline.py           # OCR 管道
│   └── train_aircraft_classifier.py  # 机型分类训练
├── src/                 # 源代码
│   ├── models/          # 模型定义
│   ├── ocr/             # OCR 模块
│   │   └── paddle_ocr.py        # PaddleOCR 实现
│   ├── trainers/        # 训练器
│   └── utils/           # 工具函数
├── checkpoints/         # 模型检查点
│   ├── stage2/          # Stage 2 检查点
│   ├── stage3/          # Stage 3 检查点
│   ├── stage4/          # Stage 4 检查点
│   ├── stage5/          # Stage 5 检查点
│   └── stage6/          # Stage 6 检查点
├── logs/               # 日志目录
├── tests/              # 测试代码
│   └── test_aircraft_classifier.py
└── docs/               # 文档目录
    ├── data_format.md  # 数据格式规范
    └── workflow.md     # 开发流程
```

## 核心功能

### 环境验证
- 检查依赖库是否正确安装
- 验证 CUDA/GPU 环境（如适用）
- 确认配置文件完整性

### 数据准备
- **飞机区域裁剪**：从原始图像中裁剪出飞机区域
- **数据集划分**：将数据集划分为训练集、验证集和测试集
- **数据验证**：检查数据格式和标注的正确性

### 模型训练
- **YOLOv8x 训练**：基于最大版本的 YOLOv8 进行训练
- **多阶段训练**：支持不同阶段的训练任务
- **超参数调优**：支持自定义训练参数

### 模型评估
- **性能指标**：计算准确率、精确率、召回率等指标
- **可视化分析**：生成混淆矩阵、训练曲线等可视化结果
- **模型对比**：支持不同模型版本的对比评估

### OCR 识别
- **PaddleOCR 集成**：基于 PaddleOCR 的文字识别
- **航司识别**：识别飞机注册号和航司信息
- **端到端管道**：完整的 OCR 处理流程

## 技术栈

### 核心框架
- **YOLOv8x**：目标检测和分类模型
- **PyTorch**：深度学习框架

### OCR
- **PaddleOCR**：开源 OCR 工具包

### 数据处理
- **Albumentations**：数据增强库
- **OpenCV**：图像处理库

### 训练与可视化
- **TensorBoard**：训练过程可视化
- **Ultralytics**：YOLOv8 训练框架

## 文档链接

- [数据格式规范](docs/data_format.md) - 详细的数据格式和标注规范
- [开发流程](docs/workflow.md) - 完整的开发和训练流程说明
- [原始规范文档](../training-by-collaborator/docs/) - 协作者提供的原始规范文档

## 注意事项

### 数据存储
- 数据实际存储在服务器 `mnt/disk/Aerovision-V1-data` 目录
- 请确保正确配置数据路径（参见 [`configs/data_paths.yaml`](configs/data_paths.yaml)）

### 模型选择
- 使用 YOLOv8x（最大版本）作为基座模型
- 确保有足够的 GPU 内存用于训练

### 训练进度
- Stage 2（机型分类）已完成
- 其他阶段的训练正在进行中

### 环境要求
- Python 3.8+
- CUDA 11.0+（如使用 GPU）
- 至少 16GB RAM（推荐 32GB）

## 许可证

本项目遵循项目的整体许可证。详见项目根目录的 [`LICENSE`](../LICENSE) 文件。

## 联系方式

如有问题或建议，请通过项目 Issue 跟踪器反馈。
