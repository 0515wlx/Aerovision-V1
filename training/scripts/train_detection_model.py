#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练 YOLOv8 检测模型
"""

from ultralytics import YOLO
from pathlib import Path


def train_detection_model(
    data_yaml: str,
    model_size: str = 'x',  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    image_size: int = 640,
    project: str = 'training/checkpoints',
    name: str = 'registration_detection',
    device: str = 'cpu'
):
    """
    训练 YOLOv8 检测模型
    
    Args:
        data_yaml: YOLO 数据集配置文件路径
        model_size: 模型大小 (n=samll, s=small, m=medium, l=large, x=extra large)
        epochs: 训练轮数
        batch_size: 批次大小
        image_size: 输入图像大小
        project: 项目目录
        name: 实验名称
        device: 设备 ('cpu' 或 'cuda:0')
    """
    print(f"{'='*60}")
    print("YOLOv8 检测模型训练")
    print(f"{'='*60}")
    print(f"数据集: {data_yaml}")
    print(f"模型大小: {model_size}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像大小: {image_size}")
    print(f"设备: {device}")
    print(f"{'='*60}\n")
    
    # 加载预训练模型
    model_name = f'yolov8{model_size}.pt'
    print(f"加载预训练模型: {model_name}")
    model = YOLO(model_name)
    
    # 训练
    print("开始训练...\n")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        project=project,
        name=name,
        device=device,
        
        # 数据增强（针对文字检测的优化）
        hsv_h=0.01,      # 色调变化（文字不需要太多色调变化）
        hsv_s=0.3,       # 饱和度变化
        hsv_v=0.3,       # 明度变化
        degrees=5,       # 旋转角度（文字不要旋转太多）
        translate=0.1,   # 平移
        scale=0.2,       # 缩放
        fliplr=0.0,      # 不左右翻转（文字方向很重要）
        flipud=0.0,      # 不上下翻转
        mosaic=0.5,      # 马赛克增强
        mixup=0.0,       # 不使用 mixup
        
        # 优化器设置
        lr0=0.01,        # 初始学习率
        lrf=0.01,        # 最终学习率因子
        momentum=0.937,  # SGD 动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,  # 预热轮数
        warmup_momentum=0.8,  # 预热动量
        warmup_bias_lr=0.1,   # 预热偏置学习率
        
        # 其他设置
        patience=20,      # 早停耐心值
        save=True,        # 保存检查点
        save_period=10,   # 每10轮保存一次
        plots=True,        # 绘制训练曲线
        verbose=True,      # 详细输出
    )
    
    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"最佳模型: {project}/{name}/weights/best.pt")
    print(f"最新模型: {project}/{name}/weights/last.pt")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='训练 YOLOv8 检测模型')
    parser.add_argument('--data', type=str, default='training/data/detection/dataset.yaml',
                        help='YOLO 数据集配置文件路径')
    parser.add_argument('--model', type=str, default='x',
                        help='模型大小 (n, s, m, l, x)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='输入图像大小')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu 或 cuda:0)')
    
    args = parser.parse_args()
    
    train_detection_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        device=args.device
    )
