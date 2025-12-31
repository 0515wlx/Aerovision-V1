#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 训练脚本：机型分类训练
基于 YOLOv8x 分类模型
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
    import torch
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:
    print(f"错误：缺少必要的依赖库: {e}")
    print("请安装: pip install ultralytics torch tensorboard")
    sys.exit(1)


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(config):
    """创建必要的目录"""
    dirs = [
        config.get('checkpoint_dir', 'training/checkpoints/stage2'),
        config.get('log_dir', 'training/logs'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def train_stage2(args):
    """执行 Stage 2 训练"""
    # 加载配置文件
    config = load_config(args.config)
    
    # 创建必要的目录
    setup_directories(config)
    
    # 设置设备
    device = f'cuda:{args.device}' if args.device != 'cpu' else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载 YOLOv8x 分类模型
    model_name = config.get('model', 'yolov8x-cls.pt')
    print(f"加载模型: {model_name}")
    model = YOLO(model_name)
    
    # 准备训练参数
    data_path = config.get('data', 'training/data/processed/aircraft_crop')
    
    # 训练参数
    train_args = {
        'data': data_path,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': args.device,
        'project': config.get('project', 'runs/classify'),
        'name': config.get('name', 'aircraft_classifier'),
        'exist_ok': config.get('exist_ok', True),
        'pretrained': config.get('pretrained', True),
        'optimizer': config.get('optimizer', 'AdamW'),
        'lr0': config.get('lr0', 0.001),
        'lrf': config.get('lrf', 0.01),
        'momentum': config.get('momentum', 0.937),
        'weight_decay': config.get('weight_decay', 0.0005),
        'warmup_epochs': config.get('warmup_epochs', 3),
        'warmup_momentum': config.get('warmup_momentum', 0.8),
        'warmup_bias_lr': config.get('warmup_bias_lr', 0.1),
        'cos_lr': config.get('cos_lr', False),
        'patience': config.get('patience', 10),
        'save': config.get('save', True),
        'save_period': config.get('save_period', 10),
        'cache': config.get('cache', False),
        'workers': config.get('workers', 8),
        'amp': config.get('amp', True),
        'fraction': config.get('fraction', 1.0),
        'profile': config.get('profile', False),
        'freeze': config.get('freeze', None),
        'plots': config.get('plots', True),
        'verbose': config.get('verbose', True),
    }
    
    # 数据增强参数
    augmentation_params = {
        'hsv_h': config.get('hsv_h', 0.015),
        'hsv_s': config.get('hsv_s', 0.7),
        'hsv_v': config.get('hsv_v', 0.4),
        'degrees': config.get('degrees', 0.0),
        'translate': config.get('translate', 0.1),
        'scale': config.get('scale', 0.5),
        'shear': config.get('shear', 0.0),
        'perspective': config.get('perspective', 0.0),
        'flipud': config.get('flipud', 0.0),
        'fliplr': config.get('fliplr', 0.5),
        'mosaic': config.get('mosaic', 0.0),
        'mixup': config.get('mixup', 0.0),
    }
    train_args.update(augmentation_params)
    
    # 打印训练配置
    print("\n" + "="*60)
    print("训练配置:")
    print("="*60)
    for key, value in train_args.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    # 开始训练
    print(f"开始训练: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"图片尺寸: {args.img_size}")
    print()
    
    try:
        # 训练模型
        results = model.train(**train_args)
        
        # 保存最佳模型到指定目录
        checkpoint_dir = config.get('checkpoint_dir', 'training/checkpoints/stage2')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 复制最佳模型
        best_model_path = os.path.join(train_args['project'], train_args['name'], 'weights', 'best.pt')
        last_model_path = os.path.join(train_args['project'], train_args['name'], 'weights', 'last.pt')
        
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, os.path.join(checkpoint_dir, 'best.pt'))
            print(f"最佳模型已保存到: {os.path.join(checkpoint_dir, 'best.pt')}")
        
        if os.path.exists(last_model_path):
            shutil.copy(last_model_path, os.path.join(checkpoint_dir, 'last.pt'))
            print(f"最终模型已保存到: {os.path.join(checkpoint_dir, 'last.pt')}")
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)
        print(f"最佳模型: {os.path.join(checkpoint_dir, 'best.pt')}")
        print(f"最终模型: {os.path.join(checkpoint_dir, 'last.pt')}")
        print(f"训练结果目录: {os.path.join(train_args['project'], train_args['name'])}")
        print("="*60)
        
        return results
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 2 训练：机型分类训练（基于 YOLOv8x）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置训练
  python train_stage2.py
  
  # 指定训练轮数和批次大小
  python train_stage2.py --epochs 50 --batch-size 64
  
  # 使用CPU训练
  python train_stage2.py --device cpu
  
  # 从检查点恢复训练
  python train_stage2.py --resume runs/classify/aircraft_classifier/weights/last.pt
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/aircraft_classify.yaml',
        help='配置文件路径 (默认: training/configs/aircraft_classify.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='训练轮数 (默认: 30)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小 (默认: 32)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='图片尺寸 (默认: 224)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='设备 (默认: 0, 使用CPU则设为cpu)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练 (默认: None)'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"错误：配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 执行训练
    train_stage2(args)


if __name__ == '__main__':
    main()
