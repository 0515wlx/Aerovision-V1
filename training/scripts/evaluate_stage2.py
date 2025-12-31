#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 评估脚本：机型分类模型评估
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import csv
import json

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        top_k_accuracy_score
    )
    from tqdm import tqdm
    from PIL import Image
except ImportError as e:
    print(f"错误：缺少必要的依赖库: {e}")
    print("请安装: pip install ultralytics scikit-learn matplotlib tqdm pillow")
    sys.exit(1)


def load_labels(labels_path):
    """加载标签文件"""
    labels = {}
    if os.path.exists(labels_path):
        with open(labels_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 假设CSV有filename和label列
                filename = row.get('filename', row.get('image', ''))
                label = row.get('label', row.get('class', ''))
                if filename and label:
                    labels[filename] = label
    return labels


def load_class_names(config_path):
    """加载类别名称"""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    names = config.get('names', {})
    # 转换为列表格式
    if isinstance(names, dict):
        class_names = [names.get(i, f'class_{i}') for i in range(len(names))]
    else:
        class_names = names
    return class_names


def evaluate_model(args):
    """评估模型性能"""
    print("="*60)
    print("机型分类模型评估")
    print("="*60)
    print(f"模型路径: {args.model}")
    print(f"测试数据路径: {args.data}")
    print(f"图片尺寸: {args.img_size}")
    print(f"批次大小: {args.batch_size}")
    print("="*60 + "\n")
    
    # 检查模型是否存在
    if not os.path.exists(args.model):
        print(f"错误：模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 加载模型
    print("加载模型...")
    model = YOLO(args.model)
    
    # 获取类别名称
    try:
        config_path = 'training/configs/aircraft_classify.yaml'
        class_names = load_class_names(config_path)
        print(f"类别数量: {len(class_names)}")
    except Exception as e:
        print(f"警告：无法加载类别名称: {e}")
        class_names = None
    
    # 收集测试图片
    test_dir = Path(args.data)
    if not test_dir.exists():
        print(f"错误：测试数据目录不存在: {args.data}")
        sys.exit(1)
    
    # 按类别组织图片
    image_files = []
    true_labels = []
    
    # 假设数据集按类别组织
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir():
            class_name = class_dir.name
            for img_file in class_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_files.append(str(img_file))
                    true_labels.append(class_name)
    
    if not image_files:
        print("错误：未找到测试图片")
        sys.exit(1)
    
    print(f"找到 {len(image_files)} 张测试图片\n")
    
    # 预测
    print("开始预测...")
    all_preds = []
    all_probs = []
    
    # 批量预测
    for i in tqdm(range(0, len(image_files), args.batch_size), desc="预测中"):
        batch_files = image_files[i:i + args.batch_size]
        results = model.predict(
            batch_files,
            imgsz=args.img_size,
            batch=args.batch_size,
            verbose=False
        )
        
        for result in results:
            if result.probs is not None:
                probs = result.probs.data.cpu().numpy()
                pred_class = int(result.probs.top1)
                all_preds.append(pred_class)
                all_probs.append(probs)
            else:
                print(f"警告：无法获取预测结果: {result.path}")
    
    # 转换标签为数值
    if class_names:
        # 创建类别到索引的映射
        unique_classes = sorted(set(true_labels))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        # 如果类别名称和模型输出不匹配，尝试匹配
        if len(class_to_idx) != len(class_names):
            print(f"警告：测试集类别数({len(class_to_idx)})与模型类别数({len(class_names)})不匹配")
        
        true_label_indices = [class_to_idx.get(label, 0) for label in true_labels]
    else:
        # 从文件路径推断标签
        true_label_indices = []
        for img_path in image_files:
            class_name = Path(img_path).parent.name
            true_label_indices.append(int(class_name) if class_name.isdigit() else 0)
    
    # 计算指标
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    # Top-1 准确率
    top1_acc = accuracy_score(true_label_indices, all_preds)
    print(f"Top-1 准确率: {top1_acc:.4f}")
    
    # Top-5 准确率
    if len(all_probs[0]) >= 5:
        all_probs_array = np.array(all_probs)
        top5_acc = top_k_accuracy_score(true_label_indices, all_probs_array, k=5)
        print(f"Top-5 准确率: {top5_acc:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print("-"*60)
    target_names = class_names if class_names else None
    report = classification_report(
        true_label_indices,
        all_preds,
        target_names=target_names,
        digits=4
    )
    print(report)
    
    # 混淆矩阵
    print("\n生成混淆矩阵...")
    cm = confusion_matrix(true_label_indices, all_preds)
    
    # 保存结果
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存评估结果到JSON
    results_dict = {
        'model_path': args.model,
        'test_data_path': args.data,
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(image_files),
        'top1_accuracy': float(top1_acc),
    }
    if len(all_probs[0]) >= 5:
        results_dict['top5_accuracy'] = float(top5_acc)
    
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"\n评估结果已保存到: {results_file}")
    
    # 保存分类报告
    report_file = output_dir / 'classification_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("机型分类模型评估报告\n")
        f.write("="*60 + "\n\n")
        f.write(f"模型路径: {args.model}\n")
        f.write(f"测试数据路径: {args.data}\n")
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"样本数量: {len(image_files)}\n\n")
        f.write("="*60 + "\n")
        f.write("评估指标\n")
        f.write("="*60 + "\n")
        f.write(f"Top-1 准确率: {top1_acc:.4f}\n")
        if len(all_probs[0]) >= 5:
            f.write(f"Top-5 准确率: {top5_acc:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("分类报告\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"分类报告已保存到: {report_file}")
    
    # 绘制混淆矩阵
    print("绘制混淆矩阵...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始混淆矩阵
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    fig.colorbar(im1, ax=axes[0])
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    fig.colorbar(im2, ax=axes[1])
    
    # 设置刻度和标签
    tick_marks = np.arange(len(class_names)) if class_names else np.arange(min(20, cm.shape[0]))
    
    for ax in axes:
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        if class_names:
            ax.set_xticklabels([class_names[i] if i < len(class_names) else str(i) for i in tick_marks], rotation=90, fontsize=8)
            ax.set_yticklabels([class_names[i] if i < len(class_names) else str(i) for i in tick_marks], fontsize=8)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    # 保存混淆矩阵
    cm_file = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_file, dpi=150, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {cm_file}")
    plt.close()
    
    # 单独保存归一化混淆矩阵
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(class_names)) if class_names else np.arange(min(20, cm.shape[0]))
    plt.xticks(tick_marks, [class_names[i] if i < len(class_names) else str(i) for i in tick_marks], rotation=90, fontsize=8)
    plt.yticks(tick_marks, [class_names[i] if i < len(class_names) else str(i) for i in tick_marks], fontsize=8)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    cm_norm_file = output_dir / 'confusion_matrix_normalized.png'
    plt.savefig(cm_norm_file, dpi=150, bbox_inches='tight')
    print(f"归一化混淆矩阵已保存到: {cm_norm_file}")
    plt.close()
    
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    print(f"结果目录: {output_dir}")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Stage 2 评估：机型分类模型评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认参数评估
  python evaluate_stage2.py
  
  # 指定模型和数据路径
  python evaluate_stage2.py --model training/checkpoints/stage2/best.pt --data training/data/processed/aircraft_crop/test
  
  # 指定输出目录
  python evaluate_stage2.py --output training/logs/eval_results
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='training/checkpoints/stage2/best.pt',
        help='模型路径 (默认: training/checkpoints/stage2/best.pt)'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='training/data/processed/aircraft_crop/test',
        help='测试数据路径 (默认: training/data/processed/aircraft_crop/test)'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        default='training/data/labels/test.csv',
        help='测试集标注路径 (默认: training/data/labels/test.csv)'
    )
    
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='图片尺寸 (默认: 224)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小 (默认: 32)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='training/logs',
        help='输出结果目录 (默认: training/logs)'
    )
    
    args = parser.parse_args()
    
    # 执行评估
    evaluate_model(args)


if __name__ == '__main__':
    main()
