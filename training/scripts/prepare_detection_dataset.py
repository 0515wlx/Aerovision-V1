#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 YOLOv8 检测数据集
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml


def prepare_yolo_dataset(
    image_dir: str,
    label_dir: str,
    output_dir: str,
    train_ratio: float = 0.8
):
    """
    准备 YOLOv8 格式的数据集
    
    Args:
        image_dir: 图片目录
        label_dir: 标注目录
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有标注文件
    label_files = list(label_path.glob('*.txt'))
    print(f"找到 {len(label_files)} 个标注文件")
    
    if len(label_files) == 0:
        print("错误: 没有找到标注文件!")
        return
    
    # 划分训练集和验证集
    train_files, val_files = train_test_split(
        label_files, 
        train_size=train_ratio, 
        random_state=42
    )
    print(f"训练集: {len(train_files)} 张")
    print(f"验证集: {len(val_files)} 张")
    
    # 处理数据
    def process_files(files, split_name):
        count = 0
        for label_file in files:
            # 对应的图片文件
            img_file = image_path / label_file.stem
            
            # 尝试不同的图片扩展名
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if (img_file.with_suffix(ext)).exists():
                    img_file = img_file.with_suffix(ext)
                    break
            else:
                print(f"警告: 找不到图片 {label_file.stem}")
                continue
            
            # 复制图片
            dst_img = output_path / 'images' / split_name / img_file.name
            shutil.copy2(img_file, dst_img)
            
            # 复制标注
            dst_label = output_path / 'labels' / split_name / label_file.name
            shutil.copy2(label_file, dst_label)
            
            count += 1
        
        return count
    
    train_count = process_files(train_files, 'train')
    val_count = process_files(val_files, 'val')
    
    print(f"\n成功处理:")
    print(f"  训练集: {train_count} 张")
    print(f"  验证集: {val_count} 张")
    
    # 创建 YOLO 配置文件
    yaml_content = f"""# YOLOv8 Dataset Configuration
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: registration
"""
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✅ 数据集配置: {yaml_path}")
    print(f"✅ 数据集准备完成!")
    
    return yaml_path


if __name__ == '__main__':
    prepare_yolo_dataset(
        image_dir='data/labeled',
        label_dir='data',
        output_dir='training/data/detection',
        train_ratio=0.8
    )
