#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备 YOLOv8 检测数据集

配置说明：
本脚本使用新的模块化配置系统，自动加载以下配置模块：
- paths.yaml: 路径配置 (data.*)
- base.yaml: 基础配置 (seed.*)

配置项：
- data.processed.labeled.images: 图片目录
- data.processed.labeled.registration: 标注目录 (YOLO格式)
- data.registration.detection_output: 输出目录

使用方法：
  # 使用默认配置
  python prepare_detection_dataset.py

  # 指定自定义路径
  python prepare_detection_dataset.py --image-dir path/to/images --label-dir path/to/labels

  # 使用自定义配置文件
  python prepare_detection_dataset.py --config my_config.yaml
"""

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml

# 添加configs模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def prepare_yolo_dataset(
    image_dir: str,
    label_dir: str,
    output_base_dir: str,
    train_ratio: float = 0.8,
    random_seed: int = 42
):
    """
    准备 YOLOv8 格式的数据集

    Args:
        image_dir: 图片目录
        label_dir: 标注目录
        output_base_dir: 输出基础目录
        train_ratio: 训练集比例
        random_seed: 随机种子
    """
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_base_dir) / f"detection_{timestamp}"

    print("=" * 60)
    print("准备 YOLOv8 检测数据集")
    print("=" * 60)
    print(f"图片目录: {image_dir}")
    print(f"标注目录: {label_dir}")
    print(f"输出目录: {output_dir}")
    print(f"训练集比例: {train_ratio}")
    print(f"随机种子: {random_seed}")
    print("=" * 60)

    image_path = Path(image_dir)
    label_path = Path(label_dir)
    output_path = Path(output_dir)

    # 创建输出目录
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # 获取所有标注文件
    label_files = list(label_path.glob('*.txt'))
    print(f"\n找到 {len(label_files)} 个标注文件")

    if len(label_files) == 0:
        print("错误: 没有找到标注文件!")
        return

    # 划分训练集和验证集
    train_files, val_files = train_test_split(
        label_files,
        train_size=train_ratio,
        random_state=random_seed
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

    print(f"\n数据集配置: {yaml_path}")
    print(f"数据集准备完成!")
    print("=" * 60)

    return yaml_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="准备 YOLOv8 检测数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置
  python prepare_detection_dataset.py

  # 指定自定义路径
  python prepare_detection_dataset.py --image-dir path/to/images --label-dir path/to/labels

  # 指定训练集比例
  python prepare_detection_dataset.py --train-ratio 0.85

  # 使用自定义配置文件
  python prepare_detection_dataset.py --config my_config.yaml

配置说明:
  本脚本使用模块化配置系统，自动加载 paths.yaml 和 base.yaml
  可以通过命令行参数覆盖配置文件中的值
        """
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='图片目录路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--label-dir',
        type=str,
        default=None,
        help='标注目录路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=None,
        help='随机种子（默认从配置文件读取）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='自定义配置文件路径'
    )

    args = parser.parse_args()

    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        # 只加载需要的模块
        config = load_config(modules=['paths'], load_all_modules=False)

    # 从配置文件获取路径（优先使用命令行参数）
    image_dir = args.image_dir or config.get('data.processed.labeled.images') or 'data/labeled'
    label_dir = args.label_dir or config.get('data.processed.labeled.registration') or 'data'
    output_base_dir = args.output_dir or config.get('data.local_data_root') or '../data'
    random_seed = args.random_seed if args.random_seed is not None else (config.get('seed.random') or 42)

    # 如果配置中的路径是相对路径，转换为绝对路径
    if image_dir and not Path(image_dir).is_absolute():
        image_dir = config.get_path('data.processed.labeled.images') or image_dir
    if label_dir and not Path(label_dir).is_absolute():
        label_dir = config.get_path('data.processed.labeled.registration') or label_dir
    if output_base_dir and not Path(output_base_dir).is_absolute():
        output_base_dir = config.get_path('data.local_data_root') or Path(output_base_dir).resolve()

    # 执行数据集准备
    prepare_yolo_dataset(
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        output_base_dir=str(output_base_dir),
        train_ratio=args.train_ratio,
        random_seed=random_seed
    )
