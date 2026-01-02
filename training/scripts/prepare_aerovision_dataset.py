#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aerovision-V1 数据集准备脚本
从labels.csv和labeled目录创建YOLOv8分类格式的数据集

功能：
1. 读取labels.csv获取机型信息
2. 从labeled目录复制图片到对应的类别目录
3. 按比例划分train/val/test
4. 生成类别映射文件
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AerovisionDatasetPreparer:
    """Aerovision-V1 数据集准备类"""

    def __init__(
        self,
        labels_csv: str,
        images_dir: str,
        output_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> None:
        """
        初始化数据集准备器

        Args:
            labels_csv: labels.csv文件路径
            images_dir: 原始图片目录（labeled目录）
            output_dir: 输出目录路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.labels_csv = Path(labels_csv)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # 验证比例总和
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"训练集、验证集、测试集比例之和必须为1.0，"
                f"当前为{train_ratio + val_ratio + test_ratio}"
            )

        # 设置输出路径
        self.processed_dir = self.output_dir / "processed" / "aircraft"
        self.labels_dir = self.output_dir / "processed" / "labels"
        self.configs_dir = self.output_dir.parent / "configs"

        # 类别映射（类别名 -> 类别ID）
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

        # 数据集信息
        self.dataset_info: Dict[str, Dict[str, int]] = {}

        # 设置随机种子
        random.seed(random_seed)

    def load_labels(self) -> Dict[str, Dict]:
        """
        从labels.csv加载标签信息

        Returns:
            字典: {filename: {typename, clarity, block, registration}}
        """
        logger.info(f"加载标签文件: {self.labels_csv}")

        labels = {}
        class_names = set()

        # 先读取文件内容，去除多余的BOM
        with open(self.labels_csv, 'rb') as f:
            content = f.read()
            # 去除BOM（可能有多个）
            while content.startswith(b'\xef\xbb\xbf'):
                content = content[3:]

        # 解码CSV内容
        import io
        csv_file = io.StringIO(content.decode('utf-8'))
        reader = csv.DictReader(csv_file)

        for row in reader:
            # 处理可能的空格问题
            filename = row.get('filename', '').strip()
            typename = row.get('typename', '').strip()
            clarity = float(row.get('clarity', 0))
            block = float(row.get('block', 0))
            registration = row.get('registration', '').strip()

            # 跳过空记录
            if not filename or not typename:
                continue

            labels[filename] = {
                'typename': typename,
                'clarity': clarity,
                'block': block,
                'registration': registration
            }
            class_names.add(typename)

        logger.info(f"加载了 {len(labels)} 个标签")
        logger.info(f"发现 {len(class_names)} 个机型类别")

        return labels, class_names

    def build_class_mapping(self, class_names: set) -> None:
        """
        构建类别映射

        Args:
            class_names: 所有类别名的集合
        """
        logger.info("构建类别映射...")

        # 按字母顺序排序类别名，并分配ID
        sorted_classes = sorted(class_names)
        self.class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}
        self.id_to_class = {idx: name for idx, name in enumerate(sorted_classes)}

        logger.info(f"共发现 {len(self.class_to_id)} 个机型类别")
        for class_id, class_name in self.id_to_class.items():
            logger.info(f"  {class_id}: {class_name}")

    def split_dataset(
        self,
        labels: Dict[str, Dict]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """
        按比例划分数据集

        Args:
            labels: {filename: {typename, clarity, block, registration}}

        Returns:
            (train_files, val_files, test_files) 每个都是 {typename: [filename列表]}
        """
        logger.info(
            f"划分数据集: 训练集{self.train_ratio*100:.0f}%, "
            f"验证集{self.val_ratio*100:.0f}%, "
            f"测试集{self.test_ratio*100:.0f}%"
        )

        # 按类别分组文件
        class_files: Dict[str, List[str]] = defaultdict(list)
        for filename, info in labels.items():
            typename = info['typename']
            class_files[typename].append(filename)

        # 划分数据集
        train_files: Dict[str, List[str]] = defaultdict(list)
        val_files: Dict[str, List[str]] = defaultdict(list)
        test_files: Dict[str, List[str]] = defaultdict(list)

        for typename, files in class_files.items():
            # 打乱该类别的文件
            shuffled_files = files.copy()
            random.shuffle(shuffled_files)

            # 计算各数据集大小
            total = len(shuffled_files)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            # 确保至少有一个验证样本（如果总样本数>=3）
            if total >= 3 and val_size == 0:
                val_size = 1
                train_size = total - val_size - 1  # 确保test至少有1个

            # 划分数据
            train_files[typename] = shuffled_files[:train_size]
            val_files[typename] = shuffled_files[train_size:train_size + val_size]
            test_files[typename] = shuffled_files[train_size + val_size:]

            logger.info(
                f"类别 {typename}: 训练集{len(train_files[typename])}, "
                f"验证集{len(val_files[typename])}, 测试集{len(test_files[typename])}"
            )

        # 统计数据集信息
        for split, split_data in [('train', train_files), ('val', val_files), ('test', test_files)]:
            total = sum(len(files) for files in split_data.values())
            self.dataset_info[split] = {
                "total": total,
                "classes": {cls: len(files) for cls, files in split_data.items()}
            }

        return train_files, val_files, test_files

    def create_directory_structure(self) -> None:
        """创建YOLOv8分类所需的目录结构"""
        logger.info("创建目录结构...")

        # 创建主目录
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # 为每个数据集和类别创建目录
        for split in ['train', 'val', 'test']:
            split_dir = self.processed_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for class_id, class_name in self.id_to_class.items():
                # 清理类别名称，替换特殊字符
                safe_class_name = class_name.replace(' ', '_').replace('/', '_')
                class_dir = split_dir / safe_class_name
                class_dir.mkdir(parents=True, exist_ok=True)

        logger.info("目录结构创建完成")

    def copy_images(
        self,
        train_files: Dict[str, List[str]],
        val_files: Dict[str, List[str]],
        test_files: Dict[str, List[str]]
    ) -> None:
        """
        复制图片到对应的类别目录

        Args:
            train_files: {typename: [filename列表]}
            val_files: {typename: [filename列表]}
            test_files: {typename: [filename列表]}
        """
        logger.info("复制图片...")

        split_files = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split, split_data in split_files.items():
            logger.info(f"\n处理 {split} 数据集...")

            copied_count = 0
            skipped_count = 0

            for typename, filenames in split_data.items():
                # 清理类别名称
                safe_class_name = typename.replace(' ', '_').replace('/', '_')

                for filename in filenames:
                    src_path = self.images_dir / filename
                    dst_path = self.processed_dir / split / safe_class_name / filename

                    # 检查源文件是否存在
                    if not src_path.exists():
                        logger.warning(f"源文件不存在: {src_path}")
                        skipped_count += 1
                        continue

                    # 复制文件
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        logger.error(f"复制文件失败 {filename}: {e}")
                        skipped_count += 1

            logger.info(
                f"{split} 数据集图片复制完成: "
                f"成功{copied_count}, 跳过{skipped_count}"
            )

        logger.info("\n所有图片复制完成")

    def save_class_mapping(self) -> None:
        """保存类别映射到JSON文件"""
        logger.info("保存类别映射...")

        output_file = self.labels_dir / "type_classes.json"

        # 准备输出数据
        class_mapping = {
            "num_classes": len(self.id_to_class),
            "classes": [
                {"id": class_id, "name": class_name}
                for class_id, class_name in self.id_to_class.items()
            ]
        }

        # 保存JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"类别映射已保存到: {output_file}")

    def save_dataset_statistics(self) -> None:
        """保存数据集统计信息到JSON文件"""
        logger.info("保存数据集统计信息...")

        output_file = self.labels_dir / "dataset_statistics.json"

        # 准备输出数据
        statistics = {
            "num_classes": len(self.id_to_class),
            "splits": self.dataset_info,
            "total_images": sum(info["total"] for info in self.dataset_info.values())
        }

        # 保存JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        logger.info(f"数据集统计信息已保存到: {output_file}")

    def create_yolo_config(self) -> None:
        """创建YOLOv8配置文件"""
        logger.info("创建YOLOv8配置文件...")

        config_file = self.configs_dir / "aircraft_classify.yaml"

        # 准备配置数据
        config = {
            # 数据集配置
            "path": str(self.processed_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",

            # 类别配置
            "names": self.id_to_class,
            "nc": len(self.id_to_class),

            # 训练参数（从统一配置文件读取）
            "epochs": 10,
            "batch_size": 32,
            "imgsz": 224,
            "patience": 50,
            "save": True,
            "device": "cpu",
            "workers": 8,
            "project": "runs/classify",
            "name": "aircraft_classifier",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "cos_lr": True,
            "amp": True,
            "val": True,
            "plots": True,
        }

        # 保存YAML文件
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"YOLOv8配置文件已保存到: {config_file}")

    def prepare(self) -> None:
        """执行完整的数据集准备流程"""
        logger.info("=" * 60)
        logger.info("开始准备Aerovision-V1数据集")
        logger.info("=" * 60)

        # 1. 加载标签
        labels, class_names = self.load_labels()

        # 2. 构建类别映射
        self.build_class_mapping(class_names)

        # 3. 划分数据集
        train_files, val_files, test_files = self.split_dataset(labels)

        # 4. 创建目录结构
        self.create_directory_structure()

        # 5. 复制图片
        self.copy_images(train_files, val_files, test_files)

        # 6. 保存类别映射
        self.save_class_mapping()

        # 7. 保存数据集统计信息
        self.save_dataset_statistics()

        # 8. 创建YOLOv8配置文件
        self.create_yolo_config()

        logger.info("=" * 60)
        logger.info("数据集准备完成!")
        logger.info("=" * 60)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从labels.csv和labeled目录创建YOLOv8分类格式的数据集"
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default='data/labels.csv',
        help='labels.csv文件路径'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default='data/labeled',
        help='原始图片目录（labeled目录）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='输出目录路径 (默认: data)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='训练集比例 (默认: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='验证集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='测试集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    args = parser.parse_args()

    # 创建数据集准备器
    preparer = AerovisionDatasetPreparer(
        labels_csv=args.labels_csv,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.random_seed
    )

    # 执行数据集准备
    preparer.prepare()


if __name__ == '__main__':
    main()
