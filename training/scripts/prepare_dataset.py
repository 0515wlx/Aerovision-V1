#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集准备脚本 - 将FGVC_Aircraft_dataset转换为YOLOv8分类格式

功能：
1. 从文件夹结构读取数据集（文件夹名作为类别）
2. 自动扫描train、val、test文件夹
3. 如果数据集已按train/val/test划分，直接使用现有划分
4. 如果只有train文件夹，按比例划分数据集（默认80%/10%/10%）
5. 生成类别映射JSON文件
6. 生成YOLOv8配置文件
"""

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """数据集准备类，用于将FGVC_Aircraft_dataset转换为YOLOv8分类格式"""

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> None:
        """
        初始化数据集准备器

        Args:
            source_dir: FGVC_Aircraft_dataset根目录
            output_dir: 输出目录路径
            train_ratio: 训练集比例（仅在需要划分时使用）
            val_ratio: 验证集比例（仅在需要划分时使用）
            test_ratio: 测试集比例（仅在需要划分时使用）
            random_seed: 随机种子
        """
        self.source_dir = Path(source_dir)
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
        self.labels_dir = self.output_dir / "labels"
        self.configs_dir = self.output_dir.parent / "configs"

        # 类别映射（类别名 -> 类别ID）
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

        # 数据集信息
        self.dataset_info: Dict[str, Dict[str, int]] = {}

        # 设置随机种子
        random.seed(random_seed)

        # 检测数据集结构
        self.dataset_structure = self._detect_dataset_structure()

    def _detect_dataset_structure(self) -> str:
        """
        检测数据集结构

        Returns:
            数据集结构类型: 'split'（已划分）、'partial_split'（部分划分）或 'single'（单文件夹）
        """
        logger.info(f"检测数据集结构: {self.source_dir}")

        # 检查是否存在train、val、test文件夹
        train_dir = self.source_dir / "train"
        val_dir = self.source_dir / "val"
        test_dir = self.source_dir / "test"

        if train_dir.exists() and train_dir.is_dir():
            if val_dir.exists() and val_dir.is_dir() and test_dir.exists() and test_dir.is_dir():
                logger.info("检测到已划分的数据集结构（train/val/test）")
                return "split"
            elif test_dir.exists() and test_dir.is_dir():
                logger.info("检测到部分划分的数据集结构（train/test，需要从train划分val）")
                return "partial_split"
            else:
                logger.info("检测到单文件夹数据集结构（仅train）")
                return "single"
        else:
            # 检查是否直接是类别文件夹
            subdirs = [d for d in self.source_dir.iterdir() if d.is_dir()]
            if subdirs:
                logger.info("检测到直接类别文件夹结构")
                return "single"
            else:
                raise ValueError(
                    f"无法识别数据集结构: {self.source_dir}。"
                    "请确保数据集包含train文件夹或直接包含类别文件夹。"
                )

    def _scan_directory(self, directory: Path) -> Dict[str, List[Path]]:
        """
        扫描目录，获取所有类别的图片

        Args:
            directory: 要扫描的目录

        Returns:
            字典: {类别名: [图片路径列表]}
        """
        logger.info(f"扫描目录: {directory}")

        class_images: Dict[str, List[Path]] = defaultdict(list)
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

        if not directory.exists():
            logger.warning(f"目录不存在: {directory}")
            return class_images

        # 遍历所有子目录（类别）
        for class_dir in directory.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            logger.info(f"处理类别: {class_name}")

            # 扫描该类别下的所有图片
            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in supported_extensions:
                    class_images[class_name].append(img_file)

            logger.info(f"  找到 {len(class_images[class_name])} 张图片")

        return class_images

    def build_class_mapping(self, all_classes: set) -> None:
        """
        构建类别映射

        Args:
            all_classes: 所有类别名的集合
        """
        logger.info("构建类别映射...")

        # 按字母顺序排序类别名，并分配ID
        sorted_classes = sorted(all_classes)
        self.class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}
        self.id_to_class = {idx: name for idx, name in enumerate(sorted_classes)}

        logger.info(f"共发现 {len(self.class_to_id)} 个机型类别")

    def load_dataset_from_structure(self) -> Dict[str, Dict[str, List[Path]]]:
        """
        从文件夹结构加载数据集

        Returns:
            字典: {split: {class_name: [图片路径列表]}}
        """
        logger.info("=" * 60)
        logger.info("从文件夹结构加载数据集")
        logger.info("=" * 60)

        dataset: Dict[str, Dict[str, List[Path]]] = {}

        if self.dataset_structure == "split":
            # 已划分的数据集结构（train/val/test）
            dataset["train"] = self._scan_directory(self.source_dir / "train")
            dataset["val"] = self._scan_directory(self.source_dir / "val")
            dataset["test"] = self._scan_directory(self.source_dir / "test")

            # 收集所有类别
            all_classes = set()
            for split_data in dataset.values():
                all_classes.update(split_data.keys())

            self.build_class_mapping(all_classes)

            # 统计数据集信息
            for split, split_data in dataset.items():
                total = sum(len(images) for images in split_data.values())
                self.dataset_info[split] = {
                    "total": total,
                    "classes": {cls: len(imgs) for cls, imgs in split_data.items()}
                }

        elif self.dataset_structure == "partial_split":
            # 部分划分的数据集结构（train/test，需要从train划分val）
            logger.info("从train和test加载数据集，将从train划分出val...")

            dataset["train"] = self._scan_directory(self.source_dir / "train")
            dataset["test"] = self._scan_directory(self.source_dir / "test")

            # 收集所有类别
            all_classes = set()
            all_classes.update(dataset["train"].keys())
            all_classes.update(dataset["test"].keys())

            self.build_class_mapping(all_classes)

            # 从train划分出val
            dataset["val"] = self._split_train_to_val(dataset["train"])

            # 统计数据集信息
            for split, split_data in dataset.items():
                total = sum(len(images) for images in split_data.values())
                self.dataset_info[split] = {
                    "total": total,
                    "classes": {cls: len(imgs) for cls, imgs in split_data.items()}
                }

        else:
            # 单文件夹结构，需要划分
            train_dir = self.source_dir / "train"
            if train_dir.exists():
                class_images = self._scan_directory(train_dir)
            else:
                class_images = self._scan_directory(self.source_dir)

            # 收集所有类别
            all_classes = set(class_images.keys())
            self.build_class_mapping(all_classes)

            # 划分数据集
            dataset = self._split_dataset(class_images)

        # 打印数据集统计信息
        self._print_dataset_statistics()

        return dataset

    def _split_train_to_val(
        self,
        train_images: Dict[str, List[Path]]
    ) -> Dict[str, List[Path]]:
        """
        从训练集中划分出验证集

        Args:
            train_images: {类别名: [图片路径列表]}

        Returns:
            字典: {class_name: [图片路径列表]} - 验证集
        """
        logger.info(f"从训练集划分验证集: 验证集比例{self.val_ratio*100:.0f}%")

        val_images: Dict[str, List[Path]] = defaultdict(list)

        # 计算train和val的比例（因为test已经存在）
        # train_ratio = train_ratio / (train_ratio + val_ratio)
        # val_ratio = val_ratio / (train_ratio + val_ratio)
        total_train_val_ratio = self.train_ratio + self.val_ratio
        adjusted_train_ratio = self.train_ratio / total_train_val_ratio
        adjusted_val_ratio = self.val_ratio / total_train_val_ratio

        logger.info(
            f"调整后的比例: 训练集{adjusted_train_ratio*100:.0f}%, "
            f"验证集{adjusted_val_ratio*100:.0f}%"
        )

        # 对每个类别的图片分别进行划分
        for class_name, images in train_images.items():
            # 打乱该类别的图片
            shuffled_images = images.copy()
            random.shuffle(shuffled_images)

            # 计算各数据集大小
            total = len(shuffled_images)
            train_size = int(total * adjusted_train_ratio)

            # 划分数据
            new_train_images = shuffled_images[:train_size]
            val_images[class_name] = shuffled_images[train_size:]

            # 更新训练集
            train_images[class_name] = new_train_images

            logger.info(
                f"类别 {class_name}: 训练集{len(new_train_images)}, 验证集{len(val_images[class_name])}"
            )

        return val_images

    def _split_dataset(
        self,
        class_images: Dict[str, List[Path]]
    ) -> Dict[str, Dict[str, List[Path]]]:
        """
        按比例划分数据集

        Args:
            class_images: {类别名: [图片路径列表]}

        Returns:
            字典: {split: {class_name: [图片路径列表]}}
        """
        logger.info(
            f"划分数据集: 训练集{self.train_ratio*100:.0f}%, "
            f"验证集{self.val_ratio*100:.0f}%, "
            f"测试集{self.test_ratio*100:.0f}%"
        )

        dataset: Dict[str, Dict[str, List[Path]]] = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }

        # 对每个类别的图片分别进行划分，保持类别平衡
        for class_name, images in class_images.items():
            # 打乱该类别的图片
            shuffled_images = images.copy()
            random.shuffle(shuffled_images)

            # 计算各数据集大小
            total = len(shuffled_images)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            # 划分数据
            train_images = shuffled_images[:train_size]
            val_images = shuffled_images[train_size:train_size + val_size]
            test_images = shuffled_images[train_size + val_size:]

            dataset["train"][class_name] = train_images
            dataset["val"][class_name] = val_images
            dataset["test"][class_name] = test_images

            logger.info(
                f"类别 {class_name}: 训练集{len(train_images)}, "
                f"验证集{len(val_images)}, 测试集{len(test_images)}"
            )

        # 统计数据集信息
        for split, split_data in dataset.items():
            total = sum(len(images) for images in split_data.values())
            self.dataset_info[split] = {
                "total": total,
                "classes": {cls: len(imgs) for cls, imgs in split_data.items()}
            }

        return dataset

    def _print_dataset_statistics(self) -> None:
        """打印数据集统计信息"""
        logger.info("=" * 60)
        logger.info("数据集统计信息")
        logger.info("=" * 60)

        for split, info in self.dataset_info.items():
            logger.info(f"\n{split.upper()} 数据集:")
            logger.info(f"  总图片数: {info['total']}")
            logger.info(f"  类别数: {len(info['classes'])}")

            # 打印每个类别的图片数
            for class_name, count in sorted(info['classes'].items()):
                logger.info(f"    {class_name}: {count} 张图片")

        logger.info("=" * 60)

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
        dataset: Dict[str, Dict[str, List[Path]]]
    ) -> None:
        """
        复制图片到对应的类别目录

        Args:
            dataset: 数据集字典
        """
        logger.info("复制图片...")

        for split, split_data in dataset.items():
            logger.info(f"\n处理 {split} 数据集...")

            copied_count = 0
            skipped_count = 0

            for class_name, images in split_data.items():
                # 清理类别名称
                safe_class_name = class_name.replace(' ', '_').replace('/', '_')

                for src_path in images:
                    # 目标文件路径
                    dst_path = self.processed_dir / split / safe_class_name / src_path.name

                    # 复制文件
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        logger.error(f"复制文件失败 {src_path.name}: {e}")
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
            "dataset_structure": self.dataset_structure,
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

            # 训练参数
            "epochs": 100,
            "batch_size": 32,
            "imgsz": 224,
            "patience": 10,
            "save": True,
            "device": 0,
            "workers": 8,
            "project": "runs/classify",
            "name": "aircraft",
            "exist_ok": True,
            "pretrained": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "close_mosaic": 0,
            "amp": True,
            "fraction": 1.0,
            "profile": False,
            "freeze": None,
            "multi_scale": False,
            "single_cls": False,
            "rect": False,
            "cos_lr": False,
            "overlap_mask": True,
            "mask_ratio": 4,
            "dropout": 0.0,
            "val": True,
            "plots": True,
            "save_json": False,
            "save_hybrid": False,
            "conf": None,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "dnn": False,
            "vid_stride": 1,
        }

        # 保存YAML文件
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"YOLOv8配置文件已保存到: {config_file}")

    def prepare(self) -> None:
        """执行完整的数据集准备流程"""
        logger.info("=" * 60)
        logger.info("开始准备数据集")
        logger.info("=" * 60)

        # 1. 从文件夹结构加载数据集
        dataset = self.load_dataset_from_structure()

        # 2. 创建目录结构
        self.create_directory_structure()

        # 3. 复制图片
        self.copy_images(dataset)

        # 4. 保存类别映射
        self.save_class_mapping()

        # 5. 保存数据集统计信息
        self.save_dataset_statistics()

        # 6. 创建YOLOv8配置文件
        self.create_yolo_config()

        logger.info("=" * 60)
        logger.info("数据集准备完成!")
        logger.info("=" * 60)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将FGVC_Aircraft_dataset转换为YOLOv8分类格式"
    )
    parser.add_argument(
        '--source-dir',
        type=str,
        required=True,
        help='FGVC_Aircraft_dataset根目录'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training/data/processed/aircraft/',
        help='输出目录路径 (默认: training/data/processed/aircraft/)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='训练集比例 (默认: 0.8)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='验证集比例 (默认: 0.1)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='测试集比例 (默认: 0.1)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )

    args = parser.parse_args()

    # 创建数据集准备器
    preparer = DatasetPreparer(
        source_dir=args.source_dir,
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
