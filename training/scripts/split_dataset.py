#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本 - 将标注好的数据集划分为 train/val/test

功能：
1. 从 aircraft_labels.csv 读取标注
2. 按比例划分数据集（70%/15%/15%）
3. 确保每个类别在各个子集中都有样本
4. 生成 train.csv, val.csv, test.csv
5. 将图片复制到对应目录

Usage:
    # 基本用法
    python split_dataset.py

    # 自定义参数
    python split_dataset.py --labels training/data/labels/aircraft_labels.csv --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import random

import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSplitter:
    """数据集划分类，将标注数据划分为 train/val/test"""

    def __init__(
        self,
        labels_file: str,
        images_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> None:
        """
        初始化数据集划分器

        Args:
            labels_file: 标注文件路径
            images_dir: 图片目录路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        self.labels_file = Path(labels_file)
        self.images_dir = Path(images_dir)
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

        # 输出目录
        self.output_dir = self.images_dir.parent
        self.labels_output_dir = self.labels_file.parent

        # 输出 CSV 文件路径
        self.train_csv = self.labels_output_dir / 'train.csv'
        self.val_csv = self.labels_output_dir / 'val.csv'
        self.test_csv = self.labels_output_dir / 'test.csv'

        # 输出图片目录
        self.train_img_dir = self.output_dir / 'train'
        self.val_img_dir = self.output_dir / 'val'
        self.test_img_dir = self.output_dir / 'test'

        # 数据集信息
        self.dataset_info = {
            'total_samples': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'classes': {}
        }

        # 设置随机种子
        random.seed(random_seed)

    def load_labels(self) -> pd.DataFrame:
        """
        加载标注文件

        Returns:
            标注数据 DataFrame
        """
        logger.info(f"加载标注文件: {self.labels_file}")

        if not self.labels_file.exists():
            raise FileNotFoundError(f"标注文件不存在: {self.labels_file}")

        # 读取 CSV 文件
        df = pd.read_csv(self.labels_file)

        # 检查必需的列
        required_columns = ['filename', 'typename', 'clarity', 'block']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"标注文件缺少必需的列: {missing_columns}")

        logger.info(f"加载了 {len(df)} 条标注记录")

        return df

    def split_by_class(
        self,
        df: pd.DataFrame,
        class_column: str = 'typename'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按类别划分数据集，确保每个类别在各个子集中都有样本

        Args:
            df: 标注数据
            class_column: 类别列名

        Returns:
            (train_df, val_df, test_df)
        """
        logger.info(
            f"按类别划分数据集: 训练集{self.train_ratio*100:.0f}%, "
            f"验证集{self.val_ratio*100:.0f}%, "
            f"测试集{self.test_ratio*100:.0f}%"
        )

        # 按类别分组
        class_groups = df.groupby(class_column)

        train_samples = []
        val_samples = []
        test_samples = []

        for class_name, class_df in class_groups:
            # 打乱该类别的样本
            shuffled_df = class_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

            total = len(shuffled_df)
            train_size = int(total * self.train_ratio)
            val_size = int(total * self.val_ratio)

            # 确保每个类别至少有 1 个样本在 train 中
            if train_size == 0:
                train_size = 1
                val_size = max(0, min(1, total - 1))
                test_size = total - train_size - val_size
            else:
                test_size = total - train_size - val_size

            # 划分数据
            train_df_class = shuffled_df.iloc[:train_size]
            val_df_class = shuffled_df.iloc[train_size:train_size + val_size]
            test_df_class = shuffled_df.iloc[train_size + val_size:]

            train_samples.append(train_df_class)
            val_samples.append(val_df_class)
            test_samples.append(test_df_class)

            logger.info(
                f"类别 {class_name}: 训练集{len(train_df_class)}, "
                f"验证集{len(val_df_class)}, 测试集{len(test_df_class)}"
            )

        # 合并所有类别的数据
        train_df = pd.concat(train_samples, ignore_index=True)
        val_df = pd.concat(val_samples, ignore_index=True)
        test_df = pd.concat(test_samples, ignore_index=True)

        # 打乱合并后的数据
        train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)

        return train_df, val_df, test_df

    def save_csv_files(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        保存划分后的 CSV 文件

        Args:
            train_df: 训练集数据
            val_df: 验证集数据
            test_df: 测试集数据
        """
        logger.info("保存 CSV 文件...")

        # 保存训练集
        train_df.to_csv(self.train_csv, index=False)
        logger.info(f"训练集已保存: {self.train_csv} ({len(train_df)} 条)")

        # 保存验证集
        val_df.to_csv(self.val_csv, index=False)
        logger.info(f"验证集已保存: {self.val_csv} ({len(val_df)} 条)")

        # 保存测试集
        test_df.to_csv(self.test_csv, index=False)
        logger.info(f"测试集已保存: {self.test_csv} ({len(test_df)} 条)")

    def copy_images(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        将图片复制到对应的目录

        Args:
            train_df: 训练集数据
            val_df: 验证集数据
            test_df: 测试集数据
        """
        logger.info("复制图片到对应目录...")

        # 创建输出目录
        for split_dir in [self.train_img_dir, self.val_img_dir, self.test_img_dir]:
            split_dir.mkdir(parents=True, exist_ok=True)

        # 复制训练集图片
        self._copy_split_images(train_df, self.train_img_dir, 'train')

        # 复制验证集图片
        self._copy_split_images(val_df, self.val_img_dir, 'val')

        # 复制测试集图片
        self._copy_split_images(test_df, self.test_img_dir, 'test')

    def _copy_split_images(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        split_name: str
    ) -> None:
        """
        复制单个数据集的图片

        Args:
            df: 数据集数据
            output_dir: 输出目录
            split_name: 数据集名称
        """
        copied_count = 0
        skipped_count = 0
        missing_count = 0

        for _, row in df.iterrows():
            filename = row['filename']
            src_path = self.images_dir / filename
            dst_path = output_dir / filename

            if not src_path.exists():
                logger.warning(f"图片不存在: {filename}")
                missing_count += 1
                continue

            # 如果目标文件已存在，跳过
            if dst_path.exists():
                skipped_count += 1
                continue

            try:
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            except Exception as e:
                logger.error(f"复制图片失败 {filename}: {e}")

        logger.info(
            f"{split_name} 数据集图片复制完成: "
            f"成功{copied_count}, 跳过{skipped_count}, 缺失{missing_count}"
        )

    def generate_class_mapping(
        self,
        df: pd.DataFrame,
        class_column: str = 'typename',
        output_file: str = 'type_classes.json'
    ) -> None:
        """
        生成类别映射文件

        Args:
            df: 标注数据
            class_column: 类别列名
            output_file: 输出文件名
        """
        logger.info(f"生成类别映射: {class_column}")

        # 获取所有类别并排序
        classes = sorted(df[class_column].unique())

        # 生成映射
        class_mapping = {
            "num_classes": len(classes),
            "classes": [
                {"id": idx, "name": name}
                for idx, name in enumerate(classes)
            ]
        }

        # 保存 JSON 文件
        output_path = self.labels_output_dir / output_file
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"类别映射已保存: {output_path} ({len(classes)} 个类别)")

    def print_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        打印数据集统计信息

        Args:
            train_df: 训练集数据
            val_df: 验证集数据
            test_df: 测试集数据
        """
        logger.info("=" * 60)
        logger.info("数据集划分统计信息")
        logger.info("=" * 60)

        total = len(train_df) + len(val_df) + len(test_df)

        logger.info(f"总样本数: {total}")
        logger.info(f"训练集: {len(train_df)} ({len(train_df)/total*100:.1f}%)")
        logger.info(f"验证集: {len(val_df)} ({len(val_df)/total*100:.1f}%)")
        logger.info(f"测试集: {len(test_df)} ({len(test_df)/total*100:.1f}%)")

        # 统计每个类别的样本数
        logger.info("\n各类别样本分布:")
        all_classes = sorted(set(train_df['typename'].unique()) |
                            set(val_df['typename'].unique()) |
                            set(test_df['typename'].unique()))

        for class_name in all_classes:
            train_count = len(train_df[train_df['typename'] == class_name])
            val_count = len(val_df[val_df['typename'] == class_name])
            test_count = len(test_df[test_df['typename'] == class_name])
            total_count = train_count + val_count + test_count

            logger.info(
                f"  {class_name}: "
                f"训练{train_count}, 验证{val_count}, 测试{test_count} "
                f"(总计{total_count})"
            )

        logger.info("=" * 60)

    def split(self) -> None:
        """执行数据集划分流程"""
        logger.info("=" * 60)
        logger.info("开始数据集划分")
        logger.info("=" * 60)

        # 1. 加载标注文件
        df = self.load_labels()

        # 2. 按类别划分数据集
        train_df, val_df, test_df = self.split_by_class(df)

        # 3. 保存 CSV 文件
        self.save_csv_files(train_df, val_df, test_df)

        # 4. 复制图片
        self.copy_images(train_df, val_df, test_df)

        # 5. 生成类别映射
        self.generate_class_mapping(df, 'typename', 'type_classes.json')

        # 如果有航司信息，也生成航司类别映射
        if 'airlinename' in df.columns and not df['airlinename'].isna().all():
            self.generate_class_mapping(df.dropna(subset=['airlinename']), 'airlinename', 'airline_classes.json')

        # 6. 打印统计信息
        self.print_statistics(train_df, val_df, test_df)

        logger.info("=" * 60)
        logger.info("数据集划分完成!")
        logger.info("=" * 60)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将标注好的数据集划分为 train/val/test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--labels',
        type=str,
        default='training/data/labels/aircraft_labels.csv',
        help='标注文件路径'
    )
    parser.add_argument(
        '--images',
        type=str,
        default='training/data/processed/aircraft_crop/unsorted',
        help='图片目录路径'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='训练集比例'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='验证集比例'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='测试集比例'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )

    args = parser.parse_args()

    # 创建数据集划分器
    splitter = DatasetSplitter(
        labels_file=args.labels,
        images_dir=args.images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )

    # 执行数据集划分
    splitter.split()


if __name__ == '__main__':
    main()
