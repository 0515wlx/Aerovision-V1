#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一数据集准备脚本
整合 Aerovision 分类数据集和检测数据集的准备流程

配置说明：
本脚本使用新的模块化配置系统，自动加载以下配置模块：
- paths.yaml: 路径配置 (labels.*, data.*)
- base.yaml: 基础配置 (seed.*)

配置项：
- labels.main: 标注文件路径 (默认: ../data/processed/labeled/labels.csv)
- data.processed.labeled.images: 图片目录 (默认: ../data/processed/labeled/images)
- data.processed.labeled.registration: 注册号标注目录 (YOLO格式)
- data.prepared_root: 统一输出根目录 (默认: ../data/prepared)
- seed.random: 随机种子 (默认: 42)

使用方法：
  # 准备所有数据集
  python prepare_dataset.py

  # 只准备分类数据集
  python prepare_dataset.py --mode aerovision

  # 只准备检测数据集
  python prepare_dataset.py --mode detection

  # 指定自定义配置
  python prepare_dataset.py --config my_config.yaml
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml

# 添加configs模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedDatasetPreparer:
    """统一数据集准备类"""

    def __init__(
        self,
        labels_csv: str,
        images_dir: str,
        registration_dir: Optional[str],
        output_root: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        detection_train_ratio: float = 0.8,
        random_seed: int = 42,
        mode: str = "all"
    ) -> None:
        """
        初始化数据集准备器

        Args:
            labels_csv: labels.csv文件路径
            images_dir: 原始图片目录
            registration_dir: 注册号标注目录（YOLO格式）
            output_root: 统一输出根目录
            train_ratio: 分类数据集训练集比例
            val_ratio: 分类数据集验证集比例
            test_ratio: 分类数据集测试集比例
            detection_train_ratio: 检测数据集训练集比例
            random_seed: 随机种子
            mode: 准备模式 ("all", "aerovision", "detection")
        """
        self.labels_csv = Path(labels_csv)
        self.images_dir = Path(images_dir)
        self.registration_dir = Path(registration_dir) if registration_dir else None
        self.output_root = Path(output_root)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.detection_train_ratio = detection_train_ratio
        self.random_seed = random_seed
        self.mode = mode

        # 验证比例总和
        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
            raise ValueError(
                f"训练集、验证集、测试集比例之和必须为1.0，"
                f"当前为{train_ratio + val_ratio + test_ratio}"
            )

        # 创建带时间戳的输出目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dataset_root = self.output_root / self.timestamp

        # Aerovision 分类数据集路径
        self.aerovision_root = self.dataset_root / "aerovision"
        self.aircraft_dir = self.aerovision_root / "aircraft"
        self.labels_dir = self.aerovision_root / "labels"

        # Detection 检测数据集路径
        self.detection_root = self.dataset_root / "detection"

        logger.info(f"数据集输出根目录: {self.dataset_root}")

        # 类别映射（类别名 -> 类别ID）
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}

        # 数据集信息
        self.dataset_info: Dict[str, Dict[str, int]] = {}

        # 设置随机种子
        random.seed(random_seed)

    def load_labels(self) -> Tuple[Dict[str, Dict], set]:
        """
        从labels.csv加载标签信息

        Returns:
            (labels字典, 类别名集合)
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
        """构建类别映射"""
        logger.info("构建类别映射...")

        # 按字母顺序排序类别名，并分配ID
        sorted_classes = sorted(class_names)
        self.class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}
        self.id_to_class = {idx: name for idx, name in enumerate(sorted_classes)}

        logger.info(f"共发现 {len(self.class_to_id)} 个机型类别")

    def split_dataset(
        self,
        labels: Dict[str, Dict]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
        """按比例划分数据集"""
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
                train_size = total - val_size - 1

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

    def prepare_aerovision(self, labels: Dict[str, Dict], class_names: set) -> None:
        """准备 Aerovision 分类数据集"""
        logger.info("=" * 60)
        logger.info("准备 Aerovision 分类数据集")
        logger.info("=" * 60)

        # 构建类别映射
        self.build_class_mapping(class_names)

        # 划分数据集
        train_files, val_files, test_files = self.split_dataset(labels)

        # 创建目录结构
        self._create_aerovision_structure()

        # 复制图片
        self._copy_aerovision_images(train_files, val_files, test_files)

        # 保存类别映射
        self._save_class_mapping()

        # 保存数据集统计
        self._save_dataset_statistics()

        # 创建YOLO配置
        self._create_aerovision_config()

        logger.info(f"Aerovision 数据集准备完成: {self.aerovision_root}")

    def _create_aerovision_structure(self) -> None:
        """创建Aerovision目录结构"""
        logger.info("创建 Aerovision 目录结构...")

        self.aircraft_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # 为每个数据集和类别创建目录
        for split in ['train', 'val', 'test']:
            split_dir = self.aircraft_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for class_id, class_name in self.id_to_class.items():
                safe_class_name = class_name.replace(' ', '_').replace('/', '_')
                class_dir = split_dir / safe_class_name
                class_dir.mkdir(parents=True, exist_ok=True)

    def _copy_aerovision_images(
        self,
        train_files: Dict[str, List[str]],
        val_files: Dict[str, List[str]],
        test_files: Dict[str, List[str]]
    ) -> None:
        """复制Aerovision图片"""
        logger.info("复制 Aerovision 图片...")

        split_files = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }

        for split, split_data in split_files.items():
            logger.info(f"处理 {split} 数据集...")

            copied_count = 0
            skipped_count = 0

            for typename, filenames in split_data.items():
                safe_class_name = typename.replace(' ', '_').replace('/', '_')

                for filename in filenames:
                    src_path = self.images_dir / filename
                    dst_path = self.aircraft_dir / split / safe_class_name / filename

                    if not src_path.exists():
                        logger.warning(f"源文件不存在: {src_path}")
                        skipped_count += 1
                        continue

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

    def _save_class_mapping(self) -> None:
        """保存类别映射"""
        output_file = self.labels_dir / "type_classes.json"

        class_mapping = {
            "num_classes": len(self.id_to_class),
            "classes": [
                {"id": class_id, "name": class_name}
                for class_id, class_name in self.id_to_class.items()
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(class_mapping, f, ensure_ascii=False, indent=2)

        logger.info(f"类别映射已保存: {output_file}")

    def _save_dataset_statistics(self) -> None:
        """保存数据集统计"""
        output_file = self.labels_dir / "dataset_statistics.json"

        statistics = {
            "num_classes": len(self.id_to_class),
            "splits": self.dataset_info,
            "total_images": sum(info["total"] for info in self.dataset_info.values()),
            "timestamp": self.timestamp
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)

        logger.info(f"数据集统计已保存: {output_file}")

    def _create_aerovision_config(self) -> None:
        """创建Aerovision配置文件"""
        config_file = self.aerovision_root / "dataset_config.yaml"

        config = {
            "path": str(self.aircraft_dir.absolute()),
            "train": "train",
            "val": "val",
            "test": "test",
            "names": self.id_to_class,
            "nc": len(self.id_to_class),
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        logger.info(f"配置文件已保存: {config_file}")

    def prepare_detection(self) -> None:
        """准备检测数据集"""
        logger.info("=" * 60)
        logger.info("准备检测数据集")
        logger.info("=" * 60)

        if not self.registration_dir or not self.registration_dir.exists():
            logger.warning(f"注册号标注目录不存在: {self.registration_dir}")
            logger.warning("跳过检测数据集准备")
            return

        # 创建目录结构
        for split in ['train', 'val']:
            (self.detection_root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.detection_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # 获取所有标注文件
        label_files = list(self.registration_dir.glob('*.txt'))
        logger.info(f"找到 {len(label_files)} 个标注文件")

        if len(label_files) == 0:
            logger.warning("没有找到标注文件，跳过检测数据集准备")
            return

        # 划分训练集和验证集
        try:
            from sklearn.model_selection import train_test_split
            train_files, val_files = train_test_split(
                label_files,
                train_size=self.detection_train_ratio,
                random_state=self.random_seed
            )
        except ImportError:
            logger.warning("sklearn 未安装，使用简单划分方法")
            train_size = int(len(label_files) * self.detection_train_ratio)
            random.shuffle(label_files)
            train_files = label_files[:train_size]
            val_files = label_files[train_size:]

        logger.info(f"训练集: {len(train_files)} 张")
        logger.info(f"验证集: {len(val_files)} 张")

        # 复制文件
        train_count = self._copy_detection_files(train_files, 'train')
        val_count = self._copy_detection_files(val_files, 'val')

        logger.info(f"成功处理: 训练集{train_count}, 验证集{val_count}")

        # 创建配置文件
        self._create_detection_config()

        logger.info(f"检测数据集准备完成: {self.detection_root}")

    def _copy_detection_files(self, files: List[Path], split_name: str) -> int:
        """复制检测数据集文件"""
        count = 0
        for label_file in files:
            # 对应的图片文件
            img_file = self.images_dir / label_file.stem

            # 尝试不同的图片扩展名
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                if (img_file.with_suffix(ext)).exists():
                    img_file = img_file.with_suffix(ext)
                    found = True
                    break

            if not found:
                logger.warning(f"找不到图片: {label_file.stem}")
                continue

            # 复制图片
            dst_img = self.detection_root / 'images' / split_name / img_file.name
            shutil.copy2(img_file, dst_img)

            # 复制标注
            dst_label = self.detection_root / 'labels' / split_name / label_file.name
            shutil.copy2(label_file, dst_label)

            count += 1

        return count

    def _create_detection_config(self) -> None:
        """创建检测配置文件"""
        yaml_content = f"""# YOLOv8 Detection Dataset Configuration
path: {self.detection_root.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: registration
"""

        yaml_path = self.detection_root / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        logger.info(f"检测配置已保存: {yaml_path}")

    def prepare(self) -> None:
        """执行完整的数据集准备流程"""
        logger.info("=" * 60)
        logger.info("开始准备数据集")
        logger.info(f"模式: {self.mode}")
        logger.info(f"时间戳: {self.timestamp}")
        logger.info(f"输出目录: {self.dataset_root}")
        logger.info("=" * 60)

        # 加载标签
        labels, class_names = self.load_labels()

        # 准备 Aerovision 数据集
        if self.mode in ["all", "aerovision"]:
            self.prepare_aerovision(labels, class_names)

        # 准备检测数据集
        if self.mode in ["all", "detection"]:
            self.prepare_detection()

        logger.info("=" * 60)
        logger.info("数据集准备完成!")
        logger.info(f"输出目录: {self.dataset_root}")
        logger.info("=" * 60)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="统一数据集准备脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 准备所有数据集
  python prepare_dataset.py

  # 只准备分类数据集
  python prepare_dataset.py --mode aerovision

  # 只准备检测数据集
  python prepare_dataset.py --mode detection

  # 指定自定义路径
  python prepare_dataset.py --labels-csv path/to/labels.csv

  # 使用自定义配置文件
  python prepare_dataset.py --config my_config.yaml

配置说明:
  本脚本使用模块化配置系统，自动加载 paths.yaml 和 base.yaml
  可以通过命令行参数覆盖配置文件中的值
        """
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'aerovision', 'detection'],
        default='all',
        help='准备模式: all(全部), aerovision(分类), detection(检测) (默认: all)'
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default=None,
        help='labels.csv文件路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--images-dir',
        type=str,
        default=None,
        help='图片目录路径（默认从配置文件读取）'
    )
    parser.add_argument(
        '--registration-dir',
        type=str,
        default=None,
        help='注册号标注目录（默认从配置文件读取）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出根目录（默认从配置文件读取）'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='分类数据集训练集比例 (默认: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='分类数据集验证集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.15,
        help='分类数据集测试集比例 (默认: 0.15)'
    )
    parser.add_argument(
        '--detection-train-ratio',
        type=float,
        default=0.8,
        help='检测数据集训练集比例 (默认: 0.8)'
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
        config = load_config(modules=['paths'], load_all_modules=False)

    # 从配置文件获取路径（优先使用命令行参数）
    labels_csv = args.labels_csv or config.get('labels.main')
    images_dir = args.images_dir or config.get('data.processed.labeled.images')
    registration_dir = args.registration_dir or config.get('data.processed.labeled.registration')
    output_root = args.output_dir or config.get('data.prepared_root') or '../data/prepared'
    random_seed = args.random_seed if args.random_seed is not None else (config.get('seed.random') or 42)

    # 如果配置中的路径是相对路径，转换为绝对路径
    if labels_csv and not Path(labels_csv).is_absolute():
        labels_csv = config.get_path('labels.main')
    if images_dir and not Path(images_dir).is_absolute():
        images_dir = config.get_path('data.processed.labeled.images')
    if registration_dir and not Path(registration_dir).is_absolute():
        registration_dir = config.get_path('data.processed.labeled.registration')
    if output_root and not Path(output_root).is_absolute():
        output_root = config.get_path('data.prepared_root') or (Path(__file__).parent.parent / 'data' / 'prepared').resolve()

    logger.info("=" * 60)
    logger.info("配置信息:")
    logger.info(f"  模式: {args.mode}")
    logger.info(f"  标注文件: {labels_csv}")
    logger.info(f"  图片目录: {images_dir}")
    logger.info(f"  注册号标注: {registration_dir}")
    logger.info(f"  输出根目录: {output_root}")
    logger.info(f"  分类 - 训练集比例: {args.train_ratio}")
    logger.info(f"  分类 - 验证集比例: {args.val_ratio}")
    logger.info(f"  分类 - 测试集比例: {args.test_ratio}")
    logger.info(f"  检测 - 训练集比例: {args.detection_train_ratio}")
    logger.info(f"  随机种子: {random_seed}")
    logger.info("=" * 60)

    # 创建数据集准备器
    preparer = UnifiedDatasetPreparer(
        labels_csv=str(labels_csv),
        images_dir=str(images_dir),
        registration_dir=str(registration_dir) if registration_dir else None,
        output_root=str(output_root),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        detection_train_ratio=args.detection_train_ratio,
        random_seed=random_seed,
        mode=args.mode
    )

    # 执行数据集准备
    preparer.prepare()


if __name__ == '__main__':
    main()
