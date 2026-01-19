#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGVC_Aircraft 数据集转换脚本

将 FGVC_Aircraft 数据集转换为项目标准格式

FGVC_Aircraft 格式：
- data/images/*.jpg (7位数字命名，如 1234567.jpg)
- data/images_variant_train.txt (训练集变体标注: image_id variant_name)
- data/images_family_train.txt (训练集族标注: image_id family_name)
- data/images_manufacturer_train.txt (训练集制造商标注: image_id manufacturer_name)
- data/images_box.txt (边界框: image_id xmin ymin xmax ymax)

项目格式要求：
- filename: 图片文件名
- typename: 机型名称 (使用 variant)
- airline: 航司/制造商 (使用 manufacturer)
- clarity: 清晰度 (默认值)
- block: 遮挡度 (默认值)
- registration: 注册号 (空)
"""

import argparse
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FGVC_AircraftConverter:
    """FGVC_Aircraft 数据集转换器"""

    def __init__(
        self,
        fgvc_data_dir: str,
        output_dir: str,
        default_clarity: float = 0.9,
        default_block: float = 0.0,
        split: str = "train",
    ) -> None:
        """
        初始化转换器

        Args:
            fgvc_data_dir: FGVC_Aircraft数据目录 (包含data子目录)
            output_dir: 输出目录
            default_clarity: 默认清晰度 (FGVC无此信息)
            default_block: 默认遮挡度 (FGVC无此信息)
            split: 使用哪个划分 (train/val/test)
        """
        self.fgvc_data_dir = Path(fgvc_data_dir)
        self.data_dir = self.fgvc_data_dir / "data"
        self.images_dir = self.data_dir / "images"
        self.output_dir = Path(output_dir)

        self.default_clarity = default_clarity
        self.default_block = default_block
        self.split = split

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_images_dir = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            "total_images": 0,
            "converted_images": 0,
            "missing_images": 0,
            "num_variants": 0,
            "num_families": 0,
            "num_manufacturers": 0,
        }

    def _parse_annotation_file(self, annotation_file: Path) -> pd.DataFrame:
        """
        解析FGVC标注文件

        格式: image_id label (每行一个)

        Args:
            annotation_file: 标注文件路径

        Returns:
            DataFrame with columns: ['image_id', 'label']
        """
        logger.info(f"解析标注文件: {annotation_file}")

        if not annotation_file.exists():
            logger.warning(f"标注文件不存在: {annotation_file}")
            return pd.DataFrame(columns=["image_id", "label"])

        data = []
        with open(annotation_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    image_id, label = parts
                    data.append({"image_id": image_id, "label": label})

        df = pd.DataFrame(data)
        logger.info(f"解析了 {len(df)} 条标注记录")

        return df

    def _parse_box_file(self, box_file: Path) -> Dict[str, List[int]]:
        """
        解析边界框文件

        格式: image_id xmin ymin xmax ymax

        Args:
            box_file: 边界框文件路径

        Returns:
            字典 {image_id: [xmin, ymin, xmax, ymax]}
        """
        logger.info(f"解析边界框文件: {box_file}")

        boxes = {}
        if not box_file.exists():
            logger.warning(f"边界框文件不存在: {box_file}")
            return boxes

        with open(box_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) == 5:
                    image_id = parts[0]
                    xmin, ymin, xmax, ymax = map(int, parts[1:])
                    boxes[image_id] = [xmin, ymin, xmax, ymax]

        logger.info(f"解析了 {len(boxes)} 个边界框")

        return boxes

    def _merge_annotations(
        self,
        variant_df: pd.DataFrame,
        family_df: pd.DataFrame,
        manufacturer_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        合并三个标注文件 (variant, family, manufacturer)

        Args:
            variant_df: 变体标注
            family_df: 族标注
            manufacturer_df: 制造商标注

        Returns:
            合并后的DataFrame
        """
        logger.info("合并标注文件...")

        # 以variant为主键进行merge
        merged = variant_df.merge(
            family_df, on="image_id", how="left", suffixes=("", "_family")
        )

        merged = merged.merge(
            manufacturer_df, on="image_id", how="left", suffixes=("", "_manufacturer")
        )

        # 重命名列
        merged = merged.rename(
            columns={
                "label": "typename",
                "label_family": "family",
                "label_manufacturer": "airline",
            }
        )

        logger.info(f"合并后共有 {len(merged)} 条记录")

        return merged

    def _filter_missing_images(
        self, df: pd.DataFrame, images_dir: Path
    ) -> pd.DataFrame:
        """
        过滤掉图片不存在的记录

        Args:
            df: 标注数据
            images_dir: 图片目录

        Returns:
            过滤后的DataFrame
        """
        logger.info("检查图片文件...")

        valid_rows = []
        missing_count = 0

        for _, row in df.iterrows():
            image_id = row["image_id"]
            image_file = images_dir / f"{image_id}.jpg"

            if image_file.exists():
                valid_rows.append(row)
            else:
                missing_count += 1

        self.stats["missing_images"] = missing_count

        filtered_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        logger.info(
            f"过滤后保留 {len(filtered_df)} 条记录，缺失 {missing_count} 张图片"
        )

        return filtered_df

    def _convert_to_project_format(
        self, df: pd.DataFrame, boxes: Dict[str, List[int]]
    ) -> pd.DataFrame:
        """
        转换为项目标准格式

        Args:
            df: 合并后的标注数据
            boxes: 边界框字典

        Returns:
            项目格式DataFrame
        """
        logger.info("转换为项目格式...")

        project_data = []

        for _, row in df.iterrows():
            image_id = row["image_id"]

            # 获取边界框信息
            airplanearea = ""
            if image_id in boxes:
                xmin, ymin, xmax, ymax = boxes[image_id]
                # 计算相对坐标 (0-1范围)
                # 注意: FGVC边界框是1-based像素坐标
                # 我们需要将边界框信息存储为字符串，或者计算相对面积
                airplanearea = f"{xmin} {ymin} {xmax} {ymax}"

            project_row = {
                "filename": f"{image_id}.jpg",
                "typename": row["typename"],
                "airline": row["airline"],
                "clarity": self.default_clarity,
                "block": self.default_block,
                "airplanearea": airplanearea,
            }

            project_data.append(project_row)

        result_df = pd.DataFrame(project_data)

        # 统计类别数量
        self.stats["num_variants"] = result_df["typename"].nunique()
        self.stats["num_families"] = (
            result_df["family"].nunique() if "family" in result_df.columns else 0
        )
        self.stats["num_manufacturers"] = (
            result_df["airline"].nunique() if "airline" in result_df.columns else 0
        )

        logger.info(f"转换完成，{len(result_df)} 条记录")
        logger.info(f"  机型变体: {self.stats['num_variants']}")
        logger.info(f"  机型族: {self.stats['num_families']}")
        logger.info(f"  制造商: {self.stats['num_manufacturers']}")

        return result_df

    def _copy_images(self, df: pd.DataFrame) -> None:
        """
        复制图片到输出目录

        Args:
            df: 包含filename列的DataFrame
        """
        logger.info("复制图片...")

        copied = 0

        for _, row in df.iterrows():
            filename = row["filename"]
            src_path = self.images_dir / filename
            dst_path = self.output_images_dir / filename

            if src_path.exists() and not dst_path.exists():
                shutil.copy2(src_path, dst_path)
                copied += 1

        self.stats["converted_images"] = copied
        logger.info(f"复制了 {copied} 张图片")

    def _save_csv(self, df: pd.DataFrame) -> None:
        """
        保存CSV文件

        Args:
            df: 项目格式DataFrame
        """
        output_csv = self.output_dir / "labels.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        logger.info(f"已保存标注文件: {output_csv}")

    def convert(self) -> None:
        """执行完整的转换流程"""
        logger.info("=" * 60)
        logger.info("开始 FGVC_Aircraft 数据集转换")
        logger.info("=" * 60)

        # 确定使用哪个split
        suffix = f"_{self.split}"

        # 解析三个标注文件
        variant_file = self.data_dir / f"images_variant{suffix}.txt"
        family_file = self.data_dir / f"images_family{suffix}.txt"
        manufacturer_file = self.data_dir / f"images_manufacturer{suffix}.txt"
        box_file = self.data_dir / "images_box.txt"

        variant_df = self._parse_annotation_file(variant_file)
        family_df = self._parse_annotation_file(family_file)
        manufacturer_df = self._parse_annotation_file(manufacturer_file)
        boxes = self._parse_box_file(box_file)

        # 合并标注
        merged_df = self._merge_annotations(variant_df, family_df, manufacturer_df)

        # 过滤缺失图片
        self.stats["total_images"] = len(merged_df)
        filtered_df = self._filter_missing_images(merged_df, self.images_dir)

        # 转换为项目格式
        project_df = self._convert_to_project_format(filtered_df, boxes)

        # 复制图片
        self._copy_images(project_df)

        # 保存CSV
        self._save_csv(project_df)

        # 打印统计
        self._print_summary()

    def _print_summary(self) -> None:
        """打印转换摘要"""
        logger.info("=" * 60)
        logger.info("转换摘要")
        logger.info("=" * 60)
        logger.info(f"原始图片数: {self.stats['total_images']}")
        logger.info(f"转换图片数: {self.stats['converted_images']}")
        logger.info(f"缺失图片数: {self.stats['missing_images']}")
        logger.info(f"机型变体数: {self.stats['num_variants']}")
        logger.info(f"机型族数: {self.stats['num_families']}")
        logger.info(f"制造商数: {self.stats['num_manufacturers']}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FGVC_Aircraft 数据集转换脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换训练集
  python convert_fgvc_aerocraft.py --fgvc-dir path/to/FGVC_Aircraft --output path/to/output --split train

  # 转换验证集
  python convert_fgvc_aerocraft.py --fgvc-dir path/to/FGVC_Aircraft --output path/to/output --split val

  # 转换测试集
  python convert_fgvc_aerocraft.py --fgvc-dir path/to/FGVC_Aircraft --output path/to/output --split test

注意:
  FGVC_Aircraft数据集格式与项目不同，本脚本会将FGVC格式转换为项目标准格式
        """,
    )

    parser.add_argument(
        "--fgvc-dir", type=str, required=True, help="FGVC_Aircraft数据目录路径"
    )
    parser.add_argument("--output", type=str, required=True, help="输出目录路径")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "trainval"],
        default="train",
        help="使用哪个划分 (默认: train)",
    )
    parser.add_argument(
        "--clarity", type=float, default=0.9, help="默认清晰度值 (默认: 0.9)"
    )
    parser.add_argument(
        "--block", type=float, default=0.0, help="默认遮挡度值 (默认: 0.0)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("转换配置:")
    logger.info(f"  FGVC目录: {args.fgvc_dir}")
    logger.info(f"  输出目录: {args.output}")
    logger.info(f"  数据划分: {args.split}")
    logger.info(f"  默认清晰度: {args.clarity}")
    logger.info(f"  默认遮挡度: {args.block}")
    logger.info("=" * 60)

    # 创建转换器并执行转换
    converter = FGVC_AircraftConverter(
        fgvc_data_dir=args.fgvc_dir,
        output_dir=args.output,
        default_clarity=args.clarity,
        default_block=args.block,
        split=args.split,
    )

    converter.convert()

    logger.info("")
    logger.info("转换完成!")
    logger.info("下一步:")
    logger.info(
        f"  python prepare_dataset.py --labels {converter.output_dir / 'labels.csv'} --images {converter.output_images_dir}"
    )
    logger.info(f"  python split_dataset.py --prepare-dir {converter.output_dir}")


if __name__ == "__main__":
    main()
