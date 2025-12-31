#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
飞机裁剪脚本 - 使用 YOLOv8 从原始图片中裁剪飞机

功能：
1. 使用 YOLOv8 检测模型检测原始图片中的飞机
2. 将检测到的飞机区域裁剪并保存
3. 支持批量处理
4. 记录裁剪日志

Usage:
    # 基本用法
    python crop_aircraft.py

    # 自定义参数
    python crop_aircraft.py --input training/data/raw --output training/data/processed/aircraft_crop/unsorted --model yolov8x.pt --conf 0.3
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json

import cv2
import numpy as np
from ultralytics import YOLO

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AircraftCropper:
    """飞机裁剪类，使用 YOLOv8 检测并裁剪图片中的飞机"""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_path: str = 'yolov8x.pt',
        conf_threshold: float = 0.25,
        device: str = '0'
    ) -> None:
        """
        初始化飞机裁剪器

        Args:
            input_dir: 输入图片目录
            output_dir: 输出裁剪图片目录
            model_path: YOLOv8 检测模型路径
            conf_threshold: 置信度阈值
            device: 设备 (0, 1, cpu, mps)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件路径
        self.log_file = self.output_dir / f'crop_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        # 裁剪统计信息
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'total_crops': 0,
            'failed_images': 0,
            'images_with_no_detection': 0,
            'crop_records': []
        }

        # 加载模型
        self._load_model()

    def _load_model(self) -> None:
        """加载 YOLOv8 检测模型"""
        logger.info(f"加载 YOLOv8 检测模型: {self.model_path}")
        self.model = YOLO(self.model_path)

        # 设置设备
        if self.device == 'cpu':
            self.model.to('cpu')
        elif self.device.isdigit():
            self.model.to(f'cuda:{self.device}')
        else:
            self.model.to(self.device)

        logger.info(f"模型加载完成，设备: {self.device}")

    def _get_image_files(self) -> List[Path]:
        """
        获取输入目录中的所有图片文件

        Returns:
            图片文件路径列表
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []

        for ext in supported_extensions:
            image_files.extend(self.input_dir.glob(f'**/*{ext}'))
            image_files.extend(self.input_dir.glob(f'**/*{ext.upper()}'))

        # 去重并排序
        image_files = sorted(list(set(image_files)))

        logger.info(f"找到 {len(image_files)} 张图片")

        return image_files

    def _crop_detection(
        self,
        image: np.ndarray,
        box: List[float],
        image_name: str,
        detection_idx: int,
        conf: float
    ) -> Optional[np.ndarray]:
        """
        裁剪单个检测区域

        Args:
            image: 原始图片
            box: 检测框 [x1, y1, x2, y2]
            image_name: 原始图片名称
            detection_idx: 检测索引
            conf: 置信度

        Returns:
            裁剪后的图片，如果裁剪失败则返回 None
        """
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # 确保坐标在图片范围内
            h, w = image.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # 检查裁剪区域是否有效
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"无效的裁剪区域: [{x1}, {y1}, {x2}, {y2}]")
                return None

            # 裁剪图片
            crop = image[y1:y2, x1:x2]

            return crop

        except Exception as e:
            logger.error(f"裁剪失败 {image_name} 检测 {detection_idx}: {e}")
            return None

    def _save_crop(
        self,
        crop: np.ndarray,
        image_name: str,
        detection_idx: int,
        conf: float
    ) -> Optional[str]:
        """
        保存裁剪后的图片

        Args:
            crop: 裁剪后的图片
            image_name: 原始图片名称
            detection_idx: 检测索引
            conf: 置信度

        Returns:
            保存的文件名，如果保存失败则返回 None
        """
        try:
            # 生成文件名: 原始名_检测序号_置信度.jpg
            stem = Path(image_name).stem
            crop_filename = f"{stem}_det{detection_idx:03d}_conf{conf:.2f}.jpg"
            crop_path = self.output_dir / crop_filename

            # 保存图片
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            return crop_filename

        except Exception as e:
            logger.error(f"保存裁剪图片失败 {crop_filename}: {e}")
            return None

    def process_image(self, image_path: Path) -> int:
        """
        处理单张图片

        Args:
            image_path: 图片路径

        Returns:
            裁剪的飞机数量
        """
        try:
            # 读取图片
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"无法读取图片: {image_path}")
                self.stats['failed_images'] += 1
                return 0

            # 进行检测
            results = self.model.predict(
                image,
                conf=self.conf_threshold,
                verbose=False
            )

            # 统计检测数量
            num_detections = len(results[0].boxes)
            self.stats['total_detections'] += num_detections

            if num_detections == 0:
                logger.debug(f"未检测到飞机: {image_path.name}")
                self.stats['images_with_no_detection'] += 1
                return 0

            # 裁剪每个检测到的飞机
            crop_count = 0
            for idx, box in enumerate(results[0].boxes):
                # 获取检测框和置信度
                box_coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # 裁剪
                crop = self._crop_detection(
                    image,
                    box_coords,
                    image_path.name,
                    idx,
                    conf
                )

                if crop is not None:
                    # 保存裁剪
                    crop_filename = self._save_crop(
                        crop,
                        image_path.name,
                        idx,
                        conf
                    )

                    if crop_filename:
                        crop_count += 1
                        self.stats['total_crops'] += 1

                        # 记录裁剪信息
                        self.stats['crop_records'].append({
                            'source_image': image_path.name,
                            'crop_filename': crop_filename,
                            'detection_index': idx,
                            'confidence': conf,
                            'box': box_coords.tolist()
                        })

            logger.info(
                f"处理完成: {image_path.name} - "
                f"检测 {num_detections} 个目标, 裁剪 {crop_count} 张图片"
            )

            return crop_count

        except Exception as e:
            logger.error(f"处理图片失败 {image_path.name}: {e}")
            self.stats['failed_images'] += 1
            return 0

    def process_batch(self) -> None:
        """批量处理所有图片"""
        logger.info("=" * 60)
        logger.info("开始批量裁剪飞机")
        logger.info("=" * 60)

        # 获取所有图片
        image_files = self._get_image_files()
        self.stats['total_images'] = len(image_files)

        if len(image_files) == 0:
            logger.warning(f"输入目录中没有找到图片: {self.input_dir}")
            return

        # 处理每张图片
        for idx, image_path in enumerate(image_files, 1):
            logger.info(f"[{idx}/{len(image_files)}] 处理: {image_path.name}")
            self.process_image(image_path)

        # 保存日志
        self._save_log()

        # 打印统计信息
        self._print_statistics()

    def _save_log(self) -> None:
        """保存裁剪日志到 JSON 文件"""
        try:
            log_data = {
                'timestamp': datetime.now().isoformat(),
                'input_dir': str(self.input_dir),
                'output_dir': str(self.output_dir),
                'model_path': self.model_path,
                'conf_threshold': self.conf_threshold,
                'statistics': {
                    'total_images': self.stats['total_images'],
                    'total_detections': self.stats['total_detections'],
                    'total_crops': self.stats['total_crops'],
                    'failed_images': self.stats['failed_images'],
                    'images_with_no_detection': self.stats['images_with_no_detection'],
                    'average_detections_per_image': (
                        self.stats['total_detections'] / self.stats['total_images']
                        if self.stats['total_images'] > 0 else 0
                    )
                },
                'crop_records': self.stats['crop_records']
            }

            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            logger.info(f"裁剪日志已保存到: {self.log_file}")

        except Exception as e:
            logger.error(f"保存日志失败: {e}")

    def _print_statistics(self) -> None:
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("裁剪统计信息")
        logger.info("=" * 60)
        logger.info(f"总图片数: {self.stats['total_images']}")
        logger.info(f"总检测数: {self.stats['total_detections']}")
        logger.info(f"总裁剪数: {self.stats['total_crops']}")
        logger.info(f"失败图片数: {self.stats['failed_images']}")
        logger.info(f"无检测图片数: {self.stats['images_with_no_detection']}")

        if self.stats['total_images'] > 0:
            avg_detections = self.stats['total_detections'] / self.stats['total_images']
            avg_crops = self.stats['total_crops'] / self.stats['total_images']
            logger.info(f"平均每张图片检测数: {avg_detections:.2f}")
            logger.info(f"平均每张图片裁剪数: {avg_crops:.2f}")

        logger.info("=" * 60)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="使用 YOLOv8 从原始图片中裁剪飞机",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        default='training/data/raw',
        help='输入图片目录'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='training/data/processed/aircraft_crop/unsorted',
        help='输出裁剪图片目录'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8x.pt',
        help='YOLOv8 检测模型路径 (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='置信度阈值'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='设备 (0, 1, cpu, mps)'
    )

    args = parser.parse_args()

    # 创建裁剪器
    cropper = AircraftCropper(
        input_dir=args.input,
        output_dir=args.output,
        model_path=args.model,
        conf_threshold=args.conf,
        device=args.device
    )

    # 批量处理
    cropper.process_batch()


if __name__ == '__main__':
    main()
