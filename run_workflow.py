#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aerovision-V1 主workflow脚本
用于跑通整个训练流程

功能：
1. 准备数据集
2. 训练检测模型
3. 评估模型
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict) -> bool:
    """
    准备数据集

    Args:
        config: 配置字典

    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("Step 1: 准备数据集")
    logger.info("=" * 60)

    try:
        # 准备命令
        cmd = [
            sys.executable,
            "training/scripts/prepare_detection_dataset.py",
            "--labels-csv", config['paths']['labels_csv'],
            "--images-dir", config['paths']['raw_images'],
            "--output-dir", config['paths']['data_root'],
            "--train-ratio", str(config['dataset']['train_ratio']),
            "--val-ratio", str(config['dataset']['val_ratio']),
            "--test-ratio", str(config['dataset']['test_ratio']),
            "--random-seed", str(config['dataset']['random_seed'])
        ]

        logger.info(f"执行命令: {' '.join(cmd)}")

        # 执行命令
        result = subprocess.run(cmd, check=True, capture_output=False)

        logger.info("数据集准备完成")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"数据集准备失败: {e}")
        return False
    except Exception as e:
        logger.error(f"数据集准备出错: {e}")
        return False


def train_detector(config: dict) -> bool:
    """
    训练检测模型

    Args:
        config: 配置字典

    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("Step 2: 训练检测模型")
    logger.info("=" * 60)

    try:
        # 获取训练参数
        detector_config = config['detection']

        # 准备命令
        cmd = [
            sys.executable,
            "training/scripts/train_detection.py",
            "--data", detector_config['data_yaml'],
            "--model", detector_config['model_size'],
            "--epochs", str(detector_config['epochs']),
            "--batch", str(detector_config['batch_size']),
            "--imgsz", str(detector_config['imgsz']),
            "--device", detector_config['device'],
            "--project", detector_config['project'],
            "--name", detector_config['name']
        ]

        logger.info(f"执行命令: {' '.join(cmd)}")

        # 执行命令
        result = subprocess.run(cmd, check=True, capture_output=False)

        logger.info("检测模型训练完成")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"检测模型训练失败: {e}")
        return False
    except Exception as e:
        logger.error(f"检测模型训练出错: {e}")
        return False


def evaluate_model(config: dict) -> bool:
    """
    评估模型

    Args:
        config: 配置字典

    Returns:
        是否成功
    """
    logger.info("=" * 60)
    logger.info("Step 3: 评估模型")
    logger.info("=" * 60)

    try:
        from ultralytics import YOLO

        # 获取模型路径
        detector_config = config['detection']
        model_path = Path(detector_config['project']) / detector_config['name'] / "weights" / "best.pt"

        if not model_path.exists():
            logger.warning(f"模型文件不存在: {model_path}")
            logger.warning("跳过评估步骤")
            return True

        logger.info(f"加载模型: {model_path}")

        # 加载模型
        model = YOLO(str(model_path))

        # 评估模型
        data_path = detector_config['data_yaml']
        metrics = model.val(data=data_path, split='val')

        logger.info("评估结果:")
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")

        logger.info("模型评估完成")
        return True

    except Exception as e:
        logger.error(f"模型评估出错: {e}")
        return False


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Aerovision-V1 主workflow脚本"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='training/configs/config/training_params.yaml',
        help='训练参数配置文件路径'
    )
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='跳过数据集准备步骤'
    )
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='跳过训练步骤'
    )
    parser.add_argument(
        '--skip-evaluate',
        action='store_true',
        help='跳过评估步骤'
    )

    args = parser.parse_args()

    # 加载配置
    logger.info(f"加载配置文件: {args.config}")
    config = load_config(args.config)

    # 执行workflow
    success = True

    # Step 1: 准备数据集
    if not args.skip_prepare:
        if not prepare_dataset(config):
            success = False
    else:
        logger.info("跳过数据集准备步骤")

    # Step 2: 训练检测模型
    if success and not args.skip_train:
        if not train_detector(config):
            success = False
    elif args.skip_train:
        logger.info("跳过训练步骤")

    # Step 3: 评估模型
    if success and not args.skip_evaluate:
        if not evaluate_model(config):
            success = False
    elif args.skip_evaluate:
        logger.info("跳过评估步骤")

    # 输出结果
    logger.info("=" * 60)
    if success:
        logger.info("Workflow 执行成功!")
    else:
        logger.error("Workflow 执行失败!")
        sys.exit(1)
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
