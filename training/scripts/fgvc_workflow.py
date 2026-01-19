#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGVC_Aircraft 数据集完整训练工作流脚本

本脚本自动化 FGVC_Aircraft 数据集从转换到训练的完整流程
"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_step(step_name: str, cmd: List[str], timeout: int = 300) -> Tuple[bool, str]:
    """
    运行单个步骤

    Args:
        step_name: 步骤名称
        cmd: 命令列表
        timeout: 超时时间（秒）

    Returns:
        (是否成功, 输出/错误信息)
    """
    logger.info(f"步骤: {step_name}")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )

        logger.info(f"✓ {step_name} 完成")
        return True, result.stdout

    except subprocess.TimeoutExpired:
        logger.error(f"✗ {step_name} 超时")
        return False, "Timeout"
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} 失败")
        logger.error(f"错误: {e.stderr}")
        return False, e.stderr
    except Exception as e:
        logger.error(f"✗ {step_name} 异常: {str(e)}")
        return False, str(e)


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FGVC_Aircraft 数据集完整训练工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整工作流
  python fgvc_workflow.py --fgvc-dir path/to/FGVC_Aircraft/raw --output-base data/fgvc

  # 只训练机型分类
  python fgvc_workflow.py --fgvc-dir path/to/FGVC_Aircraft/raw --output-base data/fgvc --task aircraft

  # 只训练航司分类
  python fgvc_workflow.py --fgvc-dir path/to/FGVC_Aircraft/raw --output-base data/fgvc --task airline
        """,
    )

    parser.add_argument(
        "--fgvc-dir", type=str, required=True, help="FGVC_Aircraft数据目录 (raw)"
    )
    parser.add_argument("--output-base", type=str, required=True, help="输出基础目录")
    parser.add_argument(
        "--task",
        type=str,
        choices=["aircraft", "airline", "all"],
        default="all",
        help="训练任务 (默认: all)",
    )
    parser.add_argument("--skip-prepare", action="store_true", help="跳过数据准备阶段")
    parser.add_argument("--skip-split", action="store_true", help="跳过数据划分阶段")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (默认: 100)")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="批次大小 (默认: 32)"
    )
    parser.add_argument("--imgsz", type=int, default=224, help="图像大小 (默认: 224)")
    parser.add_argument("--device", type=str, default="0", help="设备 (默认: 0)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FGVC_Aircraft 数据集训练工作流")
    logger.info("=" * 60)
    logger.info(f"FGVC目录: {args.fgvc_dir}")
    logger.info(f"输出基础: {args.output_base}")
    logger.info(f"训练任务: {args.task}")
    logger.info(f"训练轮数: {args.epochs}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"图像大小: {args.imgsz}")
    logger.info(f"设备: {args.device}")
    logger.info("=" * 60)
    logger.info("")

    # 脚本目录
    scripts_dir = Path(__file__).parent

    # 定义路径
    converted_base = Path(args.output_base) / "fgvc_converted"
    prepared_base = Path(args.output_base) / "fgvc_prepared"
    splits_base = Path(args.output_base) / "fgvc_splits"

    # 阶段 1: 转换 FGVC 数据集
    if not args.skip_prepare:
        logger.info("")
        logger.info("=" * 60)
        logger.info("阶段 1: 转换 FGVC 数据集")
        logger.info("=" * 60)

        # 转换三个分割集
        for split in ["train", "val", "test"]:
            output_dir = str(converted_base / split)
            cmd = [
                sys.executable,
                str(scripts_dir / "convert_fgvc_aerocraft.py"),
                "--fgvc-dir",
                args.fgvc_dir,
                "--output",
                output_dir,
                "--split",
                split,
            ]

            success, output = run_step(f"转换 {split} 集", cmd, timeout=300)

            if not success:
                logger.error(f"转换失败，停止工作流")
                return 1

        # 合并分割集
        cmd = [
            sys.executable,
            str(scripts_dir / "merge_fgvc_splits.py"),
            "--train",
            str(converted_base / "train"),
            "--val",
            str(converted_base / "val"),
            "--test",
            str(converted_base / "test"),
            "--output",
            str(converted_base / "combined"),
        ]

        success, output = run_step("合并分割集", cmd, timeout=300)

        if not success:
            logger.error(f"合并失败，停止工作流")
            return 1

    # 阶段 2: 准备数据集
    if not args.skip_prepare:
        logger.info("")
        logger.info("=" * 60)
        logger.info("阶段 2: 准备数据集 (prepare_dataset.py)")
        logger.info("=" * 60)

        cmd = [
            sys.executable,
            str(scripts_dir / "prepare_dataset.py"),
            "--labels",
            str(converted_base / "combined" / "labels.csv"),
            "--images",
            str(converted_base / "combined" / "images"),
            "--output",
            str(prepared_base),
        ]

        success, output = run_step("准备数据集", cmd, timeout=600)

        if not success:
            logger.error(f"准备失败，停止工作流")
            return 1

        # 读取最新的prepared目录
        latest_txt = prepared_base / "latest.txt"
        if latest_txt.exists():
            with open(latest_txt, "r", encoding="utf-8") as f:
                latest_prepared = Path(f.read().strip())
        else:
            logger.error(f"找不到latest.txt: {latest_txt}")
            return 1

    # 阶段 3: 划分数据集
    if not args.skip_split:
        logger.info("")
        logger.info("=" * 60)
        logger.info("阶段 3: 划分数据集 (split_dataset.py)")
        logger.info("=" * 60)

        prepare_dir = (
            latest_prepared if not args.skip_prepare else prepared_base / "latest"
        )

        cmd = [
            sys.executable,
            str(scripts_dir / "split_dataset.py"),
            "--prepare-dir",
            str(prepare_dir),
            "--output",
            str(splits_base),
            "--mode",
            "all",
        ]

        success, output = run_step("划分数据集", cmd, timeout=600)

        if not success:
            logger.error(f"划分失败，停止工作流")
            return 1

        # 读取最新的splits目录
        # 由于split_dataset.py在splits_base下创建时间戳目录
        # 我们需要找到最新的目录
        if not splits_base.exists():
            logger.error(f"找不到splits目录: {splits_base}")
            return 1

        timestamp_dirs = [d for d in splits_base.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            logger.error(f"splits目录为空: {splits_base}")
            return 1

        latest_splits = sorted(
            timestamp_dirs, key=lambda x: x.stat().st_mtime, reverse=True
        )[0]
        logger.info(f"使用splits目录: {latest_splits}")
    else:
        # 如果跳过划分，找到最新的splits目录
        if not splits_base.exists():
            logger.error(f"找不到splits目录: {splits_base}")
            return 1

        timestamp_dirs = [d for d in splits_base.iterdir() if d.is_dir()]
        if not timestamp_dirs:
            logger.error(f"splits目录为空: {splits_base}")
            return 1

        latest_splits = sorted(
            timestamp_dirs, key=lambda x: x.stat().st_mtime, reverse=True
        )[0]
        logger.info(f"使用splits目录: {latest_splits}")

    # 阶段 4: 训练
    logger.info("")
    logger.info("=" * 60)
    logger.info("阶段 4: 训练模型")
    logger.info("=" * 60)

    tasks = []
    if args.task in ["aircraft", "all"]:
        tasks.append("aircraft")
    if args.task in ["airline", "all"]:
        tasks.append("airline")

    for task in tasks:
        logger.info("")
        logger.info(f"训练任务: {task}")

        if task == "aircraft":
            data_path = latest_splits / "aerovision" / "aircraft"
            model_name = "yolov8n-cls.pt"
            output_name = "fgvc_aircraft_classifier"
        else:  # airline
            data_path = latest_splits / "aerovision" / "airline"
            model_name = "yolov8n-cls.pt"
            output_name = "fgvc_airline_classifier"

        cmd = [
            sys.executable,
            str(scripts_dir / "train_classify.py"),
            "--data",
            str(data_path),
            "--model",
            model_name,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--imgsz",
            str(args.imgsz),
            "--device",
            args.device,
            "--project",
            "output/classify",
            "--name",
            output_name,
        ]

        logger.info(f"数据路径: {data_path}")
        logger.info(f"模型: {model_name}")

        # 训练步骤不设置超时，让用户手动控制
        logger.info("开始训练... (按Ctrl+C停止)")

        try:
            result = subprocess.run(cmd, check=True)
            logger.info(f"✓ {task} 训练完成")
        except KeyboardInterrupt:
            logger.warning(f"✗ {task} 训练被中断")
            return 1
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ {task} 训练失败")
            logger.error(f"错误: {e.stderr}")
            return 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("工作流完成!")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
