#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换 FGVC_Aircraft 数据集的所有分割集
"""

import argparse
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_conversion(
    script_path: Path, fgvc_dir: str, output_dir: str, split: str
) -> Tuple[bool, str]:
    """
    运行单个分割集的转换

    Args:
        script_path: 转换脚本路径
        fgvc_dir: FGVC数据目录
        output_dir: 输出目录
        split: 数据划分

    Returns:
        (是否成功, 错误信息)
    """
    cmd = [
        "python",
        str(script_path),
        "--fgvc-dir",
        fgvc_dir,
        "--output",
        output_dir,
        "--split",
        split,
    ]

    logger.info(f"开始转换 {split} 集...")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )

        logger.info(f"{split} 集转换成功")
        return True, ""

    except subprocess.TimeoutExpired:
        return False, f"{split} 集转换超时"
    except subprocess.CalledProcessError as e:
        return False, f"{split} 集转换失败: {e.stderr}"
    except Exception as e:
        return False, f"{split} 集转换异常: {str(e)}"


def main() -> None:
    """主函数"""
    parser = argparse.ArgumentParser(
        description="批量转换 FGVC_Aircraft 数据集的所有分割集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 批量转换所有分割集
  python batch_convert_fgvc.py --fgvc-dir path/to/FGVC_Aircraft/raw --output-base path/to/fgvc_converted
        """,
    )

    parser.add_argument(
        "--fgvc-dir", type=str, required=True, help="FGVC_Aircraft数据目录"
    )
    parser.add_argument("--output-base", type=str, required=True, help="输出基础目录")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("批量转换 FGVC_Aircraft 数据集")
    logger.info("=" * 60)
    logger.info(f"FGVC目录: {args.fgvc_dir}")
    logger.info(f"输出基础目录: {args.output_base}")
    logger.info("=" * 60)
    logger.info("")

    # 转换脚本路径
    script_path = Path(__file__).parent / "convert_fgvc_aerocraft.py"

    # 定义三个分割集
    splits = ["train", "val", "test"]

    # 逐个转换
    results = {}
    for split in splits:
        output_dir = f"{args.output_base}/{split}"
        success, error = run_conversion(script_path, args.fgvc_dir, output_dir, split)
        results[split] = (success, error)
        logger.info("")

    # 打印结果
    logger.info("=" * 60)
    logger.info("转换结果")
    logger.info("=" * 60)

    all_success = True
    for split, (success, error) in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        logger.info(f"{split}: {status}")
        if not success:
            logger.error(f"  错误: {error}")
            all_success = False

    logger.info("=" * 60)

    if all_success:
        logger.info("所有分割集转换完成!")
        logger.info("")
        logger.info("下一步: 合并分割集")
        logger.info(f"  python merge_fgvc_splits.py \\")
        logger.info(f"    --train {args.output_base}/train \\")
        logger.info(f"    --val {args.output_base}/val \\")
        logger.info(f"    --test {args.output_base}/test \\")
        logger.info(f"    --output {args.output_base}/combined")
    else:
        logger.error("部分分割集转换失败，请检查错误信息")

    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
