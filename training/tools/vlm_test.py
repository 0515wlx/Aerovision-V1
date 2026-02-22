#!/usr/bin/env python3
"""
VLM测试脚本 - 用于计算VLM模型在各个任务上的准确率

支持的任务：
- 机型识别 (aircraft_type)
- 航司识别 (airline)
- OCR注册号识别 (registration)
- 质量评估 (quality)

Usage:
    # 测试机型识别
    python vlm_test.py --task aircraft_type

    # 测试航司识别
    python vlm_test.py --task airline

    # 测试OCR注册号识别
    python vlm_test.py --task registration

    # 测试质量评估
    python vlm_test.py --task quality

    # 指定数据集路径
    python vlm_test.py --task aircraft_type --data-path data/splits/latest/

    # 指定输出目录
    python vlm_test.py --task aircraft_type --output-dir results/

    # 恢复之前的测试
    python vlm_test.py --task aircraft_type --resume results/checkpoint.json

    # 跳过已处理的样本
    python vlm_test.py --task aircraft_type --skip-processed

    # 启用并发处理
    python vlm_test.py --task aircraft_type --concurrent --max-workers 10
"""

import argparse
import concurrent.futures
import csv
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config

from src.vlm.inference import VLMInference, InferenceResult


# ============================================================================
# 日志配置
# ============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_dir: 日志目录

    Returns:
        配置好的logger实例
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("VLMTest")
    logger.setLevel(logging.DEBUG)

    # 清除现有handlers
    logger.handlers.clear()

    # 文件handler - 详细日志
    log_file = log_dir / f"vlm_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台handler - info级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# ============================================================================
# 数据加载器
# ============================================================================

class DataLoader:
    """数据加载器 - 支持不同任务的数据加载"""

    def __init__(self, data_root: Path, logger: logging.Logger):
        """
        初始化数据加载器

        Args:
            data_root: 数据根目录
            logger: logger实例
        """
        self.data_root = Path(data_root)
        self.logger = logger

    def load_aircraft_type_data(self) -> List[Tuple[str, str]]:
        """
        加载机型识别数据

        Returns:
            图片路径列表和对应的真实标签
        """
        self.logger.info("加载机型识别数据...")

        test_dir = self.data_root / 'aerovision' / 'aircraft' / 'test'
        if not test_dir.exists():
            self.logger.warning(f"测试目录不存在: {test_dir}")
            return []

        data = []
        for type_dir in test_dir.iterdir():
            if type_dir.is_dir():
                aircraft_type = type_dir.name
                for img_file in type_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        data.append((str(img_file), aircraft_type))

        self.logger.info(f"加载了 {len(data)} 个机型识别样本")
        return data

    def load_airline_data(self) -> List[Tuple[str, str]]:
        """
        加载航司识别数据

        Returns:
            图片路径列表和对应的真实标签
        """
        self.logger.info("加载航司识别数据...")

        test_dir = self.data_root / 'aerovision' / 'airline' / 'test'
        if not test_dir.exists():
            self.logger.warning(f"测试目录不存在: {test_dir}")
            return []

        data = []
        for airline_dir in test_dir.iterdir():
            if airline_dir.is_dir():
                airline = airline_dir.name
                for img_file in airline_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        data.append((str(img_file), airline))

        self.logger.info(f"加载了 {len(data)} 个航司识别样本")
        return data

    def load_registration_data(self, labels_file: Optional[Path] = None) -> List[Tuple[str, str]]:
        """
        加载OCR注册号识别数据

        Args:
            labels_file: 标签文件路径，默认为 data/labels.csv

        Returns:
            图片路径列表和对应的真实注册号
        """
        self.logger.info("加载OCR注册号识别数据...")

        if labels_file is None:
            # Try multiple possible locations for labels.csv
            possible_paths = [
                Path('data/labels.csv'),  # Project root
                self.data_root.parent.parent / 'labels.csv' if self.data_root.parent else Path('data/labels.csv'),
                self.data_root / 'labels.csv',
            ]
            for path in possible_paths:
                if path.exists():
                    labels_file = path
                    break
            else:
                labels_file = Path('data/labels.csv')

        if not labels_file.exists():
            self.logger.warning(f"标签文件不存在: {labels_file}")
            return []

        data = []
        seen_paths = set()  # 用于去重：记录已处理的图片路径
        try:
            with open(labels_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle BOM in column names
                    filename = row.get('\ufefffilename') or row.get('filename') or row.get('image_path')
                    registration = row.get('registration')
                    if filename and registration:
                        # 尝试从data/labeled目录解析路径
                        labeled_dir = Path('data/labeled')
                        full_path = labeled_dir / filename
                        if full_path.exists():
                            path_str = str(full_path)
                            if path_str not in seen_paths:
                                seen_paths.add(path_str)
                                data.append((path_str, registration))
                        else:
                            # 尝试相对于data_root解析路径
                            full_path = self.data_root / filename
                            if full_path.exists():
                                path_str = str(full_path)
                                if path_str not in seen_paths:
                                    seen_paths.add(path_str)
                                    data.append((path_str, registration))
                            else:
                                # 尝试直接使用路径
                                if Path(filename).exists():
                                    path_str = filename
                                    if path_str not in seen_paths:
                                        seen_paths.add(path_str)
                                        data.append((path_str, registration))

            self.logger.info(f"加载了 {len(data)} 个注册号识别样本")
        except Exception as e:
            self.logger.error(f"加载标签文件失败: {e}")

        return data

    def load_quality_data(self, labels_file: Optional[Path] = None) -> List[Tuple[str, Dict[str, float]]]:
        """
        加载质量评估数据

        Args:
            labels_file: 标签文件路径，默认为 data/labels.csv

        Returns:
            图片路径列表和对应的真实质量值（clarity, block）
        """
        self.logger.info("加载质量评估数据...")

        if labels_file is None:
            # Try multiple possible locations for labels.csv
            possible_paths = [
                Path('data/labels.csv'),  # Project root
                self.data_root.parent.parent / 'labels.csv' if self.data_root.parent else Path('data/labels.csv'),
                self.data_root / 'labels.csv',
            ]
            for path in possible_paths:
                if path.exists():
                    labels_file = path
                    break
            else:
                labels_file = Path('data/labels.csv')

        if not labels_file.exists():
            self.logger.warning(f"标签文件不存在: {labels_file}")
            return []

        data = []
        seen_paths = set()  # 用于去重：记录已处理的图片路径
        try:
            with open(labels_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Handle BOM in column names
                    filename = row.get('\ufefffilename') or row.get('filename') or row.get('image_path')
                    clarity = row.get('clarity')
                    block = row.get('block')
                    if filename and (clarity is not None or block is not None):
                        quality = {}
                        if clarity:
                            try:
                                quality['clarity'] = float(clarity)
                            except ValueError:
                                pass
                        if block:
                            try:
                                quality['block'] = float(block)
                            except ValueError:
                                pass

                        if quality:
                            # 尝试从data/labeled目录解析路径
                            labeled_dir = Path('data/labeled')
                            full_path = labeled_dir / filename
                            if full_path.exists():
                                path_str = str(full_path)
                                if path_str not in seen_paths:
                                    seen_paths.add(path_str)
                                    data.append((path_str, quality))
                            else:
                                # 尝试相对于data_root解析路径
                                full_path = self.data_root / filename
                                if full_path.exists():
                                    path_str = str(full_path)
                                    if path_str not in seen_paths:
                                        seen_paths.add(path_str)
                                        data.append((path_str, quality))
                                else:
                                    # 尝试直接使用路径
                                    if Path(filename).exists():
                                        path_str = filename
                                        if path_str not in seen_paths:
                                            seen_paths.add(path_str)
                                            data.append((path_str, quality))

            self.logger.info(f"加载了 {len(data)} 个质量评估样本")
        except Exception as e:
            self.logger.error(f"加载标签文件失败: {e}")

        return data


# ============================================================================
# 类别映射加载器
# ============================================================================

class ClassLoader:
    """类别映射加载器"""

    def __init__(self, data_root: Path, logger: logging.Logger):
        """
        初始化类别加载器

        Args:
            data_root: 数据根目录
            logger: logger实例
        """
        self.data_root = Path(data_root)
        self.logger = logger

    def load_type_classes(self) -> List[str]:
        """
        加载机型类别映射

        Returns:
            机型类别列表
        """
        classes_file = self.data_root / 'aerovision' / 'labels' / 'type_classes.json'
        if not classes_file.exists():
            self.logger.warning(f"机型类别文件不存在: {classes_file}")
            return []

        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = json.load(f)
            self.logger.info(f"加载了 {len(classes)} 个机型类别")
            return classes
        except Exception as e:
            self.logger.error(f"加载机型类别失败: {e}")
            return []

    def load_airline_classes(self) -> List[str]:
        """
        加载航司类别映射

        Returns:
            航司类别列表
        """
        classes_file = self.data_root / 'aerovision' / 'labels' / 'airline_classes.json'
        if not classes_file.exists():
            self.logger.warning(f"航司类别文件不存在: {classes_file}")
            return []

        try:
            with open(classes_file, 'r', encoding='utf-8') as f:
                classes = json.load(f)
            self.logger.info(f"加载了 {len(classes)} 个航司类别")
            return classes
        except Exception as e:
            self.logger.error(f"加载航司类别失败: {e}")
            return []


# ============================================================================
# 准确率计算器
# ============================================================================

class AccuracyCalculator:
    """准确率计算器 - 支持不同任务的准确率计算"""

    def __init__(self, logger: logging.Logger):
        """
        初始化准确率计算器

        Args:
            logger: logger实例
        """
        self.logger = logger

    def calculate_classification_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str],
        top_k: int = 3
    ) -> Dict[str, float]:
        """
        计算分类任务的准确率

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            top_k: Top-K准确率

        Returns:
            准确率字典
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测结果和真实标签数量不匹配")

        total = len(predictions)
        if total == 0:
            return {'accuracy': 0.0, f'top{top_k}_accuracy': 0.0}

        # 精确匹配准确率
        exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
        accuracy = exact_matches / total

        # Top-K准确率（假设predictions已经是按置信度排序的列表）
        top_k_matches = 0
        for pred, gt in zip(predictions, ground_truths):
            if isinstance(pred, list):
                top_k_matches += 1 if gt in pred[:top_k] else 0
            else:
                top_k_matches += 1 if pred == gt else 0

        top_k_accuracy = top_k_matches / total

        return {
            'accuracy': accuracy,
            f'top{top_k}_accuracy': top_k_accuracy
        }

    def calculate_ocr_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """
        计算OCR任务的准确率

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表

        Returns:
            准确率字典
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测结果和真实标签数量不匹配")

        total = len(predictions)
        if total == 0:
            return {'exact_accuracy': 0.0, 'character_accuracy': 0.0}

        # 精确匹配准确率
        exact_matches = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
        exact_accuracy = exact_matches / total

        # 字符级准确率
        total_chars = 0
        correct_chars = 0
        for pred, gt in zip(predictions, ground_truths):
            # 对齐字符（简单对齐，实际可能需要更复杂的算法）
            for p_char, g_char in zip(pred, gt):
                total_chars += 1
                if p_char == g_char:
                    correct_chars += 1

        character_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

        return {
            'exact_accuracy': exact_accuracy,
            'character_accuracy': character_accuracy
        }

    def calculate_regression_metrics(
        self,
        predictions: List[Dict[str, float]],
        ground_truths: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        计算回归任务的指标

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表

        Returns:
            指标字典
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("预测结果和真实标签数量不匹配")

        total = len(predictions)
        if total == 0:
            return {'clarity_mae': 0.0, 'clarity_rmse': 0.0, 'block_mae': 0.0, 'block_rmse': 0.0}

        # 计算clarity指标
        clarity_errors = []
        for pred, gt in zip(predictions, ground_truths):
            if 'clarity' in pred and 'clarity' in gt:
                clarity_errors.append(pred['clarity'] - gt['clarity'])

        clarity_mae = np.mean(np.abs(clarity_errors)) if clarity_errors else 0.0
        clarity_rmse = np.sqrt(np.mean(np.square(clarity_errors))) if clarity_errors else 0.0

        # 计算block指标
        block_errors = []
        for pred, gt in zip(predictions, ground_truths):
            if 'block' in pred and 'block' in gt:
                block_errors.append(pred['block'] - gt['block'])

        block_mae = np.mean(np.abs(block_errors)) if block_errors else 0.0
        block_rmse = np.sqrt(np.mean(np.square(block_errors))) if block_errors else 0.0

        return {
            'clarity_mae': clarity_mae,
            'clarity_rmse': clarity_rmse,
            'block_mae': block_mae,
            'block_rmse': block_rmse
        }

    def create_confusion_matrix(
        self,
        predictions: List[str],
        ground_truths: List[str],
        classes: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """
        创建混淆矩阵

        Args:
            predictions: 预测结果列表
            ground_truths: 真实标签列表
            classes: 类别列表

        Returns:
            混淆矩阵字典
        """
        confusion = defaultdict(lambda: defaultdict(int))

        for pred, gt in zip(predictions, ground_truths):
            confusion[gt][pred] += 1

        # 转换为普通字典
        result = {}
        for gt_class in classes:
            result[gt_class] = {}
            for pred_class in classes:
                result[gt_class][pred_class] = confusion[gt_class][pred_class]

        return result


# ============================================================================
# 结果报告生成器
# ============================================================================

class ReportGenerator:
    """结果报告生成器"""

    def __init__(self, logger: logging.Logger):
        """
        初始化报告生成器

        Args:
            logger: logger实例
        """
        self.logger = logger

    def generate_json_report(
        self,
        task_type: str,
        metrics: Dict[str, float],
        results: List[Dict[str, Any]],
        confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        生成JSON格式报告

        Args:
            task_type: 任务类型
            metrics: 指标字典
            results: 结果列表
            confusion_matrix: 混淆矩阵
            errors: 错误列表
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        if output_path is None:
            output_path = Path('results') / f'{task_type}_report.json'

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'task_type': task_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'results': results,
            'confusion_matrix': confusion_matrix,
            'errors': errors or []
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"JSON报告已保存到: {output_path}")
        return output_path

    def generate_text_report(
        self,
        task_type: str,
        metrics: Dict[str, float],
        confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None,
        errors: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        生成可读的文本报告

        Args:
            task_type: 任务类型
            metrics: 指标字典
            confusion_matrix: 混淆矩阵
            errors: 错误列表
            output_path: 输出路径

        Returns:
            报告文件路径
        """
        if output_path is None:
            output_path = Path('results') / f'{task_type}_report.txt'

        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 80)
        lines.append(f"VLM测试报告 - {task_type}")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 指标
        lines.append("准确率指标:")
        lines.append("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        lines.append("")

        # 混淆矩阵
        if confusion_matrix:
            lines.append("混淆矩阵:")
            lines.append("-" * 40)
            # 获取所有类别
            classes = sorted(confusion_matrix.keys())
            # 表头
            header = "真实\\预测".ljust(15)
            for cls in classes[:10]:  # 限制显示的类别数
                header += cls.ljust(10)
            lines.append(header)
            lines.append("-" * 40)
            # 数据行
            for gt_class in classes[:10]:
                row = gt_class.ljust(15)
                for pred_class in classes[:10]:
                    row += str(confusion_matrix[gt_class][pred_class]).ljust(10)
                lines.append(row)
            lines.append("")

        # 错误分析
        if errors:
            lines.append(f"错误样本分析 (共{len(errors)}个):")
            lines.append("-" * 40)
            for i, error in enumerate(errors[:20]):  # 限制显示的样本数
                lines.append(f"  {i+1}. 图片: {error.get('image_path', 'N/A')}")
                lines.append(f"     真实: {error.get('ground_truth', 'N/A')}")
                lines.append(f"     预测: {error.get('prediction', 'N/A')}")
                lines.append(f"     错误: {error.get('error', 'N/A')}")
                lines.append("")
            if len(errors) > 20:
                lines.append(f"  ... 还有 {len(errors) - 20} 个错误样本未显示")
            lines.append("")

        lines.append("=" * 80)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self.logger.info(f"文本报告已保存到: {output_path}")
        return output_path


# ============================================================================
# 进度跟踪器
# ============================================================================

class ProgressTracker:
    """进度跟踪器 - 支持进度显示和中断恢复，支持线程安全"""

    def __init__(self, total: int, logger: logging.Logger, checkpoint_path: Optional[Path] = None):
        """
        初始化进度跟踪器

        Args:
            total: 总样本数
            logger: logger实例
            checkpoint_path: 检查点文件路径
        """
        self.total = total
        self.logger = logger
        self.checkpoint_path = checkpoint_path
        self.processed = 0
        self.start_time = time.time()
        self.results = []
        self.errors = []
        # 用于线程安全的锁
        self._lock = threading.Lock()
        # 已处理的图片路径集合（用于skip_processed逻辑）
        self.processed_paths = set()

        # 加载检查点
        if checkpoint_path and checkpoint_path.exists():
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """从检查点加载进度"""
        try:
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.processed = data.get('processed', 0)
            self.results = data.get('results', [])
            self.errors = data.get('errors', [])
            self.start_time = data.get('start_time', time.time())
            # 从检查点中恢复已处理的图片路径
            self.processed_paths = set(data.get('processed_paths', []))
            self.logger.info(f"从检查点恢复: 已处理 {self.processed}/{self.total} 个样本")
        except Exception as e:
            self.logger.warning(f"加载检查点失败: {e}")

    def _save_checkpoint(self) -> None:
        """保存检查点"""
        if not self.checkpoint_path:
            return

        try:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'total': self.total,
                'processed': self.processed,
                'results': self.results,
                'errors': self.errors,
                'start_time': self.start_time,
                'timestamp': datetime.now().isoformat(),
                'processed_paths': list(self.processed_paths)  # 保存已处理的路径
            }
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"保存检查点失败: {e}")

    def update(
        self,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None,
        save_interval: int = 10
    ) -> None:
        """
        更新进度（线程安全）

        Args:
            result: 结果字典
            error: 错误字典
            save_interval: 保存检查点的间隔
        """
        with self._lock:
            self.processed += 1

            if result:
                self.results.append(result)
                # 记录已处理的图片路径
                if 'image_path' in result:
                    self.processed_paths.add(result['image_path'])
            if error:
                self.errors.append(error)
                # 记录已处理的图片路径
                if 'image_path' in error:
                    self.processed_paths.add(error['image_path'])

            # 定期保存检查点
            if self.processed % save_interval == 0:
                self._save_checkpoint()

            # 显示进度
            self._display_progress()

    def _display_progress(self) -> None:
        """显示进度信息"""
        elapsed = time.time() - self.start_time
        progress = self.processed / self.total

        # 计算预计剩余时间
        if progress > 0:
            eta = elapsed / progress * (1 - progress)
            eta_str = self._format_time(eta)
        else:
            eta_str = "未知"

        self.logger.info(
            f"进度: {self.processed}/{self.total} ({progress*100:.1f}%) | "
            f"已用: {self._format_time(elapsed)} | "
            f"预计剩余: {eta_str}"
        )

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def is_processed(self, image_path: str) -> bool:
        """
        检查图片是否已处理（线程安全）

        Args:
            image_path: 图片路径

        Returns:
            是否已处理
        """
        with self._lock:
            return image_path in self.processed_paths

    def finish(self) -> None:
        """完成进度跟踪"""
        self._save_checkpoint()
        elapsed = time.time() - self.start_time
        self.logger.info(f"测试完成! 总耗时: {self._format_time(elapsed)}")


# ============================================================================
# VLM测试器
# ============================================================================

class VLMTester:
    """VLM测试器 - 主测试类"""

    def __init__(
        self,
        task_type: str,
        data_path: str,
        output_dir: str,
        config: Dict[str, Any],
        logger: logging.Logger,
        resume: Optional[str] = None,
        skip_processed: bool = False,
        concurrent: bool = False,
        max_workers: int = 10
    ):
        """
        初始化VLM测试器

        Args:
            task_type: 任务类型
            data_path: 数据路径
            output_dir: 输出目录
            config: 配置字典
            logger: logger实例
            resume: 恢复检查点路径
            skip_processed: 是否跳过已处理的样本
            concurrent: 是否启用并发处理
            max_workers: 最大并发数
        """
        self.task_type = task_type
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.config = config
        self.logger = logger
        self.skip_processed = skip_processed
        self.concurrent = concurrent
        self.max_workers = max_workers

        # 初始化组件
        self.data_loader = DataLoader(self.data_path, logger)
        self.class_loader = ClassLoader(self.data_path, logger)
        self.accuracy_calculator = AccuracyCalculator(logger)
        self.report_generator = ReportGenerator(logger)

        # 初始化VLM推理引擎
        self.inference = VLMInference.from_config(config)

        # 设置检查点路径
        self.checkpoint_path = self.output_dir / f'{task_type}_checkpoint.json'
        if resume:
            self.checkpoint_path = Path(resume)

        # 加载数据
        self.data = self._load_data()
        self.classes = self._load_classes()

    def _load_data(self) -> List:
        """加载测试数据"""
        if self.task_type == 'aircraft_type':
            return self.data_loader.load_aircraft_type_data()
        elif self.task_type == 'airline':
            return self.data_loader.load_airline_data()
        elif self.task_type == 'registration':
            return self.data_loader.load_registration_data()
        elif self.task_type == 'quality':
            return self.data_loader.load_quality_data()
        else:
            raise ValueError(f"未知任务类型: {self.task_type}")

    def _load_classes(self) -> List[str]:
        """加载类别映射"""
        if self.task_type == 'aircraft_type':
            return self.class_loader.load_type_classes()
        elif self.task_type == 'airline':
            return self.class_loader.load_airline_classes()
        else:
            return []

    def run(self) -> Dict[str, Any]:
        """
        运行测试

        Returns:
            测试结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info(f"开始VLM测试 - {self.task_type}")
        self.logger.info("=" * 60)
        self.logger.info(f"数据路径: {self.data_path}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(f"样本数量: {len(self.data)}")
        self.logger.info(f"并发模式: {'启用' if self.concurrent else '禁用'}")
        if self.concurrent:
            self.logger.info(f"并发数: {self.max_workers}")
        self.logger.info("=" * 60)

        # 初始化进度跟踪器
        tracker = ProgressTracker(
            total=len(self.data),
            logger=self.logger,
            checkpoint_path=self.checkpoint_path
        )

        # 过滤已处理的样本（使用路径判断而不是索引）
        if self.skip_processed:
            original_count = len(self.data)
            self.data = [(path, gt) for path, gt in self.data 
                        if not tracker.is_processed(path)]
            skipped_count = original_count - len(self.data)
            if skipped_count > 0:
                self.logger.info(f"跳过 {skipped_count} 个已处理的样本")

        # 运行推理
        results = []
        errors = []

        if self.concurrent:
            # 并发模式
            results, errors = self._run_concurrent(tracker)
        else:
            # 串行模式
            for item in self.data:
                image_path = item[0]
                ground_truth = item[1]

                try:
                    # 执行推理
                    result = self._infer_single(image_path)

                    if result.success:
                        # 应用字段名映射
                        mapped_data = self._map_field_names(result.data)
                        # 记录结果
                        result_record = {
                            'image_path': image_path,
                            'ground_truth': ground_truth,
                            'prediction': mapped_data,
                            'confidence': mapped_data.get('confidence', 0.0) if mapped_data else 0.0
                        }
                        results.append(result_record)
                        tracker.update(result=result_record)
                    else:
                        # 记录错误
                        error_record = {
                            'image_path': image_path,
                            'ground_truth': ground_truth,
                            'error': result.error
                        }
                        errors.append(error_record)
                        tracker.update(error=error_record)

                except Exception as e:
                    self.logger.error(f"处理样本失败: {image_path}, 错误: {e}")
                    error_record = {
                        'image_path': image_path,
                        'ground_truth': ground_truth,
                        'error': str(e)
                    }
                    errors.append(error_record)
                    tracker.update(error=error_record)

        # 完成进度跟踪
        tracker.finish()

        # 计算准确率
        metrics = self._calculate_metrics(results)

        # 生成混淆矩阵（仅分类任务）
        confusion_matrix = None
        if self.task_type in ['aircraft_type', 'airline']:
            # 处理prediction为None的情况
            predictions = [
                r['prediction'].get('aircraft_type' if self.task_type == 'aircraft_type' else 'airline', '')
                if r['prediction'] else ''
                for r in results
            ]
            ground_truths = [r['ground_truth'] for r in results]
            confusion_matrix = self.accuracy_calculator.create_confusion_matrix(
                predictions, ground_truths, self.classes
            )

        # 生成报告
        self._generate_reports(metrics, results, confusion_matrix, errors)

        return {
            'task_type': self.task_type,
            'metrics': metrics,
            'results': results,
            'confusion_matrix': confusion_matrix,
            'errors': errors
        }

    def _infer_single(self, image_path: str) -> InferenceResult:
        """
        执行单次推理（带重试）

        Args:
            image_path: 图片路径

        Returns:
            推理结果
        """
        max_attempts = self.config.get('glm', {}).get('retry', {}).get('max_attempts', 3)
        delay = self.config.get('glm', {}).get('retry', {}).get('delay', 1)

        for attempt in range(max_attempts):
            try:
                if self.task_type == 'aircraft_type':
                    return self.inference.infer_aircraft_type(image_path, self.classes)
                elif self.task_type == 'airline':
                    return self.inference.infer_airline(image_path, self.classes)
                elif self.task_type == 'registration':
                    return self.inference.infer_registration(image_path)
                elif self.task_type == 'quality':
                    return self.inference.infer_quality(image_path)
                else:
                    raise ValueError(f"未知任务类型: {self.task_type}")

            except Exception as e:
                if attempt < max_attempts - 1:
                    self.logger.warning(f"推理失败，重试 ({attempt + 1}/{max_attempts}): {e}")
                    time.sleep(delay * (2 ** attempt))  # 指数退避
                else:
                    raise

    def _run_concurrent(self, tracker: ProgressTracker) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        并发运行测试

        Args:
            tracker: 进度跟踪器

        Returns:
            (results, errors) 元组
        """
        results = []
        errors = []

        def process_item(item: Tuple[str, Any]) -> Optional[Dict[str, Any]]:
            """
            处理单个样本

            Args:
                item: (image_path, ground_truth) 元组

            Returns:
                结果字典或错误字典
            """
            image_path = item[0]
            ground_truth = item[1]

            try:
                # 执行推理
                result = self._infer_single(image_path)

                if result.success:
                    # 应用字段名映射
                    mapped_data = self._map_field_names(result.data)
                    # 返回结果
                    return {
                        'type': 'result',
                        'data': {
                            'image_path': image_path,
                            'ground_truth': ground_truth,
                            'prediction': mapped_data,
                            'confidence': mapped_data.get('confidence', 0.0) if mapped_data else 0.0
                        }
                    }
                else:
                    # 返回错误
                    return {
                        'type': 'error',
                        'data': {
                            'image_path': image_path,
                            'ground_truth': ground_truth,
                            'error': result.error
                        }
                    }

            except Exception as e:
                self.logger.error(f"处理样本失败: {image_path}, 错误: {e}")
                return {
                    'type': 'error',
                    'data': {
                        'image_path': image_path,
                        'ground_truth': ground_truth,
                        'error': str(e)
                    }
                }

        # 使用线程池并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_item = {executor.submit(process_item, item): item for item in self.data}

            # 等待任务完成
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    outcome = future.result()
                    if outcome['type'] == 'result':
                        results.append(outcome['data'])
                        tracker.update(result=outcome['data'])
                    else:
                        errors.append(outcome['data'])
                        tracker.update(error=outcome['data'])
                except Exception as e:
                    item = future_to_item[future]
                    self.logger.error(f"任务执行异常: {item[0]}, 错误: {e}")

        return results, errors

    def _map_field_names(self, data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        映射VLM输出的字段名以匹配data_format.md中的定义

        Args:
            data: VLM输出数据

        Returns:
            映射后的数据
        """
        if not data:
            return data

        # 创建映射后的数据副本
        mapped_data = data.copy()

        # 字段名映射：clarity_score -> clarity, occlusion_score -> block
        if 'clarity_score' in mapped_data:
            mapped_data['clarity'] = mapped_data.pop('clarity_score')
        if 'occlusion_score' in mapped_data:
            mapped_data['block'] = mapped_data.pop('occlusion_score')

        return mapped_data

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算准确率指标

        Args:
            results: 结果列表

        Returns:
            指标字典
        """
        if not results:
            return {}

        if self.task_type in ['aircraft_type', 'airline']:
            # 处理prediction为None的情况
            predictions = [
                r['prediction'].get('aircraft_type' if self.task_type == 'aircraft_type' else 'airline', '')
                if r['prediction'] else ''
                for r in results
            ]
            ground_truths = [r['ground_truth'] for r in results]
            return self.accuracy_calculator.calculate_classification_accuracy(predictions, ground_truths)

        elif self.task_type == 'registration':
            predictions = [r['prediction'].get('registration', '') if r['prediction'] else '' for r in results]
            ground_truths = [r['ground_truth'] for r in results]
            return self.accuracy_calculator.calculate_ocr_accuracy(predictions, ground_truths)

        elif self.task_type == 'quality':
            predictions = [r['prediction'] if r['prediction'] else {} for r in results]
            ground_truths = [r['ground_truth'] for r in results]
            return self.accuracy_calculator.calculate_regression_metrics(predictions, ground_truths)

        else:
            return {}

    def _generate_reports(
        self,
        metrics: Dict[str, float],
        results: List[Dict[str, Any]],
        confusion_matrix: Optional[Dict[str, Dict[str, int]]],
        errors: List[Dict[str, Any]]
    ) -> None:
        """
        生成报告

        Args:
            metrics: 指标字典
            results: 结果列表
            confusion_matrix: 混淆矩阵
            errors: 错误列表
        """
        # 生成JSON报告
        json_path = self.output_dir / f'{self.task_type}_report.json'
        self.report_generator.generate_json_report(
            self.task_type, metrics, results, confusion_matrix, errors, json_path
        )

        # 生成文本报告
        text_path = self.output_dir / f'{self.task_type}_report.txt'
        self.report_generator.generate_text_report(
            self.task_type, metrics, confusion_matrix, errors, text_path
        )


# ============================================================================
# 命令行参数解析
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    解析命令行参数

    Returns:
        解析后的参数
    """
    parser = argparse.ArgumentParser(
        description='VLM测试脚本 - 用于计算VLM模型在各个任务上的准确率',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 任务选择
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['aircraft_type', 'airline', 'registration', 'quality'],
        help='测试任务类型'
    )

    # 数据路径
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/splits/latest/',
        help='数据集路径'
    )

    # 输出目录
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/vlm_test/',
        help='输出目录'
    )

    # 恢复检查点
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='恢复检查点路径'
    )

    # 跳过已处理
    parser.add_argument(
        '--skip-processed',
        action='store_true',
        default=False,
        help='跳过已处理的样本'
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径'
    )

    # 并发处理
    parser.add_argument(
        '--concurrent',
        action='store_true',
        default=False,
        help='启用并发处理'
    )

    # 并发数
    parser.add_argument(
        '--max-workers',
        type=int,
        default=10,
        help='最大并发数（默认10）'
    )

    return parser.parse_args()


# ============================================================================
# 主函数
# ============================================================================

def main() -> None:
    """主函数"""
    # 解析参数
    args = parse_arguments()

    # 设置输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir)

    # 加载配置
    try:
        if args.config and Path(args.config).exists():
            config_obj = load_config(args.config)
        else:
            config_obj = load_config(modules=['vlm', 'paths'], load_all_modules=False)
        config = config_obj.to_dict()
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        sys.exit(1)

    # 解析数据路径
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        # 相对于项目根目录
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / args.data_path

    # 运行测试
    try:
        tester = VLMTester(
            task_type=args.task,
            data_path=str(data_path),
            output_dir=str(output_dir),
            config=config,
            logger=logger,
            resume=args.resume,
            skip_processed=args.skip_processed,
            concurrent=args.concurrent,
            max_workers=args.max_workers
        )

        results = tester.run()

        logger.info("=" * 60)
        logger.info("测试完成!")
        logger.info("=" * 60)
        logger.info(f"准确率指标: {results['metrics']}")
        logger.info(f"结果已保存到: {output_dir}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("测试被用户中断")
        sys.exit(1)

    except Exception as e:
        logger.error(f"测试失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
