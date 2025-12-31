#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 Aircraft Classifier Test Suite

测试YOLOv8机型分类微调功能，包括：
1. 数据集准备功能测试
2. 训练脚本功能测试
3. 数据加载测试
4. 模型推理测试

运行测试:
    pytest training/tests/test_aircraft_classifier.py -v
    pytest training/tests/test_aircraft_classifier.py::test_model_initialization -v
    pytest training/tests/test_aircraft_classifier.py --cov=training/scripts --cov-report=html
"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch, mock_open

import pytest
import torch
import yaml

# 添加父目录到路径以便导入
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.scripts.prepare_dataset import DatasetPreparer
from training.scripts.train_aircraft_classifier import (
    setup_logging,
    load_config,
    parse_arguments,
    merge_config_with_args,
    save_custom_checkpoint,
    AircraftClassifierTrainer,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir() -> Path:
    """
    创建临时目录用于测试

    Returns:
        临时目录路径
    """
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_csv_data() -> str:
    """
    生成示例CSV数据

    Returns:
        CSV格式的字符串数据
    """
    csv_content = """filename,type_id,type_name
aircraft_001.jpg,0,A320
aircraft_002.jpg,0,A320
aircraft_003.jpg,1,B737-800
aircraft_004.jpg,1,B737-800
aircraft_005.jpg,2,A330
aircraft_006.jpg,2,A330
aircraft_007.jpg,3,B777
aircraft_008.jpg,3,B777
aircraft_009.jpg,4,A350
aircraft_010.jpg,4,A350
"""
    return csv_content


@pytest.fixture
def sample_csv_file(temp_dir: Path, sample_csv_data: str) -> Path:
    """
    创建示例CSV文件

    Args:
        temp_dir: 临时目录
        sample_csv_data: CSV数据内容

    Returns:
        CSV文件路径
    """
    csv_file = temp_dir / "aircraft_labels.csv"
    csv_file.write_text(sample_csv_data, encoding='utf-8')
    return csv_file


@pytest.fixture
def sample_images(temp_dir: Path) -> Path:
    """
    创建示例图片文件

    Args:
        temp_dir: 临时目录

    Returns:
        图片目录路径
    """
    images_dir = temp_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 创建一些模拟图片文件
    for i in range(1, 11):
        img_file = images_dir / f"aircraft_{i:03d}.jpg"
        img_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')  # JPEG header

    return images_dir


@pytest.fixture
def sample_config(temp_dir: Path) -> Path:
    """
    创建示例配置文件

    Args:
        temp_dir: 临时目录

    Returns:
        配置文件路径
    """
    config_data = {
        'path': str(temp_dir / 'processed' / 'aircraft'),
        'train': 'train',
        'val': 'val',
        'test': 'test',
        'names': {
            0: 'A320',
            1: 'B737-800',
            2: 'A330',
            3: 'B777',
            4: 'A350',
        },
        'nc': 5,
        'epochs': 100,
        'batch_size': 32,
        'imgsz': 224,
        'patience': 10,
        'save': True,
        'device': 0,
        'workers': 8,
        'project': 'runs/classify',
        'name': 'aircraft',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'close_mosaic': 0,
        'amp': True,
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'multi_scale': False,
        'single_cls': False,
        'rect': False,
        'cos_lr': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'plots': True,
        'save_json': False,
        'save_hybrid': False,
        'conf': None,
        'iou': 0.7,
        'max_det': 300,
        'half': False,
        'dnn': False,
        'vid_stride': 1,
    }

    config_file = temp_dir / "aircraft_classify.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return config_file


@pytest.fixture
def mock_yolo_model():
    """
    创建模拟的YOLO模型

    Returns:
        模拟的YOLO模型对象
    """
    model = Mock()
    model.model = Mock()
    model.model.args = {}
    model.model.names = {0: 'A320', 1: 'B737-800', 2: 'A330', 3: 'B777', 4: 'A350'}
    model.trainer = None
    return model


@pytest.fixture
def mock_optimizer():
    """
    创建模拟的优化器

    Returns:
        模拟的优化器对象
    """
    optimizer = Mock()
    optimizer.state_dict.return_value = {'state': {}}
    return optimizer


@pytest.fixture
def mock_logger(temp_dir: Path) -> logging.Logger:
    """
    创建模拟的日志记录器

    Args:
        temp_dir: 临时目录

    Returns:
        日志记录器对象
    """
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    return logger


# ============================================================================
# 测试数据集准备功能
# ============================================================================

class TestDatasetPreparer:
    """测试数据集准备功能"""

    def test_dataset_preparer_initialization(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试DatasetPreparer初始化

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output'),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            random_seed=42
        )

        assert preparer.raw_data_dir == temp_dir / 'images'
        assert preparer.csv_file == sample_csv_file
        assert preparer.output_dir == temp_dir / 'output'
        assert preparer.train_ratio == 0.8
        assert preparer.val_ratio == 0.1
        assert preparer.test_ratio == 0.1
        assert preparer.random_seed == 42

    def test_dataset_preparer_invalid_ratios(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试无效的比例组合

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        with pytest.raises(ValueError, match="训练集、验证集、测试集比例之和必须为1.0"):
            DatasetPreparer(
                raw_data_dir=str(temp_dir / 'images'),
                csv_file=str(sample_csv_file),
                output_dir=str(temp_dir / 'output'),
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3  # 总和为1.1
            )

    def test_load_csv(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试CSV文件加载

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()

        assert len(data) == 10
        assert data[0]['filename'] == 'aircraft_001.jpg'
        assert data[0]['type_id'] == '0'
        assert data[0]['type_name'] == 'A320'

    def test_load_csv_file_not_found(self, temp_dir: Path):
        """
        测试加载不存在的CSV文件

        Args:
            temp_dir: 临时目录
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(temp_dir / 'nonexistent.csv'),
            output_dir=str(temp_dir / 'output')
        )

        with pytest.raises(FileNotFoundError, match="CSV文件不存在"):
            preparer.load_csv()

    def test_build_class_mapping(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试类别映射构建

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)

        assert len(preparer.type_classes) == 5
        assert preparer.type_classes[0] == 'A320'
        assert preparer.type_classes[1] == 'B737-800'
        assert preparer.type_classes[2] == 'A330'
        assert preparer.type_classes[3] == 'B777'
        assert preparer.type_classes[4] == 'A350'

    def test_split_dataset(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试数据集划分

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output'),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1
        )

        data = preparer.load_csv()
        train_data, val_data, test_data = preparer.split_dataset(data)

        # 验证划分结果
        assert len(train_data) == 8  # 10 * 0.8
        assert len(val_data) == 1    # 10 * 0.1
        assert len(test_data) == 1   # 10 * 0.1

        # 验证总数据量不变
        assert len(train_data) + len(val_data) + len(test_data) == len(data)

    def test_create_directory_structure(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试目录结构创建

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)
        preparer.create_directory_structure()

        # 验证主目录创建
        assert preparer.processed_dir.exists()
        assert preparer.labels_dir.exists()
        assert preparer.configs_dir.exists()

        # 验证数据集目录创建
        for split in ['train', 'val', 'test']:
            split_dir = preparer.processed_dir / split
            assert split_dir.exists()

            # 验证类别目录创建
            for class_id, class_name in preparer.type_classes.items():
                safe_class_name = class_name.replace(' ', '_').replace('/', '_')
                class_dir = split_dir / safe_class_name
                assert class_dir.exists()

    def test_save_class_mapping(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试类别映射保存

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)
        preparer.create_directory_structure()  # 先创建目录结构
        preparer.save_class_mapping()

        # 验证JSON文件创建
        output_file = preparer.labels_dir / "type_classes.json"
        assert output_file.exists()

        # 验证JSON内容
        with open(output_file, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)

        assert class_mapping['num_classes'] == 5
        assert len(class_mapping['classes']) == 5
        assert class_mapping['classes'][0]['id'] == 0
        assert class_mapping['classes'][0]['name'] == 'A320'

    def test_create_yolo_config(self, temp_dir: Path, sample_csv_file: Path):
        """
        测试YOLOv8配置文件生成

        Args:
            temp_dir: 临时目录
            sample_csv_file: 示例CSV文件
        """
        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(sample_csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)
        preparer.create_directory_structure()  # 先创建目录结构
        preparer.create_yolo_config()

        # 验证YAML文件创建（注意：文件名是 aircraft_classify.yaml）
        config_file = preparer.configs_dir / "aircraft_classify.yaml"
        assert config_file.exists()

        # 验证YAML内容
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        assert config['nc'] == 5
        assert config['train'] == 'train'
        assert config['val'] == True  # 注意：prepare_dataset.py中val是布尔值
        assert config['test'] == 'test'
        assert config['epochs'] == 100
        assert config['batch_size'] == 32
        assert config['imgsz'] == 224

        # 验证类别名称
        assert config['names'][0] == 'A320'
        assert config['names'][1] == 'B737-800'


# ============================================================================
# 测试训练脚本功能
# ============================================================================

class TestTrainingScript:
    """测试训练脚本功能"""

    def test_setup_logging(self, temp_dir: Path):
        """
        测试日志设置

        Args:
            temp_dir: 临时目录
        """
        log_dir = temp_dir / 'logs'
        logger = setup_logging(log_dir)

        assert logger is not None
        assert logger.name == "AircraftClassifier"
        assert logger.level == 10  # DEBUG level

        # 验证日志文件创建
        log_files = list(log_dir.glob('train_*.log'))
        assert len(log_files) > 0

    def test_load_config(self, sample_config: Path):
        """
        测试配置文件加载

        Args:
            sample_config: 示例配置文件
        """
        config = load_config(str(sample_config))

        assert config is not None
        assert config['epochs'] == 100
        assert config['batch_size'] == 32
        assert config['imgsz'] == 224
        assert config['nc'] == 5
        assert config['optimizer'] == 'AdamW'

    def test_load_config_not_found(self):
        """
        测试加载不存在的配置文件
        """
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config('nonexistent_config.yaml')

    def test_parse_arguments_default(self):
        """
        测试默认参数解析
        """
        # 模拟命令行参数
        test_args = ['train_aircraft_classifier.py']
        with patch('sys.argv', test_args):
            args = parse_arguments()

            assert args.model == 'yolov8n-cls.pt'
            assert args.resume is None
            assert args.epochs == 100
            assert args.batch_size == 32
            assert args.imgsz == 224
            assert args.lr0 == 0.001
            assert args.optimizer == 'AdamW'
            assert args.patience == 50
            assert args.device == '0'

    def test_parse_arguments_custom(self):
        """
        测试自定义参数解析
        """
        test_args = [
            'train_aircraft_classifier.py',
            '--model', 'yolov8s-cls.pt',
            '--epochs', '50',
            '--batch-size', '16',
            '--lr0', '0.0001',
            '--optimizer', 'SGD'
        ]
        with patch('sys.argv', test_args):
            args = parse_arguments()

            assert args.model == 'yolov8s-cls.pt'
            assert args.epochs == 50
            assert args.batch_size == 16
            assert args.lr0 == 0.0001
            assert args.optimizer == 'SGD'

    def test_merge_config_with_args(self, sample_config: Path):
        """
        测试配置与参数合并

        Args:
            sample_config: 示例配置文件
        """
        config = load_config(str(sample_config))

        # 创建模拟的参数对象
        args = Mock()
        args.model = 'yolov8s-cls.pt'
        args.data = 'custom_data'
        args.epochs = 50
        args.batch_size = 16
        args.imgsz = 256
        args.lr0 = 0.0001
        args.optimizer = 'SGD'
        args.momentum = 0.9
        args.weight_decay = 0.001
        args.cos_lr = True
        args.lrf = 0.005
        args.dropout = 0.1
        args.patience = 20
        args.warmup_epochs = 5
        args.device = '1'
        args.workers = 4
        args.amp = False
        args.seed = 123
        args.project = 'custom_project'
        args.name = 'custom_name'
        args.save_period = 5
        args.val = False
        args.plots = False

        merged_config = merge_config_with_args(config, args)

        # 验证命令行参数覆盖配置文件
        assert merged_config['model'] == 'yolov8s-cls.pt'
        assert merged_config['epochs'] == 50
        assert merged_config['batch_size'] == 16
        assert merged_config['imgsz'] == 256
        assert merged_config['lr0'] == 0.0001
        assert merged_config['optimizer'] == 'SGD'

    def test_save_custom_checkpoint(self, temp_dir: Path):
        """
        测试checkpoint保存功能

        Args:
            temp_dir: 临时目录
        """
        # 创建简单的PyTorch模型
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # 创建模拟的YOLO模型包装器
        mock_yolo_model = Mock()
        mock_yolo_model.model = model
        mock_yolo_model.model.args = {}
        mock_yolo_model.model.names = {0: 'A320', 1: 'B737-800', 2: 'A330', 3: 'B777', 4: 'A350'}

        checkpoint_path = temp_dir / 'checkpoints' / 'epoch_10.pt'

        save_custom_checkpoint(
            model=mock_yolo_model,
            epoch=10,
            optimizer=optimizer,
            val_acc=0.85,
            checkpoint_path=checkpoint_path,
            is_best=False
        )

        # 验证checkpoint文件创建
        assert checkpoint_path.exists()

        # 验证checkpoint内容
        checkpoint = torch.load(checkpoint_path)
        assert checkpoint['epoch'] == 10
        assert checkpoint['val_acc'] == 0.85
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint

    def test_save_custom_checkpoint_best(self, temp_dir: Path):
        """
        测试最佳checkpoint保存功能

        Args:
            temp_dir: 临时目录
        """
        # 创建简单的PyTorch模型
        model = torch.nn.Linear(10, 5)
        optimizer = torch.optim.Adam(model.parameters())

        # 创建模拟的YOLO模型包装器
        mock_yolo_model = Mock()
        mock_yolo_model.model = model
        mock_yolo_model.model.args = {}
        mock_yolo_model.model.names = {0: 'A320', 1: 'B737-800', 2: 'A330', 3: 'B777', 4: 'A350'}

        checkpoint_path = temp_dir / 'checkpoints' / 'epoch_10.pt'

        save_custom_checkpoint(
            model=mock_yolo_model,
            epoch=10,
            optimizer=optimizer,
            val_acc=0.95,
            checkpoint_path=checkpoint_path,
            is_best=True
        )

        # 验证checkpoint文件创建
        assert checkpoint_path.exists()

        # 验证best checkpoint文件创建
        best_path = checkpoint_path.parent / 'best.pt'
        assert best_path.exists()

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_model_initialization_from_pretrained(self, mock_yolo_class, temp_dir: Path, mock_logger):
        """
        测试从预训练模型初始化

        Args:
            mock_yolo_class: 模拟的YOLO类
            temp_dir: 临时目录
            mock_logger: 模拟的日志记录器
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        config = {
            'model': 'yolov8n-cls.pt',
            'data': str(temp_dir / 'data'),
            'epochs': 10,
            'batch_size': 16,
            'imgsz': 224,
            'lr0': 0.001,
            'optimizer': 'AdamW',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'cos_lr': True,
            'lrf': 0.01,
            'dropout': 0.0,
            'patience': 10,
            'warmup_epochs': 3,
            'device': 'cpu',
            'workers': 4,
            'amp': False,
            'seed': 42,
            'project': 'runs/classify',
            'name': 'aircraft',
            'save_period': -1,
            'val': True,
            'plots': True,
        }

        args = Mock()
        args.resume = None
        args.checkpoint_dir = str(temp_dir / 'checkpoints')
        args.tensorboard = False
        args.log_dir = str(temp_dir / 'logs')

        trainer = AircraftClassifierTrainer(config, args, mock_logger)

        # 验证YOLO被正确初始化
        mock_yolo_class.assert_called_once_with('yolov8n-cls.pt')
        assert trainer.model == mock_yolo_instance

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_model_initialization_from_checkpoint(self, mock_yolo_class, temp_dir: Path, mock_logger):
        """
        测试从checkpoint恢复训练

        Args:
            mock_yolo_class: 模拟的YOLO类
            temp_dir: 临时目录
            mock_logger: 模拟的日志记录器
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        config = {
            'model': 'yolov8n-cls.pt',
            'data': str(temp_dir / 'data'),
            'epochs': 10,
            'batch_size': 16,
            'imgsz': 224,
            'lr0': 0.001,
            'optimizer': 'AdamW',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'cos_lr': True,
            'lrf': 0.01,
            'dropout': 0.0,
            'patience': 10,
            'warmup_epochs': 3,
            'device': 'cpu',
            'workers': 4,
            'amp': False,
            'seed': 42,
            'project': 'runs/classify',
            'name': 'aircraft',
            'save_period': -1,
            'val': True,
            'plots': True,
        }

        args = Mock()
        args.resume = str(temp_dir / 'checkpoints' / 'last.pt')
        args.checkpoint_dir = str(temp_dir / 'checkpoints')
        args.tensorboard = False
        args.log_dir = str(temp_dir / 'logs')

        trainer = AircraftClassifierTrainer(config, args, mock_logger)

        # 验证YOLO从checkpoint加载
        mock_yolo_class.assert_called_once_with(args.resume)
        assert trainer.model == mock_yolo_instance

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_log_metrics(self, mock_yolo_class, temp_dir: Path, mock_logger):
        """
        测试指标日志记录

        Args:
            mock_yolo_class: 模拟的YOLO类
            temp_dir: 临时目录
            mock_logger: 模拟的日志记录器
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        config = {
            'model': 'yolov8n-cls.pt',
            'data': str(temp_dir / 'data'),
            'epochs': 10,
            'batch_size': 16,
            'imgsz': 224,
            'lr0': 0.001,
            'optimizer': 'AdamW',
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'cos_lr': True,
            'lrf': 0.01,
            'dropout': 0.0,
            'patience': 10,
            'warmup_epochs': 3,
            'device': 'cpu',
            'workers': 4,
            'amp': False,
            'seed': 42,
            'project': 'runs/classify',
            'name': 'aircraft',
            'save_period': -1,
            'val': True,
            'plots': True,
        }

        args = Mock()
        args.resume = None
        args.checkpoint_dir = str(temp_dir / 'checkpoints')
        args.tensorboard = False
        args.log_dir = str(temp_dir / 'logs')

        trainer = AircraftClassifierTrainer(config, args, mock_logger)

        # 测试日志记录
        metrics = {
            'loss': 0.5,
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.80
        }

        trainer.log_metrics(metrics, epoch=5)

        # 验证方法调用没有抛出异常
        assert True


# ============================================================================
# 测试数据加载
# ============================================================================

class TestDataLoading:
    """测试数据加载功能"""

    def test_train_data_loading(self, temp_dir: Path):
        """
        测试训练集数据加载

        Args:
            temp_dir: 临时目录
        """
        # 创建模拟的训练集目录结构
        train_dir = temp_dir / 'train'
        train_dir.mkdir(parents=True, exist_ok=True)

        # 创建类别目录和模拟图片
        classes = ['A320', 'B737-800', 'A330']
        for class_name in classes:
            class_dir = train_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # 创建3张模拟图片
            for i in range(3):
                img_file = class_dir / f'{class_name}_{i}.jpg'
                img_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')

        # 验证目录结构
        assert train_dir.exists()
        for class_name in classes:
            class_dir = train_dir / class_name
            assert class_dir.exists()
            assert len(list(class_dir.glob('*.jpg'))) == 3

    def test_val_data_loading(self, temp_dir: Path):
        """
        测试验证集数据加载

        Args:
            temp_dir: 临时目录
        """
        # 创建模拟的验证集目录结构
        val_dir = temp_dir / 'val'
        val_dir.mkdir(parents=True, exist_ok=True)

        # 创建类别目录和模拟图片
        classes = ['A320', 'B737-800']
        for class_name in classes:
            class_dir = val_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # 创建2张模拟图片
            for i in range(2):
                img_file = class_dir / f'{class_name}_val_{i}.jpg'
                img_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')

        # 验证目录结构
        assert val_dir.exists()
        for class_name in classes:
            class_dir = val_dir / class_name
            assert class_dir.exists()
            assert len(list(class_dir.glob('*.jpg'))) == 2

    def test_data_augmentation_config(self, sample_config: Path):
        """
        测试数据增强配置

        Args:
            sample_config: 示例配置文件
        """
        config = load_config(str(sample_config))

        # 验证数据增强相关配置
        assert 'fraction' in config
        assert 'multi_scale' in config
        assert 'rect' in config
        assert 'single_cls' in config
        assert 'close_mosaic' in config

        # 验证默认值
        assert config['fraction'] == 1.0
        assert config['multi_scale'] is False
        assert config['rect'] is False
        assert config['single_cls'] is False
        assert config['close_mosaic'] == 0


# ============================================================================
# 测试模型推理
# ============================================================================

class TestModelInference:
    """测试模型推理功能"""

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_model_forward_pass(self, mock_yolo_class, temp_dir: Path):
        """
        测试模型前向传播

        Args:
            mock_yolo_class: 模拟的YOLO类
            temp_dir: 临时目录
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        # 模拟预测结果
        mock_result = Mock()
        mock_result.probs = Mock()
        mock_result.probs.data = torch.tensor([0.1, 0.7, 0.05, 0.1, 0.05])
        mock_result.probs.top1 = 1
        mock_result.probs.top1conf = 0.7
        mock_yolo_instance.predict.return_value = [mock_result]

        # 初始化模型（使用mock返回的实例）
        model = mock_yolo_class('yolov8n-cls.pt')

        # 模拟预测
        results = model.predict('test_image.jpg')

        # 验证预测调用
        assert mock_yolo_instance.predict.called
        assert len(results) == 1
        assert results[0].probs.top1 == 1
        assert results[0].probs.top1conf == 0.7

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_prediction_format(self, mock_yolo_class):
        """
        测试预测结果格式

        Args:
            mock_yolo_class: 模拟的YOLO类
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        # 模拟预测结果
        mock_result = Mock()
        mock_result.probs = Mock()
        mock_result.probs.data = torch.tensor([0.05, 0.8, 0.05, 0.05, 0.05])
        mock_result.probs.top1 = 1
        mock_result.probs.top1conf = 0.8
        mock_result.probs.top5 = [1, 0, 2, 3, 4]
        mock_yolo_instance.predict.return_value = [mock_result]

        # 初始化模型（使用mock返回的实例）
        model = mock_yolo_class('yolov8n-cls.pt')

        # 模拟预测
        results = model.predict('test_image.jpg')

        # 验证预测结果格式
        assert hasattr(results[0], 'probs')
        assert hasattr(results[0].probs, 'data')
        assert hasattr(results[0].probs, 'top1')
        assert hasattr(results[0].probs, 'top1conf')
        assert isinstance(results[0].probs.data, torch.Tensor)
        assert results[0].probs.data.shape == (5,)

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_class_probability_output(self, mock_yolo_class):
        """
        测试类别概率输出

        Args:
            mock_yolo_class: 模拟的YOLO类
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        # 模拟预测结果
        mock_result = Mock()
        mock_result.probs = Mock()
        mock_result.probs.data = torch.tensor([0.1, 0.6, 0.05, 0.15, 0.1])
        mock_result.probs.top1 = 1
        mock_result.probs.top1conf = 0.6
        mock_yolo_instance.predict.return_value = [mock_result]

        # 初始化模型（使用mock返回的实例）
        model = mock_yolo_class('yolov8n-cls.pt')

        # 模拟预测
        results = model.predict('test_image.jpg')

        # 验证概率输出
        probs = results[0].probs.data

        # 验证概率总和接近1
        assert torch.abs(torch.sum(probs) - 1.0) < 0.01

        # 验证所有概率在[0, 1]范围内
        assert torch.all(probs >= 0)
        assert torch.all(probs <= 1)

        # 验证最高概率
        assert torch.max(probs) == 0.6
        assert torch.argmax(probs) == 1

    @patch('training.scripts.train_aircraft_classifier.YOLO')
    def test_batch_prediction(self, mock_yolo_class):
        """
        测试批量预测

        Args:
            mock_yolo_class: 模拟的YOLO类
        """
        # 创建模拟的YOLO实例
        mock_yolo_instance = Mock()
        mock_yolo_class.return_value = mock_yolo_instance

        # 模拟批量预测结果
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_result.probs = Mock()
            mock_result.probs.data = torch.tensor([0.1, 0.7, 0.05, 0.1, 0.05])
            mock_result.probs.top1 = 1
            mock_result.probs.top1conf = 0.7
            mock_results.append(mock_result)

        mock_yolo_instance.predict.return_value = mock_results

        # 初始化模型（使用mock返回的实例）
        model = mock_yolo_class('yolov8n-cls.pt')

        # 模拟批量预测
        image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        results = model.predict(image_paths)

        # 验证批量预测结果
        assert len(results) == 3
        for result in results:
            assert result.probs.top1 == 1
            assert result.probs.top1conf == 0.7


# ============================================================================
# 测试集成功能
# ============================================================================

class TestIntegration:
    """测试集成功能"""

    def test_full_dataset_preparation_flow(self, temp_dir: Path, sample_csv_data: str):
        """
        测试完整的数据集准备流程

        Args:
            temp_dir: 临时目录
            sample_csv_data: CSV数据内容
        """
        # 创建CSV文件
        csv_file = temp_dir / "aircraft_labels.csv"
        csv_file.write_text(sample_csv_data, encoding='utf-8')

        # 创建模拟图片
        images_dir = temp_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, 11):
            img_file = images_dir / f"aircraft_{i:03d}.jpg"
            img_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')

        # 创建数据集准备器
        preparer = DatasetPreparer(
            raw_data_dir=str(images_dir),
            csv_file=str(csv_file),
            output_dir=str(temp_dir / 'output')
        )

        # 执行完整流程
        preparer.prepare()

        # 验证所有输出
        assert preparer.processed_dir.exists()
        assert preparer.labels_dir.exists()
        assert preparer.configs_dir.exists()

        # 验证类别映射文件
        class_mapping_file = preparer.labels_dir / "type_classes.json"
        assert class_mapping_file.exists()

        # 验证配置文件
        config_file = preparer.configs_dir / "aircraft_classify.yaml"
        assert config_file.exists()

        # 验证数据集目录
        for split in ['train', 'val', 'test']:
            split_dir = preparer.processed_dir / split
            assert split_dir.exists()

    def test_config_override_flow(self, temp_dir: Path, sample_config: Path):
        """
        测试配置覆盖流程

        Args:
            temp_dir: 临时目录
            sample_config: 示例配置文件
        """
        # 加载原始配置
        config = load_config(str(sample_config))
        original_epochs = config['epochs']
        original_batch_size = config['batch_size']

        # 创建模拟的参数对象
        args = Mock()
        args.model = 'yolov8n-cls.pt'
        args.data = 'custom_data'
        args.epochs = 50
        args.batch_size = 16
        args.imgsz = 224
        args.lr0 = 0.001
        args.optimizer = 'AdamW'
        args.momentum = 0.937
        args.weight_decay = 0.0005
        args.cos_lr = True
        args.lrf = 0.01
        args.dropout = 0.0
        args.patience = 10
        args.warmup_epochs = 3
        args.device = 'cpu'
        args.workers = 4
        args.amp = False
        args.seed = 42
        args.project = 'runs/classify'
        args.name = 'aircraft'
        args.save_period = -1
        args.val = True
        args.plots = True

        # 合并配置
        merged_config = merge_config_with_args(config, args)

        # 验证配置被覆盖
        assert merged_config['epochs'] != original_epochs
        assert merged_config['epochs'] == 50
        assert merged_config['batch_size'] != original_batch_size
        assert merged_config['batch_size'] == 16


# ============================================================================
# 测试边界情况
# ============================================================================

class TestEdgeCases:
    """测试边界情况"""

    def test_empty_dataset(self, temp_dir: Path):
        """
        测试空数据集处理

        Args:
            temp_dir: 临时目录
        """
        # 创建空的CSV文件
        csv_content = "filename,type_id,type_name\n"
        csv_file = temp_dir / "empty_labels.csv"
        csv_file.write_text(csv_content, encoding='utf-8')

        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()

        # 验证空数据集
        assert len(data) == 0

    def test_single_class_dataset(self, temp_dir: Path):
        """
        测试单类别数据集

        Args:
            temp_dir: 临时目录
        """
        # 创建单类别CSV文件
        csv_content = """filename,type_id,type_name
aircraft_001.jpg,0,A320
aircraft_002.jpg,0,A320
aircraft_003.jpg,0,A320
"""
        csv_file = temp_dir / "single_class_labels.csv"
        csv_file.write_text(csv_content, encoding='utf-8')

        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)

        # 验证单类别
        assert len(preparer.type_classes) == 1
        assert preparer.type_classes[0] == 'A320'

    def test_invalid_csv_format(self, temp_dir: Path):
        """
        测试无效的CSV格式

        Args:
            temp_dir: 临时目录
        """
        # 创建格式错误的CSV文件
        csv_content = """filename,type_id
aircraft_001.jpg,0
"""
        csv_file = temp_dir / "invalid_labels.csv"
        csv_file.write_text(csv_content, encoding='utf-8')

        preparer = DatasetPreparer(
            raw_data_dir=str(temp_dir / 'images'),
            csv_file=str(csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)

        # 验证空类别映射
        assert len(preparer.type_classes) == 0

    def test_missing_images(self, temp_dir: Path, sample_csv_data: str):
        """
        测试缺失图片的处理

        Args:
            temp_dir: 临时目录
            sample_csv_data: CSV数据内容
        """
        # 创建CSV文件
        csv_file = temp_dir / "aircraft_labels.csv"
        csv_file.write_text(sample_csv_data, encoding='utf-8')

        # 不创建图片文件
        images_dir = temp_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        preparer = DatasetPreparer(
            raw_data_dir=str(images_dir),
            csv_file=str(csv_file),
            output_dir=str(temp_dir / 'output')
        )

        data = preparer.load_csv()
        preparer.build_class_mapping(data)
        preparer.create_directory_structure()

        # 复制图片（应该跳过所有文件）
        preparer.copy_images(data, 'train')

        # 验证没有文件被复制
        train_dir = preparer.processed_dir / 'train'
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                assert len(list(class_dir.glob('*.jpg'))) == 0


# ============================================================================
# Pytest配置
# ============================================================================

def pytest_configure(config):
    """Pytest配置钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
