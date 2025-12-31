#!/usr/bin/env python3
"""
YOLOv8 Aircraft Classifier Fine-tuning Training Script

This script fine-tunes a pre-trained YOLOv8 classification model for aircraft type classification.
It supports custom configurations via YAML file and command-line arguments.

Usage:
    # Basic usage
    python train_aircraft_classifier.py

    # Custom parameters
    python train_aircraft_classifier.py --epochs 100 --batch-size 32 --imgsz 224 --device 0

    # Resume from checkpoint
    python train_aircraft_classifier.py --resume checkpoints/stage2/last.pt
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


def setup_logging(log_dir: Path) -> logging.Logger:
    """
    Configure structured logging for the training process.

    Args:
        log_dir: Directory to save log files.

    Returns:
        Configured logger instance.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger("AircraftClassifier")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler - detailed logs
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - info level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to the configuration YAML file.

    Returns:
        Dictionary containing configuration parameters.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for training configuration.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8 model for aircraft type classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n-cls.pt',
        help='Pre-trained model path or model name'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    # Data arguments
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/aircraft',
        help='Path to dataset root directory'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=224,
        help='Input image size'
    )

    # Optimizer arguments
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.001,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='AdamW',
        choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'],
        help='Optimizer type'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD momentum or Adam beta1'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='Weight decay (L2 regularization)'
    )

    # Learning rate scheduler
    parser.add_argument(
        '--cos-lr',
        action='store_true',
        default=True,
        help='Use cosine learning rate scheduler'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate fraction'
    )

    # Regularization
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout for classification head'
    )

    # Early stopping
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs without improvement)'
    )

    # Warmup
    parser.add_argument(
        '--warmup-epochs',
        type=float,
        default=3.0,
        help='Warmup epochs'
    )

    # Device and performance
    parser.add_argument(
        '--device',
        type=str,
        default='0',
        help='Device to use (e.g., 0, 1, cpu, mps)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of dataloader workers'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='Use Automatic Mixed Precision training'
    )

    # Reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Saving
    parser.add_argument(
        '--project',
        type=str,
        default='runs/classify',
        help='Project name for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='aircraft_classifier',
        help='Experiment name'
    )
    parser.add_argument(
        '--save-period',
        type=int,
        default=-1,
        help='Save checkpoint every N epochs (-1 to disable)'
    )

    # Checkpoint directory (custom)
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/stage2',
        help='Directory to save custom checkpoints'
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='configs/aircraft_classify.yaml',
        help='Path to configuration YAML file'
    )

    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs',
        help='Directory to save log files'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        default=True,
        help='Enable TensorBoard logging'
    )

    # Validation
    parser.add_argument(
        '--val',
        action='store_true',
        default=True,
        help='Run validation during training'
    )

    # Plots
    parser.add_argument(
        '--plots',
        action='store_true',
        default=True,
        help='Save plots and images during training'
    )

    return parser.parse_args()


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Merge configuration from YAML file with command-line arguments.

    Command-line arguments take precedence over configuration file values.

    Args:
        config: Configuration dictionary from YAML file.
        args: Parsed command-line arguments.

    Returns:
        Merged configuration dictionary.
    """
    # Create a mapping from argument names to config keys
    arg_to_config = {
        'model': 'model',
        'data': 'data',
        'epochs': 'epochs',
        'batch_size': 'batch_size',
        'imgsz': 'imgsz',
        'lr0': 'lr0',
        'optimizer': 'optimizer',
        'momentum': 'momentum',
        'weight_decay': 'weight_decay',
        'cos_lr': 'cos_lr',
        'lrf': 'lrf',
        'dropout': 'dropout',
        'patience': 'patience',
        'warmup_epochs': 'warmup_epochs',
        'device': 'device',
        'workers': 'workers',
        'amp': 'amp',
        'seed': 'seed',
        'project': 'project',
        'name': 'name',
        'save_period': 'save_period',
        'val': 'val',
        'plots': 'plots',
    }

    # Update config with command-line arguments
    for arg_name, config_key in arg_to_config.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            config[config_key] = arg_value

    return config


def save_custom_checkpoint(
    model: YOLO,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    val_acc: float,
    checkpoint_path: Path,
    is_best: bool = False
) -> None:
    """
    Save custom checkpoint with specific format.

    Args:
        model: YOLO model instance.
        epoch: Current epoch number.
        optimizer: Optimizer instance.
        val_acc: Validation accuracy.
        checkpoint_path: Path to save the checkpoint.
        is_best: Whether this is the best model so far.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the underlying PyTorch model
    pt_model = model.model

    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': pt_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'model_args': pt_model.args if hasattr(pt_model, 'args') else {},
        'names': pt_model.names if hasattr(pt_model, 'names') else {},
        'date': datetime.now().isoformat(),
    }

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save best checkpoint separately
    if is_best:
        best_path = checkpoint_path.parent / 'best.pt'
        torch.save(checkpoint, best_path)


class AircraftClassifierTrainer:
    """
    Wrapper class for training aircraft classifier with custom logging and checkpointing.
    """

    def __init__(self, config: Dict[str, Any], args: argparse.Namespace, logger: logging.Logger):
        """
        Initialize the aircraft classifier trainer.

        Args:
            config: Training configuration dictionary.
            args: Parsed command-line arguments.
            logger: Logger instance.
        """
        self.config = config
        self.args = args
        self.logger = logger
        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.best_val_acc = 0.0

        # Setup directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model
        self._init_model()

        # Setup TensorBoard
        self.tb_writer = None
        if args.tensorboard:
            self._setup_tensorboard()

    def _init_model(self) -> None:
        """Initialize the YOLO model for training."""
        if self.args.resume:
            self.logger.info(f"Resuming training from checkpoint: {self.args.resume}")
            self.model = YOLO(self.args.resume)
        else:
            self.logger.info(f"Loading pre-trained model: {self.config['model']}")
            self.model = YOLO(self.config['model'])

    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard writer for logging."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = Path(self.args.log_dir) / 'tensorboard' / datetime.now().strftime('%Y%m%d_%H%M%S')
            self.tb_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard logging enabled: {log_dir}")
        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.tb_writer = None

    def log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """
        Log training metrics to both file and TensorBoard.

        Args:
            metrics: Dictionary of metric names and values.
            epoch: Current epoch number.
        """
        # Log to file
        metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {epoch} - {metric_str}")

        # Log to TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f'train/{key}', value, epoch)
            self.tb_writer.flush()

    def train(self) -> None:
        """
        Execute the training process.

        This method runs the complete training loop including validation,
        checkpointing, and logging.
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Aircraft Classifier Fine-tuning")
        self.logger.info("=" * 60)

        # Log configuration
        self.logger.info("Configuration:")
        for key, value in self.config.items():
            self.logger.info(f"  {key}: {value}")

        # Prepare training arguments for YOLO
        train_args = {
            'data': self.config['data'],
            'epochs': self.config['epochs'],
            'batch': self.config['batch_size'],
            'imgsz': self.config['imgsz'],
            'lr0': self.config['lr0'],
            'optimizer': self.config['optimizer'],
            'momentum': self.config['momentum'],
            'weight_decay': self.config['weight_decay'],
            'cos_lr': self.config['cos_lr'],
            'lrf': self.config['lrf'],
            'dropout': self.config['dropout'],
            'patience': self.config['patience'],
            'warmup_epochs': self.config['warmup_epochs'],
            'device': self.config['device'],
            'workers': self.config['workers'],
            'amp': self.config['amp'],
            'seed': self.config['seed'],
            'project': self.config['project'],
            'name': self.config['name'],
            'save_period': self.config['save_period'],
            'val': self.config['val'],
            'plots': self.config['plots'],
            'verbose': True,
        }

        # Start training
        self.logger.info("Starting training...")
        start_time = datetime.now()

        try:
            # Train the model
            results = self.model.train(**train_args)

            # Log final results
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Training time: {datetime.now() - start_time}")

            if results:
                self.logger.info("Final metrics:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {key}: {value:.4f}")

            # Save final checkpoint
            if hasattr(self.model, 'trainer') and self.model.trainer:
                self._save_final_checkpoint()

        except Exception as e:
            self.logger.error(f"Training failed with error: {e}", exc_info=True)
            raise

        finally:
            # Close TensorBoard writer
            if self.tb_writer:
                self.tb_writer.close()
                self.logger.info("TensorBoard writer closed")

    def _save_final_checkpoint(self) -> None:
        """Save the final checkpoint after training completion."""
        trainer = self.model.trainer

        # Get validation accuracy from metrics
        val_acc = 0.0
        if hasattr(trainer, 'metrics') and trainer.metrics:
            # Try to find accuracy in metrics
            for key in ['metrics/accuracy_top1', 'accuracy_top1', 'acc', 'val_acc']:
                if key in trainer.metrics:
                    val_acc = float(trainer.metrics[key])
                    break

        # Get optimizer
        optimizer = trainer.optimizer if hasattr(trainer, 'optimizer') else None

        # Save last checkpoint
        last_path = self.checkpoint_dir / 'last.pt'
        if optimizer:
            save_custom_checkpoint(
                self.model,
                trainer.epoch,
                optimizer,
                val_acc,
                last_path,
                is_best=False
            )
            self.logger.info(f"Saved last checkpoint to: {last_path}")

        # Save best checkpoint if available
        best_path = self.checkpoint_dir / 'best.pt'
        if hasattr(trainer, 'best') and trainer.best.exists():
            import shutil
            shutil.copy(trainer.best, best_path)
            self.logger.info(f"Copied best checkpoint to: {best_path}")


def main() -> None:
    """
    Main entry point for the training script.

    This function:
    1. Parses command-line arguments
    2. Loads configuration from YAML file
    3. Merges config with command-line arguments
    4. Sets up logging
    5. Initializes and runs the trainer
    """
    # Parse arguments
    args = parse_arguments()

    # Load configuration from file
    config = load_config(args.config)

    # Merge config with command-line arguments
    config = merge_config_with_args(config, args)

    # Setup logging
    log_dir = Path(args.log_dir)
    logger = setup_logging(log_dir)

    # Log startup information
    logger.info("=" * 60)
    logger.info("Aircraft Classifier Training Script")
    logger.info("=" * 60)
    logger.info(f"Configuration file: {args.config}")
    logger.info(f"Dataset path: {config['data']}")
    logger.info(f"Model: {config['model']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['lr0']}")
    logger.info(f"Optimizer: {config['optimizer']}")
    logger.info("=" * 60)

    # Verify dataset paths
    train_path = Path(config['data']) / 'train'
    val_path = Path(config['data']) / 'val'

    if not train_path.exists():
        logger.warning(f"Training path not found: {train_path}")
    else:
        logger.info(f"Training path verified: {train_path}")

    if not val_path.exists():
        logger.warning(f"Validation path not found: {val_path}")
    else:
        logger.info(f"Validation path verified: {val_path}")

    # Initialize and run trainer
    try:
        trainer = AircraftClassifierTrainer(config, args, logger)
        trainer.train()

        logger.info("=" * 60)
        logger.info("Training script completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
