#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-Annotation Script for Aircraft Dataset

This script provides a complete automated pipeline for labeling
aircraft images using trained classification models.

Usage:
    python auto_annotate.py --config config/auto_annotate.yaml

Requirements:
    - YOLOv8 aircraft model: /home/wlx/yolo26x-cls-aircraft.pt
    - YOLOv8 airline model: /home/wlx/yolo26x-cls-airline.pt
    - Raw images: /mnt/disk/AeroVision/images

Output:
    - High confidence (>=95%): /mnt/disk/AeroVision/labeled/<class_name>/
    - New class candidates: /mnt/disk/AeroVision/filtered_new_class/
    - Low confidence (<95%): /mnt/disk/AeroVision/filtered_95/
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_annotate.pipeline import AutoAnnotatePipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(output_dir: Path) -> None:
    """
    Setup logging to file and console.

    Args:
        output_dir: Directory to save log file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = output_dir / f"auto_annotate_{timestamp}.log"

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)
    logger.info(f"Log file: {log_file}")


def load_config_from_yaml(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    except ImportError:
        logger.warning("PyYAML not installed, using default config")
        return get_default_config()
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using default")
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        # Data paths
        "raw_images_dir": "/mnt/disk/AeroVision/images",
        "labeled_dir": "/mnt/disk/AeroVision/labeled",
        "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class",
        "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95",

        # Model paths
        "aircraft_model_path": "/home/wlx/yolo26x-cls-aircraft.pt",
        "airline_model_path": "/home/wlx/yolo26x-cls-airline.pt",

        # Confidence thresholds
        "high_confidence_threshold": 0.95,
        "low_confidence_threshold": 0.80,

        # HDBSCAN parameters
        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True
        },

        # Inference parameters
        "device": "cpu",
        "batch_size": 32,
        "imgsz": 640
    }


def save_run_info(output_dir: Path, result: Dict[str, Any]) -> None:
    """
    Save run information to JSON file.

    Args:
        output_dir: Output directory
        result: Pipeline result dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "timestamp": datetime.now().isoformat(),
        "config": result.get("config", {}),
        "statistics": result.get("statistics", {}),
        "duration_seconds": result.get("duration_seconds", 0),
        "success": result.get("success", False)
    }

    if not result.get("success"):
        run_info["error"] = result.get("error", "Unknown error")

    output_file = output_dir / "run_info.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    logger.info(f"Run info saved to: {output_file}")


def print_summary(result: Dict[str, Any]) -> None:
    """
    Print pipeline execution summary.

    Args:
        result: Pipeline result dictionary
    """
    print("\n" + "=" * 60)
    print("AUTO-ANNOTATION PIPELINE SUMMARY")
    print("=" * 60)

    if not result.get("success"):
        print(f"Status: FAILED")
        print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        stats = result.get("statistics", {})
        print(f"Status: SUCCESS")
        print(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
        print("")
        print(f"Total images: {stats.get('total', 0)}")
        print(f"Auto-labeled (high confidence): {stats.get('high_confidence_count', 0)}")
        print(f"Manual review (filtered_95): {stats.get('filtered_95_count', 0)}")
        print(f"New class candidates: {stats.get('new_class_count', 0)}")
        print("")
        print(f"Auto-label rate: {stats.get('high_confidence_ratio', 0)*100:.1f}%")
        print(f"Manual review rate: {stats.get('filtered_95_ratio', 0)*100:.1f}%")
        print(f"New class rate: {stats.get('new_class_ratio', 0)*100:.1f}%")

        organizer_stats = result.get("organizer_stats", {})
        print("")
        print("File organization:")
        print(f"  Labeled: {organizer_stats.get('labeled_count', 0)}")
        print(f"  Filtered 95: {organizer_stats.get('filtered_95_count', 0)}")
        print(f"  New class: {organizer_stats.get('new_class_count', 0)}")
        print(f"  Skipped: {organizer_stats.get('skipped_count', 0)}")
        print(f"  Errors: {organizer_stats.get('error_count', 0)}")

    print("")
    print("Output directories:")
    print(f"  Labeled: {result.get('config', {}).get('labeled_dir')}")
    print(f"  Filtered 95: {result.get('config', {}).get('filtered_95_dir')}")
    print(f"  New class: {result.get('config', {}).get('filtered_new_class_dir')}")

    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Auto-annotation pipeline for aircraft dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python auto_annotate.py

  # Run with custom configuration file
  python auto_annotate.py --config /path/to/config.yaml

  # Run with custom paths
  python auto_annotate.py \\
      --raw-images /mnt/disk/AeroVision/images \\
      --labeled-dir /mnt/disk/AeroVision/labeled \\
      --aircraft-model /home/wlx/yolo26x-cls-aircraft.pt

  # Run on GPU
  python auto_annotate.py --device cuda:0
        """
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )

    # Data paths
    parser.add_argument(
        '--raw-images',
        type=str,
        default=None,
        help='Directory containing raw images'
    )
    parser.add_argument(
        '--labeled-dir',
        type=str,
        default=None,
        help='Output directory for high confidence images'
    )
    parser.add_argument(
        '--filtered-new-class-dir',
        type=str,
        default=None,
        help='Output directory for new class candidates'
    )
    parser.add_argument(
        '--filtered-95-dir',
        type=str,
        default=None,
        help='Output directory for low confidence images'
    )

    # Model paths
    parser.add_argument(
        '--aircraft-model',
        type=str,
        default=None,
        help='Path to aircraft classification model'
    )
    parser.add_argument(
        '--airline-model',
        type=str,
        default=None,
        help='Path to airline classification model'
    )

    # Confidence thresholds
    parser.add_argument(
        '--high-threshold',
        type=float,
        default=None,
        help='High confidence threshold (default: 0.95)'
    )
    parser.add_argument(
        '--low-threshold',
        type=float,
        default=None,
        help='Low confidence threshold (default: 0.80)'
    )

    # Inference parameters
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for inference (e.g., cpu, cuda:0)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Image size for inference'
    )

    # Logging
    parser.add_argument(
        '--log-dir',
        type=str,
        default=None,
        help='Directory for log files'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = load_config_from_yaml(Path(args.config))
    else:
        # Try to load config.yaml from script directory
        script_dir = Path(__file__).parent
        default_config_path = script_dir / "config.yaml"
        if default_config_path.exists():
            config = load_config_from_yaml(default_config_path)
            logger.info(f"Using default config: {default_config_path}")
        else:
            config = get_default_config()
            logger.info("Using hardcoded default config")

    # Override configuration with command-line arguments
    if args.raw_images:
        config["raw_images_dir"] = args.raw_images
    if args.labeled_dir:
        config["labeled_dir"] = args.labeled_dir
    if args.filtered_new_class_dir:
        config["filtered_new_class_dir"] = args.filtered_new_class_dir
    if args.filtered_95_dir:
        config["filtered_95_dir"] = args.filtered_95_dir

    if args.aircraft_model:
        config["aircraft_model_path"] = args.aircraft_model
    if args.airline_model:
        config["airline_model_path"] = args.airline_model

    if args.high_threshold is not None:
        config["high_confidence_threshold"] = args.high_threshold
    if args.low_threshold is not None:
        config["low_confidence_threshold"] = args.low_threshold

    if args.device:
        config["device"] = args.device
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.imgsz:
        config["imgsz"] = args.imgsz

    # Setup logging
    log_dir = Path(args.log_dir) if args.log_dir else Path(config["labeled_dir"])
    setup_logging(log_dir)

    # Print configuration
    logger.info("=" * 60)
    logger.info("AUTO-ANNOTATION PIPELINE CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Raw images: {config['raw_images_dir']}")
    logger.info(f"Labeled dir: {config['labeled_dir']}")
    logger.info(f"Filtered new class: {config['filtered_new_class_dir']}")
    logger.info(f"Filtered 95: {config['filtered_95_dir']}")
    logger.info(f"Aircraft model: {config['aircraft_model_path']}")
    logger.info(f"Airline model: {config['airline_model_path']}")
    logger.info(f"High threshold: {config['high_confidence_threshold']}")
    logger.info(f"Low threshold: {config['low_confidence_threshold']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info("=" * 60)

    # Create pipeline
    pipeline = AutoAnnotatePipeline(config)

    # Load models
    pipeline.load_models()

    # Run pipeline
    result = pipeline.run()

    # Save result with config
    result["config"] = config

    # Save run info
    save_run_info(log_dir, result)

    # Save pipeline statistics
    pipeline.save_statistics(result)

    # Print summary
    print_summary(result)


if __name__ == '__main__':
    main()
