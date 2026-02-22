#!/usr/bin/env python3
"""
Example usage of auto-annotation pipeline.

This script demonstrates how to use the auto-annotation pipeline
programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_annotate.pipeline import AutoAnnotatePipeline


def example_basic_usage():
    """基本使用示例"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)

    # 配置
    config = {
        "raw_images_dir": "/mnt/disk/AeroVision/images",
        "labeled_dir": "/mnt/disk/AeroVision/labeled",
        "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class",
        "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95",

        "aircraft_model_path": "/home/wlx/yolo26x-cls-aircraft.pt",
        "airline_model_path": "/home/wlx/yolo26x-cls-airline.pt",

        "high_confidence_threshold": 0.95,
        "low_confidence_threshold": 0.80,

        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3,
            "metric": "euclidean"
        },

        "device": "cpu",
        "batch_size": 32,
        "imgsz": 640
    }

    # 创建流水线
    pipeline = AutoAnnotatePipeline(config)

    # 加载模型
    pipeline.load_models()

    # 运行流水线
    result = pipeline.run()

    # 打印结果
    print(f"\nPipeline completed successfully!")
    print(f"Total images: {result['statistics']['total']}")
    print(f"Auto-labeled: {result['statistics']['high_confidence_count']}")
    print(f"Manual review: {result['statistics']['filtered_95_count']}")
    print(f"New class candidates: {result['statistics']['new_class_count']}")
    print(f"Duration: {result['duration_seconds']:.2f} seconds")

    return result


def example_custom_thresholds():
    """自定义置信度阈值示例"""
    print("\n" + "="*60)
    print("Example 2: Custom Confidence Thresholds")
    print("="*60)

    # 使用更严格的阈值
    config = {
        "raw_images_dir": "/mnt/disk/AeroVision/images",
        "labeled_dir": "/mnt/disk/AeroVision/labeled_strict",
        "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class_strict",
        "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95_strict",

        "aircraft_model_path": "/home/wlx/yolo26x-cls-aircraft.pt",
        "airline_model_path": "/home/wlx/yolo26x-cls-airline.pt",

        # 更严格的阈值：需要98%的置信度才自动标注
        "high_confidence_threshold": 0.98,
        "low_confidence_threshold": 0.85,

        "hdbscan": {
            "min_cluster_size": 10,
            "min_samples": 5,
            "metric": "cosine"  # 使用余弦距离
        },

        "device": "cpu",
        "batch_size": 32,
        "imgsz": 640
    }

    pipeline = AutoAnnotatePipeline(config)
    pipeline.load_models()
    result = pipeline.run()

    print(f"\nStrict mode completed!")
    print(f"Auto-labeled (≥98%): {result['statistics']['high_confidence_count']}")
    print(f"Manual review: {result['statistics']['filtered_95_count']}")

    return result


def example_gpu_usage():
    """GPU使用示例"""
    print("\n" + "="*60)
    print("Example 3: GPU Usage")
    print("="*60)

    # 使用GPU进行加速
    config = {
        "raw_images_dir": "/mnt/disk/AeroVision/images",
        "labeled_dir": "/mnt/disk/AeroVision/labeled_gpu",
        "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class_gpu",
        "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95_gpu",

        "aircraft_model_path": "/home/wlx/yolo26x-cls-aircraft.pt",
        "airline_model_path": "/home/wlx/yolo26x-cls-airline.pt",

        "high_confidence_threshold": 0.95,
        "low_confidence_threshold": 0.80,

        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3
        },

        # 使用GPU
        "device": "cuda:0",
        "batch_size": 64,  # GPU可以处理更大的批量
        "imgsz": 640
    }

    try:
        pipeline = AutoAnnotatePipeline(config)
        pipeline.load_models()
        result = pipeline.run()

        print(f"\nGPU mode completed!")
        print(f"Duration: {result['duration_seconds']:.2f} seconds")

        return result
    except Exception as e:
        print(f"\nGPU mode failed: {e}")
        print("Please ensure CUDA is available or use CPU mode.")
        return None


def example_batch_processing():
    """批量处理示例"""
    print("\n" + "="*60)
    print("Example 4: Batch Processing with Step-by-Step Control")
    print("="*60)

    config = {
        "raw_images_dir": "/mnt/disk/AeroVision/images",
        "labeled_dir": "/mnt/disk/AeroVision/labeled_batch",
        "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class_batch",
        "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95_batch",

        "aircraft_model_path": "/home/wlx/yolo26x-cls-aircraft.pt",
        "airline_model_path": "/home/wlx/yolo26x-cls-airline.pt",

        "high_confidence_threshold": 0.95,
        "low_confidence_threshold": 0.80,

        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3
        },

        "device": "cpu",
        "batch_size": 32,
        "imgsz": 640
    }

    pipeline = AutoAnnotatePipeline(config)
    pipeline.load_models()

    # 步骤1：收集图片
    print("\nStep 1: Collecting images...")
    image_files = pipeline._collect_image_files()
    print(f"Found {len(image_files)} images")

    # 步骤2：预测
    print("\nStep 2: Running predictions...")
    predictions = pipeline.predict_batch(image_files)
    print(f"Generated {len(predictions)} predictions")

    # 步骤3：检测新类别
    print("\nStep 3: Detecting new classes...")
    embeddings = pipeline._embeddings
    new_class_indices = pipeline._detect_new_classes()
    print(f"Found {len(new_class_indices)} new class candidates")

    # 步骤4：置信度过滤
    print("\nStep 4: Filtering by confidence...")
    filtered = pipeline._filter_by_confidence(predictions)
    print(f"High confidence: {len(filtered['high_confidence'])}")
    print(f"Medium confidence: {len(filtered['medium_confidence'])}")
    print(f"Low confidence: {len(filtered['low_confidence'])}")

    # 步骤5：组织文件
    print("\nStep 5: Organizing files...")
    pipeline.file_organizer.organize_labeled_images(filtered["high_confidence"])
    pipeline.file_organizer.organize_new_class_images(
        [predictions[i] for i in new_class_indices]
    )
    pipeline.file_organizer.organize_filtered_95_images(
        filtered["medium_confidence"] + filtered["low_confidence"]
    )

    # 步骤6：保存详情
    print("\nStep 6: Saving prediction details...")
    pipeline._save_prediction_details(
        predictions, new_class_indices, "new_class"
    )
    pipeline._save_prediction_details(
        filtered["medium_confidence"] + filtered["low_confidence"],
        None, "filtered_95"
    )

    print("\nBatch processing completed!")

    return {
        "predictions": predictions,
        "new_class_indices": new_class_indices,
        "filtered": filtered
    }


def example_verify_output():
    """验证输出示例"""
    print("\n" + "="*60)
    print("Example 5: Verify Output")
    print("="*60)

    config = {
        "raw_images_dir": "/mnt/disk/AeroVision/images",
        "labeled_dir": "/mnt/disk/AeroVision/labeled",
        "filtered_new_class_dir": "/mnt/disk/AeroVision/filtered_new_class",
        "filtered_95_dir": "/mnt/disk/AeroVision/filtered_95",

        "aircraft_model_path": "/home/wlx/yolo26x-cls-aircraft.pt",
        "airline_model_path": "/home/wlx/yolo26x-cls-airline.pt",

        "high_confidence_threshold": 0.95,
        "low_confidence_threshold": 0.80,

        "hdbscan": {
            "min_cluster_size": 5,
            "min_samples": 3
        },

        "device": "cpu",
        "batch_size": 32,
        "imgsz": 640
    }

    pipeline = AutoAnnotatePipeline(config)

    # 验证输出
    print("\nVerifying output directories...")
    verification = pipeline.verify_output()

    print(f"\nLabeled directory:")
    print(f"  Number of classes: {verification['labeled'].get('num_classes', 0)}")
    print(f"  Total images: {verification['labeled'].get('total_images', 0)}")
    if verification['labeled'].get('classes'):
        print(f"  Classes: {verification['labeled']['classes'][:5]}...")

    print(f"\nFiltered new class directory:")
    print(f"  Number of images: {verification['new_class'].get('num_images', 0)}")

    print(f"\nFiltered 95 directory:")
    print(f"  Number of images: {verification['filtered_95'].get('num_images', 0)}")

    return verification


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Auto-Annotation Pipeline Examples")
    print("="*60)
    print("\nThese examples demonstrate various usage patterns.")
    print("Note: Actual execution requires model files and images.")
    print("\nUncomment the example you want to run:")
    print("  - example_basic_usage()")
    print("  - example_custom_thresholds()")
    print("  - example_gpu_usage()")
    print("  - example_batch_processing()")
    print("  - example_verify_output()")
    print("="*60)

    # 取消注释以运行示例

    # result = example_basic_usage()
    # result = example_custom_thresholds()
    # result = example_gpu_usage()
    # result = example_batch_processing()
    # result = example_verify_output()
