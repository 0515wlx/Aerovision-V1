"""
AeroVision-V1 配置系统快速使用示例
====================================

演示如何使用新的模块化配置系统
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_config


def example_basic_usage():
    """示例1: 基本使用"""
    print("=" * 60)
    print("示例1: 基本使用")
    print("=" * 60)

    # 加载默认配置（base.yaml + 所有模块）
    config = load_config()

    # 访问配置
    print(f"项目名称: {config.get('project.name')}")
    print(f"默认设备: {config.get('device.default')}")
    print(f"YOLO置信度阈值: {config.get('detection.conf_threshold')}")
    print(f"训练批次大小: {config.get('basic.batch_size')}")
    print(f"裁剪padding: {config.get('crop.padding')}")
    print()


def example_load_specific_modules():
    """示例2: 只加载特定模块"""
    print("=" * 60)
    print("示例2: 只加载特定模块（提高加载速度）")
    print("=" * 60)

    # 只加载需要的模块
    config = load_config(modules=['yolo', 'crop'], load_all_modules=False)

    print(f"YOLO模型: {config.get('model.size')}")
    print(f"YOLO置信度: {config.get('detection.conf_threshold')}")
    print(f"裁剪padding: {config.get('crop.padding')}")
    print()


def example_path_handling():
    """示例3: 路径处理（重要！）"""
    print("=" * 60)
    print("示例3: 路径处理 - 所有路径相对于/training/configs")
    print("=" * 60)

    config = load_config()

    # 使用get_path()获取绝对路径
    data_root = config.get_path('paths.data_root')
    model_path = config.get_path('paths.yolo_model')
    logs_root = config.get_path('logs.root')

    print(f"数据根目录: {data_root}")
    print(f"YOLO模型路径: {model_path}")
    print(f"日志目录: {logs_root}")
    print()
    print("说明:")
    print("  - 所有yaml中的相对路径都相对于 /training/configs 目录")
    print("  - 无论在哪里运行脚本，../data 都表示 /training/data")
    print("  - 使用 get_path() 自动转换为绝对路径")
    print()


def example_runtime_override():
    """示例4: 运行时覆盖配置"""
    print("=" * 60)
    print("示例4: 运行时覆盖配置")
    print("=" * 60)

    # 加载时覆盖配置
    config = load_config(
        device={'default': 'cpu'},
        detection={'conf_threshold': 0.8},
        basic={'batch_size': 64}
    )

    print(f"设备: {config.get('device.default')}")
    print(f"YOLO置信度: {config.get('detection.conf_threshold')}")
    print(f"批次大小: {config.get('basic.batch_size')}")
    print()


def example_practical_use_case():
    """示例5: 实际使用场景 - 裁剪飞机图片"""
    print("=" * 60)
    print("示例5: 实际使用场景 - 裁剪飞机图片")
    print("=" * 60)

    # 加载需要的模块
    config = load_config(modules=['yolo', 'crop', 'paths'])

    # 获取裁剪所需的所有配置
    input_dir = config.get_path('data.raw')
    output_dir = config.get_path('data.processed.aircraft_crop.unsorted')
    yolo_model = config.get('model.weights')
    conf_threshold = config.get('detection.conf_threshold')
    padding = config.get('crop.padding')
    min_size = config.get('crop.min_size')

    print("裁剪配置:")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  YOLO模型: {yolo_model}")
    print(f"  置信度阈值: {conf_threshold}")
    print(f"  边界框padding: {padding}")
    print(f"  最小尺寸: {min_size}")
    print()


def example_training_config():
    """示例6: 训练配置"""
    print("=" * 60)
    print("示例6: 训练配置")
    print("=" * 60)

    # 加载训练相关配置
    config = load_config(modules=['training', 'augmentation', 'paths'])

    # 训练参数
    batch_size = config.get('basic.batch_size')
    learning_rate = config.get('basic.learning_rate')
    num_epochs = config.get('basic.num_epochs')
    optimizer_type = config.get('optimizer.type')

    # 数据增强
    aug_enabled = config.get('augmentation.enabled')
    h_flip = config.get('geometric.horizontal_flip.enabled')
    rotation = config.get('geometric.rotation.enabled')

    print("训练配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  学习率: {learning_rate}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  优化器: {optimizer_type}")
    print()
    print("数据增强:")
    print(f"  启用: {aug_enabled}")
    print(f"  水平翻转: {h_flip}")
    print(f"  旋转: {rotation}")
    print()


if __name__ == "__main__":
    print("\n")
    print("#" * 60)
    print("# AeroVision-V1 配置系统使用示例")
    print("#" * 60)
    print()

    # 运行所有示例
    example_basic_usage()
    example_load_specific_modules()
    example_path_handling()
    example_runtime_override()
    example_practical_use_case()
    example_training_config()

    print("=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
    print()
    print("快速参考:")
    print("  1. 加载配置: config = load_config()")
    print("  2. 访问配置: config.get('key.subkey')")
    print("  3. 获取路径: config.get_path('paths.key')")
    print("  4. 覆盖配置: config = load_config(key={'subkey': value})")
    print()
    print("详细文档: training/configs/README.md")
    print()
