#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示从边界框识别注册号的功能
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from ocr.paddle_ocr import create_ocr


def demo_bbox_recognition():
    """演示边界框识别功能"""

    print("=" * 60)
    print("边界框识别功能演示")
    print("=" * 60)

    # 创建OCR实例
    print("\n[1] 创建OCR实例...")
    try:
        ocr = create_ocr(
            lang='en',
            use_gpu=False,  # 演示使用CPU
            specialized_mode=True
        )
        print("    [OK] OCR实例创建成功")
    except Exception as e:
        print(f"    [ERROR] 创建失败: {e}")
        return

    # 示例1: 使用YOLO格式边界框
    print("\n[2] 示例1: 使用YOLO格式边界框")
    print("    假设图片: aircraft.jpg (1920x1080)")
    print("    注册号位置: 中心(1632, 702), 大小(230x43)")
    print("    YOLO格式: [0.85, 0.65, 0.12, 0.04]")

    # 模拟边界框
    bbox_yolo = [0.85, 0.65, 0.12, 0.04]
    print(f"    边界框: {bbox_yolo}")
    print("    说明: 这会裁剪出注册号区域，然后进行OCR识别")
    print("    优势: 速度快3-5倍，准确率更高（无干扰）")

    # 示例2: 从txt文件读取
    print("\n[3] 示例2: 从txt文件读取边界框")
    print("    txt文件格式 (YOLO):")
    print("    ```")
    print("    0 0.85 0.65 0.12 0.04")
    print("    ```")
    print("    说明: class_id x_center y_center width height")

    # 示例3: 识别多个边界框
    print("\n[4] 示例3: 识别多个边界框")
    print("    txt文件格式:")
    print("    ```")
    print("    0 0.85 0.65 0.12 0.04  # 第一个注册号")
    print("    0 0.45 0.32 0.10 0.03  # 第二个注册号")
    print("    ```")
    print("    使用 ocr_from_bbox_file() 可以一次识别所有")

    # 代码示例
    print("\n[5] 代码示例:")
    print("    ```python")
    print("    from ocr.paddle_ocr import create_ocr")
    print("")
    print("    # 创建OCR实例")
    print("    ocr = create_ocr(lang='en', use_gpu=True)")
    print("")
    print("    # 方式1: 直接提供边界框")
    print("    bbox = [0.85, 0.65, 0.12, 0.04]")
    print("    result = ocr.ocr_from_bbox('aircraft.jpg', bbox, bbox_format='yolo')")
    print("    print(result['text'])  # 输出: B-1234")
    print("")
    print("    # 方式2: 从txt文件读取")
    print("    results = ocr.ocr_from_bbox_file('aircraft.jpg', 'aircraft.txt')")
    print("    for r in results:")
    print("        print(f\"{r['text']} (置信度: {r['confidence']:.2f})\")")
    print("    ```")

    # 性能对比
    print("\n[6] 性能对比:")
    print("    ┌─────────────────┬──────────┬──────────┬──────────┐")
    print("    │ 方式            │ 处理区域 │ 速度     │ 准确率   │")
    print("    ├─────────────────┼──────────┼──────────┼──────────┤")
    print("    │ 全图识别        │ 整张图片 │ 慢 (1x)  │ 低       │")
    print("    │ 边界框识别      │ 注册号区 │ 快 (3-5x)│ 高       │")
    print("    └─────────────────┴──────────┴──────────┴──────────┘")

    # 推荐工作流程
    print("\n[7] 推荐工作流程:")
    print("    步骤1: 使用YOLOv8检测注册号区域")
    print("           → 生成 aircraft.txt (YOLO格式)")
    print("")
    print("    步骤2: 使用OCR识别注册号")
    print("           → ocr.ocr_from_bbox_file('aircraft.jpg', 'aircraft.txt')")
    print("")
    print("    步骤3: 获取识别结果")
    print("           → 文本、置信度、位置信息")

    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)
    print("\n提示: 如果你有实际的图片和txt文件，可以运行:")
    print("      python demo_bbox_ocr.py <image_path> <bbox_txt_path>")


def test_with_real_files(image_path: str, bbox_txt: str):
    """使用真实文件测试"""

    print("\n" + "=" * 60)
    print("使用真实文件测试")
    print("=" * 60)

    # 检查文件
    if not Path(image_path).exists():
        print(f"[ERROR] 图片不存在: {image_path}")
        return

    if not Path(bbox_txt).exists():
        print(f"[ERROR] txt文件不存在: {bbox_txt}")
        return

    # 创建OCR实例
    print("\n[1] 创建OCR实例...")
    try:
        ocr = create_ocr(lang='en', use_gpu=False, specialized_mode=True)
        print("    [OK] 创建成功")
    except Exception as e:
        print(f"    [ERROR] 创建失败: {e}")
        return

    # 读取txt文件内容
    print(f"\n[2] 读取边界框文件: {bbox_txt}")
    with open(bbox_txt, 'r') as f:
        lines = f.readlines()
        print(f"    找到 {len(lines)} 个边界框")
        for i, line in enumerate(lines[:5], 1):  # 只显示前5个
            print(f"    [{i}] {line.strip()}")
        if len(lines) > 5:
            print(f"    ... 还有 {len(lines) - 5} 个")

    # 识别
    print(f"\n[3] 识别图片: {image_path}")
    try:
        results = ocr.ocr_from_bbox_file(
            image=image_path,
            bbox_txt=bbox_txt,
            class_id=0,
            padding=0.1
        )

        print(f"    [OK] 识别完成，共 {len(results)} 个结果")

        # 显示结果
        print("\n[4] 识别结果:")
        for i, result in enumerate(results, 1):
            print(f"    [{i}] 文本: '{result['text']}'")
            print(f"        置信度: {result['confidence']:.4f}")
            print(f"        位置: {result['bbox']}")
            print(f"        大小: {result['cropped_size']}")
            print()

    except Exception as e:
        print(f"    [ERROR] 识别失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""

    if len(sys.argv) >= 3:
        # 使用真实文件测试
        image_path = sys.argv[1]
        bbox_txt = sys.argv[2]
        test_with_real_files(image_path, bbox_txt)
    else:
        # 演示功能
        demo_bbox_recognition()


if __name__ == '__main__':
    main()
