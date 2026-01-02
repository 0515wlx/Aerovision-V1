"""
推理测试脚本
测试推理服务的完整流程
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

# 导入推理服务
import sys
sys.path.append(str(Path(__file__).parent))
from services.inference_service import InferenceService


def print_result(result: dict):
    """打印推理结果"""
    print("\n" + "="*80)
    print("推理结果")
    print("="*80)
    
    # 打印汇总信息
    summary = result.get('summary', {})
    print(f"\n【汇总信息】")
    print(f"  图片路径: {summary.get('image_path', 'N/A')}")
    print(f"  图片尺寸: {summary.get('image_shape', 'N/A')}")
    print(f"  检测数量: {summary.get('total_detections', 0)}")
    
    # 打印检测结果
    print(f"\n【检测结果】")
    detections = result.get('detections', [])
    for i, det in enumerate(detections, 1):
        print(f"\n  检测 #{i}:")
        print(f"    类别ID: {det.get('class_id', 'N/A')}")
        print(f"    类别名称: {det.get('class_name', 'N/A')}")
        print(f"    置信度: {det.get('confidence', 0):.4f}")
        print(f"    边界框(YOLO格式): {det.get('bbox_xywh', [])}")
        print(f"    边界框(像素坐标): {det.get('bbox_xyxy', [])}")
    
    # 打印OCR结果
    print(f"\n【OCR识别结果】")
    ocr_results = result.get('ocr_results', [])
    for i, ocr in enumerate(ocr_results, 1):
        print(f"\n  OCR #{i} (对应检测 #{ocr.get('detection_index', 'N/A') + 1}):")
        print(f"    识别文字: {ocr.get('ocr_text', 'N/A')}")
        print(f"    置信度: {ocr.get('ocr_confidence', 0):.4f}")
    
    print("\n" + "="*80 + "\n")


def save_result(result: dict, output_path: str):
    """保存推理结果到JSON文件"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='推理服务测试脚本')
    parser.add_argument(
        'image_path',
        type=str,
        help='输入图片路径（从data目录中选择）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/inference.yaml',
        help='配置文件路径 (默认: configs/inference.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出JSON文件路径 (默认: 不保存)'
    )
    
    args = parser.parse_args()
    
    # 检查图片是否存在
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"错误: 图片不存在: {args.image_path}")
        return 1
    
    print(f"正在加载推理服务...")
    print(f"配置文件: {args.config}")
    
    try:
        # 初始化推理服务
        service = InferenceService(config_path=args.config)
        print("推理服务加载成功!")
        
        # 执行推理
        print(f"\n正在推理: {args.image_path}")
        result = service.infer(str(image_path))
        
        # 打印结果
        print_result(result)
        
        # 保存结果（如果指定了输出路径）
        if args.output:
            save_result(result, args.output)
        
        return 0
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
