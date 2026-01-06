#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注册号 OCR 识别器
使用 PP-OCRv4_server_rec_doc 模型
只对 YOLO 边界框裁剪区域进行识别

使用方法:
    python paddle_ocr.py <image_path> <bbox_txt>
"""

import os
import sys
import re
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

# 设置环境变量（必须最先设置）
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEX_LOG_LEVEL'] = 'ERROR'

# Patch importlib.util.find_spec 来阻止 torch 被检测到
import importlib.util
_original_find_spec = importlib.util.find_spec

def _patched_find_spec(name, *args, **kwargs):
    if name == 'torch':
        return None  # 假装 torch 不存在
    return _original_find_spec(name, *args, **kwargs)

importlib.util.find_spec = _patched_find_spec

import numpy as np
from PIL import Image
import cv2

# Monkey patch for PaddlePaddle 2.6.2 compatibility
try:
    import paddle
    if hasattr(paddle, 'base') and hasattr(paddle.base, 'libpaddle'):
        AnalysisConfig = paddle.base.libpaddle.AnalysisConfig
        if not hasattr(AnalysisConfig, 'set_optimization_level'):
            AnalysisConfig.set_optimization_level = lambda self, level: None
            print("[INFO] Applied PaddlePaddle 2.6.2 compatibility patch")
except Exception as e:
    print(f"[WARNING] Could not apply compatibility patch: {e}")

# 现在可以安全导入 paddleocr
from paddleocr import PaddleOCR


class RegistrationOCR:
    """
    注册号 OCR 识别器

    只对裁剪后的区域进行识别，提高速度和准确率
    """

    def __init__(
        self,
        lang: str = 'en',
        rec_model_name: str = 'PP-OCRv4_server_rec_doc',
    ):
        """
        初始化 OCR

        Args:
            lang: 语言 ('en' 或 'ch')
            rec_model_name: 识别模型名称
        """
        # 注册号配置
        self.whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        self.min_confidence = 0.5
        self.min_chars = 4
        self.max_chars = 10

        print(f"[INFO] 初始化 PaddleOCR...")
        print(f"[INFO] 识别模型: {rec_model_name}")
        print(f"[INFO] 语言: {lang}")

        self.ocr = PaddleOCR(
            lang=lang,
            text_recognition_model_name=rec_model_name,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print(f"[OK] PaddleOCR 初始化完成")

    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """加载图片为 numpy 数组"""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"无法读取图片: {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return image
        else:
            raise ValueError(f"不支持的图片类型: {type(image)}")

    def _crop_by_yolo_bbox(
        self,
        image: np.ndarray,
        bbox: List[float],
        padding: float = 0.1
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """根据 YOLO 格式边界框裁剪图片"""
        h, w = image.shape[:2]
        x_center, y_center, box_w, box_h = bbox

        # 转换为像素坐标
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # 添加 padding
        if padding > 0:
            pad_w = int((x2 - x1) * padding)
            pad_h = int((y2 - y1) * padding)
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)

        cropped = image[y1:y2, x1:x2]
        return cropped, (x1, y1, x2, y2)

    def _load_bbox_from_txt(self, txt_path: str, class_id: int = 0) -> List[List[float]]:
        """从 YOLO 格式 txt 文件读取边界框"""
        path = Path(txt_path)
        if not path.exists():
            return []

        bboxes = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                box_class = int(parts[0])
                if class_id is not None and box_class != class_id:
                    continue

                bbox = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]
                bboxes.append(bbox)

        return bboxes

    def _postprocess(self, text: str, confidence: float) -> Tuple[str, bool]:
        """后处理识别结果"""
        text = text.upper().replace(' ', '').replace('.', '').replace(',', '')
        text = ''.join(c for c in text if c in self.whitelist)

        if not (self.min_chars <= len(text) <= self.max_chars):
            return text, False

        if confidence < self.min_confidence:
            return text, False

        return text, True

    def _parse_ocr_result(self, result) -> Tuple[str, float]:
        """解析 OCR 结果"""
        if not result or not result[0]:
            return '', 0.0

        texts = []
        confs = []

        for line in result[0]:
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    texts.append(text_info[0])
                    confs.append(float(text_info[1]))
                elif isinstance(text_info, str):
                    texts.append(text_info)
                    confs.append(1.0)

        if not texts:
            return '', 0.0

        return ''.join(texts), sum(confs) / len(confs)

    def recognize(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bbox: Optional[List[float]] = None,
        bbox_txt: Optional[str] = None,
        class_id: int = 0,
        padding: float = 0.1,
    ) -> List[Dict]:
        """识别注册号"""
        img = self._load_image(image)

        # 获取边界框
        if bbox_txt:
            bboxes = self._load_bbox_from_txt(bbox_txt, class_id)
        elif bbox:
            bboxes = [bbox]
        else:
            bboxes = [[0.5, 0.5, 1.0, 1.0]]

        results = []

        for box in bboxes:
            cropped, pixel_bbox = self._crop_by_yolo_bbox(img, box, padding)

            if cropped.size == 0:
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'bbox': pixel_bbox,
                    'valid': False,
                })
                continue

            try:
                ocr_result = self.ocr.ocr(cropped)
            except Exception as e:
                print(f"[ERROR] OCR 识别失败: {e}")
                results.append({
                    'text': '',
                    'confidence': 0.0,
                    'bbox': pixel_bbox,
                    'valid': False,
                    'error': str(e),
                })
                continue

            text, confidence = self._parse_ocr_result(ocr_result)
            processed_text, is_valid = self._postprocess(text, confidence)

            results.append({
                'text': processed_text,
                'raw_text': text,
                'confidence': confidence,
                'bbox': pixel_bbox,
                'valid': is_valid,
            })

        return results

    def recognize_from_bbox(self, image, bbox, padding=0.1) -> Dict:
        """从单个边界框识别"""
        results = self.recognize(image, bbox=bbox, padding=padding)
        return results[0] if results else {'text': '', 'confidence': 0.0, 'valid': False}

    def recognize_from_txt(self, image, bbox_txt, class_id=0, padding=0.1) -> List[Dict]:
        """从 txt 文件批量识别"""
        return self.recognize(image, bbox_txt=bbox_txt, class_id=class_id, padding=padding)


# 兼容旧 API
PaddleOCRWrapper = RegistrationOCR


def create_ocr(
    lang: str = 'en',
    rec_model_name: str = 'PP-OCRv4_server_rec_doc',
    **kwargs
) -> RegistrationOCR:
    """创建 OCR 实例"""
    return RegistrationOCR(lang=lang, rec_model_name=rec_model_name)


if __name__ == '__main__':
    print("=" * 60)
    print("注册号 OCR 测试")
    print("=" * 60)

    if len(sys.argv) < 3:
        print("\n用法: python paddle_ocr.py <image_path> <bbox_txt>")
        print("示例: python paddle_ocr.py aircraft.jpg aircraft.txt")
        sys.exit(1)

    image_path = sys.argv[1]
    bbox_txt = sys.argv[2]

    if not Path(image_path).exists():
        print(f"[ERROR] 图片不存在: {image_path}")
        sys.exit(1)

    if not Path(bbox_txt).exists():
        print(f"[ERROR] txt 文件不存在: {bbox_txt}")
        sys.exit(1)

    print(f"\n图片: {Path(image_path).name}")
    print(f"边界框: {Path(bbox_txt).name}")

    # 读取边界框
    with open(bbox_txt, 'r') as f:
        lines = f.readlines()
        print(f"找到 {len(lines)} 个边界框")

    # 创建 OCR 实例
    print("\n创建 OCR 实例...")
    ocr = create_ocr(lang='en')

    # 识别
    print("\n开始识别...")
    results = ocr.recognize_from_txt(image_path, bbox_txt)

    print(f"\n识别结果 ({len(results)} 个):")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] 文本: '{r['text']}'")
        print(f"      原始: '{r.get('raw_text', '')}'")
        print(f"      置信度: {r['confidence']:.4f}")
        print(f"      有效: {r['valid']}")
        print(f"      位置: {r['bbox']}")
        print()

    print("=" * 60)
    print("测试完成")
    print("=" * 60)
