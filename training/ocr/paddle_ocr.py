#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 封装类
提供统一的OCR识别接口
"""

import os
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    import numpy as np
    from PIL import Image
except ImportError as e:
    raise ImportError(
        f"缺少必要的依赖库: {e}\n"
        "请安装: pip install paddlepaddle paddleocr"
    )


class PaddleOCRWrapper:
    """
    PaddleOCR 封装类
    
    支持中文和英文识别，返回识别结果和置信度
    """
    
    def __init__(
        self,
        use_angle_cls: bool = True,
        # TODO: 仅需识别英文字母 中划线 和 数字即可
        lang: str = 'ch',
        use_gpu: bool = False,
        show_log: bool = False,
        det_model_dir: Optional[str] = None,
        rec_model_dir: Optional[str] = None,
        cls_model_dir: Optional[str] = None,
    ):
        """
        初始化 PaddleOCR
        
        Args:
            use_angle_cls: 是否使用方向分类器
            lang: 语言类型 ('ch': 中文, 'en': 英文, 'japan': 日文等)
            use_gpu: 是否使用GPU（通过环境变量控制）
            show_log: 是否显示日志（已弃用，通过环境变量控制）
            det_model_dir: 检测模型路径
            rec_model_dir: 识别模型路径
            cls_model_dir: 分类模型路径
        """
        self.use_angle_cls = use_angle_cls
        self.lang = lang
        self.use_gpu = use_gpu
        
        # 通过环境变量控制设备使用
        # PaddleOCR 新版本使用环境变量控制 GPU/CPU
        if use_gpu:
            os.environ['FLAGS_use_gpu'] = 'true'
        else:
            os.environ['FLAGS_use_gpu'] = 'false'
        
        # 通过环境变量控制日志输出
        if not show_log:
            os.environ['FLAGS_ocr_debug_mode'] = '0'
        
        # 初始化 PaddleOCR（只传递支持的参数）
        ocr_params = {
            'use_angle_cls': use_angle_cls,
            'lang': lang,
        }
        
        # 添加可选的模型路径参数
        if det_model_dir:
            ocr_params['det_model_dir'] = det_model_dir
        if rec_model_dir:
            ocr_params['rec_model_dir'] = rec_model_dir
        if cls_model_dir:
            ocr_params['cls_model_dir'] = cls_model_dir
        
        self.ocr = PaddleOCR(**ocr_params)
        
        print(f"PaddleOCR 初始化完成 (语言: {lang}, GPU: {use_gpu})")
    
    def ocr_text(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_details: bool = False
    ) -> Union[str, List[Dict]]:
        """
        识别图片中的文字
        
        Args:
            image: 输入图片（文件路径、numpy数组或PIL Image）
            return_details: 是否返回详细信息
        
        Returns:
            如果 return_details=False: 返回识别的文字字符串
            如果 return_details=True: 返回详细信息列表
        """
        # 加载图片
        img_array = self._load_image(image)
        
        # 执行OCR（新版本不支持 cls 参数）
        result = self.ocr.ocr(img_array)
        
        if not result or result[0] is None:
            if return_details:
                return []
            return ""
        
        # 解析结果
        if return_details:
            return self._parse_result_details(result)
        else:
            return self._parse_result_text(result)
    
    def ocr_text_with_boxes(
        self,
        image: Union[str, np.ndarray, Image.Image]
    ) -> List[Dict]:
        """
        识别图片中的文字并返回文本框信息
        
        Args:
            image: 输入图片
        
        Returns:
            包含文本框和识别结果的列表
        """
        img_array = self._load_image(image)
        result = self.ocr.ocr(img_array)
        
        if not result or result[0] is None:
            return []
        
        return self._parse_result_details(result)
    
    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """
        加载图片为numpy数组
        
        Args:
            image: 输入图片
        
        Returns:
            numpy数组格式的图片
        """
        if isinstance(image, str):
            # 文件路径
            img = Image.open(image).convert('RGB')
            return np.array(img)
        elif isinstance(image, Image.Image):
            # PIL Image
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # numpy数组
            return image
        else:
            raise ValueError(f"不支持的图片类型: {type(image)}")
    
    def _parse_result_text(self, result: List) -> str:
        """
        解析OCR结果，返回纯文本
        
        Args:
            result: PaddleOCR原始结果
        
        Returns:
            识别的文字字符串
        """
        texts = []
        for line in result[0]:
            if line:
                text = line[1][0]
                texts.append(text)
        return ' '.join(texts)
    
    def _parse_result_details(self, result: List) -> List[Dict]:
        """
        解析OCR结果，返回详细信息
        
        Args:
            result: PaddleOCR原始结果
        
        Returns:
            包含详细信息的字典列表
        """
        details = []
        
        # 检查结果格式
        if not result or not result[0]:
            return details
        
        for line in result[0]:
            if line:
                box = line[0]  # 文本框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                text_info = line[1]  # (text, confidence) 或 text
                
                # 处理不同的返回格式
                if isinstance(text_info, (list, tuple)):
                    text = text_info[0]
                    confidence = float(text_info[1]) if len(text_info) > 1 else 0.0
                else:
                    text = str(text_info)
                    confidence = 0.0
                
                details.append({
                    'text': text,
                    'confidence': confidence,
                    'box': box,
                })
        return details
    
    def crop_text_region(
        self,
        image: Union[str, np.ndarray, Image.Image],
        box: List[List[int]]
    ) -> np.ndarray:
        """
        根据文本框坐标裁剪文字区域
        
        Args:
            image: 输入图片
            box: 文本框坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
        Returns:
            裁剪后的图片区域
        """
        img_array = self._load_image(image)
        
        # 计算最小外接矩形
        x_coords = [point[0] for point in box]
        y_coords = [point[1] for point in box]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        # 裁剪区域
        cropped = img_array[y_min:y_max, x_min:x_max]
        return cropped
    
    def batch_ocr(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        return_details: bool = False
    ) -> List[Union[str, List[Dict]]]:
        """
        批量识别图片中的文字
        
        Args:
            images: 输入图片列表
            return_details: 是否返回详细信息
        
        Returns:
            识别结果列表
        """
        results = []
        for image in images:
            result = self.ocr_text(image, return_details=return_details)
            results.append(result)
        return results
    
    def __call__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        return_details: bool = False
    ) -> Union[str, List[Dict]]:
        """
        使实例可调用
        
        Args:
            image: 输入图片
            return_details: 是否返回详细信息
        
        Returns:
            识别结果
        """
        return self.ocr_text(image, return_details=return_details)


def create_ocr(
    lang: str = 'ch',
    use_gpu: bool = False,
    use_angle_cls: bool = True
) -> PaddleOCRWrapper:
    """
    工厂函数：创建PaddleOCR实例
    
    Args:
        lang: 语言类型
        use_gpu: 是否使用GPU
        use_angle_cls: 是否使用方向分类器
    
    Returns:
        PaddleOCRWrapper实例
    """
    return PaddleOCRWrapper(
        lang=lang,
        use_gpu=use_gpu,
        use_angle_cls=use_angle_cls
    )


# 示例用法
if __name__ == '__main__':
    # 创建OCR实例
    ocr = create_ocr(lang='ch', use_gpu=False)

    # TODO: 图片传入方式为 numpy 数组，按照 txt 从对应图片中截取
    
    # 识别单张图片
    # result = ocr.ocr_text('test_image.jpg')
    # print(f"识别结果: {result}")
    
    # 获取详细信息
    # details = ocr.ocr_text_with_boxes('test_image.jpg')
    # for item in details:
    #     print(f"文字: {item['text']}, 置信度: {item['confidence']:.4f}")
