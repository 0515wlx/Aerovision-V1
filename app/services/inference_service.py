"""
推理服务核心模块
实现完整的推理流程：输入图片 -> 模型推理 -> 裁剪 -> OCR识别 -> 输出结果
"""

import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ultralytics import YOLO

# 导入现有的PaddleOCR模块
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "training" / "src"))
from ocr.paddle_ocr import PaddleOCRWrapper


class InferenceService:
    """推理服务类"""
    
    def __init__(self, config_path: str = "configs/inference.yaml"):
        """
        初始化推理服务
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.ocr = None
        self._load_models()
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _load_models(self):
        """加载模型"""
        # 加载YOLOv8检测模型
        detector_config = self.config['models']['detector']
        model_path = detector_config['path']
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(detector_config['device'])
        
        # 加载PaddleOCR
        ocr_config = self.config['ocr']
        self.ocr = PaddleOCRWrapper(
            use_angle_cls=ocr_config['use_angle_cls'],
            lang=ocr_config['lang'],
            use_gpu=ocr_config['use_gpu'],
            show_log=ocr_config['show_log']
        )
    
    def _crop_image(self, image: np.ndarray, bbox: List[float], padding: int = 10) -> np.ndarray:
        """
        根据边界框裁剪图片
        
        Args:
            image: 原始图片 (H, W, C)
            bbox: 边界框 [x_center, y_center, width, height] (YOLO格式)
            padding: 裁剪时的padding
            
        Returns:
            裁剪后的图片
        """
        h, w = image.shape[:2]
        
        # YOLO格式转换为像素坐标
        x_center, y_center, box_w, box_h = bbox
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)
        
        # 添加padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # 裁剪
        cropped = image[y1:y2, x1:x2]
        return cropped
    
    def infer(self, image_path: str) -> Dict[str, Any]:
        """
        执行完整的推理流程
        
        Args:
            image_path: 输入图片路径
            
        Returns:
            推理结果字典，包含：
            - detections: 检测结果列表（包含边界框、类别、置信度）
            - ocr_results: OCR识别结果列表
            - summary: 汇总信息
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 模型推理
        detector_config = self.config['models']['detector']
        results = self.model(
            image,
            conf=detector_config['conf_threshold'],
            iou=detector_config['iou_threshold'],
            max_det=detector_config['max_det']
        )
        
        # 处理检测结果
        detections = []
        ocr_results = []
        
        print(f"推理结果数量: {len(results)}")
        for i, result in enumerate(results):
            boxes = result.boxes
            print(f"结果 {i}: boxes={boxes}, type={type(boxes)}")
            if boxes is None:
                print(f"结果 {i}: boxes is None")
                continue
            print(f"结果 {i}: boxes length={len(boxes)}")
            for box in boxes:
                # 获取边界框、类别、置信度
                bbox_xywh = box.xywhn[0].cpu().numpy().tolist()  # [x_center, y_center, width, height]
                bbox_xyxy = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                detection = {
                    "bbox_xywh": bbox_xywh,
                    "bbox_xyxy": bbox_xyxy,
                    "class_id": cls_id,
                    "confidence": conf,
                    "class_name": self.model.names.get(cls_id, f"class_{cls_id}")
                }
                detections.append(detection)
                
                # 裁剪图片并进行OCR识别
                padding = self.config['image']['padding']
                cropped = self._crop_image(image, bbox_xywh, padding)
                
                # 只处理前3个检测框，避免OCR卡住
                if len(detections) >= 3:
                    print(f"跳过第 {len(detections)+1} 个检测框，只处理前3个")
                    ocr_text = ""
                    ocr_details = []
                    ocr_confidence = 0.0
                else:
                    # OCR识别 - 使用 ocr_text 方法
                    print(f"正在OCR识别检测框 {len(detections)+1}...")
                    ocr_text = self.ocr.ocr_text(cropped)
                    ocr_details = self.ocr.ocr_text_with_boxes(cropped)
                
                # 获取置信度（如果有识别结果）
                ocr_confidence = 0.0
                if ocr_details:
                    # 计算平均置信度
                    confidences = [item['confidence'] for item in ocr_details]
                    ocr_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                ocr_results.append({
                    "detection_index": len(detections) - 1,
                    "ocr_text": ocr_text,
                    "ocr_confidence": ocr_confidence,
                    "ocr_details": ocr_details
                })
        
        # 汇总结果
        summary = {
            "total_detections": len(detections),
            "image_path": image_path,
            "image_shape": image.shape
        }
        
        return {
            "detections": detections,
            "ocr_results": ocr_results,
            "summary": summary
        }
    
    def infer_batch(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            推理结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.infer(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "image_path": image_path
                })
        return results
