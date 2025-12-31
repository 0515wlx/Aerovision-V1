# 阶段 6：OCR 注册号识别

> ⏱️ 预计时间：3-4 天
> 🎯 目标：实现飞机注册号的检测和识别
> 📌 核心概念：检测 + OCR 两阶段流程

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 注册号区域检测模型（YOLOv8）
- [ ] OCR 识别能力（PaddleOCR 或自训练）
- [ ] 完整的检测→识别 Pipeline
- [ ] 注册号完全正确率 > 75%

---

## 核心概念：为什么 OCR 单独做？

注册号识别与图像分类是完全不同的任务：

| 方面 | 图像分类 | OCR |
|------|----------|-----|
| 输入 | 整张图片 | 文字区域 |
| 输出 | 固定类别 | 可变长度字符串 |
| 难点 | 类间差异小 | 字符变形、遮挡 |
| 方法 | CNN/Transformer | 检测 + 序列识别 |

### 两阶段流程

```
原始图片
    │
    ▼
┌─────────────────┐
│  注册号检测      │  ← YOLOv8 或你标注的 registrationarea
│  (定位)         │
└────────┬────────┘
         │
    检测到的区域
         │
         ▼
┌─────────────────┐
│  OCR 识别       │  ← PaddleOCR / TrOCR / 自训练
│  (识别文字)     │
└────────┬────────┘
         │
         ▼
    "B-1234"
```

---

## 第一步：准备注册号检测数据

### 1.1 理解你的标注

你的数据中有 `registrationarea` 字段，格式是 YOLO 格式：
```
x_center y_center width height
0.85 0.65 0.12 0.04
```

### 1.2 生成 YOLO 检测数据集

```python
# training/scripts/prepare_registration_detection.py
"""准备注册号检测数据集"""

import pandas as pd
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def prepare_detection_dataset(
    csv_path: str,
    image_dir: str,
    output_dir: str
):
    """
    将标注转换为 YOLO 检测格式
    
    YOLO 格式目录结构：
    output_dir/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    output_path = Path(output_dir)
    
    # 创建目录
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 读取标注
    df = pd.read_csv(csv_path)
    
    # 只保留有注册号区域标注的
    df = df[df['registrationarea'].notna() & (df['registrationarea'] != '')]
    print(f"有注册号区域标注的图片: {len(df)} 张")
    
    if len(df) == 0:
        print("⚠️ 没有找到注册号区域标注！")
        print("  请确保 CSV 中有 registrationarea 列")
        return
    
    # 划分训练/验证
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    image_path = Path(image_dir)
    
    def process_split(split_df, split_name):
        count = 0
        for _, row in split_df.iterrows():
            img_file = image_path / row['filename']
            if not img_file.exists():
                continue
            
            # 复制图片
            dst_img = output_path / 'images' / split_name / row['filename']
            shutil.copy2(img_file, dst_img)
            
            # 创建标签文件
            # YOLO 格式: class_id x_center y_center width height
            label_content = f"0 {row['registrationarea']}\n"
            
            label_file = output_path / 'labels' / split_name / (Path(row['filename']).stem + '.txt')
            label_file.write_text(label_content)
            
            count += 1
        
        print(f"  {split_name}: {count} 张")
        return count
    
    print("\n生成检测数据集:")
    process_split(train_df, 'train')
    process_split(val_df, 'val')
    
    # 创建 YOLO 配置文件
    yaml_content = f"""
# Registration Detection Dataset
path: {output_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: registration
"""
    
    yaml_path = output_path / 'dataset.yaml'
    yaml_path.write_text(yaml_content)
    print(f"\n✅ 数据集配置: {yaml_path}")


if __name__ == "__main__":
    prepare_detection_dataset(
        csv_path="training/data/labels/aircraft_labels.csv",
        image_dir="training/data/processed/aircraft_crop/train",
        output_dir="training/data/registration_detection"
    )
```

---

## 第二步：训练注册号检测模型

### 2.1 使用 YOLOv8 训练

```python
# training/scripts/train_registration_detector.py
"""训练注册号检测模型"""

from ultralytics import YOLO
from pathlib import Path

def train_detector():
    # 加载预训练模型
    model = YOLO('yolov8m.pt')  # 中等大小
    
    # 训练
    results = model.train(
        data='training/data/registration_detection/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='registration_detector',
        project='training/checkpoints/stage6',
        
        # 数据增强（文字检测不要太激进的增强）
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5,
        translate=0.1,
        scale=0.2,
        fliplr=0.0,  # 不要左右翻转（会影响文字方向）
        flipud=0.0,
        mosaic=0.5,
    )
    
    print("\n训练完成！")
    print(f"最佳模型: training/checkpoints/stage6/registration_detector/weights/best.pt")


if __name__ == "__main__":
    train_detector()
```

### 2.2 测试检测效果

```python
# training/scripts/test_registration_detector.py
"""测试注册号检测"""

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

def test_detector(model_path: str, image_dir: str, n_samples: int = 10):
    model = YOLO(model_path)
    
    image_path = Path(image_dir)
    images = list(image_path.glob("*.jpg"))[:n_samples]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for ax, img_file in zip(axes, images):
        # 检测
        results = model(str(img_file), verbose=False)[0]
        
        # 显示结果
        img = Image.open(img_file)
        ax.imshow(img)
        
        # 画框
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().item()
            
            rect = plt.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                fill=False, color='red', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f'{conf:.2f}', color='red', fontsize=8)
        
        ax.set_title(img_file.name[:15])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('training/logs/registration_detection_test.png', dpi=150)
    print("结果保存到 training/logs/registration_detection_test.png")


if __name__ == "__main__":
    test_detector(
        model_path="training/checkpoints/stage6/registration_detector/weights/best.pt",
        image_dir="training/data/processed/aircraft_crop/val"
    )
```

---

## 第三步：OCR 识别

### 3.1 方案选择

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **PaddleOCR** | 开箱即用、效果好 | 模型较大 | 快速实现 |
| **EasyOCR** | 简单易用 | 准确率稍低 | 简单场景 |
| **TrOCR** | 效果好 | 需要微调 | 高精度需求 |
| **自训练 CRNN** | 完全可控 | 需要大量数据 | 特殊字体 |

### 3.2 使用 PaddleOCR（推荐）

```python
# training/src/ocr/paddle_ocr.py
"""使用 PaddleOCR 识别注册号"""

from paddleocr import PaddleOCR
import re

class RegistrationOCR:
    """注册号 OCR 识别器"""
    
    def __init__(self, use_gpu: bool = True):
        self.ocr = PaddleOCR(
            use_angle_cls=True,  # 使用方向分类
            lang='en',           # 英文（注册号是字母+数字）
            use_gpu=use_gpu,
            show_log=False
        )
        
        # 注册号格式正则（根据实际情况调整）
        # 中国: B-xxxx
        # 美国: N xxxxx
        # 欧洲: 各种格式
        self.patterns = [
            r'B-\d{4}',           # 中国
            r'B-\d{3}[A-Z]',      # 中国
            r'N\d{1,5}[A-Z]{0,2}', # 美国
            r'[A-Z]-[A-Z]{4}',    # 欧洲
            r'[A-Z]{2}-[A-Z]{3}', # 欧洲
        ]
    
    def recognize(self, image):
        """
        识别图片中的注册号
        
        Args:
            image: PIL Image 或 numpy array 或 文件路径
        
        Returns:
            str: 识别的注册号，未识别到返回空字符串
        """
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return ""
        
        # 合并所有识别文本
        texts = []
        for line in result[0]:
            text = line[1][0]  # 文本内容
            conf = line[1][1]  # 置信度
            if conf > 0.5:
                texts.append(text.upper().replace(' ', ''))
        
        full_text = ''.join(texts)
        
        # 尝试匹配注册号格式
        for pattern in self.patterns:
            match = re.search(pattern, full_text)
            if match:
                return match.group()
        
        # 如果没有匹配到标准格式，返回清理后的文本
        # 只保留字母、数字和连字符
        cleaned = re.sub(r'[^A-Z0-9-]', '', full_text)
        return cleaned if len(cleaned) >= 4 else ""
    
    def recognize_with_confidence(self, image):
        """识别并返回置信度"""
        result = self.ocr.ocr(image, cls=True)
        
        if not result or not result[0]:
            return "", 0.0
        
        texts = []
        confs = []
        for line in result[0]:
            texts.append(line[1][0])
            confs.append(line[1][1])
        
        full_text = ''.join(texts).upper().replace(' ', '')
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        
        # 清理
        cleaned = re.sub(r'[^A-Z0-9-]', '', full_text)
        
        return cleaned, avg_conf
```

### 3.3 完整 Pipeline

```python
# training/src/ocr/pipeline.py
"""注册号识别完整 Pipeline"""

from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

class RegistrationPipeline:
    """
    注册号识别 Pipeline
    
    流程: 检测 → 裁剪 → OCR
    """
    
    def __init__(
        self,
        detector_path: str,
        use_gpu: bool = True
    ):
        # 加载检测模型
        self.detector = YOLO(detector_path)
        
        # 加载 OCR
        from .paddle_ocr import RegistrationOCR
        self.ocr = RegistrationOCR(use_gpu=use_gpu)
        
        print("✅ Pipeline 初始化完成")
    
    def process(self, image, conf_threshold: float = 0.5):
        """
        处理单张图片
        
        Args:
            image: PIL Image、numpy array 或文件路径
        
        Returns:
            dict: {
                'registration': str,  # 识别结果
                'confidence': float,  # 置信度
                'bbox': list,         # 检测框 [x1, y1, x2, y2]
            }
        """
        # 加载图片
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # 检测注册号区域
        results = self.detector(img, verbose=False)[0]
        
        if len(results.boxes) == 0:
            return {
                'registration': '',
                'confidence': 0.0,
                'bbox': None,
                'detected': False
            }
        
        # 选择置信度最高的检测框
        best_idx = results.boxes.conf.argmax()
        box = results.boxes.xyxy[best_idx].cpu().numpy()
        det_conf = results.boxes.conf[best_idx].cpu().item()
        
        if det_conf < conf_threshold:
            return {
                'registration': '',
                'confidence': 0.0,
                'bbox': None,
                'detected': False
            }
        
        # 裁剪区域（稍微扩大一点）
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        padding = 0.1
        
        x1 = max(0, x1 - w * padding)
        y1 = max(0, y1 - h * padding)
        x2 = min(img.width, x2 + w * padding)
        y2 = min(img.height, y2 + h * padding)
        
        crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
        
        # OCR 识别
        text, ocr_conf = self.ocr.recognize_with_confidence(np.array(crop))
        
        # 综合置信度
        final_conf = det_conf * ocr_conf
        
        return {
            'registration': text,
            'confidence': final_conf,
            'bbox': [x1, y1, x2, y2],
            'detected': True
        }
    
    def process_batch(self, images):
        """批量处理"""
        return [self.process(img) for img in images]
```

---

## 第四步：评估 OCR 效果

```python
# training/scripts/evaluate_ocr.py
"""评估 OCR 效果"""

import sys
sys.path.append('training/src')

import pandas as pd
from pathlib import Path
from tqdm import tqdm

def evaluate_ocr(pipeline, csv_path: str, image_dir: str):
    """评估 OCR 准确率"""
    
    df = pd.read_csv(csv_path)
    
    # 只评估有注册号标注的
    df = df[df['registration'].notna() & (df['registration'] != '')]
    print(f"评估样本数: {len(df)}")
    
    correct = 0
    detected = 0
    char_correct = 0
    char_total = 0
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = Path(image_dir) / row['filename']
        if not img_path.exists():
            continue
        
        # 预测
        pred = pipeline.process(str(img_path))
        
        # 真实值
        gt = row['registration'].upper().replace(' ', '')
        pred_text = pred['registration']
        
        # 统计
        if pred['detected']:
            detected += 1
        
        if pred_text == gt:
            correct += 1
        
        # 字符级准确率
        for i, c in enumerate(gt):
            char_total += 1
            if i < len(pred_text) and pred_text[i] == c:
                char_correct += 1
        
        results.append({
            'filename': row['filename'],
            'ground_truth': gt,
            'prediction': pred_text,
            'correct': pred_text == gt,
            'confidence': pred['confidence']
        })
    
    # 统计
    total = len(df)
    detection_rate = detected / total if total > 0 else 0
    accuracy = correct / total if total > 0 else 0
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    
    print("\n" + "=" * 50)
    print("OCR 评估结果")
    print("=" * 50)
    print(f"检测率: {detection_rate:.2%} ({detected}/{total})")
    print(f"完全正确率: {accuracy:.2%} ({correct}/{total})")
    print(f"字符准确率: {char_accuracy:.2%}")
    
    # 保存详细结果
    results_df = pd.DataFrame(results)
    results_df.to_csv('training/logs/ocr_evaluation.csv', index=False)
    print(f"\n详细结果保存到: training/logs/ocr_evaluation.csv")
    
    # 显示错误样例
    errors = results_df[~results_df['correct']].head(10)
    if len(errors) > 0:
        print("\n错误样例:")
        for _, row in errors.iterrows():
            print(f"  {row['filename']}: GT={row['ground_truth']}, Pred={row['prediction']}")
    
    return accuracy, char_accuracy


if __name__ == "__main__":
    from ocr.pipeline import RegistrationPipeline
    
    pipeline = RegistrationPipeline(
        detector_path="training/checkpoints/stage6/registration_detector/weights/best.pt"
    )
    
    evaluate_ocr(
        pipeline,
        csv_path="training/data/processed/aircraft_crop/test.csv",
        image_dir="training/data/processed/aircraft_crop/test"
    )
```

---

## ✅ 过关标准

- [ ] 注册号检测率 > 90%
- [ ] 注册号完全正确率 > 75%
- [ ] 字符准确率 > 95%
- [ ] Pipeline 能端到端运行

---

## ❌ 禁止事项

- ❌ 用分类方法做 OCR（把每个字符当成类别）
- ❌ 不做检测直接对整图 OCR
- ❌ 忽略注册号格式校验

---

## 💡 提升技巧

### 数据增强

```python
# 针对 OCR 的数据增强
import albumentations as A

ocr_augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 30), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.Perspective(scale=(0.02, 0.05), p=0.3),
])
```

### 后处理

```python
def postprocess_registration(text):
    """后处理注册号"""
    # 常见 OCR 错误修正
    corrections = {
        'O': '0',  # O → 0
        'I': '1',  # I → 1
        'S': '5',  # S → 5
        'Z': '2',  # Z → 2
    }
    
    # 只在数字位置做替换
    # 例如 B-1234 中的 1234 部分
    # ...
    
    return text
```

---

## 🔜 下一步

完成所有检查项后，进入 [阶段 7：联合集成](stage7_integration.md)

