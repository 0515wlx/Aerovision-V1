# 阶段 7：联合集成与部署

> ⏱️ 预计时间：2 天
> 🎯 目标：将所有模块整合成完整的推理 Pipeline，进行置信度校准
> 📌 核心概念：模型集成、置信度校准、部署优化

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 完整的推理 Pipeline（一个接口输出所有结果）
- [ ] 置信度校准（预测的置信度反映真实准确率）
- [ ] 性能优化（推理速度可接受）
- [ ] 可部署的模型文件

---

## 核心概念：为什么需要集成？

### 当前状态

经过前面的阶段，你有：
- **分类模型**：机型、航司预测
- **回归模型**：清晰度、遮挡程度预测
- **OCR Pipeline**：注册号检测 + 识别

### 目标状态

```
                    输入图片
                       │
                       ▼
              ┌────────────────┐
              │   统一 Pipeline │
              └────────┬───────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
┌────────┐       ┌──────────┐       ┌─────────┐
│ Hybrid │       │ Quality  │       │   OCR   │
│ Model  │       │ Model    │       │Pipeline │
└────┬───┘       └────┬─────┘       └────┬────┘
     │                │                  │
     ▼                ▼                  ▼
┌─────────────────────────────────────────────┐
│              统一输出                        │
│  {                                          │
│    "type": "B737-800",                      │
│    "type_confidence": 0.92,                 │
│    "airline": "China Eastern",              │
│    "airline_confidence": 0.85,              │
│    "clarity": 0.88,                         │
│    "block": 0.12,                           │
│    "registration": "B-1234",                │
│    "registration_confidence": 0.78          │
│  }                                          │
└─────────────────────────────────────────────┘
```

---

## 第一步：创建统一推理接口

### 1.1 完整 Pipeline 类

```python
# training/src/inference/pipeline.py
"""完整推理 Pipeline"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import json

class AircraftRecognitionPipeline:
    """
    飞机识别完整 Pipeline
    
    整合所有模型，提供统一的推理接口
    """
    
    def __init__(
        self,
        model_path: str,
        ocr_detector_path: str = None,
        class_names_dir: str = None,
        device: str = None,
        use_ocr: bool = True
    ):
        """
        初始化 Pipeline
        
        Args:
            model_path: 主模型权重路径（Hybrid 或 FullMultiTask）
            ocr_detector_path: OCR 检测模型路径
            class_names_dir: 类别名称文件目录
            device: 运行设备
            use_ocr: 是否启用 OCR
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_ocr = use_ocr
        
        print(f"初始化 Pipeline, 设备: {self.device}")
        
        # ===== 加载主模型 =====
        self._load_main_model(model_path)
        
        # ===== 加载类别名称 =====
        self._load_class_names(class_names_dir)
        
        # ===== 加载 OCR Pipeline =====
        if use_ocr and ocr_detector_path:
            self._load_ocr(ocr_detector_path)
        else:
            self.ocr_pipeline = None
        
        # ===== 图像预处理 =====
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print("✅ Pipeline 初始化完成")
    
    def _load_main_model(self, model_path: str):
        """加载主模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', {})
        
        # 根据配置决定模型类型
        if 'transformer_backbone' in config:
            # Hybrid 模型
            from models.hybrid import HybridModel
            self.model = HybridModel(
                num_types=config.get('num_types', 10),
                num_airlines=config.get('num_airlines', 12),
                cnn_backbone=config.get('cnn_backbone', 'convnext_base'),
                transformer_backbone=config.get('transformer_backbone', 'swin_base_patch4_window7_224'),
                fusion_method=config.get('fusion_method', 'concat')
            )
        else:
            # FullMultiTask 模型
            from models.full_model import FullMultiTaskModel
            self.model = FullMultiTaskModel(
                num_types=config.get('num_types', 10),
                num_airlines=config.get('num_airlines', 12),
                backbone_name=config.get('backbone', 'convnext_base')
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  主模型加载完成: {model_path}")
    
    def _load_class_names(self, class_names_dir: str):
        """加载类别名称"""
        if class_names_dir:
            dir_path = Path(class_names_dir)
            
            with open(dir_path / 'type_classes.json', 'r') as f:
                self.type_names = json.load(f)['classes']
            
            with open(dir_path / 'airline_classes.json', 'r') as f:
                self.airline_names = json.load(f)['classes']
        else:
            self.type_names = [f"type_{i}" for i in range(10)]
            self.airline_names = [f"airline_{i}" for i in range(12)]
        
        print(f"  类别: {len(self.type_names)} 机型, {len(self.airline_names)} 航司")
    
    def _load_ocr(self, detector_path: str):
        """加载 OCR Pipeline"""
        try:
            from ocr.pipeline import RegistrationPipeline
            self.ocr_pipeline = RegistrationPipeline(
                detector_path=detector_path,
                use_gpu=(self.device == 'cuda')
            )
            print(f"  OCR Pipeline 加载完成")
        except Exception as e:
            print(f"  ⚠️ OCR 加载失败: {e}")
            self.ocr_pipeline = None
    
    def preprocess(self, image):
        """预处理图片"""
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert('RGB')
        
        tensor = self.transform(image).unsqueeze(0)  # [1, 3, 224, 224]
        return tensor.to(self.device), image
    
    @torch.no_grad()
    def predict(self, image, include_ocr: bool = True):
        """
        预测单张图片
        
        Args:
            image: PIL Image、numpy array 或文件路径
            include_ocr: 是否包含 OCR 结果
        
        Returns:
            dict: 预测结果
        """
        # 预处理
        tensor, pil_image = self.preprocess(image)
        
        # 主模型推理
        outputs = self.model(tensor)
        
        # ===== 处理分类结果 =====
        type_probs = F.softmax(outputs['type'], dim=1)[0]
        type_conf, type_idx = type_probs.max(dim=0)
        type_name = self.type_names[type_idx.item()]
        
        airline_probs = F.softmax(outputs['airline'], dim=1)[0]
        airline_conf, airline_idx = airline_probs.max(dim=0)
        airline_name = self.airline_names[airline_idx.item()]
        
        # ===== 处理回归结果 =====
        clarity = outputs['clarity'][0].item()
        block = outputs['block'][0].item()
        
        # ===== 构建结果 =====
        result = {
            # 机型
            'type': type_name,
            'type_confidence': round(type_conf.item(), 4),
            'type_top5': self._get_top5(type_probs, self.type_names),
            
            # 航司
            'airline': airline_name,
            'airline_confidence': round(airline_conf.item(), 4),
            
            # 质量指标
            'clarity': round(clarity, 4),
            'block': round(block, 4),
            
            # 综合质量评分
            'quality_score': round(clarity * (1 - block), 4),
        }
        
        # ===== OCR =====
        if include_ocr and self.ocr_pipeline:
            ocr_result = self.ocr_pipeline.process(pil_image)
            result['registration'] = ocr_result['registration']
            result['registration_confidence'] = round(ocr_result['confidence'], 4)
            result['registration_detected'] = ocr_result['detected']
        else:
            result['registration'] = ''
            result['registration_confidence'] = 0.0
            result['registration_detected'] = False
        
        return result
    
    def _get_top5(self, probs, names):
        """获取 Top-5 预测"""
        top5_probs, top5_indices = probs.topk(5)
        return [
            {'name': names[idx.item()], 'confidence': round(prob.item(), 4)}
            for prob, idx in zip(top5_probs, top5_indices)
        ]
    
    def predict_batch(self, images, include_ocr: bool = True):
        """批量预测"""
        return [self.predict(img, include_ocr) for img in images]


def load_pipeline(
    model_path: str = "training/checkpoints/stage5/best.pth",
    ocr_detector_path: str = "training/checkpoints/stage6/registration_detector/weights/best.pt",
    class_names_dir: str = "training/data/labels"
):
    """便捷函数：加载 Pipeline"""
    return AircraftRecognitionPipeline(
        model_path=model_path,
        ocr_detector_path=ocr_detector_path,
        class_names_dir=class_names_dir
    )
```

### 1.2 使用示例

```python
# training/scripts/demo_pipeline.py
"""Pipeline 使用演示"""

import sys
sys.path.append('training/src')

from inference.pipeline import load_pipeline
from pathlib import Path
import json

def demo():
    # 加载 Pipeline
    pipeline = load_pipeline(
        model_path="training/checkpoints/stage5/best.pth",
        ocr_detector_path="training/checkpoints/stage6/registration_detector/weights/best.pt",
        class_names_dir="training/data/labels"
    )
    
    # 预测单张图片
    image_path = "training/data/processed/aircraft_crop/test/sample.jpg"
    result = pipeline.predict(image_path)
    
    print("\n预测结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 输出示例：
    # {
    #   "type": "B737-800",
    #   "type_confidence": 0.9234,
    #   "type_top5": [
    #     {"name": "B737-800", "confidence": 0.9234},
    #     {"name": "A320", "confidence": 0.0412},
    #     ...
    #   ],
    #   "airline": "China Eastern",
    #   "airline_confidence": 0.8567,
    #   "clarity": 0.8823,
    #   "block": 0.1245,
    #   "quality_score": 0.7724,
    #   "registration": "B-1234",
    #   "registration_confidence": 0.7821,
    #   "registration_detected": true
    # }


if __name__ == "__main__":
    demo()
```

---

## 第二步：置信度校准

### 2.1 为什么需要校准？

模型输出的 softmax 概率**不一定**反映真实准确率：
- 预测置信度 0.9 的样本，实际准确率可能只有 0.7
- 这对用户决策很重要（高置信度应该更可靠）

### 2.2 温度缩放校准

```python
# training/src/inference/calibration.py
"""置信度校准"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class TemperatureScaling(nn.Module):
    """
    温度缩放校准
    
    通过学习一个温度参数 T，使得 softmax(logits/T) 更好地反映真实概率
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, model, val_loader, device='cuda'):
        """
        在验证集上学习最优温度
        
        Args:
            model: 训练好的模型
            val_loader: 验证数据加载器
            device: 设备
        """
        model.eval()
        self.to(device)
        
        # 收集所有 logits 和标签
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="收集预测"):
                images = images.to(device)
                outputs = model(images)
                
                # 假设是机型分类的 logits
                all_logits.append(outputs['type'].cpu())
                all_labels.append(labels['type'] if isinstance(labels, dict) else labels)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # 优化温度
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(all_logits.to(device))
            loss = criterion(scaled_logits, all_labels.to(device))
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        print(f"校准完成，最优温度: {self.temperature.item():.4f}")
        return self.temperature.item()


def evaluate_calibration(probs, labels, n_bins=10):
    """
    评估校准效果
    
    Returns:
        ECE: Expected Calibration Error
    """
    confidences = probs.max(dim=1)[0].numpy()
    predictions = probs.argmax(dim=1).numpy()
    labels = labels.numpy()
    
    accuracies = predictions == labels
    
    ece = 0.0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    
    return ece
```

### 2.3 应用校准

```python
# training/scripts/calibrate_model.py
"""校准模型置信度"""

import sys
sys.path.append('training/src')

import torch
from torch.utils.data import DataLoader
from inference.calibration import TemperatureScaling, evaluate_calibration

def calibrate():
    # 加载模型和数据
    # ... 省略加载代码 ...
    
    # 校准
    calibrator = TemperatureScaling()
    optimal_temp = calibrator.calibrate(model, val_loader)
    
    # 保存校准参数
    torch.save({
        'temperature': optimal_temp
    }, 'training/checkpoints/calibration.pth')
    
    print(f"校准参数已保存")

if __name__ == "__main__":
    calibrate()
```

---

## 第三步：性能优化

### 3.1 模型导出（ONNX）

```python
# training/scripts/export_onnx.py
"""导出 ONNX 模型"""

import torch
import sys
sys.path.append('training/src')

def export_to_onnx(model_path: str, output_path: str):
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型（根据你的模型类型）
    from models.full_model import FullMultiTaskModel
    model = FullMultiTaskModel(num_types=10, num_airlines=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 示例输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 导出
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['type', 'airline', 'clarity', 'block'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'type': {0: 'batch_size'},
            'airline': {0: 'batch_size'},
            'clarity': {0: 'batch_size'},
            'block': {0: 'batch_size'},
        },
        opset_version=14
    )
    
    print(f"✅ ONNX 模型导出到: {output_path}")

if __name__ == "__main__":
    export_to_onnx(
        model_path="training/checkpoints/stage5/best.pth",
        output_path="training/checkpoints/model.onnx"
    )
```

### 3.2 推理优化

```python
# training/src/inference/optimized.py
"""优化的推理代码"""

import torch
from torch.cuda.amp import autocast

class OptimizedPipeline:
    """优化的推理 Pipeline"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device).eval()
        self.device = device
        
        # 预热
        self._warmup()
    
    def _warmup(self):
        """预热 GPU"""
        dummy = torch.randn(1, 3, 224, 224).to(self.device)
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy)
        torch.cuda.synchronize()
    
    @torch.no_grad()
    def predict(self, tensor):
        """单张预测"""
        with autocast():  # 混合精度推理
            return self.model(tensor)
    
    @torch.no_grad()
    def predict_batch(self, tensors):
        """批量预测（更高效）"""
        with autocast():
            return self.model(tensors)
```

### 3.3 推理速度测试

```python
# training/scripts/benchmark.py
"""推理速度测试"""

import torch
import time
import sys
sys.path.append('training/src')

def benchmark(model_path: str, n_iterations: int = 100):
    # 加载模型
    from models.full_model import FullMultiTaskModel
    
    checkpoint = torch.load(model_path)
    model = FullMultiTaskModel(num_types=10, num_airlines=12)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda().eval()
    
    # 预热
    dummy = torch.randn(1, 3, 224, 224).cuda()
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy)
    torch.cuda.synchronize()
    
    # 测试
    times = []
    for _ in range(n_iterations):
        x = torch.randn(1, 3, 224, 224).cuda()
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # 转为毫秒
    
    avg_time = sum(times) / len(times)
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"FPS: {1000 / avg_time:.1f}")

if __name__ == "__main__":
    benchmark("training/checkpoints/stage5/best.pth")
```

---

## 第四步：最终测试

### 4.1 端到端测试

```python
# training/scripts/final_test.py
"""最终端到端测试"""

import sys
sys.path.append('training/src')

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from inference.pipeline import load_pipeline
import json

def final_test(test_csv: str, test_dir: str, output_path: str):
    """完整测试"""
    
    # 加载 Pipeline
    pipeline = load_pipeline()
    
    # 读取测试数据
    df = pd.read_csv(test_csv)
    
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="测试"):
        img_path = Path(test_dir) / row['filename']
        if not img_path.exists():
            continue
        
        # 预测
        pred = pipeline.predict(str(img_path))
        
        # 记录结果
        result = {
            'filename': row['filename'],
            # 真实值
            'gt_type': row['typename'],
            'gt_airline': row.get('airlinename', ''),
            'gt_clarity': row.get('clarity', 1.0),
            'gt_block': row.get('block', 0.0),
            'gt_registration': row.get('registration', ''),
            # 预测值
            'pred_type': pred['type'],
            'pred_type_conf': pred['type_confidence'],
            'pred_airline': pred['airline'],
            'pred_clarity': pred['clarity'],
            'pred_block': pred['block'],
            'pred_registration': pred['registration'],
            # 是否正确
            'type_correct': pred['type'] == row['typename'],
        }
        results.append(result)
    
    # 统计
    results_df = pd.DataFrame(results)
    
    type_acc = results_df['type_correct'].mean()
    clarity_mae = (results_df['pred_clarity'] - results_df['gt_clarity']).abs().mean()
    block_mae = (results_df['pred_block'] - results_df['gt_block']).abs().mean()
    
    # 注册号准确率
    reg_df = results_df[results_df['gt_registration'].notna() & (results_df['gt_registration'] != '')]
    if len(reg_df) > 0:
        reg_acc = (reg_df['pred_registration'] == reg_df['gt_registration']).mean()
    else:
        reg_acc = 0.0
    
    print("\n" + "=" * 60)
    print("最终测试结果")
    print("=" * 60)
    print(f"机型准确率: {type_acc:.2%}")
    print(f"清晰度 MAE: {clarity_mae:.4f}")
    print(f"遮挡 MAE: {block_mae:.4f}")
    print(f"注册号准确率: {reg_acc:.2%}")
    print("=" * 60)
    
    # 保存详细结果
    results_df.to_csv(output_path, index=False)
    print(f"\n详细结果保存到: {output_path}")
    
    return {
        'type_accuracy': type_acc,
        'clarity_mae': clarity_mae,
        'block_mae': block_mae,
        'registration_accuracy': reg_acc
    }


if __name__ == "__main__":
    final_test(
        test_csv="training/data/processed/aircraft_crop/test.csv",
        test_dir="training/data/processed/aircraft_crop/test",
        output_path="training/logs/final_test_results.csv"
    )
```

---

## ✅ 最终过关标准

恭喜你完成了整个训练流程！确保达到以下指标：

| 任务 | 指标 | 目标值 | 你的结果 |
|------|------|--------|----------|
| 机型分类 | Top-1 准确率 | > 85% | _____ |
| 机型分类 | Top-5 准确率 | > 95% | _____ |
| 航司分类 | 准确率 | > 75% | _____ |
| 清晰度 | MAE | < 0.15 | _____ |
| 遮挡程度 | MAE | < 0.15 | _____ |
| 注册号 | 完全正确率 | > 75% | _____ |
| 推理速度 | 单张耗时 | < 100ms | _____ |

---

## 📦 部署清单

### 需要的文件

```
deployment/
├── models/
│   ├── main_model.pth         # 主模型权重
│   ├── ocr_detector.pt        # OCR 检测模型
│   └── calibration.pth        # 校准参数
├── configs/
│   ├── type_classes.json      # 机型类别
│   └── airline_classes.json   # 航司类别
├── src/
│   ├── inference/             # 推理代码
│   └── ocr/                   # OCR 代码
└── requirements.txt           # 依赖
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
timm>=1.0.0
ultralytics>=8.0.0
paddlepaddle-gpu
paddleocr
pillow
numpy
```

---

## 🎉 恭喜完成！

你已经完成了从零开始训练飞机识别系统的完整流程：

1. ✅ 阶段 0：环境配置
2. ✅ 阶段 1：数据准备
3. ✅ 阶段 2：单任务训练
4. ✅ 阶段 3：多 Head 训练
5. ✅ 阶段 4：清晰度 + 遮挡
6. ✅ 阶段 5：Hybrid 融合
7. ✅ 阶段 6：OCR
8. ✅ 阶段 7：联合集成

---

## 🔄 后续优化方向

1. **增加数据量** - 更多数据通常能带来更好效果
2. **更大的模型** - 如果显存允许，尝试更大的 backbone
3. **知识蒸馏** - 用大模型指导小模型，提升推理速度
4. **持续学习** - 随着新机型出现，持续更新模型
5. **主动学习** - 选择模型最不确定的样本进行标注

祝你的飞机识别系统越来越强大！🚀✈️

