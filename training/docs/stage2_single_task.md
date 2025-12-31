# 阶段 2：单任务训练 - 机型分类

> ⏱️ 预计时间：2-3 天
> 🎯 目标：使用 ConvNeXt 完成机型分类任务，Top-1 准确率 > 80%
> 📌 核心原则：先跑通，再优化

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 能正常加载数据的 Dataset 类
- [ ] 能跑通的训练循环
- [ ] 验证集 Top-1 准确率 > 80%
- [ ] 保存的模型权重文件

---

## 第一步：创建 Dataset 类

### 1.1 理解 Dataset

PyTorch 的 Dataset 类负责：
1. 告诉训练器有多少数据（`__len__`）
2. 根据索引返回一个样本（`__getitem__`）

```
DataLoader 工作流程：
┌─────────────────────────────────────────────────────────────┐
│  DataLoader                                                  │
│    │                                                        │
│    ├── 从 Dataset 获取索引 0, 1, 2, ..., batch_size-1       │
│    ├── 调用 Dataset.__getitem__(idx) 获取每个样本           │
│    ├── 将 batch_size 个样本打包成一个 batch                 │
│    └── 返回 (images, labels) 张量                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 实现 Dataset

```python
# training/src/data/dataset.py
"""航空照片数据集"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import json

class AircraftDataset(Dataset):
    """
    航空照片数据集
    
    Args:
        csv_path: 标注 CSV 文件路径
        image_dir: 图片目录
        transform: 图片变换（数据增强）
        task: 任务类型 'type' | 'airline' | 'multi'
    """
    
    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform=None,
        task: str = 'type'
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.task = task
        
        # 读取标注
        self.df = pd.read_csv(csv_path)
        
        # 过滤无效数据
        if task == 'type' or task == 'multi':
            self.df = self.df[self.df['typename'].notna() & (self.df['typename'] != '')]
        
        # 重置索引
        self.df = self.df.reset_index(drop=True)
        
        print(f"加载数据集: {len(self.df)} 个样本, 任务: {task}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """返回一个样本"""
        row = self.df.iloc[idx]
        
        # 加载图片
        img_path = self.image_dir / row['filename']
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 根据任务返回不同的标签
        if self.task == 'type':
            label = int(row['typeid'])
            return image, label
        
        elif self.task == 'airline':
            label = int(row['airlineid']) if pd.notna(row['airlineid']) else 0
            return image, label
        
        elif self.task == 'multi':
            # 多任务：返回字典
            labels = {
                'type': int(row['typeid']),
                'airline': int(row['airlineid']) if pd.notna(row['airlineid']) else 0,
            }
            return image, labels
        
        else:
            raise ValueError(f"未知任务: {self.task}")
    
    @property
    def num_types(self):
        """机型类别数"""
        return self.df['typeid'].nunique()
    
    @property
    def num_airlines(self):
        """航司类别数"""
        return self.df['airlineid'].nunique()


def get_class_names(labels_dir: str, task: str = 'type'):
    """获取类别名称列表"""
    labels_path = Path(labels_dir)
    
    if task == 'type':
        json_path = labels_path / 'type_classes.json'
    elif task == 'airline':
        json_path = labels_path / 'airline_classes.json'
    else:
        raise ValueError(f"未知任务: {task}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data['classes']
```

### 1.3 测试 Dataset

```python
# training/scripts/test_dataset.py
"""测试 Dataset 是否正常工作"""

import sys
sys.path.append('training/src')

from data.dataset import AircraftDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def test_dataset():
    # 定义变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = AircraftDataset(
        csv_path="training/data/processed/aircraft_crop/train.csv",
        image_dir="training/data/processed/aircraft_crop/train",
        transform=transform,
        task='type'
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"机型类别数: {dataset.num_types}")
    
    # 获取一个样本
    image, label = dataset[0]
    print(f"图片形状: {image.shape}")  # [3, 224, 224]
    print(f"标签: {label}")
    
    # 测试 DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    for batch_images, batch_labels in dataloader:
        print(f"Batch 图片形状: {batch_images.shape}")  # [16, 3, 224, 224]
        print(f"Batch 标签形状: {batch_labels.shape}")  # [16]
        break
    
    print("\n✅ Dataset 测试通过！")


if __name__ == "__main__":
    test_dataset()
```

---

## 第二步：数据增强

### 2.1 为什么需要数据增强？

数据增强可以：
- 增加数据多样性，减少过拟合
- 模拟真实世界的变化（光照、角度等）
- 让模型学习更鲁棒的特征

### 2.2 实现数据增强

```python
# training/src/data/transforms.py
"""数据变换与增强"""

from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_train_transform(image_size: int = 224):
    """训练时的数据增强"""
    return A.Compose([
        # 尺寸调整
        A.LongestMaxSize(max_size=image_size + 32),
        A.RandomCrop(height=image_size, width=image_size),
        
        # 翻转（飞机可以左右翻转，但不要上下翻转）
        A.HorizontalFlip(p=0.5),
        
        # 颜色增强
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        
        # 模糊（模拟不同清晰度）
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        
        # 噪声
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        
        # 仿射变换（轻微）
        A.Affine(
            scale=(0.95, 1.05),
            rotate=(-5, 5),
            shear=(-3, 3),
            p=0.3
        ),
        
        # 归一化
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transform(image_size: int = 224):
    """验证/测试时的变换（不做增强）"""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, 
                      border_mode=0, value=(128, 128, 128)),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


class AlbumentationsWrapper:
    """将 Albumentations 变换包装成 torchvision 风格"""
    
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, image):
        # PIL Image → numpy array
        image = np.array(image)
        # 应用变换
        augmented = self.transform(image=image)
        return augmented['image']
```

### 2.3 更新 Dataset 使用 Albumentations

```python
# 在 dataset.py 中修改 __getitem__

def __getitem__(self, idx):
    row = self.df.iloc[idx]
    
    # 加载图片
    img_path = self.image_dir / row['filename']
    image = Image.open(img_path).convert('RGB')
    
    # 转换为 numpy array（Albumentations 需要）
    image = np.array(image)
    
    # 应用变换
    if self.transform:
        augmented = self.transform(image=image)
        image = augmented['image']
    
    # ... 返回标签
```

---

## 第三步：创建模型

### 3.1 理解模型结构

```
ConvNeXt 模型结构：
┌────────────────────────────────────────────────────────────┐
│  输入: [B, 3, 224, 224]                                     │
│         ↓                                                  │
│  ┌──────────────────────┐                                  │
│  │     Backbone         │  ← ConvNeXt 预训练权重           │
│  │  (特征提取)          │                                  │
│  └──────────────────────┘                                  │
│         ↓                                                  │
│  特征: [B, 1024]                                           │
│         ↓                                                  │
│  ┌──────────────────────┐                                  │
│  │     Head (分类头)     │  ← 我们要训练的部分              │
│  │  Linear(1024, N)     │                                  │
│  └──────────────────────┘                                  │
│         ↓                                                  │
│  输出: [B, N]  (N = 类别数)                                 │
└────────────────────────────────────────────────────────────┘
```

### 3.2 实现模型

```python
# training/src/models/classifier.py
"""分类模型"""

import torch
import torch.nn as nn
import timm

class AircraftClassifier(nn.Module):
    """
    飞机分类模型
    
    Args:
        num_classes: 类别数
        backbone_name: 骨干网络名称
        pretrained: 是否使用预训练权重
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 创建骨干网络
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0  # 不要分类头，只要特征
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        print(f"Backbone: {backbone_name}, 特征维度: {self.feature_dim}")
        
        # 分类头
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """前向传播"""
        # 提取特征
        features = self.backbone(x)  # [B, feature_dim]
        
        # 分类
        logits = self.head(features)  # [B, num_classes]
        
        return logits
    
    def get_features(self, x):
        """只返回特征（用于可视化等）"""
        return self.backbone(x)


def create_model(num_classes: int, config: dict = None):
    """工厂函数：创建模型"""
    config = config or {}
    
    return AircraftClassifier(
        num_classes=num_classes,
        backbone_name=config.get('backbone', 'convnext_base'),
        pretrained=config.get('pretrained', True),
        dropout=config.get('dropout', 0.2)
    )
```

### 3.3 测试模型

```python
# training/scripts/test_model.py
"""测试模型是否正常工作"""

import sys
sys.path.append('training/src')

import torch
from models.classifier import AircraftClassifier

def test_model():
    # 创建模型
    model = AircraftClassifier(num_classes=10, backbone_name="convnext_base")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f}M")
    print(f"可训练参数: {trainable_params / 1e6:.2f}M")
    
    # 测试前向传播
    x = torch.randn(4, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 验证输出
    assert output.shape == (4, 10), f"输出形状错误: {output.shape}"
    
    print("\n✅ 模型测试通过！")


if __name__ == "__main__":
    test_model()
```

---

## 第四步：训练循环

### 4.1 创建训练器

```python
# training/src/trainers/base_trainer.py
"""基础训练器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

class BaseTrainer:
    """
    基础训练器
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 模型
        self.model = model.to(self.device)
        
        # 数据
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30),
            eta_min=config.get('lr', 1e-4) * 0.01
        )
        
        # 保存目录
        self.save_dir = Path(config.get('save_dir', 'training/checkpoints/stage2'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'history': self.history
        }
        
        # 保存最新的
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        # 保存最好的
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"  💾 保存最佳模型 (acc: {self.best_val_acc:.4f})")
    
    def train(self, epochs: int = None):
        """完整训练流程"""
        epochs = epochs or self.config.get('epochs', 30)
        
        print(f"\n{'='*60}")
        print(f"开始训练: {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # 打印结果
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存检查点
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(is_best)
        
        print(f"\n{'='*60}")
        print(f"训练完成！最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"{'='*60}")
        
        # 保存训练历史
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
```

### 4.2 主训练脚本

```python
# training/scripts/train_stage2.py
"""阶段 2 训练脚本：机型分类"""

import sys
sys.path.append('training/src')

import torch
from torch.utils.data import DataLoader

from data.dataset import AircraftDataset
from data.transforms import get_train_transform, get_val_transform, AlbumentationsWrapper
from models.classifier import AircraftClassifier
from trainers.base_trainer import BaseTrainer

def main():
    # ============ 配置 ============
    config = {
        # 数据
        'train_csv': 'training/data/processed/aircraft_crop/train.csv',
        'val_csv': 'training/data/processed/aircraft_crop/val.csv',
        'train_dir': 'training/data/processed/aircraft_crop/train',
        'val_dir': 'training/data/processed/aircraft_crop/val',
        'image_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        
        # 模型
        'backbone': 'convnext_base',
        'pretrained': True,
        'dropout': 0.2,
        
        # 训练
        'epochs': 30,
        'lr': 1e-4,
        'weight_decay': 0.01,
        
        # 保存
        'save_dir': 'training/checkpoints/stage2',
    }
    
    print("配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # ============ 数据 ============
    print("\n加载数据...")
    
    train_transform = AlbumentationsWrapper(get_train_transform(config['image_size']))
    val_transform = AlbumentationsWrapper(get_val_transform(config['image_size']))
    
    train_dataset = AircraftDataset(
        csv_path=config['train_csv'],
        image_dir=config['train_dir'],
        transform=train_transform,
        task='type'
    )
    
    val_dataset = AircraftDataset(
        csv_path=config['val_csv'],
        image_dir=config['val_dir'],
        transform=val_transform,
        task='type'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ============ 模型 ============
    print("\n创建模型...")
    
    num_classes = train_dataset.num_types
    print(f"类别数: {num_classes}")
    
    model = AircraftClassifier(
        num_classes=num_classes,
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        dropout=config['dropout']
    )
    
    # ============ 训练 ============
    trainer = BaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    history = trainer.train(epochs=config['epochs'])
    
    print("\n训练完成！")


if __name__ == "__main__":
    main()
```

---

## 第五步：运行训练

### 5.1 开始训练

```bash
cd F:\bian\pyproject\Aerovision-V1
python training/scripts/train_stage2.py
```

### 5.2 监控训练

你应该看到类似输出：

```
配置:
  train_csv: training/data/processed/aircraft_crop/train.csv
  ...

加载数据...
加载数据集: 2500 个样本, 任务: type
加载数据集: 500 个样本, 任务: type

创建模型...
类别数: 10
Backbone: convnext_base, 特征维度: 1024

使用设备: cuda

============================================================
开始训练: 30 epochs
============================================================

Epoch 1 [Train]: 100%|██████████| 79/79 [00:45<00:00, loss: 2.1234, acc: 25.30%]
Epoch 1 [Val]: 100%|██████████| 16/16 [00:05<00:00]

Epoch 1/30
  Train Loss: 2.0123, Train Acc: 0.2530
  Val Loss: 1.8234, Val Acc: 0.3450
  LR: 0.000099
  💾 保存最佳模型 (acc: 0.3450)

...
```

### 5.3 常见问题排查

**问题 1：CUDA out of memory**
```python
# 减小 batch_size
config['batch_size'] = 16  # 或更小
```

**问题 2：loss 不降**
```python
# 检查数据
for images, labels in train_loader:
    print(f"Images range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"Labels: {labels[:10]}")
    break

# 确保 labels 是有效的类别索引 (0 到 num_classes-1)
```

**问题 3：训练太慢**
```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, labels in train_loader:
    with autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 第六步：评估模型

### 6.1 评估脚本

```python
# training/scripts/evaluate_stage2.py
"""评估阶段 2 模型"""

import sys
sys.path.append('training/src')

import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from data.dataset import AircraftDataset, get_class_names
from data.transforms import get_val_transform, AlbumentationsWrapper
from models.classifier import AircraftClassifier

def evaluate(checkpoint_path: str, test_csv: str, test_dir: str, labels_dir: str):
    """评估模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载检查点
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 数据
    transform = AlbumentationsWrapper(get_val_transform(config.get('image_size', 224)))
    
    test_dataset = AircraftDataset(
        csv_path=test_csv,
        image_dir=test_dir,
        transform=transform,
        task='type'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    
    # 模型
    num_classes = test_dataset.num_types
    model = AircraftClassifier(
        num_classes=num_classes,
        backbone_name=config.get('backbone', 'convnext_base'),
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # 预测
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("开始评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 获取类别名称
    class_names = get_class_names(labels_dir, task='type')
    
    # 指标
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Top-1 和 Top-5 准确率
    top1_correct = (all_preds == all_labels).sum()
    top1_acc = top1_correct / len(all_labels)
    
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_correct = sum(label in pred for label, pred in zip(all_labels, top5_preds))
    top5_acc = top5_correct / len(all_labels)
    
    print(f"\nTop-1 准确率: {top1_acc:.4f}")
    print(f"Top-5 准确率: {top5_acc:.4f}")
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('混淆矩阵')
    plt.tight_layout()
    
    # 保存
    output_dir = Path(checkpoint_path).parent
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    print(f"\n混淆矩阵已保存: {output_dir / 'confusion_matrix.png'}")
    
    return top1_acc, top5_acc


if __name__ == "__main__":
    evaluate(
        checkpoint_path="training/checkpoints/stage2/best.pth",
        test_csv="training/data/processed/aircraft_crop/test.csv",
        test_dir="training/data/processed/aircraft_crop/test",
        labels_dir="training/data/labels"
    )
```

---

## ✅ 过关标准

在进入阶段 3 之前，确保：

- [ ] 训练能正常运行不报错
- [ ] 验证集 Top-1 准确率 > 80%
- [ ] 验证集 Top-5 准确率 > 95%
- [ ] loss 曲线正常下降
- [ ] `training/checkpoints/stage2/best.pth` 已保存

---

## ❌ 禁止事项

在本阶段，**不要**：

- ❌ 添加多任务（航司、清晰度等）
- ❌ 尝试 Swin 或其他高级模型
- ❌ 做 Hybrid 模型
- ❌ 过度调参（先跑通！）

---

## 💡 调参建议

如果准确率不达标：

1. **首先检查数据**
   - 各类别样本是否均衡？
   - 标注是否有错误？

2. **增加 epochs**（30 → 50）

3. **调整学习率**
   - 太高：loss 震荡
   - 太低：收敛太慢
   - 试试 3e-4 或 5e-5

4. **增加数据增强**

---

## 🔜 下一步

完成所有检查项后，进入 [阶段 3：多 Head 训练](stage3_multi_head.md)

