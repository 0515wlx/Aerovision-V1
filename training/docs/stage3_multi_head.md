# 阶段 3：多 Head 训练 - 机型 + 航司

> ⏱️ 预计时间：2 天
> 🎯 目标：在一个模型中同时输出机型和航司预测
> 📌 核心原则：共享 Backbone，独立 Head

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 多任务 Dataset（同时返回机型和航司标签）
- [ ] 多 Head 模型
- [ ] 多任务训练器（处理多个 loss）
- [ ] 两个任务的准确率都不低于单任务

---

## 核心概念：多任务学习

### 为什么要多任务学习？

```
单任务方式（需要两个模型）：
┌─────────────┐     ┌─────────────┐
│ ConvNeXt    │     │ ConvNeXt    │
│ (机型模型)   │     │ (航司模型)   │
└─────────────┘     └─────────────┘
      ↓                   ↓
   机型预测            航司预测

多任务方式（一个模型）：
┌─────────────────────────────────┐
│         ConvNeXt Backbone       │  ← 共享特征提取
└─────────────────────────────────┘
         ↓              ↓
   ┌──────────┐   ┌──────────┐
   │ Type Head│   │Airline   │      ← 独立分类头
   │          │   │Head      │
   └──────────┘   └──────────┘
         ↓              ↓
     机型预测       航司预测
```

**优点：**
- 减少计算量（共享特征提取）
- 特征共享可能提升效果（飞机外形和航司涂装有关联）
- 部署更简单（一个模型输出多个结果）

---

## 第一步：更新 Dataset

### 1.1 修改 Dataset 支持多任务

```python
# training/src/data/dataset.py（更新）
"""航空照片数据集 - 多任务版本"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np

class AircraftDataset(Dataset):
    """
    航空照片数据集（支持多任务）
    
    Args:
        csv_path: 标注 CSV 文件路径
        image_dir: 图片目录
        transform: 图片变换
        task: 'type' | 'airline' | 'multi'
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
        if task in ['type', 'multi']:
            self.df = self.df[self.df['typename'].notna() & (self.df['typename'] != '')]
        
        self.df = self.df.reset_index(drop=True)
        
        print(f"加载数据集: {len(self.df)} 个样本, 任务: {task}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 加载图片
        img_path = self.image_dir / row['filename']
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # 应用变换
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        # 根据任务返回不同的标签
        if self.task == 'type':
            label = int(row['typeid'])
            return image, label
        
        elif self.task == 'airline':
            label = int(row['airlineid']) if pd.notna(row.get('airlineid')) else 0
            return image, label
        
        elif self.task == 'multi':
            # 多任务：返回字典
            labels = {
                'type': int(row['typeid']),
                'airline': int(row['airlineid']) if pd.notna(row.get('airlineid')) else 0,
            }
            return image, labels
        
        else:
            raise ValueError(f"未知任务: {self.task}")
    
    @property
    def num_types(self):
        return int(self.df['typeid'].max()) + 1 if 'typeid' in self.df.columns else 0
    
    @property
    def num_airlines(self):
        return int(self.df['airlineid'].max()) + 1 if 'airlineid' in self.df.columns else 0
```

### 1.2 自定义 collate_fn

由于多任务返回字典，需要自定义打包函数：

```python
# training/src/data/dataset.py（添加）

def multi_task_collate_fn(batch):
    """多任务数据打包函数"""
    images = torch.stack([item[0] for item in batch])
    
    # 检查是否是多任务（字典标签）
    if isinstance(batch[0][1], dict):
        labels = {
            key: torch.tensor([item[1][key] for item in batch])
            for key in batch[0][1].keys()
        }
    else:
        labels = torch.tensor([item[1] for item in batch])
    
    return images, labels
```

---

## 第二步：创建多 Head 模型

### 2.1 模型结构

```python
# training/src/models/multi_head.py
"""多任务分类模型"""

import torch
import torch.nn as nn
import timm

class MultiHeadClassifier(nn.Module):
    """
    多 Head 分类模型
    
    共享一个 Backbone，每个任务有独立的分类头
    """
    
    def __init__(
        self,
        num_types: int,
        num_airlines: int,
        backbone_name: str = "convnext_base",
        pretrained: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 共享的 Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        self.feature_dim = self.backbone.num_features
        
        # 独立的 Head
        self.type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_types)
        )
        
        self.airline_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_airlines)
        )
        
        self.num_types = num_types
        self.num_airlines = num_airlines
        
        print(f"MultiHeadClassifier: backbone={backbone_name}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  机型类别: {num_types}")
        print(f"  航司类别: {num_airlines}")
    
    def forward(self, x):
        """
        前向传播
        
        Returns:
            dict: {'type': logits, 'airline': logits}
        """
        # 共享特征
        features = self.backbone(x)
        
        # 各任务预测
        type_logits = self.type_head(features)
        airline_logits = self.airline_head(features)
        
        return {
            'type': type_logits,
            'airline': airline_logits
        }
    
    def get_features(self, x):
        """返回特征向量"""
        return self.backbone(x)
```

### 2.2 测试多 Head 模型

```python
# training/scripts/test_multi_head.py
"""测试多 Head 模型"""

import sys
sys.path.append('training/src')

import torch
from models.multi_head import MultiHeadClassifier

def test():
    model = MultiHeadClassifier(
        num_types=10,
        num_airlines=12,
        backbone_name="convnext_base"
    )
    
    x = torch.randn(4, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"机型输出形状: {outputs['type'].shape}")  # [4, 10]
    print(f"航司输出形状: {outputs['airline'].shape}")  # [4, 12]
    
    print("\n✅ 多 Head 模型测试通过！")

if __name__ == "__main__":
    test()
```

---

## 第三步：多任务训练器

### 3.1 关键问题：Loss 加权

多任务学习的核心问题是**如何平衡各任务的 loss**。

```
总 Loss = w1 × Loss_type + w2 × Loss_airline

问题：
- 如果 Loss_type ≈ 2.0, Loss_airline ≈ 0.5
- 使用 w1=w2=1.0，模型会更关注机型（因为 loss 大）
- 航司任务会被忽视

解决方案：
1. 固定权重：根据任务重要性设置
2. 动态权重：根据 loss 量级自动调整
3. 不确定性加权：基于任务不确定性
```

### 3.2 实现多任务训练器

```python
# training/src/trainers/multi_task_trainer.py
"""多任务训练器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json

class MultiTaskTrainer:
    """
    多任务训练器
    
    Args:
        model: 多 Head 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置
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
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 各任务的损失函数
        self.criterion_type = nn.CrossEntropyLoss()
        self.criterion_airline = nn.CrossEntropyLoss()
        
        # Loss 权重
        self.loss_weights = {
            'type': config.get('type_weight', 1.0),
            'airline': config.get('airline_weight', 1.0)
        }
        print(f"Loss 权重: {self.loss_weights}")
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30),
            eta_min=config.get('lr', 1e-4) * 0.01
        )
        
        # 保存目录
        self.save_dir = Path(config.get('save_dir', 'training/checkpoints/stage3'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 状态
        self.best_val_acc = 0.0
        self.current_epoch = 0
        self.history = {
            'train_loss': [], 'train_type_acc': [], 'train_airline_acc': [],
            'val_loss': [], 'val_type_acc': [], 'val_airline_acc': []
        }
    
    def compute_loss(self, outputs, labels):
        """计算加权总 loss"""
        loss_type = self.criterion_type(outputs['type'], labels['type'])
        loss_airline = self.criterion_airline(outputs['airline'], labels['airline'])
        
        total_loss = (
            self.loss_weights['type'] * loss_type +
            self.loss_weights['airline'] * loss_airline
        )
        
        return total_loss, {'type': loss_type.item(), 'airline': loss_airline.item()}
    
    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0.0
        type_correct = 0
        airline_correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # 前向
            outputs = self.model(images)
            loss, loss_dict = self.compute_loss(outputs, labels)
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            
            _, type_pred = outputs['type'].max(1)
            type_correct += type_pred.eq(labels['type']).sum().item()
            
            _, airline_pred = outputs['airline'].max(1)
            airline_correct += airline_pred.eq(labels['airline']).sum().item()
            
            total += batch_size
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'type': f"{100. * type_correct / total:.1f}%",
                'airline': f"{100. * airline_correct / total:.1f}%"
            })
        
        return (
            total_loss / total,
            type_correct / total,
            airline_correct / total
        )
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        
        total_loss = 0.0
        type_correct = 0
        airline_correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            outputs = self.model(images)
            loss, _ = self.compute_loss(outputs, labels)
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            
            _, type_pred = outputs['type'].max(1)
            type_correct += type_pred.eq(labels['type']).sum().item()
            
            _, airline_pred = outputs['airline'].max(1)
            airline_correct += airline_pred.eq(labels['airline']).sum().item()
            
            total += batch_size
        
        return (
            total_loss / total,
            type_correct / total,
            airline_correct / total
        )
    
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
        
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"  💾 保存最佳模型")
    
    def train(self, epochs: int = None):
        """完整训练"""
        epochs = epochs or self.config.get('epochs', 30)
        
        print(f"\n{'='*60}")
        print(f"开始多任务训练: {epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # 训练
            train_loss, train_type_acc, train_airline_acc = self.train_epoch()
            
            # 验证
            val_loss, val_type_acc, val_airline_acc = self.validate()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['train_type_acc'].append(train_type_acc)
            self.history['train_airline_acc'].append(train_airline_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_type_acc'].append(val_type_acc)
            self.history['val_airline_acc'].append(val_airline_acc)
            
            # 打印
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train - Loss: {train_loss:.4f}, Type: {train_type_acc:.4f}, Airline: {train_airline_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Type: {val_type_acc:.4f}, Airline: {val_airline_acc:.4f}")
            
            # 保存（以机型准确率为主）
            combined_acc = 0.7 * val_type_acc + 0.3 * val_airline_acc  # 加权平均
            is_best = combined_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = combined_acc
            self.save_checkpoint(is_best)
        
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"  最佳机型准确率: {max(self.history['val_type_acc']):.4f}")
        print(f"  最佳航司准确率: {max(self.history['val_airline_acc']):.4f}")
        print(f"{'='*60}")
        
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
```

---

## 第四步：训练脚本

```python
# training/scripts/train_stage3.py
"""阶段 3 训练脚本：多任务（机型 + 航司）"""

import sys
sys.path.append('training/src')

import torch
from torch.utils.data import DataLoader

from data.dataset import AircraftDataset, multi_task_collate_fn
from data.transforms import get_train_transform, get_val_transform, AlbumentationsWrapper
from models.multi_head import MultiHeadClassifier
from trainers.multi_task_trainer import MultiTaskTrainer

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
        
        # 多任务权重
        'type_weight': 1.0,      # 机型任务权重
        'airline_weight': 0.5,   # 航司任务权重（次要任务，权重低一些）
        
        # 保存
        'save_dir': 'training/checkpoints/stage3',
    }
    
    # ============ 数据 ============
    print("加载数据...")
    
    train_transform = AlbumentationsWrapper(get_train_transform(config['image_size']))
    val_transform = AlbumentationsWrapper(get_val_transform(config['image_size']))
    
    train_dataset = AircraftDataset(
        csv_path=config['train_csv'],
        image_dir=config['train_dir'],
        transform=train_transform,
        task='multi'  # 多任务模式
    )
    
    val_dataset = AircraftDataset(
        csv_path=config['val_csv'],
        image_dir=config['val_dir'],
        transform=val_transform,
        task='multi'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=multi_task_collate_fn  # 使用自定义打包函数
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # ============ 模型 ============
    print("\n创建模型...")
    
    model = MultiHeadClassifier(
        num_types=train_dataset.num_types,
        num_airlines=train_dataset.num_airlines,
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        dropout=config['dropout']
    )
    
    # ============ 加载阶段 2 权重（可选但推荐）============
    stage2_checkpoint = 'training/checkpoints/stage2/best.pth'
    try:
        checkpoint = torch.load(stage2_checkpoint, map_location='cpu')
        # 只加载 backbone 权重
        backbone_state = {k.replace('backbone.', ''): v 
                         for k, v in checkpoint['model_state_dict'].items() 
                         if k.startswith('backbone.')}
        model.backbone.load_state_dict(backbone_state, strict=False)
        print(f"✅ 加载阶段 2 backbone 权重: {stage2_checkpoint}")
    except Exception as e:
        print(f"⚠️ 未能加载阶段 2 权重: {e}")
        print("  将使用 ImageNet 预训练权重")
    
    # ============ 训练 ============
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
```

---

## 第五步：运行与验证

### 5.1 运行训练

```bash
python training/scripts/train_stage3.py
```

### 5.2 预期输出

```
加载数据...
加载数据集: 2500 个样本, 任务: multi

创建模型...
MultiHeadClassifier: backbone=convnext_base
  特征维度: 1024
  机型类别: 10
  航司类别: 12

✅ 加载阶段 2 backbone 权重

Loss 权重: {'type': 1.0, 'airline': 0.5}

============================================================
开始多任务训练: 30 epochs
============================================================

Epoch 1 [Train]: loss: 1.8234, type: 45.2%, airline: 38.1%
Epoch 1 [Val]: ...

Epoch 1/30
  Train - Loss: 1.7234, Type: 0.4520, Airline: 0.3810
  Val   - Loss: 1.5234, Type: 0.5230, Airline: 0.4120
  💾 保存最佳模型
```

---

## ✅ 过关标准

在进入阶段 4 之前，确保：

- [ ] 多任务训练能正常运行
- [ ] 机型准确率 ≥ 阶段 2 的 95%（不能大幅下降）
- [ ] 航司准确率 > 70%
- [ ] `training/checkpoints/stage3/best.pth` 已保存

---

## ❌ 禁止事项

- ❌ 添加 clarity/block 回归任务（下一阶段）
- ❌ 尝试动态 loss 加权（先用固定权重）
- ❌ 同时调多个超参数

---

## 💡 常见问题

### Q1: 机型准确率下降了？

可能原因：
1. **航司任务抢占了学习能力** → 降低 `airline_weight`
2. **没有加载阶段 2 权重** → 确保正确加载
3. **batch_size 太小** → 增大 batch_size

### Q2: 航司准确率很低？

可能原因：
1. **航司标注质量差** → 检查标注
2. **类别不平衡** → 使用加权 CrossEntropyLoss
3. **权重太低** → 提高 `airline_weight`

### Q3: 如何调整 loss 权重？

经验法则：
- 如果两个任务 loss 量级相近，用 1:1
- 如果差异大，调整权重使加权后量级接近
- 主任务（机型）权重应该更高

---

## 🔜 下一步

完成所有检查项后，���入 [阶段 4：清晰度 + 遮挡检测](stage4_quality_block.md)

