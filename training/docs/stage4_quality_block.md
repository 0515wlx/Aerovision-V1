# 阶段 4：清晰度 + 遮挡检测

> ⏱️ 预计时间：2 天
> 🎯 目标：添加 clarity（清晰度）和 block（遮挡程度）两个回归任务
> 📌 核心概念：分类 vs 回归

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 支持回归任务的 Dataset
- [ ] 包含 4 个 Head 的模型（机型、航司、清晰度、遮挡）
- [ ] 理解分类 loss 和回归 loss 的区别
- [ ] clarity MAE < 0.15, block MAE < 0.15

---

## 核心概念：分类 vs 回归

### 两种任务的区别

| 方面 | 分类任务 | 回归任务 |
|------|----------|----------|
| 输出 | 离散类别（0, 1, 2, ...） | 连续值（0.0 ~ 1.0） |
| 输出层 | Linear(dim, num_classes) | Linear(dim, 1) |
| 激活函数 | 无（或 Softmax） | Sigmoid（限制到 0-1） |
| 损失函数 | CrossEntropyLoss | MSELoss 或 L1Loss |
| 评估指标 | 准确率、F1 | MAE、RMSE、相关系数 |

### 模型结构

```
                    ┌─────────────────────────────────────────┐
                    │           ConvNeXt Backbone              │
                    └─────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
              ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
              │           │      │           │      │           │
         ┌────┴────┐ ┌────┴────┐ │           │ ┌────┴────┐ ┌────┴────┐
         │  Type   │ │ Airline │ │           │ │ Clarity │ │  Block  │
         │  Head   │ │  Head   │ │           │ │  Head   │ │  Head   │
         └────┬────┘ └────┬────┘ │           │ └────┬────┘ └────┬────┘
              │           │      │           │      │           │
         [B, 10]     [B, 12]     │           │   [B, 1]      [B, 1]
              │           │      │           │      │           │
          Softmax     Softmax    │           │   Sigmoid    Sigmoid
              │           │      │           │      │           │
           分类         分类      │           │   0.0~1.0    0.0~1.0
                                 │           │
                            ┌────┴────┐ ┌────┴────┐
                            │ 未来扩展 │ │ 未来扩展 │
                            └─────────┘ └─────────┘
```

---

## 第一步：更新 Dataset

### 1.1 添加回归标签支持

```python
# training/src/data/dataset.py（更新 __getitem__ 方法）

def __getitem__(self, idx):
    row = self.df.iloc[idx]
    
    # 加载图片
    img_path = self.image_dir / row['filename']
    image = Image.open(img_path).convert('RGB')
    image = np.array(image)
    
    if self.transform:
        augmented = self.transform(image=image)
        image = augmented['image']
    
    if self.task == 'type':
        return image, int(row['typeid'])
    
    elif self.task == 'airline':
        return image, int(row['airlineid']) if pd.notna(row.get('airlineid')) else 0
    
    elif self.task == 'multi':
        # 分类标签
        labels = {
            'type': int(row['typeid']),
            'airline': int(row['airlineid']) if pd.notna(row.get('airlineid')) else 0,
        }
        return image, labels
    
    elif self.task == 'full':
        # 完整标签（包含回归任务）
        labels = {
            # 分类任务
            'type': int(row['typeid']),
            'airline': int(row['airlineid']) if pd.notna(row.get('airlineid')) else 0,
            # 回归任务
            'clarity': float(row['clarity']) if pd.notna(row.get('clarity')) else 1.0,
            'block': float(row['block']) if pd.notna(row.get('block')) else 0.0,
        }
        return image, labels
    
    else:
        raise ValueError(f"未知任务: {self.task}")
```

### 1.2 更新 collate_fn

```python
# training/src/data/dataset.py（更新）

def full_task_collate_fn(batch):
    """完整多任务数据打包（包含回归任务）"""
    images = torch.stack([item[0] for item in batch])
    
    labels = {}
    sample_labels = batch[0][1]
    
    for key in sample_labels.keys():
        values = [item[1][key] for item in batch]
        if key in ['type', 'airline']:
            # 分类任务：整数
            labels[key] = torch.tensor(values, dtype=torch.long)
        else:
            # 回归任务：浮点数
            labels[key] = torch.tensor(values, dtype=torch.float32)
    
    return images, labels
```

---

## 第二步：创建完整多任务模型

### 2.1 四 Head 模型

```python
# training/src/models/full_model.py
"""完整多任务模型（分类 + 回归）"""

import torch
import torch.nn as nn
import timm

class FullMultiTaskModel(nn.Module):
    """
    完整多任务模型
    
    包含：
    - 机型分类 Head
    - 航司分类 Head
    - 清晰度回归 Head
    - 遮挡程度回归 Head
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
        
        # 共享 Backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        self.feature_dim = self.backbone.num_features
        
        # ===== 分类 Head =====
        self.type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_types)
        )
        
        self.airline_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_airlines)
        )
        
        # ===== 回归 Head =====
        # 清晰度：0（模糊）~ 1（清晰）
        self.clarity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 限制输出到 0-1
        )
        
        # 遮挡程度：0（无遮挡）~ 1（完全遮挡）
        self.block_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.num_types = num_types
        self.num_airlines = num_airlines
        
        print(f"FullMultiTaskModel 创建完成")
        print(f"  Backbone: {backbone_name}")
        print(f"  特征维度: {self.feature_dim}")
        print(f"  机型类别: {num_types}")
        print(f"  航司类别: {num_airlines}")
        print(f"  回归任务: clarity, block")
    
    def forward(self, x):
        """前向传播"""
        features = self.backbone(x)
        
        return {
            # 分类
            'type': self.type_head(features),
            'airline': self.airline_head(features),
            # 回归
            'clarity': self.clarity_head(features).squeeze(-1),  # [B, 1] -> [B]
            'block': self.block_head(features).squeeze(-1),
        }
    
    def get_features(self, x):
        return self.backbone(x)
```

### 2.2 测试模型

```python
# training/scripts/test_full_model.py
"""测试完整多任务模型"""

import sys
sys.path.append('training/src')

import torch
from models.full_model import FullMultiTaskModel

def test():
    model = FullMultiTaskModel(
        num_types=10,
        num_airlines=12
    )
    
    x = torch.randn(4, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\n输出检查:")
    print(f"  type 形状: {outputs['type'].shape}")        # [4, 10]
    print(f"  airline 形状: {outputs['airline'].shape}")  # [4, 12]
    print(f"  clarity 形状: {outputs['clarity'].shape}")  # [4]
    print(f"  block 形状: {outputs['block'].shape}")      # [4]
    
    print(f"\n回归输出范围检查:")
    print(f"  clarity: [{outputs['clarity'].min():.3f}, {outputs['clarity'].max():.3f}]")
    print(f"  block: [{outputs['block'].min():.3f}, {outputs['block'].max():.3f}]")
    
    # 确保回归输出在 0-1 范围
    assert outputs['clarity'].min() >= 0 and outputs['clarity'].max() <= 1
    assert outputs['block'].min() >= 0 and outputs['block'].max() <= 1
    
    print("\n✅ 完整模型测试通过！")

if __name__ == "__main__":
    test()
```

---

## 第三步：多任务训练器（含回归）

### 3.1 理解不同的 Loss

```python
# 分类任务使用 CrossEntropyLoss
# 输入：logits [B, C], 标签 [B]（类别索引）
criterion_cls = nn.CrossEntropyLoss()
loss = criterion_cls(logits, labels)  # labels 是整数

# 回归���务使用 MSELoss 或 L1Loss（SmoothL1Loss）
# 输入：预测值 [B], 真实值 [B]
criterion_reg = nn.MSELoss()  # 或 nn.L1Loss()
loss = criterion_reg(predictions, targets)  # 都是浮点数
```

### 3.2 完整训练器

```python
# training/src/trainers/full_trainer.py
"""完整多任务训练器"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np

class FullMultiTaskTrainer:
    """
    完整多任务训练器（分类 + 回归）
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
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # ===== 损失函数 =====
        # 分类任务
        self.criterion_type = nn.CrossEntropyLoss()
        self.criterion_airline = nn.CrossEntropyLoss()
        
        # 回归任务（使用 SmoothL1Loss，对异常值更鲁棒）
        self.criterion_clarity = nn.SmoothL1Loss()
        self.criterion_block = nn.SmoothL1Loss()
        
        # ===== Loss 权重 =====
        self.loss_weights = {
            'type': config.get('type_weight', 1.0),
            'airline': config.get('airline_weight', 0.5),
            'clarity': config.get('clarity_weight', 0.3),
            'block': config.get('block_weight', 0.3),
        }
        print(f"Loss 权重: {self.loss_weights}")
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 30)
        )
        
        self.save_dir = Path(config.get('save_dir', 'training/checkpoints/stage4'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_score = 0.0
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_type_acc': [],
            'val_airline_acc': [],
            'val_clarity_mae': [],
            'val_block_mae': [],
        }
    
    def compute_loss(self, outputs, labels):
        """计算总 loss"""
        losses = {}
        
        # 分类 loss
        losses['type'] = self.criterion_type(outputs['type'], labels['type'])
        losses['airline'] = self.criterion_airline(outputs['airline'], labels['airline'])
        
        # 回归 loss
        losses['clarity'] = self.criterion_clarity(outputs['clarity'], labels['clarity'])
        losses['block'] = self.criterion_block(outputs['block'], labels['block'])
        
        # 加权总和
        total_loss = sum(
            self.loss_weights[k] * v for k, v in losses.items()
        )
        
        return total_loss, {k: v.item() for k, v in losses.items()}
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            outputs = self.model(images)
            loss, loss_dict = self.compute_loss(outputs, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return total_loss / n_samples
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        
        total_loss = 0.0
        n_samples = 0
        
        # 收集预测和真实值
        all_type_preds, all_type_labels = [], []
        all_airline_preds, all_airline_labels = [], []
        all_clarity_preds, all_clarity_labels = [], []
        all_block_preds, all_block_labels = [], []
        
        for images, labels in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
            images = images.to(self.device)
            labels_device = {k: v.to(self.device) for k, v in labels.items()}
            
            outputs = self.model(images)
            loss, _ = self.compute_loss(outputs, labels_device)
            
            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size
            
            # 收集分类预测
            _, type_pred = outputs['type'].max(1)
            all_type_preds.extend(type_pred.cpu().numpy())
            all_type_labels.extend(labels['type'].numpy())
            
            _, airline_pred = outputs['airline'].max(1)
            all_airline_preds.extend(airline_pred.cpu().numpy())
            all_airline_labels.extend(labels['airline'].numpy())
            
            # 收集回归预测
            all_clarity_preds.extend(outputs['clarity'].cpu().numpy())
            all_clarity_labels.extend(labels['clarity'].numpy())
            
            all_block_preds.extend(outputs['block'].cpu().numpy())
            all_block_labels.extend(labels['block'].numpy())
        
        # 计算指标
        avg_loss = total_loss / n_samples
        
        type_acc = np.mean(np.array(all_type_preds) == np.array(all_type_labels))
        airline_acc = np.mean(np.array(all_airline_preds) == np.array(all_airline_labels))
        
        clarity_mae = np.mean(np.abs(np.array(all_clarity_preds) - np.array(all_clarity_labels)))
        block_mae = np.mean(np.abs(np.array(all_block_preds) - np.array(all_block_labels)))
        
        return {
            'loss': avg_loss,
            'type_acc': type_acc,
            'airline_acc': airline_acc,
            'clarity_mae': clarity_mae,
            'block_mae': block_mae,
        }
    
    def save_checkpoint(self, is_best: bool = False):
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history,
        }
        
        torch.save(checkpoint, self.save_dir / 'latest.pth')
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pth')
            print(f"  💾 保存最佳模型")
    
    def train(self, epochs: int = None):
        epochs = epochs or self.config.get('epochs', 30)
        
        print(f"\n{'='*60}")
        print(f"开始完整多任务训练")
        print(f"  任务: 机型分类, 航司分类, 清晰度回归, 遮挡回归")
        print(f"  Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_type_acc'].append(val_metrics['type_acc'])
            self.history['val_airline_acc'].append(val_metrics['airline_acc'])
            self.history['val_clarity_mae'].append(val_metrics['clarity_mae'])
            self.history['val_block_mae'].append(val_metrics['block_mae'])
            
            # 打印
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  分类 - Type Acc: {val_metrics['type_acc']:.4f}, Airline Acc: {val_metrics['airline_acc']:.4f}")
            print(f"  回归 - Clarity MAE: {val_metrics['clarity_mae']:.4f}, Block MAE: {val_metrics['block_mae']:.4f}")
            
            # 综合评分（用于选择最佳模型）
            # 准确率越高越好，MAE 越低越好
            score = (
                0.5 * val_metrics['type_acc'] +
                0.2 * val_metrics['airline_acc'] +
                0.15 * (1 - val_metrics['clarity_mae']) +
                0.15 * (1 - val_metrics['block_mae'])
            )
            
            is_best = score > self.best_score
            if is_best:
                self.best_score = score
            self.save_checkpoint(is_best)
        
        print(f"\n{'='*60}")
        print(f"训练完成！")
        print(f"  最佳 Type Acc: {max(self.history['val_type_acc']):.4f}")
        print(f"  最佳 Clarity MAE: {min(self.history['val_clarity_mae']):.4f}")
        print(f"  最佳 Block MAE: {min(self.history['val_block_mae']):.4f}")
        print(f"{'='*60}")
        
        with open(self.save_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
```

---

## 第四步：训练脚本

```python
# training/scripts/train_stage4.py
"""阶段 4 训练脚本：完整多任务（分类 + 回归）"""

import sys
sys.path.append('training/src')

import torch
from torch.utils.data import DataLoader

from data.dataset import AircraftDataset, full_task_collate_fn
from data.transforms import get_train_transform, get_val_transform, AlbumentationsWrapper
from models.full_model import FullMultiTaskModel
from trainers.full_trainer import FullMultiTaskTrainer

def main():
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
        
        # 任务权重
        'type_weight': 1.0,
        'airline_weight': 0.5,
        'clarity_weight': 0.3,
        'block_weight': 0.3,
        
        'save_dir': 'training/checkpoints/stage4',
    }
    
    # 数据
    train_transform = AlbumentationsWrapper(get_train_transform(config['image_size']))
    val_transform = AlbumentationsWrapper(get_val_transform(config['image_size']))
    
    train_dataset = AircraftDataset(
        csv_path=config['train_csv'],
        image_dir=config['train_dir'],
        transform=train_transform,
        task='full'  # 完整任务模式
    )
    
    val_dataset = AircraftDataset(
        csv_path=config['val_csv'],
        image_dir=config['val_dir'],
        transform=val_transform,
        task='full'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=full_task_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=full_task_collate_fn
    )
    
    # 模型
    model = FullMultiTaskModel(
        num_types=train_dataset.num_types,
        num_airlines=train_dataset.num_airlines,
        backbone_name=config['backbone'],
        pretrained=config['pretrained'],
        dropout=config['dropout']
    )
    
    # 加载阶段 3 权重
    try:
        checkpoint = torch.load('training/checkpoints/stage3/best.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("✅ 加载阶段 3 权重成功")
    except Exception as e:
        print(f"⚠️ 未能加载阶段 3 权重: {e}")
    
    # 训练
    trainer = FullMultiTaskTrainer(
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

## 第五步：评估回归任务

```python
# training/scripts/evaluate_regression.py
"""评估回归任务（clarity 和 block）"""

import sys
sys.path.append('training/src')

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def evaluate_regression(checkpoint_path, test_csv, test_dir):
    """评估回归任务"""
    # ... 加载模型和数据 ...
    
    # 收集预测和真实值
    clarity_preds, clarity_labels = [], []
    block_preds, block_labels = [], []
    
    # ... 推理循环 ...
    
    # 计算指标
    def calc_metrics(preds, labels, name):
        preds = np.array(preds)
        labels = np.array(labels)
        
        mae = np.mean(np.abs(preds - labels))
        rmse = np.sqrt(np.mean((preds - labels) ** 2))
        corr, _ = stats.pearsonr(preds, labels)
        
        print(f"\n{name} 回归指标:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  相关系数: {corr:.4f}")
        
        # 绘制散点图
        plt.figure(figsize=(6, 6))
        plt.scatter(labels, preds, alpha=0.5, s=10)
        plt.plot([0, 1], [0, 1], 'r--', label='理想预测')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.title(f'{name}: 预测 vs 真实')
        plt.legend()
        plt.savefig(f'training/logs/{name.lower()}_scatter.png')
        
        return mae, rmse, corr
    
    calc_metrics(clarity_preds, clarity_labels, 'Clarity')
    calc_metrics(block_preds, block_labels, 'Block')
```

---

## ✅ 过关标准

- [ ] 机型准确率仍 > 80%（不能大幅下降）
- [ ] Clarity MAE < 0.15
- [ ] Block MAE < 0.15
- [ ] 回归预测与真实值相关系数 > 0.7

---

## ❌ 禁止事项

- ❌ 使用分类方法做回归（把 0-1 分成 10 个区间再分类）
- ❌ 回归 Head 不加 Sigmoid（输出可能超出 0-1）
- ❌ 跳过本阶段直接做 Hybrid

---

## 💡 常见问题

### Q1: MAE 很高，回归效果差？

1. **检查标注是否一致** - 不同标注者对 clarity/block 理解可能不同
2. **检查数据分布** - 是否大部分都是高清晰度？
3. **尝试不同 loss** - MSELoss vs SmoothL1Loss vs L1Loss

### Q2: 分类准确率下降了？

1. **降低回归任务权重** - `clarity_weight` 和 `block_weight` 设为 0.1
2. **分阶段训练** - 先冻结 backbone 只训练回归 Head

---

## 🔜 下一步

完成所有检查项后，进入 [阶段 5：Hybrid 模型融合](stage5_hybrid.md)

