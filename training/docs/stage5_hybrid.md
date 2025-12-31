# 阶段 5：Hybrid 模型融合

> ⏱️ 预计时间：2-3 天
> 🎯 目标：融合 ConvNeXt 和 Swin Transformer 的特征，提升模型表现
> 📌 核心概念：CNN + Transformer 互补

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 理解 CNN 和 Transformer 的区别
- [ ] Hybrid 模型能正常运行
- [ ] 相比阶段 4，准确率有提升（至少 1-2%）
- [ ] 模型大小和推理速度在可接受范围

---

## 核心概念：为什么要 Hybrid？

### CNN vs Transformer

| 特性 | CNN (ConvNeXt) | Transformer (Swin) |
|------|----------------|-------------------|
| 归纳偏置 | 局部性、平移等变性 | 较少归纳偏置 |
| 感受野 | 逐层扩大（局部→全局） | 一开始就能看全局 |
| 擅长 | 纹理、边缘、局部特征 | 形状、结构、全局关系 |
| 计算效率 | 较高 | 较低 |
| 数据需求 | 较少 | 较多 |

### 为什么融合有效？

```
飞机识别需要：
├── 局部特征（CNN 擅长）
│   ├── 发动机形状
│   ├── 翼尖小翼细节
│   ├── 舱门数量和位置
│   └── 涂装颜色纹理
│
└── 全局特征（Transformer 擅长）
    ├── 机身长宽比
    ├── 机翼形状
    ├── 整体轮廓
    └── 部件间的空间关系
```

### Hybrid 架构

```
                    输入图片
                       │
          ┌────────────┴────────────┐
          │                         │
    ┌─────▼─────┐            ┌─────▼─────┐
    │  ConvNeXt │            │   Swin    │
    │  Backbone │            │  Backbone │
    └─────┬─────┘            └─────┬─────┘
          │                         │
     [B, 1024]                 [B, 1024]
          │                         │
          └────────────┬────────────┘
                       │
                  ┌────▼────┐
                  │ Fusion  │  ← 特征融合
                  │ Module  │
                  └────┬────┘
                       │
                  [B, 2048] 或 [B, 1024]
                       │
          ┌────────────┼────────────┐
          │            │            │
     ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
     │  Type   │ │ Clarity │ │  ...    │
     │  Head   │ │  Head   │ │         │
     └─────────┘ └─────────┘ └─────────┘
```

---

## 第一步：实现 Hybrid 模型

### 1.1 基础 Hybrid 模型

```python
# training/src/models/hybrid.py
"""Hybrid 模型：ConvNeXt + Swin Transformer"""

import torch
import torch.nn as nn
import timm

class HybridModel(nn.Module):
    """
    混合模型：结合 CNN 和 Transformer 的优势
    
    Args:
        num_types: 机型类别数
        num_airlines: 航司类别数
        cnn_backbone: CNN 骨干网络名称
        transformer_backbone: Transformer 骨干网络名称
        fusion_method: 融合方法 'concat' | 'add' | 'attention'
        pretrained: 是否使用预训练权重
        dropout: Dropout 比例
    """
    
    def __init__(
        self,
        num_types: int,
        num_airlines: int,
        cnn_backbone: str = "convnext_base",
        transformer_backbone: str = "swin_base_patch4_window7_224",
        fusion_method: str = "concat",
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # ===== 两个 Backbone =====
        print(f"加载 CNN backbone: {cnn_backbone}")
        self.cnn_backbone = timm.create_model(
            cnn_backbone,
            pretrained=pretrained,
            num_classes=0
        )
        self.cnn_dim = self.cnn_backbone.num_features
        
        print(f"加载 Transformer backbone: {transformer_backbone}")
        self.transformer_backbone = timm.create_model(
            transformer_backbone,
            pretrained=pretrained,
            num_classes=0
        )
        self.transformer_dim = self.transformer_backbone.num_features
        
        print(f"  CNN 特征维度: {self.cnn_dim}")
        print(f"  Transformer 特征维度: {self.transformer_dim}")
        
        # ===== 特征融合 =====
        if fusion_method == "concat":
            self.fused_dim = self.cnn_dim + self.transformer_dim
            self.fusion = None  # 直接拼接
        
        elif fusion_method == "add":
            # 需要投影到相同维度
            assert self.cnn_dim == self.transformer_dim, \
                f"add 融合要求维度相同: {self.cnn_dim} vs {self.transformer_dim}"
            self.fused_dim = self.cnn_dim
            self.fusion = None
        
        elif fusion_method == "attention":
            # 使用注意力机制融合
            self.fused_dim = self.cnn_dim  # 输出与 CNN 维度相同
            self.fusion = FusionAttention(self.cnn_dim, self.transformer_dim)
        
        else:
            raise ValueError(f"未知融合方法: {fusion_method}")
        
        print(f"融合方法: {fusion_method}, 融合后维度: {self.fused_dim}")
        
        # ===== 任务 Head =====
        # 分类
        self.type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, num_types)
        )
        
        self.airline_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, num_airlines)
        )
        
        # 回归
        self.clarity_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.block_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.num_types = num_types
        self.num_airlines = num_airlines
    
    def forward(self, x):
        # 提取两个 backbone 的特征
        cnn_features = self.cnn_backbone(x)          # [B, cnn_dim]
        transformer_features = self.transformer_backbone(x)  # [B, transformer_dim]
        
        # 融合
        if self.fusion_method == "concat":
            fused = torch.cat([cnn_features, transformer_features], dim=1)
        elif self.fusion_method == "add":
            fused = cnn_features + transformer_features
        elif self.fusion_method == "attention":
            fused = self.fusion(cnn_features, transformer_features)
        
        # 各任务预测
        return {
            'type': self.type_head(fused),
            'airline': self.airline_head(fused),
            'clarity': self.clarity_head(fused).squeeze(-1),
            'block': self.block_head(fused).squeeze(-1),
        }
    
    def get_features(self, x):
        """返回融合后的特征"""
        cnn_features = self.cnn_backbone(x)
        transformer_features = self.transformer_backbone(x)
        
        if self.fusion_method == "concat":
            return torch.cat([cnn_features, transformer_features], dim=1)
        elif self.fusion_method == "add":
            return cnn_features + transformer_features
        elif self.fusion_method == "attention":
            return self.fusion(cnn_features, transformer_features)


class FusionAttention(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, cnn_dim: int, transformer_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 投影到相同维度
        self.cnn_proj = nn.Linear(cnn_dim, hidden_dim)
        self.trans_proj = nn.Linear(transformer_dim, hidden_dim)
        
        # 注意力权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, cnn_dim)
    
    def forward(self, cnn_feat, trans_feat):
        # 投影
        cnn_proj = self.cnn_proj(cnn_feat)      # [B, hidden]
        trans_proj = self.trans_proj(trans_feat)  # [B, hidden]
        
        # 计算注意力权重
        combined = torch.cat([cnn_proj, trans_proj], dim=1)  # [B, hidden*2]
        weights = self.attention(combined)  # [B, 2]
        
        # 加权融合
        w_cnn = weights[:, 0:1]   # [B, 1]
        w_trans = weights[:, 1:2]  # [B, 1]
        
        fused = w_cnn * cnn_proj + w_trans * trans_proj  # [B, hidden]
        
        # 输出
        return self.output_proj(fused)
```

### 1.2 测试 Hybrid 模型

```python
# training/scripts/test_hybrid.py
"""测试 Hybrid 模型"""

import sys
sys.path.append('training/src')

import torch
from models.hybrid import HybridModel

def test():
    print("测试 Hybrid 模型...")
    
    # 测试不同融合方法
    for fusion in ['concat', 'attention']:
        print(f"\n--- 融合方法: {fusion} ---")
        
        model = HybridModel(
            num_types=10,
            num_airlines=12,
            cnn_backbone="convnext_base",
            transformer_backbone="swin_base_patch4_window7_224",
            fusion_method=fusion
        )
        
        # 参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数量: {total_params / 1e6:.2f}M")
        
        # 测试前向传播
        x = torch.randn(2, 3, 224, 224)
        
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        print(f"输出形状:")
        for k, v in outputs.items():
            print(f"  {k}: {v.shape}")
    
    print("\n✅ Hybrid 模型测试通过！")

if __name__ == "__main__":
    test()
```

---

## 第二步：训练策略

### 2.1 分阶段训练（推荐）

由于 Hybrid 模型参数量大，建议分阶段训练：

```python
# 阶段 A：冻结 Backbone，只训练 Head（5 epochs）
for param in model.cnn_backbone.parameters():
    param.requires_grad = False
for param in model.transformer_backbone.parameters():
    param.requires_grad = False

# 阶段 B：解冻 Transformer，微调（10 epochs）
for param in model.transformer_backbone.parameters():
    param.requires_grad = True

# 阶段 C：全部解冻，小学习率微调（15 epochs）
for param in model.cnn_backbone.parameters():
    param.requires_grad = True
```

### 2.2 训练脚本

```python
# training/scripts/train_stage5.py
"""阶段 5 训练脚本：Hybrid 模型"""

import sys
sys.path.append('training/src')

import torch
from torch.utils.data import DataLoader

from data.dataset import AircraftDataset, full_task_collate_fn
from data.transforms import get_train_transform, get_val_transform, AlbumentationsWrapper
from models.hybrid import HybridModel
from trainers.full_trainer import FullMultiTaskTrainer

def freeze_backbone(model, freeze_cnn=True, freeze_transformer=True):
    """冻结 backbone"""
    for param in model.cnn_backbone.parameters():
        param.requires_grad = not freeze_cnn
    for param in model.transformer_backbone.parameters():
        param.requires_grad = not freeze_transformer
    
    # 统计可训练参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

def main():
    config = {
        # 数据
        'train_csv': 'training/data/processed/aircraft_crop/train.csv',
        'val_csv': 'training/data/processed/aircraft_crop/val.csv',
        'train_dir': 'training/data/processed/aircraft_crop/train',
        'val_dir': 'training/data/processed/aircraft_crop/val',
        'image_size': 224,
        'batch_size': 16,  # Hybrid 模型更大，减小 batch
        'num_workers': 4,
        
        # 模型
        'cnn_backbone': 'convnext_base',
        'transformer_backbone': 'swin_base_patch4_window7_224',
        'fusion_method': 'concat',
        'dropout': 0.3,
        
        # 训练
        'epochs': 30,
        'lr': 5e-5,  # Hybrid 用更小的学习率
        'weight_decay': 0.01,
        
        # 任务权重
        'type_weight': 1.0,
        'airline_weight': 0.5,
        'clarity_weight': 0.3,
        'block_weight': 0.3,
        
        'save_dir': 'training/checkpoints/stage5',
    }
    
    # 数据加载
    train_transform = AlbumentationsWrapper(get_train_transform(config['image_size']))
    val_transform = AlbumentationsWrapper(get_val_transform(config['image_size']))
    
    train_dataset = AircraftDataset(
        csv_path=config['train_csv'],
        image_dir=config['train_dir'],
        transform=train_transform,
        task='full'
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
    
    # 创建模型
    model = HybridModel(
        num_types=train_dataset.num_types,
        num_airlines=train_dataset.num_airlines,
        cnn_backbone=config['cnn_backbone'],
        transformer_backbone=config['transformer_backbone'],
        fusion_method=config['fusion_method'],
        dropout=config['dropout']
    )
    
    # ===== 分阶段训练 =====
    
    # 阶段 A：冻结 backbone，训练 Head
    print("\n" + "="*60)
    print("阶段 A：训练 Head（冻结 Backbone）")
    print("="*60)
    
    freeze_backbone(model, freeze_cnn=True, freeze_transformer=True)
    
    config_a = config.copy()
    config_a['epochs'] = 5
    config_a['lr'] = 1e-3
    config_a['save_dir'] = 'training/checkpoints/stage5/phase_a'
    
    trainer_a = FullMultiTaskTrainer(model, train_loader, val_loader, config_a)
    trainer_a.train()
    
    # 阶段 B：解冻 Transformer
    print("\n" + "="*60)
    print("阶段 B：微调 Transformer")
    print("="*60)
    
    freeze_backbone(model, freeze_cnn=True, freeze_transformer=False)
    
    config_b = config.copy()
    config_b['epochs'] = 10
    config_b['lr'] = 5e-5
    config_b['save_dir'] = 'training/checkpoints/stage5/phase_b'
    
    trainer_b = FullMultiTaskTrainer(model, train_loader, val_loader, config_b)
    trainer_b.train()
    
    # 阶段 C：全部解冻
    print("\n" + "="*60)
    print("阶段 C：全模型微调")
    print("="*60)
    
    freeze_backbone(model, freeze_cnn=False, freeze_transformer=False)
    
    config_c = config.copy()
    config_c['epochs'] = 15
    config_c['lr'] = 1e-5
    config_c['save_dir'] = 'training/checkpoints/stage5'
    
    trainer_c = FullMultiTaskTrainer(model, train_loader, val_loader, config_c)
    trainer_c.train()
    
    print("\n🎉 Hybrid 模型训练完成！")


if __name__ == "__main__":
    main()
```

---

## 第三步：模型对比

### 3.1 对比脚本

```python
# training/scripts/compare_models.py
"""对比不同模型的效果"""

import sys
sys.path.append('training/src')

import torch
from pathlib import Path

def compare():
    models = {
        'Stage 2 (ConvNeXt only)': 'training/checkpoints/stage2/best.pth',
        'Stage 4 (Full Multi-task)': 'training/checkpoints/stage4/best.pth',
        'Stage 5 (Hybrid)': 'training/checkpoints/stage5/best.pth',
    }
    
    print("=" * 70)
    print("模型对比")
    print("=" * 70)
    print(f"{'模型':<30} {'Type Acc':<12} {'Clarity MAE':<12} {'Block MAE':<12}")
    print("-" * 70)
    
    for name, path in models.items():
        if Path(path).exists():
            ckpt = torch.load(path, map_location='cpu')
            history = ckpt.get('history', {})
            
            type_acc = max(history.get('val_type_acc', [0]))
            clarity_mae = min(history.get('val_clarity_mae', [1]))
            block_mae = min(history.get('val_block_mae', [1]))
            
            print(f"{name:<30} {type_acc:<12.4f} {clarity_mae:<12.4f} {block_mae:<12.4f}")
        else:
            print(f"{name:<30} {'(未找到)':<12}")
    
    print("=" * 70)

if __name__ == "__main__":
    compare()
```

---

## ✅ 过关标准

- [ ] Hybrid 模型能正常训练
- [ ] 机型准确率比阶段 4 提升 ≥ 1%
- [ ] 推理速度可接受（< 100ms/张）
- [ ] GPU 显存占用 < 20GB

---

## ❌ 禁止事项

- ❌ 从头训练（必须加载预训练权重）
- ❌ 一开始就解冻全部参数
- ❌ batch_size 太大导致显存不足

---

## 💡 优化建议

### 显存优化

```python
# 1. 使用混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = compute_loss(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 2. 梯度累积
accumulation_steps = 4
for i, (images, labels) in enumerate(dataloader):
    loss = compute_loss(model(images), labels) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 模型选择

| 模型组合 | 参数量 | 显存需求 | 效果 |
|----------|--------|----------|------|
| convnext_small + swin_small | ~100M | ~10GB | 适中 |
| convnext_base + swin_base | ~180M | ~16GB | 较好 |
| convnext_large + swin_large | ~400M | ~24GB | 最佳（需大显存） |

---

## 🔜 下一步

完成所有检查项后，进入 [阶段 6：OCR 注册号识别](stage6_ocr.md)

