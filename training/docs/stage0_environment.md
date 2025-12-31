# 阶段 0：环境配置与基础认知

> ⏱️ 预计时间：1 天
> 🎯 目标：跑通环境，理解深度学习基础概念

---

## 📋 本阶段检查清单

开始前，确保你有：
- [ ] 一台带 NVIDIA GPU 的电脑（至少 RTX 3060 12GB）
- [ ] 安装了 Python 3.9+
- [ ] 安装了 CUDA 11.8+ 和 cuDNN

---

## 第一步：创建项目环境

### 1.1 创建虚拟环境

```bash
# 进入项目目录
cd F:\bian\pyproject\Aerovision-V1

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境（Windows）
venv\Scripts\activate

# 激活后命令行前面会出现 (venv)
```

**为什么要虚拟环境？**
- 隔离项目依赖，避免不同项目包冲突
- 方便复现环境

### 1.2 安装 PyTorch

```bash
# 安装 CUDA 12.1 版本的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**验证 PyTorch 安装：**

```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
```

如果显示 `CUDA 可用: True`，说明安装成功。

### 1.3 安装其他依赖

```bash
# 安装训练相关包
pip install timm==1.0.3           # 预训练模型库（很重要！）
pip install ultralytics==8.1.0    # YOLOv8（用于裁剪飞机）
pip install albumentations==1.4.0 # 数据增强
pip install pandas                # 数据处理
pip install scikit-learn          # 机器学习工具
pip install matplotlib            # 可视化
pip install tqdm                  # 进度条
pip install tensorboard           # 训练可视化
pip install pyyaml                # 配置文件

# 可选：实验追踪（推荐）
pip install wandb
```

---

## 第二步：验证环境

创建并运行以下脚本：

```python
# training/scripts/verify_env.py
"""环境验证脚本 - 运行这个确保一切正常"""

import sys

def check_import(module_name, package_name=None):
    """检查模块是否可以导入"""
    try:
        module = __import__(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: 未安装 - {e}")
        return False

def main():
    print("=" * 50)
    print("环境检查")
    print("=" * 50)
    
    all_ok = True
    
    # 检查 Python 版本
    py_version = sys.version_info
    if py_version >= (3, 9):
        print(f"✅ Python: {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"❌ Python: {py_version.major}.{py_version.minor} (需要 3.9+)")
        all_ok = False
    
    # 检查必要的包
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('timm', 'timm'),
        ('ultralytics', 'ultralytics'),
        ('albumentations', 'albumentations'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
    ]
    
    for module, name in packages:
        if not check_import(module, name):
            all_ok = False
    
    print()
    
    # 检查 CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA 可用")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA 不可用 - 训练会非常慢！")
        all_ok = False
    
    print()
    
    # 测试模型加载
    print("测试模型加载...")
    try:
        import timm
        model = timm.create_model("convnext_base", pretrained=True)
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        print(f"✅ ConvNeXt 模型加载成功，输出形状: {y.shape}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        all_ok = False
    
    print()
    print("=" * 50)
    if all_ok:
        print("🎉 所有检查通过！可以开始下一阶段")
    else:
        print("⚠️ 有些检查未通过，请修复后再继续")
    print("=" * 50)

if __name__ == "__main__":
    main()
```

运行：
```bash
python training/scripts/verify_env.py
```

---

## 第三步：理解基础概念

### 3.1 什么是深度学习模型？

一个图像分类模型可以简化为：

```
输入图片 → [Backbone 提取特征] → [Head 输出预测] → 预测结果
   │              │                    │              │
224x224x3    1024维向���           10个类别概率     "A320"
```

**关键概念：**

| 概念 | 解释 | 类比 |
|------|------|------|
| **Backbone** | 提取图片特征的网络 | 相当于"眼睛"，看图片提取信息 |
| **Head** | 根据特征做预测的层 | 相当于"大脑"，根据信息做判断 |
| **Loss** | 预测值和真实值的差距 | 相当于"分数"，越低越好 |
| **Optimizer** | 更新模型参数的方法 | 相当于"老师"，指导如何进步 |
| **Epoch** | 遍历整个数据集一次 | 相当于"做完一套题" |
| **Batch** | 一次处理的图片数量 | 相当于"每次看几道题" |

### 3.2 动手理解：一个最简单的 forward

```python
# training/scripts/simple_forward.py
"""最简单的模型前向传播示例"""

import torch
import timm

# 1. 创建模型
# pretrained=True 表示使用在 ImageNet 上预训练的权重
model = timm.create_model("convnext_base", pretrained=True)
print(f"模型类型: {type(model).__name__}")

# 2. 创建一个假的输入图片
# 形状: [batch_size, channels, height, width]
# 这里是 1 张 224x224 的 RGB 图片
x = torch.randn(1, 3, 224, 224)
print(f"输入形状: {x.shape}")

# 3. 前向传播
model.eval()  # 设置为评估模式
with torch.no_grad():  # 不计算梯度（推理时）
    y = model(x)

print(f"输出形状: {y.shape}")  # [1, 1000] - ImageNet 有 1000 个类别

# 4. 获取预测类别
pred_class = y.argmax(dim=1).item()
print(f"预测类别索引: {pred_class}")

# 5. 查看概率分布
probs = torch.softmax(y, dim=1)
top5_probs, top5_indices = probs.topk(5)
print(f"Top-5 预测:")
for prob, idx in zip(top5_probs[0], top5_indices[0]):
    print(f"  类别 {idx.item()}: {prob.item():.4f}")
```

运行这个脚本，观察输出，确保你理解每一步。

### 3.3 理解训练流程

```python
# 伪代码 - 不需要运行，理解流程即可
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # 1. 前向传播：模型看图片
        predictions = model(images)
        
        # 2. 计算损失：预测和真实值差多少
        loss = loss_function(predictions, labels)
        
        # 3. 反向传播：计算每个参数该怎么调整
        loss.backward()
        
        # 4. 更新参数：按计算出的方向调整
        optimizer.step()
        
        # 5. 清零梯度：为下一次迭代准备
        optimizer.zero_grad()
```

---

## 第四步：创建目录结构

```bash
# 在项目根目录运行
mkdir -p training/configs
mkdir -p training/data/raw
mkdir -p training/data/processed/aircraft_crop/unsorted
mkdir -p training/data/processed/aircraft_crop/train
mkdir -p training/data/processed/aircraft_crop/val
mkdir -p training/data/processed/aircraft_crop/test
mkdir -p training/data/labels/registration
mkdir -p training/src/data
mkdir -p training/src/models
mkdir -p training/src/trainers
mkdir -p training/src/utils
mkdir -p training/scripts
mkdir -p training/checkpoints
mkdir -p training/logs
```

或者用 Python：

```python
# training/scripts/setup_directories.py
from pathlib import Path

dirs = [
    "training/configs",
    "training/data/raw",
    "training/data/processed/aircraft_crop/unsorted",
    "training/data/processed/aircraft_crop/train",
    "training/data/processed/aircraft_crop/val",
    "training/data/processed/aircraft_crop/test",
    "training/data/labels/registration",
    "training/src/data",
    "training/src/models",
    "training/src/trainers",
    "training/src/utils",
    "training/scripts",
    "training/checkpoints",
    "training/logs",
]

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✅ 创建目录: {d}")

print("\n目录结构创建完成！")
```

---

## ✅ 过关标准

在进入阶段 1 之前，确保：

- [ ] `python training/scripts/verify_env.py` 全部通过
- [ ] 理解 Backbone、Head、Loss 的概念
- [ ] 能运行 `simple_forward.py` 并理解输出
- [ ] 目录结构已创建

---

## ❌ 禁止事项

在本阶段，**不要**：

- ❌ 开始收集数据
- ❌ 研究 Swin、Hybrid 等高级模型
- ❌ 想多任务学习
- ❌ 写训练代码

**专注于理解基础！**

---

## 🔜 下一步

完成所有检查项后，进入 [阶段 1：数据准备](stage1_data_preparation.md)

