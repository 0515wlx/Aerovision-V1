# 阶段 1：数据准备与标注

> ⏱️ 预计时间：3-5 天
> 🎯 目标：获得干净的飞机裁剪图 + 完成标注
> ⚠️ 这是最重要的阶段，80% 的模型失败都死在数据上！

---

## 📋 本阶段检查清单

完成本阶段后，你需要有：
- [ ] 裁剪好的飞机图片（每张图只有飞机主体）
- [ ] 完整的标注 CSV 文件
- [ ] 注册号区域标注文件（YOLO 格式 txt）
- [ ] 类别映射 JSON 文件
- [ ] 数据质量验证通过

---

## 第一步：理解你的数据格式

### 1.1 标注文件结构

你的标注数据由**两部分**组成：

```
training/data/labels/
├── aircraft_labels.csv          # 主标注文件
├── type_classes.json            # 机型类别映射
├── airline_classes.json         # 航司类别映射
└── registration/                # 注册号区域标注
    ├── IMG_0001.txt
    ├── IMG_0002.txt
    └── ...
```

### 1.2 主标注文件 (CSV)

**文件**：`aircraft_labels.csv`

```csv
filename,typeid,typename,airlineid,airlinename,clarity,block,registration
IMG_0001.jpg,0,A320,1,China Eastern,0.95,0.0,B-1234
IMG_0002.jpg,1,B737-800,0,Air China,0.80,0.15,B-5678
IMG_0003.jpg,7,A380,8,Emirates,0.70,0.40,
IMG_0004.jpg,4,B787-9,3,Hainan Airlines,0.50,0.60,
```

**字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `filename` | string | ✅ | 图片文件名 |
| `typeid` | int | ❌ | 机型编号（可自动生成） |
| `typename` | string | ✅ | 机型名称，如 `A320`、`B737-800` |
| `airlineid` | int | ❌ | 航司编号（可自动生成） |
| `airlinename` | string | ❌ | 航司名称，如 `China Eastern` |
| `clarity` | float | ✅ | 清晰度 0.0-1.0（1.0=最清晰，0.0=最模糊） |
| `block` | float | ✅ | 遮挡程度 0.0-1.0（0.0=无遮挡，1.0=完全遮挡） |
| `registration` | string | ❌ | 注册号文字，如 `B-1234`，看不清则留空 |

### 1.3 注册号区域标注 (YOLO 格式 txt)

**目录**：`registration/`  
**文件命名**：与图片同名，扩展名改为 `.txt`

```
图片: training/data/processed/aircraft_crop/unsorted/IMG_0001.jpg
标注: training/data/labels/registration/IMG_0001.txt
```

**文件内容格式（YOLO 格式）**：

```
class_id x_center y_center width height
```

**示例**：
```
# IMG_0001.txt - 单个注册号
0 0.85 0.65 0.12 0.04

# IMG_0005.txt - 多个注册号（机身有多处）
0 0.25 0.55 0.10 0.03
0 0.82 0.48 0.08 0.025
```

**字段详解**：

| 字段 | 含义 | 范围 | 说明 |
|------|------|------|------|
| `class_id` | 类别ID | 0 | 固定为 0（只有一个类：registration） |
| `x_center` | 框中心 X | 0.0-1.0 | 相对于图片宽度的归一化值 |
| `y_center` | 框中心 Y | 0.0-1.0 | 相对于图片高度的归一化值 |
| `width` | 框宽度 | 0.0-1.0 | 相对于图片宽度的归一化值 |
| `height` | 框高度 | 0.0-1.0 | 相对于图片高度的归一化值 |

**坐标计算示例**：

```
假设图片尺寸: 1000 x 600 像素
注册号区域像素坐标: 左上(800, 360), 右下(920, 384)

计算过程：
- 框宽度 = 920 - 800 = 120 像素
- 框高度 = 384 - 360 = 24 像素
- x_center = (800 + 120/2) / 1000 = 0.86
- y_center = (360 + 24/2) / 600 = 0.62
- width = 120 / 1000 = 0.12
- height = 24 / 600 = 0.04

txt 文件内容:
0 0.86 0.62 0.12 0.04
```

**重要规则**：
- ⚠️ 如果图片中注册号**不可见**，则**不创建**对应的 `.txt` 文件
- ⚠️ 注册号的**文字内容**存在 CSV 的 `registration` 列，不是 txt 文件中
- ⚠️ txt 文件只存储**位置信息**，用于训练检测模型

---

## 第二步：飞机裁剪

### 2.1 为什么要裁剪？

原始图片通常包含大量背景（天空、机场、地面），直接用于训练会让模型学到很多无用信息。

```
原始图片                    裁剪后
┌─────────────────────┐    ┌─────────────────┐
│     天空 天空 天空    │    │                 │
│  ═══╦═══════════╦═══│ →  │  ═══════════════│
│     ║   飞机    ║   │    │     飞机         │
│  ═══╩═══════════╩═══│    │  ═══════════════│
│     跑道 跑道 跑道    │    │                 │
└─────────────────────┘    └─────────────────┘
```

### 2.2 使用 YOLOv8 自动裁剪

创建裁剪脚本：

```python
# training/scripts/crop_aircraft.py
"""使用 YOLOv8 检测并裁剪飞机"""

from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import shutil
from tqdm import tqdm

def crop_aircraft(
    input_dir: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    padding: float = 0.1,
    min_size: int = 224
):
    """
    检测并裁剪飞机
    
    Args:
        input_dir: 原始图片目录
        output_dir: 输出目录
        conf_threshold: 检测置信度阈值
        padding: 边界框扩展比例（避免裁太紧）
        min_size: 最小输出尺寸
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载 YOLOv8（COCO 预训练，包含 airplane 类别）
    model = YOLO("yolov8m.pt")  # 中等大小，平衡速度和精度
    
    # COCO 数据集中 airplane 的类别 ID 是 4
    AIRPLANE_CLASS = 4
    
    # 统计
    total = 0
    success = 0
    no_detection = 0
    too_small = 0
    
    # 获取所有图片
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")) + list(input_path.glob("*.png"))
    
    print(f"找到 {len(image_files)} 张图片")
    
    for img_file in tqdm(image_files, desc="裁剪飞机"):
        total += 1
        
        try:
            # 检测
            results = model(str(img_file), verbose=False)[0]
            
            # 筛选飞机检测结果
            boxes = results.boxes
            airplane_boxes = []
            
            for i, cls in enumerate(boxes.cls):
                if int(cls) == AIRPLANE_CLASS and boxes.conf[i] >= conf_threshold:
                    airplane_boxes.append({
                        'box': boxes.xyxy[i].cpu().numpy(),
                        'conf': boxes.conf[i].cpu().item()
                    })
            
            if not airplane_boxes:
                no_detection += 1
                continue
            
            # 选择置信度最高的（或最大的）
            best_box = max(airplane_boxes, key=lambda x: x['conf'])
            x1, y1, x2, y2 = best_box['box']
            
            # 打开原图
            img = Image.open(img_file)
            img_w, img_h = img.size
            
            # 添加 padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = box_w * padding
            pad_h = box_h * padding
            
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(img_w, x2 + pad_w)
            y2 = min(img_h, y2 + pad_h)
            
            # 检查尺寸
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                too_small += 1
                continue
            
            # 裁剪并保存
            cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
            output_file = output_path / img_file.name
            cropped.save(output_file, quality=95)
            success += 1
            
        except Exception as e:
            print(f"处理 {img_file.name} 时出错: {e}")
            continue
    
    # 打印统计
    print("\n" + "=" * 50)
    print(f"处理完成！")
    print(f"  总数: {total}")
    print(f"  成功: {success}")
    print(f"  未检测到飞机: {no_detection}")
    print(f"  太小跳过: {too_small}")
    print(f"  输出目录: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    crop_aircraft(
        input_dir="training/data/raw",
        output_dir="training/data/processed/aircraft_crop/unsorted",
        conf_threshold=0.5,
        padding=0.1
    )
```

运行：
```bash
python training/scripts/crop_aircraft.py
```

### 2.3 手动检查裁剪结果

裁剪后，**必须**人工检查一遍：

```python
# training/scripts/review_crops.py
"""简单的图片浏览脚本，用于检查裁剪结果"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random

def review_random_samples(image_dir: str, n_samples: int = 20):
    """随机查看一些裁剪结果"""
    image_path = Path(image_dir)
    images = list(image_path.glob("*.jpg"))
    
    if len(images) == 0:
        print("未找到图片！")
        return
    
    samples = random.sample(images, min(n_samples, len(images)))
    
    # 显示图片网格
    cols = 5
    rows = (len(samples) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
    
    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_path.name[:15] + "...", fontsize=8)
        ax.axis('off')
    
    # 隐藏多余的子图
    for ax in axes[len(samples):]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("training/logs/crop_review.png", dpi=150)
    plt.show()
    print(f"已保存到 training/logs/crop_review.png")

if __name__ == "__main__":
    review_random_samples("training/data/processed/aircraft_crop/unsorted")
```

**检查要点：**
- [ ] 飞机主体完整（没有被裁掉机翼、尾翼）
- [ ] 没有裁到其他飞机
- [ ] 边界适中（不要太紧也不要太松）

---

## 第三步：数据标注

### 3.1 标注工作流程

```
裁剪后的图片
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  标注工具（Label Studio / 自定义工具）                │
│                                                     │
│  对每张图片标注：                                    │
│  ├── typename (必填) - 选择机型                     │
│  ├── airlinename - 选择航司                         │
│  ├── clarity (必填) - 滑块选择 0-1                  │
│  ├── block (必填) - 滑块选择 0-1                    │
│  ├── registration - 输入注册号文字（看不清留空）      │
│  └── 注册号区域 - 画框标注位置（看不清不画）          │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  导出为：                                            │
│  ├── aircraft_labels.csv (主标注)                   │
│  └── registration/*.txt (注册号区域，YOLO格式)       │
└─────────────────────────────────────────────────────┘
```

### 3.2 标注规范

#### typename（机型）标注规范

| 规则 | 说明 | 示例 |
|------|------|------|
| 使用标准简写 | ICAO 代码简写 | A320, B737-800 |
| 区分子型号 | 不同型号分开 | A320 ≠ A321, B737-800 ≠ B737-900 |
| 不确定标 Unknown | 宁缺毋滥 | Unknown |

#### clarity（清晰度）评分标准

| 分数 | 描述 | 示例情况 |
|------|------|----------|
| 0.9-1.0 | 非常清晰 | 细节锐利，可以看清小字 |
| 0.7-0.9 | 清晰 | 整体清晰，细节略有模糊 |
| 0.5-0.7 | 一般 | 能辨认机型，但不够锐利 |
| 0.3-0.5 | 模糊 | 勉强能辨认 |
| 0.0-0.3 | 非常模糊 | 几乎无法辨认 |

#### block（遮挡程度）评分标准

| 分数 | 描述 | 示例情况 |
|------|------|----------|
| 0.0 | 无遮挡 | 飞机完全可见 |
| 0.1-0.3 | 轻微遮挡 | 一小部分被遮挡（如起落架被地面挡住） |
| 0.3-0.5 | 部分遮挡 | 约 1/3 被遮挡（如被其他飞机部分挡住） |
| 0.5-0.7 | 明显遮挡 | 约一半被遮挡 |
| 0.7-1.0 | 严重遮挡 | 大部分被遮挡，难以辨认 |

#### registration（注册号）标注规范

| 规则 | 说明 |
|------|------|
| 全大写 | `B-1234` 不是 `b-1234` |
| 保留连字符 | `B-1234` 不是 `B1234` |
| 看不清留空 | 不要猜测 |
| 多个注册号 | 只填最清晰的那个 |

### 3.3 使用 Label Studio 标注

**安装：**
```bash
pip install label-studio
label-studio start --port 8080
```

**创建项目配置 XML：**
```xml
<View>
  <Image name="image" value="$image" zoom="true"/>
  
  <!-- 机型分类 -->
  <Header value="机型 Aircraft Type"/>
  <Choices name="typename" toName="image" choice="single" required="true">
    <Choice value="A319"/><Choice value="A320"/><Choice value="A321"/>
    <Choice value="A330-200"/><Choice value="A330-300"/>
    <Choice value="A350-900"/><Choice value="A350-1000"/>
    <Choice value="A380"/>
    <Choice value="B737-700"/><Choice value="B737-800"/><Choice value="B737-900"/>
    <Choice value="B737-MAX8"/><Choice value="B737-MAX9"/>
    <Choice value="B747-400"/><Choice value="B747-8"/>
    <Choice value="B777-200"/><Choice value="B777-300ER"/>
    <Choice value="B787-8"/><Choice value="B787-9"/><Choice value="B787-10"/>
    <Choice value="ARJ21"/><Choice value="C919"/>
    <Choice value="E190"/><Choice value="E195"/>
    <Choice value="CRJ900"/>
    <Choice value="Unknown"/>
  </Choices>
  
  <!-- 航司分类 -->
  <Header value="航空公司 Airline"/>
  <Choices name="airlinename" toName="image" choice="single">
    <Choice value="Air China"/><Choice value="China Eastern"/>
    <Choice value="China Southern"/><Choice value="Hainan Airlines"/>
    <Choice value="Xiamen Airlines"/><Choice value="Shenzhen Airlines"/>
    <Choice value="Sichuan Airlines"/><Choice value="Spring Airlines"/>
    <Choice value="Juneyao Airlines"/><Choice value="China United"/>
    <Choice value="Cathay Pacific"/><Choice value="EVA Air"/>
    <Choice value="Singapore Airlines"/><Choice value="Emirates"/>
    <Choice value="Other"/><Choice value="Unknown"/>
  </Choices>
  
  <!-- 清晰度 -->
  <Header value="清晰度 Clarity (1=最清晰, 10=最模糊)"/>
  <Rating name="clarity_rating" toName="image" maxRating="10"/>
  
  <!-- 遮挡程度 -->
  <Header value="遮挡程度 Block (1=无遮挡, 10=完全遮挡)"/>
  <Rating name="block_rating" toName="image" maxRating="10"/>
  
  <!-- 注册号文字 -->
  <Header value="注册号 Registration (看不清留空)"/>
  <TextArea name="registration" toName="image" placeholder="B-1234" maxSubmissions="1"/>
  
  <!-- 注册号区域框 -->
  <Header value="注册号区域 (看不清不画)"/>
  <RectangleLabels name="registration_bbox" toName="image">
    <Label value="registration" background="#FF0000"/>
  </RectangleLabels>
</View>
```

### 3.4 导出并转换格式

```python
# training/scripts/convert_labelstudio.py
"""将 Label Studio 导出转换为训练格式"""

import json
import pandas as pd
from pathlib import Path

def convert_export(export_json: str, output_dir: str):
    """
    转换 Label Studio JSON 导出为训练格式
    
    输出:
    - aircraft_labels.csv (主标注)
    - registration/*.txt (注册号区域，YOLO格式)
    - type_classes.json (机型类别映射)
    - airline_classes.json (航司类别映射)
    """
    
    with open(export_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建注册号区域目录
    reg_dir = output_path / 'registration'
    reg_dir.mkdir(exist_ok=True)
    
    records = []
    
    for item in data:
        filename = Path(item['data']['image']).name
        results = item.get('annotations', [{}])[0].get('result', [])
        
        record = {
            'filename': filename,
            'typename': '',
            'airlinename': '',
            'clarity': 1.0,
            'block': 0.0,
            'registration': ''
        }
        
        bboxes = []
        
        for r in results:
            rtype = r.get('type', '')
            from_name = r.get('from_name', '')
            
            if rtype == 'choices':
                if from_name == 'typename':
                    choices = r.get('value', {}).get('choices', [])
                    record['typename'] = choices[0] if choices else ''
                elif from_name == 'airlinename':
                    choices = r.get('value', {}).get('choices', [])
                    record['airlinename'] = choices[0] if choices else ''
            
            elif rtype == 'rating':
                rating = r.get('value', {}).get('rating', 5)
                if from_name == 'clarity_rating':
                    # 1=最清晰 → 1.0, 10=最模糊 → 0.0
                    record['clarity'] = 1.0 - (rating - 1) / 9.0
                elif from_name == 'block_rating':
                    # 1=无遮挡 → 0.0, 10=完全遮挡 → 1.0
                    record['block'] = (rating - 1) / 9.0
            
            elif rtype == 'textarea' and from_name == 'registration':
                text_list = r.get('value', {}).get('text', [])
                text = text_list[0] if text_list else ''
                record['registration'] = text.upper().replace(' ', '')
            
            elif rtype == 'rectanglelabels' and from_name == 'registration_bbox':
                # 提取边界框
                value = r.get('value', {})
                x = value.get('x', 0) / 100  # Label Studio 用百分比
                y = value.get('y', 0) / 100
                w = value.get('width', 0) / 100
                h = value.get('height', 0) / 100
                
                # 转换为 YOLO 格式 (中心点)
                x_center = x + w / 2
                y_center = y + h / 2
                
                bboxes.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        records.append(record)
        
        # 保存注册号区域 txt 文件（如果有标注）
        if bboxes:
            txt_filename = Path(filename).stem + '.txt'
            txt_path = reg_dir / txt_filename
            txt_path.write_text('\n'.join(bboxes))
    
    # 创建 DataFrame
    df = pd.DataFrame(records)
    
    # 生成类别 ID
    types = sorted([t for t in df['typename'].unique() if t and t != 'Unknown'])
    airlines = sorted([a for a in df['airlinename'].unique() if a and a != 'Unknown'])
    
    # 确保 Unknown 在最后
    if 'Unknown' in df['typename'].values:
        types.append('Unknown')
    if 'Unknown' in df['airlinename'].values:
        airlines.append('Unknown')
    
    type_to_id = {t: i for i, t in enumerate(types)}
    airline_to_id = {a: i for i, a in enumerate(airlines)}
    
    df['typeid'] = df['typename'].map(type_to_id)
    df['airlineid'] = df['airlinename'].map(airline_to_id)
    
    # 重新排列列顺序
    columns = ['filename', 'typeid', 'typename', 'airlineid', 'airlinename', 
               'clarity', 'block', 'registration']
    df = df[columns]
    
    # 保存 CSV
    csv_path = output_path / 'aircraft_labels.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存标注: {csv_path} ({len(df)} 条)")
    
    # 保存类别映射
    type_classes = {'classes': types, 'num_classes': len(types)}
    with open(output_path / 'type_classes.json', 'w', encoding='utf-8') as f:
        json.dump(type_classes, f, indent=2, ensure_ascii=False)
    print(f"✅ 机型类别: {len(types)} 个")
    
    airline_classes = {'classes': airlines, 'num_classes': len(airlines)}
    with open(output_path / 'airline_classes.json', 'w', encoding='utf-8') as f:
        json.dump(airline_classes, f, indent=2, ensure_ascii=False)
    print(f"✅ 航司类别: {len(airlines)} 个")
    
    # 统计注册号区域标注
    reg_files = list(reg_dir.glob('*.txt'))
    print(f"✅ 注册号区域标注: {len(reg_files)} 个文件")


if __name__ == "__main__":
    convert_export(
        export_json="export.json",  # Label Studio 导出的 JSON 文件
        output_dir="training/data/labels"
    )
```

---

## 第四步：数据集划分

### 4.1 划分原则

| 集合 | 比例 | 用途 |
|------|------|------|
| 训练集 (train) | 70% | 模型学习 |
| 验证集 (val) | 15% | 调参、Early Stopping |
| 测试集 (test) | 15% | 最终评估（只用一次） |

**重要原则：**
- 同一架飞机的照片应该在同一个集合（避免数据泄露）
- 各类别在各集合中比例应该接近（分层抽样）

### 4.2 划分脚本

```python
# training/scripts/split_dataset.py
"""数据集划分"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

def split_dataset(
    csv_path: str,
    image_dir: str,
    output_base: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    划分数据集
    
    Args:
        csv_path: 标注 CSV 文件
        image_dir: 原始图片目录
        output_base: 输出基础目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例（剩余为测试集）
        random_seed: 随机种子（保证可复现）
    """
    # 读取标注
    df = pd.read_csv(csv_path)
    print(f"总样本数: {len(df)}")
    
    # 过滤掉没有机型标注的
    df = df[df['typename'].notna() & (df['typename'] != '')]
    print(f"有效样本数: {len(df)}")
    
    # 分层划分（按机型）
    # 先分出测试集
    test_ratio = 1 - train_ratio - val_ratio
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_ratio,
        stratify=df['typename'],
        random_state=random_seed
    )
    
    # 再从训练+验证中分出验证集
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df['typename'],
        random_state=random_seed
    )
    
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")
    
    # 复制图片到对应目录
    image_path = Path(image_dir)
    output_path = Path(output_base)
    
    def copy_images(subset_df, split_name):
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f"复制 {split_name}"):
            src = image_path / row['filename']
            if src.exists():
                dst = split_dir / row['filename']
                shutil.copy2(src, dst)
    
    copy_images(train_df, 'train')
    copy_images(val_df, 'val')
    copy_images(test_df, 'test')
    
    # 保存划分后的 CSV
    train_df.to_csv(output_path / 'train.csv', index=False)
    val_df.to_csv(output_path / 'val.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    
    print("\n✅ 数据集划分完成！")
    
    # 打印各类别分布
    print("\n各机型分布:")
    for typename in sorted(df['typename'].unique()):
        train_count = len(train_df[train_df['typename'] == typename])
        val_count = len(val_df[val_df['typename'] == typename])
        test_count = len(test_df[test_df['typename'] == typename])
        print(f"  {typename:15} Train:{train_count:4} Val:{val_count:3} Test:{test_count:3}")


if __name__ == "__main__":
    split_dataset(
        csv_path="training/data/labels/aircraft_labels.csv",
        image_dir="training/data/processed/aircraft_crop/unsorted",
        output_base="training/data/processed/aircraft_crop"
    )
```

---

## 第五步：数据质量验证

### 5.1 验证脚本

```python
# training/scripts/verify_data.py
"""数据质量验证"""

import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter

def verify_dataset(data_dir: str, labels_dir: str):
    """验证数据集质量"""
    
    data_path = Path(data_dir)
    labels_path = Path(labels_dir)
    
    # 读取主标注文件
    csv_path = labels_path / 'aircraft_labels.csv'
    if not csv_path.exists():
        print(f"❌ 找不到标注文件: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    
    issues = []
    warnings = []
    
    print("=" * 60)
    print("数据质量检查")
    print("=" * 60)
    
    # 1. 检查图片是否存在
    print("\n📁 检查图片文件...")
    missing_images = []
    for split in ['train', 'val', 'test', 'unsorted']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
        
        split_csv = data_path / f'{split}.csv'
        if split_csv.exists():
            split_df = pd.read_csv(split_csv)
            check_df = split_df
        else:
            check_df = df
        
        for filename in check_df['filename']:
            img_path = split_dir / filename
            if not img_path.exists():
                # 尝试其他扩展名
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                    alt_path = split_dir / (Path(filename).stem + ext)
                    if alt_path.exists():
                        found = True
                        break
                if not found:
                    missing_images.append(str(img_path))
    
    if missing_images:
        issues.append(f"❌ {len(missing_images)} 个图片文件缺失")
        for p in missing_images[:5]:
            print(f"   缺失: {p}")
        if len(missing_images) > 5:
            print(f"   ... 还有 {len(missing_images) - 5} 个")
    else:
        print("✅ 所有图片文件存在")
    
    # 2. 检查标注完整性
    print("\n📋 检查标注完整性...")
    empty_typename = df[df['typename'].isna() | (df['typename'] == '')]
    if len(empty_typename) > 0:
        issues.append(f"❌ {len(empty_typename)} 条记录缺少 typename")
    else:
        print("✅ 所有记录都有 typename")
    
    # 3. 检查 clarity 和 block 范围
    print("\n📊 检查数值范围...")
    if 'clarity' in df.columns:
        invalid_clarity = df[(df['clarity'] < 0) | (df['clarity'] > 1)]
        if len(invalid_clarity) > 0:
            issues.append(f"❌ {len(invalid_clarity)} 条 clarity 不在 0-1 范围")
        else:
            print(f"✅ clarity 范围正确 [0, 1]，均值: {df['clarity'].mean():.2f}")
    
    if 'block' in df.columns:
        invalid_block = df[(df['block'] < 0) | (df['block'] > 1)]
        if len(invalid_block) > 0:
            issues.append(f"❌ {len(invalid_block)} 条 block 不在 0-1 范围")
        else:
            print(f"✅ block 范围正确 [0, 1]，均值: {df['block'].mean():.2f}")
    
    # 4. 检查类别分布
    print("\n📈 机型分布:")
    type_counts = Counter(df['typename'].dropna())
    
    min_samples = 50
    for typename, count in type_counts.most_common():
        bar = "█" * (count // 20)
        status = "⚠️" if count < min_samples else "  "
        print(f"  {status} {typename:15} {count:4} {bar}")
        if count < min_samples:
            warnings.append(f"⚠️ {typename} 只有 {count} 个样本，建议增加到 {min_samples}+")
    
    # 5. 检查注册号区域标注
    print("\n📍 检查注册号区域标注...")
    reg_dir = labels_path / 'registration'
    if reg_dir.exists():
        reg_files = list(reg_dir.glob('*.txt'))
        print(f"  注册号区域标注文件: {len(reg_files)} 个")
        
        # 检查格式
        format_errors = 0
        for txt_file in reg_files[:100]:  # 抽样检查
            try:
                content = txt_file.read_text().strip()
                if content:
                    for line in content.split('\n'):
                        parts = line.strip().split()
                        if len(parts) != 5:
                            format_errors += 1
                            break
                        # 检查数值范围
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            format_errors += 1
                            break
            except Exception as e:
                format_errors += 1
        
        if format_errors > 0:
            issues.append(f"❌ {format_errors} 个注册号区域标注格式错误")
        else:
            print("✅ 注册号区域标注格式正确")
        
        # 检查与 CSV 的对应关系
        reg_count_in_csv = df['registration'].notna().sum() - (df['registration'] == '').sum()
        print(f"  CSV 中有注册号的记录: {reg_count_in_csv} 条")
    else:
        print("  ⚠️ 注册号区域目录不存在（如果不需要 OCR 可忽略）")
    
    # 6. 检查重复
    print("\n🔍 检查重复...")
    duplicates = df[df.duplicated(subset=['filename'], keep=False)]
    if len(duplicates) > 0:
        issues.append(f"❌ 发现 {len(duplicates)} 条重复记录")
    else:
        print("✅ 无重复记录")
    
    # 汇总
    print("\n" + "=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    
    if issues:
        print("\n❌ 严重问题（必须修复）:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️ 警告（建议处理）:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n🎉 所有检查通过！数据质量良好")
    elif not issues:
        print("\n✅ 无严重问题，可以继续（建议处理警告）")
    else:
        print("\n❌ 请修复严重问题后再继续")
    
    return len(issues) == 0


if __name__ == "__main__":
    verify_dataset(
        data_dir="training/data/processed/aircraft_crop",
        labels_dir="training/data/labels"
    )
```

---

## ✅ 过关标准

在进入阶段 2 之前，确保：

- [ ] 有至少 1000+ 张裁剪好的飞机图片
- [ ] 每个机型至少 50+ 张图片
- [ ] `aircraft_labels.csv` 包含所有必要字段（filename, typename, clarity, block）
- [ ] `type_classes.json` 和 `airline_classes.json` 已生成
- [ ] 注册号区域标注文件格式正确（`registration/*.txt`，YOLO 格式）
- [ ] 数据已划分为 train/val/test
- [ ] `verify_data.py` 无严重错误

---

## 📦 最终文件结构

```
training/data/
├── processed/
│   └── aircraft_crop/
│       ├── unsorted/          # 裁剪后待标注（标注时使用）
│       │   ├── IMG_0001.jpg
│       │   └── ...
│       ├── train/             # 训练集（划分后）
│       ├── val/               # 验证集
│       └── test/              # 测试集
│
└── labels/
    ├── aircraft_labels.csv    # 主标注文件
    ├── type_classes.json      # 机型类别映射
    ├── airline_classes.json   # 航司类别映射
    ├── train.csv              # 训练集标注
    ├── val.csv                # 验证集标注
    ├── test.csv               # 测试集标注
    └── registration/          # 注册号区域标注 (YOLO 格式)
        ├── IMG_0001.txt       # 0 x_center y_center width height
        ├── IMG_0002.txt
        └── ...
```

---

## ❌ 禁止事项

在本阶段，**不要**：

- ❌ 开始写训练代码
- ❌ 纠结于完美标注（先完成，再完美）
- ❌ 同时标注所有字段（先标 typename 和 clarity/block）

---

## 💡 小技巧

1. **批量标注**：先按机型分组，一次性标注同一机型的所有图片
2. **不确定就标 Unknown**：宁缺毋滥，错误标注比没有标注更糟糕
3. **定期备份**：每标注 100 张就导出一次
4. **记录问题图片**：遇到不确定的图片，记下来稍后处理
5. **注册号区域**：如果看不清注册号，**不要画框也不要填文字**

---

## 🔜 下一步

完成所有检查项后，进入 [阶段 2：单任务训练](stage2_single_task.md)

