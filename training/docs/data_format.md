# AeroVision 数据格式规范

## 数据集组织结构

```
training/data/
├── raw/                                    # 原始图片（未裁剪）
├── processed/aircraft_crop/               # 裁剪后的飞机图片
│   ├── unsorted/                           # 待标注的裁剪图片
│   ├── train/                              # 训练集（70%）
│   ├── val/                                # 验证集（15%）
│   └── test/                               # 测试集（15%）
└── labels/                                 # 标注文件
    ├── aircraft_labels.csv                 # 主标注文件
    ├── type_classes.json                   # 机型类别映射
    ├── airline_classes.json                # 航司类别映射
    ├── train.csv                           # 训练集标注
    ├── val.csv                             # 验证集标注
    ├── test.csv                            # 测试集标注
    └── registration/                       # 注册号区域标注（YOLO格式）
        ├── IMG_0001.txt
        └── ...
```

## 主标注文件格式 (aircraft_labels.csv)

### CSV 字段说明

| 字段名 | 类型 | 必填 | 说明 | 示例 |
|--------|------|------|------|------|
| filename | string | ✅ | 图片文件名 | IMG_0001.jpg |
| typeid | int | ❌ | 机型编号（可自动生成） | 0 |
| typename | string | ✅ | 机型名称 | A320, B737-800 |
| airlineid | int | ❌ | 航司编号（可自动生成） | 1 |
| airlinename | string | ❌ | 航司名称 | China Eastern |
| clarity | float | ✅ | 清晰度 0.0-1.0 | 0.95 |
| block | float | ✅ | 遮挡程度 0.0-1.0 | 0.0 |
| registration | string | ❌ | 注册号文字 | B-1234 |

### CSV 示例

```csv
filename,typeid,typename,airlineid,airlinename,clarity,block,registration
IMG_0001.jpg,0,A320,1,China Eastern,0.95,0.0,B-1234
IMG_0002.jpg,1,B737-800,0,Air China,0.80,0.15,B-5678
```

## 注册号区域标注格式 (YOLO)

### 目录位置
`training/data/labels/registration/`

### 文件命名
与图片同名，扩展名改为 `.txt`

### 格式
```
class_id x_center y_center width height
```

### 字段说明

| 字段 | 含义 | 范围 | 说明 |
|------|------|------|------|
| class_id | 类别ID | 0 | 固定为 0（registration 类） |
| x_center | 框中心 X | 0.0-1.0 | 相对图片宽度的归一化值 |
| y_center | 框中心 Y | 0.0-1.0 | 相对图片高度的归一化值 |
| width | 框宽度 | 0.0-1.0 | 相对图片宽度的归一化值 |
| height | 框高度 | 0.0-1.0 | 相对图片高度的归一化值 |

### 示例
```txt
0 0.85 0.65 0.12 0.04
```

## 类别映射文件

### type_classes.json
```json
{
  "classes": ["A320", "B737-800", "A380", ...],
  "num_classes": 10
}
```

### airline_classes.json
```json
{
  "classes": ["Air China", "China Eastern", ...],
  "num_classes": 12
}
```

## 数据质量标准

### 清晰度评分标准
- 0.9-1.0：非常清晰（细节锐利，可看清小字）
- 0.7-0.9：清晰（整体清晰，细节略有模糊）
- 0.5-0.7：一般（能辨认机型，但不够锐利）
- 0.3-0.5：模糊（勉强能辨认）
- 0.0-0.3：非常模糊（几乎无法辨认）

### 遮挡程度评分标准
- 0.0：无遮挡（飞机完全可见）
- 0.1-0.3：轻微遮挡（一小部分被遮挡）
- 0.3-0.5：部分遮挡（约 1/3 被遮挡）
- 0.5-0.7：明显遮挡（约一半被遮挡）
- 0.7-1.0：严重遮挡（大部分被遮挡）

## 数据集划分比例

- 训练集 (train): 70%
- 验证集 (val): 15%
- 测试集 (test): 15%
