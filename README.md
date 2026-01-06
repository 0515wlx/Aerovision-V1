# AeroVision-V1

航空摄影智能审核系统 - AI 微服务

## 简介

AeroVision 是一个面向航空摄影社区（类似 JetPhotos）的 AI 审核微服务，为上传的航空照片提供自动化质量审核和内容识别。

## 功能特性

| 功能 | 说明 |
|------|------|
| 图片质量评估 | 清晰度、曝光、构图、噪点、色彩 |
| 飞机识别 | 是否包含飞机、机型识别、航司识别 |
| 注册号识别 | OCR 识别、清晰度评估、格式验证 |
| 遮挡检测 | 主体遮挡比例、关键部位遮挡 |
| 违规检测 | 水印、敏感内容、过度后期 |

## 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| Web 框架 | FastAPI |
| 深度学习 | PyTorch, YOLOv8, timm |
| OCR | PaddleOCR |
| 图像处理 | Pillow, OpenCV, albumentations |
| 部署 | Docker, Uvicorn |

## 项目架构

```
AeroVision-V1/
│
├── app/                    # API 服务层
│   ├── main.py            # FastAPI 应用入口
│   ├── api/               # API 路由
│   │   └── routes/
│   │       ├── review.py  # 审核接口
│   │       └── health.py  # 健康检查
│   ├── core/              # 核心配置
│   │   ├── config.py
│   │   └── logging.py
│   ├── schemas/           # 请求/响应模型
│   │   ├── request.py
│   │   └── response.py
│   └── services/          # 业务逻辑
│       └── review_service.py
│
├── infer/                  # 推理服务层
│   ├── __init__.py
│   ├── classifier.py      # 分类推理（机型、航司）
│   ├── detector.py        # 检测推理（注册号区域）
│   ├── ocr.py             # OCR 推理（注册号识别）
│   ├── quality.py         # 质量评估
│   └── models/            # 模型加载器
│       ├── loader.py
│       └── registry.py
│
├── training/               # 模型训练层
│   ├── configs/           # 训练配置
│   ├── scripts/           # 训练脚本
│   ├── ocr/               # OCR 模块
│   ├── data/              # 数据目录
│   ├── model/             # 预训练模型
│   ├── ckpt/              # 检查点
│   ├── logs/              # 训练日志
│   ├── output/            # 训练输出
│   └── README.md          # 训练模块文档
│
├── models/                 # 生产模型存放
├── tests/                  # 测试
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── conductor.md            # 训练路线图
```

## 模块说明

### app - API 服务

FastAPI 构建的 RESTful API 服务，对外提供审核接口。

**职责**:
- 接收审核请求（图片 URL 或 Base64）
- 调用 infer 模块执行推理
- 聚合结果并返回响应
- 健康检查与监控

**主要接口**:

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/review` | 图片审核 |
| GET | `/api/v1/health` | 健康检查 |

### infer - 推理服务

封装所有模型推理逻辑，提供统一的推理接口供 app 层调用。

**职责**:
- 加载训练好的模型
- 执行图片推理
- 输出结构化结果

**核心组件**:

| 组件 | 功能 |
|------|------|
| `classifier.py` | 机型分类、航司识别 |
| `detector.py` | 注册号区域检测 |
| `ocr.py` | 注册号文字识别 |
| `quality.py` | 图片质量评估 |

### training - 模型训练

模型训练与数据处理模块，详见 [training/README.md](training/README.md)。

**训练任务**:

| 任务 | 模型 | 脚本 |
|------|------|------|
| 机型分类 | YOLOv8-cls | `train_classify.py` |
| 航司识别 | YOLOv8-cls | `train_airline.py` |
| 注册号检测 | YOLOv8 | `train_detection.py` |
| 注册号 OCR | PaddleOCR | `paddle_ocr.py` |

## 快速开始

### 环境要求

- Python 3.11+
- CUDA 11.8+ (GPU 推理/训练)
- Docker & Docker Compose (生产部署)

### 安装

```bash
# 克隆仓库
git clone https://github.com/AeroVision-Lab/Aerovision-V1.git
cd Aerovision-V1

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 填写配置
```

**环境变量**:

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 服务端口 | 8000 |
| `LOG_LEVEL` | 日志级别 | INFO |
| `MODEL_DIR` | 模型目录 | ./models |
| `DEVICE` | 推理设备 | cuda:0 |
| `QUALITY_THRESHOLD` | 质量阈值 | 0.70 |
| `REGISTRATION_CLARITY_THRESHOLD` | 注册号清晰度阈值 | 0.80 |
| `OCCLUSION_THRESHOLD` | 遮挡比例阈值 | 0.20 |

### 启动服务

```bash
# 开发模式
uvicorn app.main:app --reload --port 8000

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker 部署

```bash
# 构建镜像
docker build -t aerovision:latest .

# 启动服务
docker-compose up -d
```

## API 文档

启动服务后访问: `http://localhost:8000/docs`

### 审核接口

```
POST /api/v1/review
```

**请求**:

```json
{
  "image_url": "https://example.com/photo.jpg",
  "image_base64": "...",
  "review_types": ["quality", "aircraft", "registration", "occlusion", "violation"]
}
```

**响应**:

```json
{
  "success": true,
  "review_id": "550e8400-e29b-41d4-a716-446655440000",
  "results": {
    "overall_pass": true,
    "quality": {
      "pass": true,
      "score": 0.85,
      "details": {
        "sharpness": 0.90,
        "exposure": 0.80,
        "composition": 0.85
      }
    },
    "aircraft": {
      "pass": true,
      "is_aircraft": true,
      "confidence": 0.98,
      "aircraft_type": "Boeing 737-800",
      "airline": "China Eastern"
    },
    "registration": {
      "pass": true,
      "detected": true,
      "value": "B-1234",
      "confidence": 0.95,
      "clarity_score": 0.88
    },
    "occlusion": {
      "pass": true,
      "occlusion_percentage": 0.05
    },
    "violation": {
      "pass": true,
      "has_watermark": false,
      "has_sensitive_content": false
    }
  },
  "fail_reasons": []
}
```

### 健康检查

```
GET /api/v1/health
```

```json
{
  "status": "healthy",
  "models": {
    "classifier": "loaded",
    "detector": "loaded",
    "ocr": "loaded"
  },
  "device": "cuda:0"
}
```

## 审核模块详情

### 质量评估 (Quality)

| 指标 | 说明 | 权重 |
|------|------|:----:|
| sharpness | 清晰度/对焦准确性 | 30% |
| exposure | 曝光正确性 | 25% |
| composition | 构图质量 | 20% |
| noise | 噪点程度 | 15% |
| color | 色彩还原 | 10% |

### 飞机识别 (Aircraft)

- 是否包含飞机
- 飞机类别: 客机、货机、公务机、军机
- 机型识别: Boeing 737-800、Airbus A320 等
- 航司涂装识别

### 注册号识别 (Registration)

- 区域检测 (YOLOv8)
- OCR 文字识别 (PaddleOCR)
- 清晰度评分 (0-1)
- 格式验证 (各国注册号格式)

### 遮挡检测 (Occlusion)

- 主体遮挡比例
- 关键部位: 机头、机尾、发动机、注册号
- 遮挡物类型识别

### 违规检测 (Violation)

- 水印/Logo 检测
- 敏感内容检测
- 过度后期处理检测

## 模型文件

训练好的模型放置在 `models/` 目录:

```
models/
├── aircraft_classifier.pt     # 机型分类模型
├── airline_classifier.pt      # 航司识别模型
├── registration_detector.pt   # 注册号检测模型
└── ocr/                       # OCR 模型缓存
```

## 开发指南

### 代码规范

- 遵循 PEP 8
- 使用 type hints
- 使用 Pydantic 数据验证
- 异步优先 (async/await)

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 文件名 | snake_case | `review_service.py` |
| 类名 | PascalCase | `ReviewService` |
| 函数名 | snake_case | `analyze_image` |
| 常量 | UPPER_SNAKE | `MAX_IMAGE_SIZE` |

### 测试

```bash
# 运行测试
pytest

# 覆盖率报告
pytest --cov=app --cov=infer --cov-report=html
```

### 代码质量

```bash
# 格式化
black .

# 代码检查
ruff check .

# 类型检查
mypy app/ infer/
```

## 相关文档

| 文档 | 说明 |
|------|------|
| [training/README.md](training/README.md) | 模型训练指南 |
| [conductor.md](conductor.md) | 训练路线图 |
| [CLAUDE.md](CLAUDE.md) | AI 辅助开发指南 |

## 相关项目

| 项目 | 说明 |
|------|------|
| QuanPhotos-backend | 图库后端服务 |
| QuanPhotos-web | 图库前端应用 |

## License

[LICENSE](LICENSE)
