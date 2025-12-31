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

    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")

    # 加载 YOLOv8（COCO 预训练，包含 airplane 类别）
    model = YOLO("../model/yolov8m.pt")  # 中等大小，平衡速度和精度

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
        input_dir="../data/raw",
        output_dir="../data/processed/aircraft_crop/unsorted",
        conf_threshold=0.5,
        padding=0.1
    )