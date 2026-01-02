# training/scripts/review_crops.py
"""简单的图片浏览脚本，用于检查裁剪结果"""

import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import random
import sys
from datetime import datetime

# 添加configs模块路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import load_config


def review_random_samples(
        image_dir: str = None,
        n_samples: int = None,
        config_path: str = None
):
    """
    随机查看一些裁剪结果

    Args:
        image_dir: 图片目录（如果为None则从配置读取）
        n_samples: 样本数量（如果为None则从配置读取）
        config_path: 自定义配置文件路径
    """
    # 加载配置
    config = load_config(config_path)

    # 使用参数或配置值
    image_dir = image_dir or config.get('paths.aircraft_crop')
    n_samples = n_samples if n_samples is not None else config.get('review.n_samples')
    cols = config.get('review.grid_cols', 5)
    fig_width = config.get('review.fig_width', 15)
    row_height = config.get('review.row_height', 3)
    dpi = config.get('review.dpi', 150)
    title_length = config.get('review.title_length', 15)

    image_path = Path(image_dir)
    images = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

    if len(images) == 0:
        print("未找到图片！")
        return

    samples = random.sample(images, min(n_samples, len(images)))

    # 显示图片网格
    rows = (len(samples) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, row_height * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(img_path.name[:title_length] + "...", fontsize=8)
        ax.axis('off')

    # 隐藏多余的子图
    for ax in axes[len(samples):]:
        ax.axis('off')

    plt.tight_layout()

    # 获取保存路径并替换时间戳占位符
    save_path_template = config.get('paths.crop_review')

    # 生成UTC时间戳：格式为 YYYYMMDD_HHMMSS
    utc_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    # 替换{timestamp}占位符
    save_path_str = save_path_template.replace('{timestamp}', utc_timestamp)

    # 转换为Path对象并解析相对路径
    save_path = Path(save_path_str)
    if not save_path.is_absolute():
        config_base = Path(config._config_base_path)
        save_path = (config_base / save_path).resolve()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=dpi)
    plt.show()
    print(f"已保存到 {save_path}")


if __name__ == "__main__":
    review_random_samples()
