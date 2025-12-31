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
    plt.savefig("../logs/crop_review.png", dpi=150)
    plt.show()
    print(f"已保存到 training/logs/crop_review.png")


if __name__ == "__main__":
    review_random_samples("../data/processed/aircraft_crop/unsorted")