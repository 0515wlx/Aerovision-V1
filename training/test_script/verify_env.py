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
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
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