# training/config/config_loader.py
"""配置文件加载器"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os


class Config:
    """配置类，用于加载和访问YAML配置"""

    def __init__(self, config_dict: Dict[str, Any], config_file_path: Optional[Path] = None):
        """
        初始化配置对象

        Args:
            config_dict: 配置字典
            config_file_path: 配置文件路径（用于解析相对路径）
        """
        self._config = config_dict
        self._config_base_path = config_file_path.parent if config_file_path else Path.cwd()

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self._config[key]

    def __getattr__(self, key: str) -> Any:
        """支持属性式访问"""
        if key.startswith('_'):
            return object.__getattribute__(self, key)
        return self._config.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键（如 'paths.data_root'）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_path(self, key: str, create: bool = False) -> Path:
        """
        获取路径配置并转换为Path对象
        相对路径会相对于yaml配置文件所在目录进行解析

        Args:
            key: 配置键
            create: 是否创建目录（如果不存在）

        Returns:
            Path对象（绝对路径）
        """
        path_str = self.get(key)
        if path_str is None:
            raise ValueError(f"配置键 '{key}' 未找到")

        path = Path(path_str)

        # 如果是相对路径，则相对于配置文件所在目录
        if not path.is_absolute():
            path = (self._config_base_path / path).resolve()

        if create and not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        return path

    def update(self, updates: Dict[str, Any]):
        """
        更新配置

        Args:
            updates: 要更新的配置字典
        """
        self._deep_update(self._config, updates)

    @staticmethod
    def _deep_update(base_dict: Dict, update_dict: Dict):
        """递归更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                Config._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: Optional[str] = None, **kwargs) -> Config:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，如果为None则加载默认配置
        **kwargs: 额外的配置覆盖项

    Returns:
        Config对象

    Example:
        >>> config = load_config()
        >>> config = load_config('custom_config.yaml')
        >>> config = load_config(yolo={'conf_threshold': 0.6})
    """
    # 确定配置文件路径
    if config_path is None:
        # 使用默认配置
        current_dir = Path(__file__).parent
        config_path = current_dir / "default.yaml"
    else:
        config_path = Path(config_path)

    # 检查文件是否存在
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    # 加载YAML文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    # 创建Config对象
    config = Config(config_dict, config_path)

    # 应用额外的配置覆盖
    if kwargs:
        config.update(kwargs)

    return config


def save_config(config: Config, save_path: str):
    """
    保存配置到文件

    Args:
        config: Config对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    print(f"配置已保存到: {save_path}")


# 便捷函数：获取默认配置
def get_default_config() -> Config:
    """获取默认配置"""
    return load_config()


if __name__ == "__main__":
    # 测试代码
    print("=== 测试配置加载器 ===\n")

    # 加载默认配置
    config = load_config()

    # 测试不同的访问方式
    print("1. 字典式访问:")
    print(f"   data_root = {config['paths']['data_root']}")

    print("\n2. get方法（支持点号分隔）:")
    print(f"   yolo.conf_threshold = {config.get('yolo.conf_threshold')}")
    print(f"   crop.padding = {config.get('crop.padding')}")

    print("\n3. get_path方法:")
    print(f"   logs_root = {config.get_path('paths.logs_root')}")

    print("\n4. 属性式访问:")
    print(f"   yolo配置 = {config.yolo}")

    print("\n5. 配置覆盖:")
    config2 = load_config(yolo={'conf_threshold': 0.8})
    print(f"   新的conf_threshold = {config2.get('yolo.conf_threshold')}")

    print("\n6. 保存配置:")
    save_path = Path(__file__).parent / "test_config.yaml"
    save_config(config, str(save_path))

    print("\n=== 测试完成 ===")
