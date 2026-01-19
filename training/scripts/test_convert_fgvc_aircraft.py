#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGVC_Aircraft 转换脚本测试模块
"""

import tempfile
import shutil
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.convert_fgvc_aerocraft import FGVC_AircraftConverter


def test_parse_annotation_file():
    """测试标注文件解析功能"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试标注文件
        annotation_file = tmpdir / "test_annotations.txt"
        annotation_file.write_text(
            "1234567 707-320\n2345678 A320-200\n3456789 Boeing 777-300ER\n"
        )

        converter = FGVC_AircraftConverter(
            fgvc_data_dir=str(tmpdir), output_dir=str(tmpdir / "output")
        )

        # 解析标注文件
        df = converter._parse_annotation_file(annotation_file)

        assert len(df) == 3, f"期望3条记录，实际{len(df)}条"
        assert df.iloc[0]["image_id"] == "1234567", f"第一条记录image_id错误"
        assert df.iloc[0]["label"] == "707-320", f"第一条记录label错误"
        assert df.iloc[1]["label"] == "A320-200", f"第二条记录label错误"
        assert df.iloc[2]["label"] == "Boeing 777-300ER", f"第三条记录label错误"

        print("✓ 测试通过: parse_annotation_file")


def test_convert_to_project_format():
    """测试转换为项目格式"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试数据目录结构
        fgvc_dir = tmpdir / "fgvc_data"
        fgvc_dir.mkdir()
        fgvc_data_dir = fgvc_dir / "data"
        fgvc_data_dir.mkdir()

        # 创建图片目录
        fgvc_images_dir = fgvc_data_dir / "images"
        fgvc_images_dir.mkdir()

        # 创建测试图片
        test_image = fgvc_images_dir / "1234567.jpg"
        test_image.write_bytes(b"fake_image_data")

        # 创建标注文件
        variant_file = fgvc_data_dir / "images_variant_train.txt"
        variant_file.write_text("1234567 707-320\n")

        family_file = fgvc_data_dir / "images_family_train.txt"
        family_file.write_text("1234567 Boeing 707\n")

        manufacturer_file = fgvc_data_dir / "images_manufacturer_train.txt"
        manufacturer_file.write_text("1234567 Boeing\n")

        box_file = fgvc_data_dir / "images_box.txt"
        box_file.write_text("1234567 100 100 200 200\n")

        # 创建输出目录
        output_dir = tmpdir / "output"

        # 执行转换
        converter = FGVC_AircraftConverter(
            fgvc_data_dir=str(fgvc_dir),
            output_dir=str(output_dir),
            default_clarity=0.9,
            default_block=0.0,
        )

        converter.convert()

        # 验证输出
        output_csv = output_dir / "labels.csv"
        assert output_csv.exists(), "输出CSV文件不存在"

        df = pd.read_csv(output_csv)

        # 验证必需字段
        assert "filename" in df.columns, "缺少filename列"
        assert "typename" in df.columns, "缺少typename列"
        assert "clarity" in df.columns, "缺少clarity列"
        assert "block" in df.columns, "缺少block列"

        # 验证数据
        assert len(df) == 1, f"期望1条记录，实际{len(df)}条"
        assert df.iloc[0]["filename"] == "1234567.jpg", f"filename错误"
        assert df.iloc[0]["typename"] == "707-320", f"typename错误"
        assert df.iloc[0]["airline"] == "Boeing", f"airline错误"
        assert df.iloc[0]["clarity"] == 0.9, f"clarity错误"
        assert df.iloc[0]["block"] == 0.0, f"block错误"

        # 验证图片已复制
        output_images = output_dir / "images"
        assert (output_images / "1234567.jpg").exists(), "图片未复制"

        print("✓ 测试通过: convert_to_project_format")


def test_box_file_parsing():
    """测试边界框文件解析"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试边界框文件
        box_file = tmpdir / "test_box.txt"
        box_file.write_text("1234567 100 100 200 200\n2345678 50 50 150 150\n")

        converter = FGVC_AircraftConverter(
            fgvc_data_dir=str(tmpdir), output_dir=str(tmpdir / "output")
        )

        boxes = converter._parse_box_file(box_file)

        assert "1234567" in boxes, "缺少图片1234567的边界框"
        assert "2345678" in boxes, "缺少图片2345678的边界框"
        assert boxes["1234567"] == [100, 100, 200, 200], "边界框数据错误"

        print("✓ 测试通过: box_file_parsing")


def test_merge_annotations():
    """测试多个标注文件的合并"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试数据
        converter = FGVC_AircraftConverter(
            fgvc_data_dir=str(tmpdir), output_dir=str(tmpdir / "output")
        )

        # 模拟三个数据源
        variant_df = pd.DataFrame(
            {
                "image_id": ["001", "002", "003"],
                "label": ["Variant1", "Variant2", "Variant3"],
            }
        )

        family_df = pd.DataFrame(
            {
                "image_id": ["001", "002", "003"],
                "label": ["Family1", "Family2", "Family3"],
            }
        )

        manufacturer_df = pd.DataFrame(
            {"image_id": ["001", "002", "003"], "label": ["Maker1", "Maker2", "Maker3"]}
        )

        merged_df = converter._merge_annotations(variant_df, family_df, manufacturer_df)

        assert len(merged_df) == 3, "合并后的记录数错误"
        assert "typename" in merged_df.columns, "缺少typename列"
        assert "family" in merged_df.columns, "缺少family列"
        assert "airline" in merged_df.columns, "缺少airline列"

        assert merged_df.iloc[0]["typename"] == "Variant1", "typename合并错误"
        assert merged_df.iloc[0]["family"] == "Family1", "family合并错误"
        assert merged_df.iloc[0]["airline"] == "Maker1", "airline合并错误"

        print("✓ 测试通过: merge_annotations")


def test_filter_missing_images():
    """测试过滤缺失图片"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # 创建测试数据
        fgvc_data_dir = tmpdir / "fgvc" / "data"
        fgvc_data_dir.mkdir(parents=True)
        fgvc_images_dir = fgvc_data_dir / "images"
        fgvc_images_dir.mkdir()

        # 只创建一张图片
        (fgvc_images_dir / "001.jpg").write_bytes(b"img1")

        # 创建3条标注
        df = pd.DataFrame(
            {"image_id": ["001", "002", "003"], "label": ["Type1", "Type2", "Type3"]}
        )

        converter = FGVC_AircraftConverter(
            fgvc_data_dir=str(fgvc_data_dir.parent), output_dir=str(tmpdir / "output")
        )

        filtered_df = converter._filter_missing_images(df, fgvc_images_dir)

        assert len(filtered_df) == 1, f"应该只有1条有效记录，实际{len(filtered_df)}条"
        assert filtered_df.iloc[0]["image_id"] == "001", "保留的图片ID错误"

        print("✓ 测试通过: filter_missing_images")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行 FGVC_Aircraft 转换测试")
    print("=" * 60 + "\n")

    tests = [
        test_parse_annotation_file,
        test_convert_to_project_format,
        test_box_file_parsing,
        test_merge_annotations,
        test_filter_missing_images,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ 测试失败: {test.__name__}")
            print(f"  错误: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
