#!/bin/bash
# 批量转换 FGVC_Aircraft 数据集的所有分割

set -e

FGVC_DIR="D:\Users\34737\PycharmProgram\Aerovision-Lab\Aerovision-V1\FGVC_Aircraft\raw"
OUTPUT_BASE="D:\Users\34737\PycharmProgram\Aerovision-Lab\Aerovision-V1\data\fgvc_converted"

echo "============================================================"
echo "开始批量转换 FGVC_Aircraft 数据集"
echo "============================================================"
echo ""

# 转换训练集
echo "转换训练集..."
cd "D:\Users\34737\PycharmProgram\Aerovision-Lab\Aerovision-V1\training\scripts"
python convert_fgvc_aerocraft.py --fgvc-dir "$FGVC_DIR" --output "$OUTPUT_BASE/train" --split train

echo ""
echo "转换验证集..."
python convert_fgvc_aerocraft.py --fgvc-dir "$FGVC_DIR" --output "$OUTPUT_BASE/val" --split val

echo ""
echo "转换测试集..."
python convert_fgvc_aerocraft.py --fgvc-dir "$FGVC_DIR" --output "$OUTPUT_BASE/test" --split test

echo ""
echo "============================================================"
echo "所有转换完成!"
echo "============================================================"
echo ""
echo "合并所有数据集..."

# 合并所有分割集的标签文件
OUTPUT_DIR="$OUTPUT_BASE/combined"
mkdir -p "$OUTPUT_DIR/images"

# 复制图片
echo "复制训练集图片..."
cp -r "$OUTPUT_BASE/train/images/"* "$OUTPUT_DIR/images/" 2>/dev/null || true

echo "复制验证集图片..."
cp -r "$OUTPUT_BASE/val/images/"* "$OUTPUT_DIR/images/" 2>/dev/null || true

echo "复制测试集图片..."
cp -r "$OUTPUT_BASE/test/images/"* "$OUTPUT_DIR/images/" 2>/dev/null || true

# 合并CSV文件
echo "合并CSV文件..."
cat "$OUTPUT_BASE/train/labels.csv" > "$OUTPUT_DIR/labels.csv"
tail -n +2 "$OUTPUT_BASE/val/labels.csv" >> "$OUTPUT_DIR/labels.csv"
tail -n +2 "$OUTPUT_BASE/test/labels.csv" >> "$OUTPUT_DIR/labels.csv"

echo ""
echo "============================================================"
echo "合并完成!"
echo "输出目录: $OUTPUT_DIR"
echo "============================================================"
echo ""
echo "下一步：运行 prepare_dataset.py"
echo "python prepare_dataset.py --labels $OUTPUT_DIR/labels.csv --images $OUTPUT_DIR/images"
