#!/bin/bash
# 自动检测 utils 目录下 A 开头的 txt 列表文件并批量下载 FY-3D GNOS 数据

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOWNLOAD_SH="$SCRIPT_DIR/download_normal.sh"
OUTPUT_DIR="${1:-$SCRIPT_DIR/../Data/Sample/FY3D}"

# 检查 download_normal.sh 是否存在
if [ ! -f "$DOWNLOAD_SH" ]; then
    echo "错误: 找不到 $DOWNLOAD_SH"
    exit 1
fi

chmod 777 "$DOWNLOAD_SH"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 查找所有 A 开头的 txt 文件
TXT_FILES=("$SCRIPT_DIR"/A*.txt)

if [ ! -e "${TXT_FILES[0]}" ]; then
    echo "未找到 A 开头的 txt 列表文件，退出。"
    exit 0
fi

echo "找到 ${#TXT_FILES[@]} 个列表文件，开始下载..."

for TXT in "${TXT_FILES[@]}"; do
    echo ""
    echo ">>> 处理: $(basename "$TXT")"
    "$DOWNLOAD_SH" "$TXT" "$OUTPUT_DIR"
    if [ $? -eq 0 ]; then
        echo "下载完成，删除列表文件: $(basename "$TXT")"
        rm "$TXT"
    else
        echo "警告: 下载过程中出现错误，保留列表文件: $(basename "$TXT")"
    fi
done

echo ""
echo "全部完成，数据保存在: $OUTPUT_DIR"
