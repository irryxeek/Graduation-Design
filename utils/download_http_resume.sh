#!/bin/bash
set -u

if [ $# -lt 2 ]; then
  echo "用法: $0 <url_list.txt> <output_dir> [start_line] [end_line]"
  exit 1
fi

LIST_FILE="$1"
OUTPUT_DIR="$2"
START_LINE="${3:-1}"
END_LINE="${4:-0}"

if [ ! -f "$LIST_FILE" ]; then
  echo "错误: 清单文件不存在: $LIST_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

SUCCESS_LOG="$OUTPUT_DIR/download_success.log"
FAIL_LOG="$OUTPUT_DIR/download_fail.log"
RUN_LOG="$OUTPUT_DIR/download_run.log"

touch "$SUCCESS_LOG" "$FAIL_LOG" "$RUN_LOG"

line_no=0
while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  line_no=$((line_no + 1))
  if [ "$line_no" -lt "$START_LINE" ]; then
    continue
  fi
  if [ "$END_LINE" -gt 0 ] && [ "$line_no" -gt "$END_LINE" ]; then
    break
  fi

  url="$(printf '%s' "$raw_line" | sed 's/\r$//')"
  if [ -z "$url" ]; then
    continue
  fi

  file_name="${url%%\?*}"
  file_name="${file_name##*/}"
  out_path="$OUTPUT_DIR/$file_name"

  echo "[$(date '+%F %T')] line=$line_no start $file_name" | tee -a "$RUN_LOG"

  wget -c \
    --tries=3 \
    --waitretry=3 \
    --timeout=30 \
    --read-timeout=30 \
    --dns-timeout=15 \
    --no-verbose \
    -O "$out_path" \
    "$url" >>"$RUN_LOG" 2>&1
  status=$?

  if [ "$status" -eq 0 ]; then
    size=$(stat -c%s "$out_path" 2>/dev/null || echo 0)
    echo "$line_no|$file_name|$size|$url" >> "$SUCCESS_LOG"
    echo "[$(date '+%F %T')] line=$line_no ok size=$size $file_name" | tee -a "$RUN_LOG"
  else
    echo "$line_no|exit=$status|$file_name|$url" >> "$FAIL_LOG"
    echo "[$(date '+%F %T')] line=$line_no fail exit=$status $file_name" | tee -a "$RUN_LOG"
  fi
done < "$LIST_FILE"

echo "完成: list=$LIST_FILE out=$OUTPUT_DIR" | tee -a "$RUN_LOG"
