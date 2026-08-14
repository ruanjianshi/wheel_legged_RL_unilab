#!/usr/bin/env bash
# 一键编译脚本：使用 xelatex 编译，输出到 out/ 目录
set -e
cd "$(dirname "$0")"

latexmk -xelatex main.tex

echo ""
echo "✅ 编译完成，PDF 位于：out/main.pdf"
echo "   (使用 'latexmk -c' 清理编译中间文件)"
