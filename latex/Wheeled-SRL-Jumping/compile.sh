#!/bin/bash
# Wheeled-SRL 论文编译脚本
# 用法: ./compile.sh         # 编译PDF
#       ./compile.sh clean   # 清理
#       ./compile.sh view    # 编译并打开PDF

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

case "$1" in
  clean)
    echo "🧹 清理中间文件..."
    latexmk -c
    rm -f out/main.pdf
    echo "✅ 清理完成"
    ;;
  view)
    echo "📄 编译并打开PDF..."
    latexmk
    open out/main.pdf
    echo "✅ 完成"
    ;;
  *)
    echo "📄 编译论文..."
    latexmk
    echo "✅ PDF: out/main.pdf"
    ;;
esac
