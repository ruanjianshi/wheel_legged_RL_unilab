#!/bin/bash
"""
-*- coding: utf-8 -*-
============================================================================
功能描述：sim2real 项目发布打包脚本

本脚本用于将 sim2real 项目打包成可部署的 OTA 升级包，主要完成以下任务：
1. 从 version.json 读取版本号
2. 创建临时目录并复制必要的文件
3. 清理不需要的文件（如测试文件、多余的 .cpp 文件）
4. 生成 tar.gz 压缩包
5. 通过 SCP 将压缩包上传到远程服务器

打包内容：
- install/           : 安装目录
- src/sim2real/      : 主模块源码（仅保留 main.cpp）
- src/sim2real_master/ : 主控模块源码（仅保留 main.cpp）
- src/robot_driver/  : 机器人驱动模块
- version.json       : 版本信息
- install.sh         : 安装脚本

使用方式：
  bash pack_release.sh

注意：需要提前配置 SSH 免密登录到目标服务器（172.20.7.12）
============================================================================
"""

set -e  # 遇到错误立即退出

# 定义变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION=""
if [ -f "${SCRIPT_DIR}/version.json" ]; then
    # VERSION=$(jq -r '.[0]' version.json)
    VERSION=$(grep -oP '(?<=").*(?=")' version.json)
    echo $VERSION
fi

# OTA_PACKAGE_NAME="sim2real_release_$(date +%Y%m%d_%H%M%S)"
OTA_PACKAGE_NAME="sim2real_release_${VERSION}"
PACKAGE_NAME="sim2real"
TEMP="/tmp"
TEMP_DIR="${TEMP}/${PACKAGE_NAME}"
ARCHIVE_NAME="${PACKAGE_NAME}.tar.gz"
OTA_ARCHIVE_NAME="${OTA_PACKAGE_NAME}.tar.gz"
INSTALL_DELETE_FILES="custom_action.yaml *.launch"
SIM2REAL_DELETE_FILES="*.json"

echo "Start packaging..."
echo "Working directory: ${SCRIPT_DIR}"
echo "Temp directory: ${TEMP_DIR}"
echo "Archive name: ${ARCHIVE_NAME}"
echo "OTA Archive name: ${OTA_ARCHIVE_NAME}"

# 创建临时目录
rm -rf "${TEMP_DIR}"
mkdir -p "${TEMP_DIR}"

# 1. 复制 install 文件夹（如果存在）
if [ -d "${SCRIPT_DIR}/install" ]; then
    cp -r "${SCRIPT_DIR}/install" "${TEMP_DIR}/"

    for PATTERN in $INSTALL_DELETE_FILES; do
        # 关键：PATTERN 不要加引号在 find 的 -name 后面，或者用单引号包裹
        # 这里使用 -name "$PATTERN" 是最标准的做法
        # 它会将 *.launch 传递给 find，由 find 负责在文件系统中匹配
        find "${TEMP_DIR}/install" -name "$PATTERN" -exec rm -rf {} + 2>/dev/null
        echo "  Processing deletion pattern: $PATTERN"
    done
else
    echo "ERROR: install folder not found, skip"
fi

# 2. 复制 src/sim2real 文件夹，但只保留 main.cpp
if [ -d "${SCRIPT_DIR}/src/sim2real" ]; then
    echo "Copy src/sim2real (keep only main.cpp)..."
    mkdir -p "${TEMP_DIR}/src/sim2real"
    
    # 复制整个文件夹结构
    cp -r "${SCRIPT_DIR}/src/sim2real"/* "${TEMP_DIR}/src/sim2real/"
    
    for PATTERN in $SIM2REAL_DELETE_FILES; do
        # 关键：PATTERN 不要加引号在 find 的 -name 后面，或者用单引号包裹
        # 这里使用 -name "$PATTERN" 是最标准的做法
        # 它会将 *.launch 传递给 find，由 find 负责在文件系统中匹配
        find "${TEMP_DIR}/src/sim2real" -name "$PATTERN" -exec rm -rf {} + 2>/dev/null
        echo "  Processing deletion pattern: $PATTERN"
    done

    # 删除 src 目录下除了 main.cpp 之外的所有 .cpp 文件
    if [ -d "${TEMP_DIR}/src/sim2real/src" ]; then
        find "${TEMP_DIR}/src/sim2real/src" -name "*.cpp" ! -name "main.cpp" -delete
        echo "Deleted all .cpp except main.cpp in src/sim2real/src"
    fi
    # 删除 test 目录
    if [ -d "${TEMP_DIR}/src/sim2real/test" ]; then
        rm -rf "${TEMP_DIR}/src/sim2real/test"
        echo "Removed src/sim2real/test directory"
    fi
else
    echo "WARNING: src/sim2real folder not found, skip"
fi

# 3. 复制 sim2real_master 文件夹，但只保留 main.cpp
if [ -d "${SCRIPT_DIR}/src/sim2real_master" ]; then
    echo "Copy src/sim2real_master (keep only main.cpp)..."
    mkdir -p "${TEMP_DIR}/src/sim2real_master"

    # 复制整个文件夹结构
    cp -r "${SCRIPT_DIR}/src/sim2real_master"/* "${TEMP_DIR}/src/sim2real_master/"
    
    # 删除 src 目录下除了 main.cpp 之外的所有 .cpp 文件
    if [ -d "${TEMP_DIR}/src/sim2real_master/src" ]; then
        find "${TEMP_DIR}/src/sim2real_master/src" -name "*.cpp" ! -name "main.cpp" -delete
        echo "Deleted all .cpp except main.cpp in src/sim2real_master/src"
    fi
else
    echo "WARNING: src/sim2real_master folder not found, skip"
fi

# 4. 复制 robot_driver 文件夹
if [ -d "${SCRIPT_DIR}/src/robot_driver" ]; then
    echo "Copy robot_driver folder..."
    cp -r "${SCRIPT_DIR}/src/robot_driver" "${TEMP_DIR}/src/"
else
    echo "WARNING: src/robot_driver folder not found, skip"
fi

# 5. 复制 version.json 文件
if [ -f "${SCRIPT_DIR}/version.json" ]; then
    echo "Copy version.json file..."
    cp -r "${SCRIPT_DIR}/version.json" "${TEMP_DIR}/"
else
    echo "WARNING: version.json file not found, skip"
fi

# 5. 复制 install.sh 文件
if [ -f "${SCRIPT_DIR}/install.sh" ]; then
    echo "Copy install.sh file..."
    cp -r "${SCRIPT_DIR}/install.sh" "${TEMP}/"
else
    echo "WARNING: install.sh file not found, skip"
fi


# 6. 创建sim2real压缩包
echo "Creating archive..."
cd ${TEMP}
pwd
echo "tar -czf '${ARCHIVE_NAME}' '${PACKAGE_NAME}' "
tar -czf "${ARCHIVE_NAME}" "${PACKAGE_NAME}"
rm sim2real -rf
mkdir sim2real
mv install.sh ${ARCHIVE_NAME} sim2real -f
echo "tar -czf '${SCRIPT_DIR}/${OTA_ARCHIVE_NAME}' 'sim2real' "
tar -czf ${SCRIPT_DIR}/${OTA_ARCHIVE_NAME} sim2real


# 清理临时目录
rm -rf "${TEMP_DIR}"

# 显示结果
echo ""
echo "Packaging completed!"
echo "Archive path: ${SCRIPT_DIR}/${OTA_ARCHIVE_NAME}"
echo "Archive size: $(du -h "${SCRIPT_DIR}/${OTA_ARCHIVE_NAME}" | cut -f1)"

# 显示压缩包内容
echo ""
echo "Archive content:"
tar -tzf "${SCRIPT_DIR}/${OTA_ARCHIVE_NAME}" | head -20
if [ $(tar -tzf "${SCRIPT_DIR}/${OTA_ARCHIVE_NAME}" | wc -l) -gt 20 ]; then
    echo "... (more files)"
fi

echo ""
echo "Extract with:"
echo "tar -xzf ${OTA_ARCHIVE_NAME} && ./install.sh"
echo "start scp Archive "
ls -l ${SCRIPT_DIR}/${OTA_ARCHIVE_NAME}
scp -o StrictHostKeyChecking=no ${SCRIPT_DIR}/${OTA_ARCHIVE_NAME} root@172.20.7.12:/data/app/hightorque-install/download/${OTA_ARCHIVE_NAME} 
echo "end scp Archive "

