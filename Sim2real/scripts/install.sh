#!/bin/bash

echo "=============================================="
echo "       机器人开发环境部署脚本 - 增量更新版"
echo " 注意: 如果 $WORKSPACE_DIR/sim2real"
echo "       将不会覆盖自定义文件如launch custom_action.yaml"
echo "=============================================="


# 获取 pinocchio 的版本号，并将错误输出重定向到空（防止没安装时报错）
VERSION=$(pkg-config --modversion pinocchio 2>/dev/null)

# 1. 首先检查是否能获取到版本号
if [ -z "$VERSION" ]; then
    echo "错误: 系统中未找到 pinocchio 的 pkg-config 信息。"
    echo "请执行 sudo apt install ros-noetic-pinocchio"
    exit 1
fi

# 2. 判断版本是否不等于 3.3.0
if [ "$VERSION" != "3.3.0" ]; then
    echo "--------------------------------------------------------"
    echo "警告: 当前 Pinocchio 版本为 [$VERSION]，不是要求的 [3.3.0]"
    echo "请执行 sudo apt install ros-noetic-pinocchio"
    echo "--------------------------------------------------------"
    exit 1
else
    echo "验证通过: 当前 Pinocchio 版本是 3.3.0"
fi

# 获取用户家目录
USER_HOME=$(eval echo ~$SUDO_USER)
echo "用户家目录: $USER_HOME"

tar -xzf ./sim2real.tar.gz -C $USER_HOME/

echo "=============================================="
echo " 部署完成！"
echo " 机器人将在系统启动时自动运行"
echo " 手动启动: ./${USER_HOME}/sim2real/sim2real_xxx.sh (在普通用户下执行)"
echo "=============================================="
