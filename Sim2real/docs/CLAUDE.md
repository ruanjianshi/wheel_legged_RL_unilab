# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个 ROS Noetic (C++14/Python3) 工作区，用于将强化学习（RL）策略部署到实体人形机器人（"sim2real"）。支持多硬件平台：x86_64、Jetson Orin (TensorRT) 和 Rockchip RK3588s (RKNN)。机器人通过手柄控制，状态机管理运行模式。

## 构建与测试

```bash
# 设置 ROS 环境并构建
source /opt/ros/noetic/setup.bash
catkin build -DCMAKE_CXX_FLAGS="-DRELEASE"

# 带测试构建（包含 lcov 覆盖率）
catkin build -DCMAKE_CXX_FLAGS="-DRELEASE" --catkin-make-args tests
catkin run_tests sim2real --no-deps

# 平台特定构建宏由 CMakeLists 自动检测：
# PLATFORM_X86_64, PLATFORM_JETSON, PLATFORM_RK

# 运行单个测试
catkin run_tests sim2real --no-deps --this

# 清理并重新构建
catkin clean -b -y
catkin config --install
catkin build -DCMAKE_CXX_FLAGS="-DRELEASE"

# 完整测试 + 覆盖率流程（参见 build_script.sh）：
lcov --directory build/sim2real --zerocounters
catkin run_tests sim2real --no-deps
lcov --directory build/sim2real --capture --output-file coverage.info
lcov --remove coverage.info '/usr/*' '*/test/*' --output-file coverage_filtered.info
genhtml coverage_filtered.info --output-directory coverage_report
```

## 架构

### 包依赖关系

```
sim2real_msg（自定义 ROS 消息/服务/动作）
     |
     v
sim2real（核心 RL-to-real 控制器：PD 循环、RL 推理、运动学、轨迹）
     |
     v
sim2real_master（基于 FSM 的模式管理、手柄输入、控制器调度）
     |
     +--> robot_driver（硬件通信，Python）
     +--> sim2real_dev（远程可视化，关节级调试）
     +--> sim2real_data_log（数据记录工具）
```

### `sim2real` 核心抽象

- **`Sim2Real` 类**（`src/sim2real/include/sim2real.h`）：中心控制器，管理三个并行循环 —— `execPdLoop()`（PD 位置控制）、`execRlLoop()`（RL 策略推理）、`execTrajLoop()`（路点轨迹回放）。使用 Pinocchio 进行前向运动学和雅可比计算。

- **策略继承体系**：`PolicyBase` → 具体策略（`AMPPolicy`、`PBHCPolicy`、`HumanoidGymPolicy`、`FootstepPolicy`、`DreamwaqPolicy`、`LRPolicy`、`BYDMMCPolicy`）。策略根据观测值（关节状态、IMU、速度指令）输出关节目标位置。

- **推理引擎**（编译时根据平台选择）：
  - `PLATFORM_X86_64` → `OpenVINOInferenceEngine`
  - `PLATFORM_JETSON` → `TensorRTInferenceEngine`
  - `PLATFORM_RK` → `RKNNInferenceEngine`
  - `LRInferenceEngine`（始终可用，用于线性回归风格策略）

- **`RobotMotorGroup`** 类（`PiMotorGroup`、`PiPlusMotorGroup`、`HiMotorGroup`）：不同机器人型号（pi、pi_plus、hi）的电机类型抽象。

### 模式状态机（`sim2real_master`）

FSM（Boost MSM）通过 `MasterNode` 管理机器人运行模式。手柄按键组合通过 `sim2real_msg::Joy` 消息触发模式切换。各模式对应的控制器接口：

| 模式 | 类 | 功能 |
|------|-------|---------|
| Default（默认） | `DefaultControllerInterface` | 使用 RL 策略的站立/行走/奔跑 |
| Custom（自定义） | `CustomControllerInterface` | 用户提供的策略和配置，从 `custom_config/` 加载 |
| Develop（开发者） | `DevelopControllerInterface` | 电机级别直接控制，参数调试 |
| Remote（远程） | `RemoteControllerInterface` | 通过 ROS 话题进行远程控制 |
| Teach（示教） | `TeachControllerInterfaceN` | 录制/回放关节轨迹 |
| Calibration（校零） | `ZeroCalibrationControllerInterface` | 关节零点校准 |
| Protection（保护） | `ProtectionControllerInterface` | 故障检测时关机 |

状态包括：`CANDIDATE_*`（通过手柄方向键选择模式）、`EXEC_*`（执行中）、`PROTECTION_SHUTDOWN`（保护关机）。

### 配置系统

两个 YAML 文件定义所有行为（路径通过 ROS 参数设置）：
- **PD 配置**（`pdInfo_`）：关节限位、Kp/Kd 增益、电机 ID、URDF 路径、路点/动作定义
- **RL 配置**（`rlConfigGroup_`）：观测维度、动作缩放、策略文件、手柄按键映射（用于策略切换）

支持运行时策略切换 —— `Sim2Real` 中的 `changePolicy()` 处理切换过程。

### 自定义 ROS 消息（`sim2real_msg`）

- `Joy.msg` —— 手柄状态（按键、摇杆）
- `RbtState.msg`、`MtrState.msg` —— 机器人/电机遥测
- `Orient.msg` —— 姿态指令
- `WayPoint.msg` —— 轨迹路点
- `lowlevel_controller.action` —— 轨迹执行的 ActionLib 接口

### 关键 ROS 话题

- `/joy_msg` → 手柄输入
- `/fsm_state` → 当前 FSM 状态（发布为 `std_msgs/Int32`）
- `/error_joint_states` → 关节状态反馈
- `/imu/data` → IMU 数据
- `/cmd_vel` → 速度指令（行走时使用）

## 部署目标

- **x86_64**：开发/仿真，使用 OpenVINO
- **Jetson Orin**：机器人端部署，使用 TensorRT（`sim2real_pi_plus_orin.sh`）
- **RK3588s**：低成本机器人端部署，使用 RKNN（`sim2real_pi_plus_rknn.sh`）

launch 文件命名规则：`joy_control_{机器人型号}_{推理平台}.launch`。带 `_waist` 的变体包含腰部关节控制。shell 脚本（`sim2real_pi_plus*.sh`）是一行启动器，执行 `source devel/setup.bash && roslaunch`。

## OTA 打包

`pack_release.sh` 构建部署用的 tar 包，仅包含发布产物（`install/`、main.cpp 源码、`robot_driver/`、`version.json`），移除测试文件，并通过 SCP 上传到 `172.20.7.12:/data/app/hightorque-install/download/`。

## 依赖

- ROS Noetic + catkin
- Pinocchio 3.3.0（刚体动力学，通过 `ros-noetic-pinocchio` 安装）
- LibTorch（构建时放置于工作区根目录的父目录）
- `livelybot_serial`、`livelybot_logger`（机器人 SDK 包）
- 平台特定：OpenVINO、TensorRT 或 RKNN SDK（用于推理）
- nlohmann/json（JSON 解析）
