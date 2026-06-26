# Sim2Real 部署开发文档

> 将强化学习策略部署到实体人形机器人的完整流程与架构说明。

## 目录

1. [架构概览](#1-架构概览)
2. [三线程控制模型](#2-三线程控制模型)
3. [观测输入构造](#3-观测输入构造)
4. [动作输出流程](#4-动作输出流程)
5. [模型转换管线](#5-模型转换管线)
6. [RK3588s NPU 部署](#6-rk3588s-npu-部署)
7. [配置系统详解](#7-配置系统详解)
8. [手柄操作说明](#8-手柄操作说明)
9. [新机器人适配指南](#9-新机器人适配指南)
10. [常见问题排查](#10-常见问题排查)

---

## 1. 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  训练服务器 (x86 + GPU)                                    │
│  PPO/AMP训练 → .pt (TorchScript)                         │
│       ↓ tools/pt2onnx.py                                       │
│  .onnx → tools/onnx2rknn.py → .rknn (或 .trt for Jetson)       │
└──────────────────────────────────────────────────────────┘
                          │ scp 部署
                          ▼
┌──────────────────────────────────────────────────────────┐
│  RK3588s 机器人 (Ubuntu 20.04 + ROS Noetic)               │
│                                                            │
│  joy_control_pi_plus_rknn.launch 启动:                     │
│  ├─ lr_control           → 底层串口+CAN驱动               │
│  ├─ yesense_imu_node     → /imu/data (姿态+角速度)        │
│  ├─ power_node           → /battery_voltage/current       │
│  ├─ joy_teleop           → /joy_msg (手柄输入)             │
│  └─ sim2real_master_node → C++ 核心: FSM + 三线程控制      │
│                                                            │
│  硬件总线:                                                  │
│  /dev/ttyACM (4Mbaud) → STM32H730 → CAN-FD BRS → 电机    │
│  /dev/ttyS7  (460800)  → Yesense IMU                      │
│  /dev/input/js0         → 蓝牙手柄                         │
└──────────────────────────────────────────────────────────┘
```

## 2. 三线程控制模型

### 2.1 线程关系

```
execRlLoop (50Hz)                     execPdLoop (1kHz)              电机控制器
─────────────────                    ─────────────────             ─────────
1. 读取电机位置/速度/力矩            1. 读 motorOutput_             内部1-10kHz
2. Motor→Robot 坐标变换              2. 送电机指令 (ZOH)            PD闭环
3. 构造观测 (69维)                   3. sendMotors()
4. 帧堆叠 (×5 = 345维)              4. 坠落检测
5. RKNN NPU 推理 (20维action)       5. 状态机切换
6. 动作裁剪+缩放+重排
7. Robot→Motor 逆变换
8. 写入 motorOutput_ ←── outputMutex_ (shared_timed_mutex) ──→ 读取
```

**补偿机制**: PD 1kHz vs RL 50Hz — 无插值，纯零阶保持（ZOH）。RL线程每20ms更新一次 `motorOutput_`，PD线程在20ms内重复发送10次相同指令。电机自身高频PD闭环平滑过渡。

### 2.2 状态机流转

```
手柄 LT+RT + Dpad ← → : 选模式 (Default → Custom → Remote → Calibration → Teaching)
手柄 LT+RT + A        : 确认进入
手柄 LB               : RL行走 开/关
手柄 L                : 站立归零 (3秒轨迹)
手柄 RB               : 坐下 (3秒轨迹)
手柄 B                : 紧急停机
```

```
状态图:
INIT → STANDING → STANDBY → RUNNING
                  ↑            ↓ (坠落)
                  └── PRE_STANDING ← POLICY_STANDING
```

## 3. 观测输入构造

### 3.1 AMPPolicy (pi_plus 22DOF, 策略20DOF)

```cpp
// 单帧: ang_vel(3) + proj_gravity(3) + cmd_vel(3) + q(20) + dq(20) + last_act(20) = 69维

obs[0:3]   = base_ang_vel × rbt_ang_vel_scale    // IMU角速度(基座坐标系)
obs[3:6]   = quat.rotate(0,0,-1)                  // 重力方向投影
obs[6]     = filtered_vx × cmdVelLineScale        // 手柄vx (IIR滤波)
obs[7]     = filtered_vy × cmdVelLineScale        // 手柄vy
obs[8]     = filtered_dyaw × cmdVelAngleScale     // 手柄角速度
obs[9:29]  = q_policy[:20] × rbtLinPoseScale      // 20个策略关节位置
obs[29:49] = dq_policy[:20] × rbtLinVelScale      // 20个策略关节速度
obs[49:69] = last_action[:20]                      // 上一帧20个策略输出
                                              ────
帧堆叠 ×5                                    = 69维
                                              ────
模型输入                                     345维 (1×345)
```

### 3.2 HumanoidgymPolicy (pi 12DOF)

```cpp
// 单帧: sin/cos(2) + cmd(3) + q(12) + dq(12) + last(12) + ang_vel(3) + euler(3) = 47维
obs[0]  = sin(2π·t/frequency)
obs[1]  = cos(2π·t/frequency)
obs[2:5] = cmd_vel × scale
obs[5:17] = q[:12] × poseScale
obs[17:29]= dq[:12] × velScale
obs[29:41]= last_action[:12]
obs[41:44]= base_ang_vel × angVelScale
obs[44:47]= euler_angles
```

### 3.3 帧堆叠机制 (PolicyBase)

```cpp
// policy_base.h
std::deque<Eigen::VectorXd> histObs_;   // 滑动窗口

MatrixXd getFinalObsInput() {
    histObs_.push_back(obs_);           // 推入新帧
    if (histObs_.size() > frameStack_)  // 超过堆叠数则弹出最旧帧
        histObs_.pop_front();
    // 拼接: [obs_t | obs_t-1 | obs_t-2 | obs_t-3 | obs_t-4]
    for (int i = 0; i < frameStack_; i++)
        output.block(0, i*numSingleObs, 1, numSingleObs) = histObs_[i].transpose();
}
```

## 4. 动作输出流程

### 4.1 AMPPolicy 输出链

```
RKNN推理 → 20个float
    ↓
clipActions()         → 截断到 [-10, 10]
    ↓
postModifyOutput()    → action[i] = NN_out[i] × actionScale[i] + defaultPose[i]
                         (默认: scale=1.0, pose=0.0)
    ↓
policy→actual 重排    → 20维策略顺序 → 22维实际机器人顺序
    ↓                   不在策略中的2个关节 = -999
execRlLoop()
    target_q[i] = action[i] × action_scale(0.25)   [仅 ≠ -999 的关节]
    ↓
Robot→Motor 变换
    motor_q = (target_q + urdf_offset) × direction
    ↓
motorOutputSetKpKd()  → 设置 p_kp, p_kd
    ↓
写入 motorOutput_
```

### 4.2 完整末端公式

```
电机收到的位置指令 = [ (NN_output × 1.0 + 0.0) × 0.25 + urdf_offset ] × direction
电机收到的Kp/Kd   = p_kp, p_kd (从PD配置)
```

### 4.3 状态坐标变换

```cpp
// Motor → Robot (传感器方向)
robot_q[i] = motor_q[i] × direction[map_index[i]] - urdf_offset[map_index[i]]

// Robot → Motor (指令方向)  
motor_q[i] = (robot_q[i] + urdf_offset[map_index[i]]) × direction[map_index[i]]
```

其中 `map_index` 将 `joint_names` 顺序映射到硬件电机ID。

## 5. 模型转换管线

### 5.1 TorchScript → ONNX

```python
# tools/pt2onnx.py
model = torch.jit.load("policy.pt", map_location="cpu")
dummy_input = torch.randn(1, 270)   # AMP=345, Humanoidgym=270

torch.onnx.export(model, dummy_input, "policy.onnx",
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"])
```

### 5.2 ONNX → RKNN (x86上执行)

```bash
pip install rknn-toolkit2
```

```python
# tools/onnx2rknn.py
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform="rk3588s")
rknn.load_onnx("policy.onnx",
    inputs=["input"],
    input_size_list=[[1, 345]])   # batch=1, 输入维度
rknn.build(do_quantization=False)  # FP32, 不做INT8量化
rknn.export_rknn("policy.rknn")
```

### 5.3 ONNX → TensorRT (Jetson Orin)

```bash
# 在Jetson上执行
trtexec --onnx=policy.onnx \
        --saveEngine=policy.trt \
        --fp16                        # FP16加速
```

## 6. RK3588s NPU 部署

### 6.1 硬件环境

```
RK3588s + Ubuntu 20.04 aarch64
├── ROS Noetic (需手动编译安装)
├── Pinocchio 3.3.0  (ros-noetic-pinocchio)
├── yaml-cpp, Eigen3, nlohmann-json
├── RKNN SDK 运行时: librknn_api.so, librknnrt.so, rknn_api.h
└── livelybot_serial SDK (电机驱动)
```

### 6.2 目录结构

```
Sim2real/
├── src/sim2real/policy/walk/     ← .rknn 策略文件
├── src/sim2real/config/          ← PD/RL YAML配置
├── src/sim2real/robot_param/     ← 电机CAN拓扑参数
├── rknn/                         ← RKNN SDK so + header
├── tools/onnx2rknn.py                   # ONNX→RKNN 转换
├── tools/pt2onnx.py                     # TorchScript→ONNX 转换
└── sim2real_pi_plus_rknn.sh      # 一键启动脚本
```

### 6.3 CMake 自动平台检测

```cmake
# CMakeLists.txt
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    find_path(TENSORRT_HEADER NvInfer.h /usr/include/aarch64-linux-gpu)
    if(TENSORRT_HEADER)
        add_definitions(-DPLATFORM_JETSON)    # 有TensorRT → Jetson
    else()
        add_definitions(-DPLATFORM_RK)         # 无TensorRT → RK3588
    endif()
endif()

# RK平台链接RKNN库
target_link_libraries(... ${RKNN_PATH}/librknn_api.so ${RKNN_PATH}/librknnrt.so)
```

### 6.4 RKNNInferenceEngine 核心代码

```cpp
// 构造: 加载.rknn文件
RKNNInferenceEngine(path, obsSize, dofs, lower, upper) {
    auto modelData = readFileData("policy.rknn");
    rknn_init(&ctx_, modelData, size, 0, NULL);
    rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &ioNum_, ...);

    rknnInputs_[0].index = 0;
    rknnInputs_[0].size  = obsSize × sizeof(float);
    rknnInputs_[0].type  = RKNN_TENSOR_FLOAT32;
    rknnInputs_[0].fmt   = RKNN_TENSOR_NHWC;
}

// 推理: 每帧调用 (50Hz)
bool updateAction(MatrixXd input, Robot_Output& robotOut) {
    // Eigen → float[]
    for (i = 0; i < obsSize; i++)
        inputData[i] = input(i);

    rknnInputs_[0].buf = inputData.data();
    rknn_inputs_set(ctx_, 1, rknnInputs_);     // 送入NPU
    rknn_run(ctx_, nullptr);                    // 执行推理
    rknn_outputs_get(ctx_, 1, rknnOutputs_, ..); // 取结果

    // float[] → Robot_Output.action
    for (i = 0; i < dofs; i++)
        robotOut.action[i] = outputData[i];
}
```

### 6.5 启动命令

```bash
# 一键启动
./sim2real_pi_plus_rknn.sh

# 等价于:
source devel/setup.bash
roslaunch sim2real_master joy_control_pi_plus_rknn.launch
```

## 7. 配置系统详解

### 7.1 YAML 文件层级

```
rl_config.yaml (配置文件注册表)
 ├─ walk: 列表
 │    ├─ path: walk/amp_pi_plus_20dof.yaml
 │    ├─ path: walk/lr.yaml
 │    └─ path: walk/footstep.yaml
 ├─ motion: 列表
 └─ up: 列表

walk/amp_pi_plus_20dof.yaml (具体策略配置)
 ├─ policy_name: "walk/pi_plus_amp_1110_policy_1.rknn"
 ├─ algorithm: "amp"
 ├─ dofs: 20
 ├─ num_single_obs: 69
 ├─ frame_stack: 5
 ├─ action_scale: 0.25
 └─ ...

pd_config.yaml (全机器人共用)
 ├─ dofs: 22
 ├─ joint_names: [...]
 ├─ map_index: [...]      ← joint_names → 硬件电机ID
 ├─ direction: [±1,...]    ← 电机旋转方向
 ├─ urdf_offset: [...]     ← 零点偏移
 ├─ kp/kd, p_kp/p_kd      ← PD增益
 └─ lower/upper            ← 关节限位
```

### 7.2 关键字段说明

| 字段 | 含义 | 示例 |
|------|------|------|
| `map_index` | `joint_names[i]` → 硬件电机ID `map_index[i]` | `[5,4,3,2,1,0,...]` |
| `direction` | 电机旋向: +1正向, -1反向 | `[1,1,-1,-1,...]` |
| `urdf_offset` | 电机零点 → 策略home点的偏移(rad) | `[0,0,0,1.95,-1.57,...]` |
| `kp/kd` | PD基础增益(默认模式) | 腿 110/1.1, 臂 6/0.6 |
| `p_kp/p_kd` | 策略融合PD增益(RUNNING时) | 腿 80/1.9, 臂 30/1.1 |
| `action_scale` | RL输出阻尼系数(sim2real关键) | 0.25 (越小越稳) |
| `policy2ActualMap` | 策略关节→实际关节重排 | `[?,?,...]` |
| `clip_obs` | 观测截断阈值 | 18.0 |

### 7.3 机器人参数 YAML (robot_param)

```yaml
# 22dof_STM32H730_pi_plus_ctrl_params.yaml
SDK_version: 2
serial: /dev/ttyACM, baudrate: 4000000

CAN_BOARD_1:
  serial_id: 1
  port_1:  [L_low_foot, L_up_foot, L_calf, L_hip_pitch, L_hip_roll, L_hip_yaw]
    motor: 5047_36_2, ids: [1,2,3,4,5,6]
  port_2:  [R_low_foot, R_up_foot, R_calf, R_hip_pitch, R_hip_roll, R_hip_yaw]
    motor: 5047_36_2, ids: [1,2,3,4,5,6]
  port_3:  [L_shoulder, L_arm_1, L_arm_2, L_wrist]
    motor: 4438_30, ids: [1,2,3,4]
  port_4:  [R_shoulder, R_arm_1, R_arm_2, R_wrist]
    motor: 4438_30, ids: [1,2,3,4]
  port_5:  [Head_Yaw, Head_Pitch]
    motor: 3536_32, ids: [1,2]
```

## 8. 手柄控制详解

### 8.1 物理连接与数据流

```
蓝牙手柄 (Xbox/F710) → /dev/input/js0
    ↓
joy_node (ROS, deadzone=0.001, 50ms合并间隔)
    ↓ sensor_msgs/Joy → /joy_input (8轴 + 11按键)
    ↓
humanoid_driver (Python, IIR低通滤波, alpha=0.6)
    ↓ 过滤后 sensor_msgs/Joy → /joy
    ↓
joy_teleop.py (按 joy.yaml 拆分为两路)
    ↓
    ├─ geometry_msgs/Twist → /cmd_vel  (速度指令, 被RL线程消费做观测输入)
    └─ sim2real_msg/Joy   → /joy_msg  (模式切换, 被FSM和Sim2Real消费)
```

### 8.2 摇杆→行走命令映射

```yaml
# joy.yaml 配置
左摇杆 Y (axis 1) → cmd_vel linear.x  ×1.5    # 前进后退, 最大1.5m/s
左摇杆 X (axis 0) → cmd_vel linear.y  ×0.7    # 横移, 最大0.7m/s
右摇杆 X (axis 3) → cmd_vel angular.z ×1.57   # 转向, 最大1.57rad/s (~90°/s)
```

`Sim2Real::cmdVelCallback()` 接收 `/cmd_vel`，存入 `cmdVel_` 结构体:
```cpp
cmdVel_.vx   = msg->linear.x;    // +前进
cmdVel_.vy   = msg->linear.y;    // +左移
cmdVel_.dyaw = msg->angular.z;   // +左转
cmdVel_.joy  = joy_;             // 附带按键状态
```

RL推理时，`cmdVel_` 被传入策略的 `updateObservation()`，成为观测的第6-8维:
```cpp
// AMPPolicy: 做IIR滤波后缩放入观测
obs[6] = filtered_vx × cmdVelLineScale;
obs[7] = filtered_vy × cmdVelLineScale;
obs[8] = filtered_dyaw × cmdVelAngleScale;
```

### 8.3 手柄滤波 (HumanoidDriver)

```python
# 一阶IIR低通，防摇杆抖动
v_output = 0.6 * v_input + 0.4 * v_output_last

# 低速死区（过滤微小漂移）
if abs(vx) < 0.08:   filtered_vx = 0
if abs(vy) < 0.08:   filtered_vy = 0
if abs(dyaw) < 0.17: filtered_dyaw = 0
```

### 8.4 Xbox手柄→Joy.msg字段对应

```
物理控制            Joy.msg字段        用途
─────────────────────────────────────────────────
左摇杆 X (axis 0) → l_horizontal    → /cmd_vel (横移)
左摇杆 Y (axis 1) → l_vertical      → /cmd_vel (前进)
LT   (axis 2)     → lt              → 模式切换组合键
右摇杆 X (axis 3) → r_horizontal    → /cmd_vel (转向)
右摇杆 Y (axis 4) → r_vertical      → (未用)
RT   (axis 5)     → rt              → 模式切换组合键
Dpad 水平 (axis 6)→ dpad_horizontal → 模式选择/策略切换
Dpad 垂直 (axis 7)→ dpad_vertical   → 模式选择

按钮0  = A        按钮3  = Y        按钮6  = Back    按钮9  = L (左摇杆按下)
按钮1  = B        按钮4  = LB       按钮7  = Start   按钮10 = R (右摇杆按下)
按钮2  = X        按钮5  = RB       按钮8  = Center
```

### 8.5 键码值范围（逐层）

**第1层: Linux内核 `/dev/input/js0` — 原始硬件值**

```
轴 (axes):    js_event.value → 有符号 16 位整数 → [-32767, +32767]
按钮 (buttons): js_event.value → 有符号 16 位整数 → 0 或 1
```

各轴原始值语义:

| 轴 | 物理杆 | 推到底 | 居中 | 推到反方向 |
|----|--------|--------|------|-----------|
| 左X (axis 0) | 左摇杆左右 | +32767 (右) | 0 | -32767 (左) |
| 左Y (axis 1) | 左摇杆上下 | +32767 (上) | 0 | -32767 (下) |
| LT  (axis 2) | 左扳机 | **-32767** (按死) | +32767 | — |
| 右X (axis 3) | 右摇杆左右 | +32767 (右) | 0 | -32767 (左) |
| 右Y (axis 4) | 右摇杆上下 | +32767 (上) | 0 | -32767 (下) |
| RT  (axis 5) | 右扳机 | **-32767** (按死) | +32767 | — |
| DpadH(axis 6)| 十字键水平 | -32767 (左) | 0 | +32767 (右) |
| DpadV(axis 7)| 十字键垂直 | -32767 (下) | 0 | +32767 (上) |

> **关键**: 扳机 (LT/RT) 的值区间反向——完全释放时为 `+32767`，按死时为 `-32767`。

**第2层: joy_node 归一化 → `sensor_msgs/Joy.axes[]`**

```cpp
// joy_node 内部转换
normalized = event.value / 32767.0;  // int16_t → float32 [-1.0, +1.0]
```

| 轴 | 推到极限 | 居中 | 推到反方向 |
|----|---------|------|-----------|
| 摇杆 X/Y | +1.0 | 0.0 | -1.0 |
| 扳机 LT/RT | **-1.0** (按死) | +1.0 (释放) | — |
| Dpad | -1.0 (左/下) | 0.0 | +1.0 (右/上) |
| 按钮 | 1.0 (按下) | 0.0 (松开) | — |

**第3层: joy_teleop 乘 scale 后的 `cmd_vel` 范围**

```yaml
# joy.yaml 的 scale → 最终速度指令范围
左Y ×1.5  → cmd_vel linear.x  = [-1.5,  +1.5]  m/s (前进/后退)
左X ×0.7  → cmd_vel linear.y  = [-0.7,  +0.7]  m/s (左移/右移)
右X ×1.57 → cmd_vel angular.z = [-1.57, +1.57] rad/s (左转/右转)
```

**第4层: 代码中的阈值判断**

```cpp
// 模式切换: 用 -0.1 判断 LT/RT 是否按下（注意扳机按死=负值）
msg->lt < -0.1   → LT 按了半程以上
msg->lt < -0.5   → LT 快按死了（用于运动加速）
msg->rt < -0.5   → RT 快按死了

// 按键: 直接和 1 比较
msg->a  == 1     → A 按下
msg->lb == 1     → LB 按下

// Dpad: 直接和 ±1 比较
msg->dpad_horizontal == 1   → 十字键右
msg->dpad_horizontal == -1  → 十字键左
msg->dpad_vertical   == 1   → 十字键上
msg->dpad_vertical   == -1  → 十字键下
```

**扳机符号速查**

```
LT/RT 释放 → axes[2/5] = +32767 → joy_node → +1.0  → lt/rt = +1.0
LT/RT 半按 → axes[2/5] =     0  →            →  0.0  → lt/rt =  0.0
LT/RT 按死 → axes[2/5] = -32767 →            → -1.0  → lt/rt = -1.0

代码判断含义:
  lt/rt < -0.1  → 扳机按了半程以上
  lt/rt < -0.5  → 扳机快按死了
```

### 8.6 按键操作表

| 组合键 | 功能 | 处理方 |
|--------|------|--------|
| **Center** 单独按 | 强制回 Default 模式 | MasterNode FSM |
| **LT+RT 按住 + Dpad →** | 切换下一模式 | MasterNode FSM |
| **LT+RT 按住 + Dpad ←** | 切换上一模式 | MasterNode FSM |
| **LT+RT 按住 + A** | 确认进入当前模式 | MasterNode FSM |
| **LT+RT 按住 + B** | 紧急停机 (电机泄力) | MasterNode FSM |
| **L** (左摇杆按下) | 站立归零 (3秒轨迹→STANDBY) | Sim2Real |
| **LB** | RL行走 开/关 (RUNNING↔STANDBY) | Sim2Real |
| **RB** | 坐下 (3秒轨迹→SITTING) | Sim2Real |
| **LT+RT + Start** | 站立 (同 L) | DefaultController |
| **LT+RT + Dpad ← →** | STANDBY下切换walk/motion/up策略 | DefaultController |
| **LT(+RT) + A/B/X/Y** | 触发预设动作(踢球/跳舞等) | DefaultController |
| **LT + Dpad ↑** | 切换 normal/walk 模式 | AMPPolicy |
| **左摇杆** | 行走方向 (前后/左右) | 连续控制 |
| **右摇杆** | 转动方向 (角速度) | 连续控制 |

### 8.7 模式列表

```
手柄 LT+RT + Dpad ← → 循环:
  Default (RL行走)  →  Custom (自定义策略)  →  Remote (远程遥控)
      ↑                                                    ↓
  Develop (电机调试) ←  Teaching (示教录制)  ←  Calibration (校零)
```

确认进入后，各模式对应的控制器接口：
- Default → `DefaultControllerInterface` (行走/站立/动作)
- Custom → `CustomControllerInterface` (用户策略)
- Remote → `RemoteControllerInterface` (ROS话题操控)
- Calibration → `ZeroCalibrationControllerInterface` (零点校准)
- Teaching → `TeachControllerInterfaceN` (录制/回放轨迹)
- Develop → `DevelopControllerInterface` (逐个电机调试)

## 9. 新机器人适配指南

适配新机器人需要以下步骤：

### 步骤1: 训练RL策略

确保训练时的observation和action维度与真机一致。

### 步骤2: 模型转换

```bash
python3 tools/pt2onnx.py    # 改 PT_MODEL_PATH, NUM_OBS, NUM_ACTIONS
python3 tools/onnx2rknn.py  # 改 ONNX_PATH, input_size_list, OUT_DIR
```

### 步骤3: 编写 PD 配置

```yaml
# myrobot_Ndof_pd_config.yaml
joint_names: [关节1, 关节2, ...]          # 必须与仿真顺序一致
map_index: [电机ID1, 电机ID2, ...]       # joint_names[i] → 电机序号
direction: [±1, ±1, ...]                 # 实测每个电机方向
urdf_offset: [0, 0, ...]                 # 电机零点偏移
kp: [...]                                 # 各关节PD比例增益
kd: [...]                                 # 各关节PD微分增益
lower: [...]                              # 关节最小角度
upper: [...]                              # 关节最大角度
```

**实测 direction 的方法**: 对每个电机发送 `target_q = 0.1`，观察实际旋转方向。若关节 angle 增大则为正向(+1)，减小则为反向(-1)。

**实测 urdf_offset 的方法**: 将机器人摆到训练时的默认姿态(直立)，读取各电机当前位置，该值即为 `urdf_offset`。

### 步骤4: 编写 RL 配置

```yaml
# myrobot_rl_config.yaml
algorithm: "humanoidgym" (或 "amp")
policy_name: "walk/myrobot_policy.rknn"
dofs: N                    # 策略控制关节数
frame_stack: 5
num_single_obs: M          # 单帧观测维度
action_scale: 0.25         # 从0.1起步逐步增大
pd_ctrl_f: 1000            # PD频率
rl_ctrl_f: 50              # RL频率
...
```

### 步骤5: 实现 MotorGroup (如需要新电机协议)

```cpp
// myrobot_motor_group.h
class MyRobotMotorGroup : public RobotMotorGroup {
public:
    MyRobotMotorGroup(robot_ptr, mapIndex, joint_names, limits);
    
    // 电机发送函数
    void setMotorGroupByName(name, motorOutput) override {
        for (auto& idx : groups_[name]) {
            motors_[idx]->setPosition(motorOutput.target_q[idx]);
            motors_[idx]->setKpKd(motorOutput.target_kp[idx], motorOutput.target_kd[idx]);
        }
    }
    void sendMotors() override {
        serial_->flush();   // 发送到硬件
    }
};
```

### 步骤6: 编写机器人参数 YAML

定义电机拓扑 (CAN总线/串口分配、电机型号、ID等)。

### 步骤7: 编写 Launch 文件

```xml
<!-- joy_control_myrobot_rknn.launch -->
<launch>
  <arg name="model_type" value="myrobot"/>
  <arg name="dof_type" default="N"/>
  
  <rosparam file="...params.yaml" command="load"/>
  <!-- 硬件驱动、IMU、手柄 -->
  <include file="$(find lr_control)/launch/..."/>
  <node ... yesense_imu_node .../>
  <include file="$(find sim2real_master)/launch/joy_teleop.launch"/>
  
  <!-- sim2real核心 -->
  <node pkg="sim2real_master" type="sim2real_master_node" ...>
    <param name="model_type" value="myrobot"/>
    <param name="pd_config_file" value="...pd_config.yaml"/>
    <param name="rl_config_file" value="...rl_config.yaml"/>
  </node>
</launch>
```

### 步骤8: 构建与部署

```bash
# RK3588
catkin build -DCMAKE_CXX_FLAGS="-DRELEASE"
./sim2real_myrobot_rknn.sh

# Jetson Orin
catkin build -DCMAKE_CXX_FLAGS="-DRELEASE"
./sim2real_myrobot_orin.sh
```

## 10. 常见问题排查

### 10.1 机器人乱动/抽搐

检查 `direction` 符号: 实测每个电机方向，确保 `robot_q = motor_q × direction - offset` 的符号正确。

### 10.2 机器人无法站立

- 检查 `urdf_offset` 是否等于电机在默认姿态下的实际位置
- 检查 `action_scale` 是否太小 (从0.1起步，逐步增大到0.5)
- 检查 `kp/kd` 是否太小 (腿≥80, 臂≥6)

### 10.3 仿真和真机行为不一致

- 确保 `num_single_obs` 和仿真观测维一致
- 确保 `joint_names` 顺序与训练时action顺序一致
- 减小 `action_scale` (0.1~0.25) 增加sim2real阻尼

### 10.4 RKNN推理失败

```cpp
// 常见错误码
RKNN_ERR_MODEL_INVALID (-3)  → .rknn文件损坏或平台不匹配
RKNN_ERR_DEVICE_UNAVAILABLE  → NPU驱动未加载，检查 dmesg | grep rknn
```

### 10.5 电机不响应

```bash
# 检查串口
ls -la /dev/ttyACM*        # 电机串口
ls -la /dev/ttyS*          # IMU/OLED串口
dmesg | grep tty            # 内核日志

# 检查ROS话题
rostopic list | grep imu
rostopic echo /imu/data
rostopic echo /joy_msg
```

### 10.6 观测构造校验

在C++代码中打印对比仿真和真机的第一帧观测:
```cpp
ROS_INFO_STREAM("obs[0:3] ang_vel: " << obs_.segment(0,3).transpose());
ROS_INFO_STREAM("obs[3:6] gravity: " << obs_.segment(3,3).transpose());
ROS_INFO_STREAM("obs[6:9] cmd:     " << obs_.segment(6,3).transpose());
// ... 与训练代码对比
```
