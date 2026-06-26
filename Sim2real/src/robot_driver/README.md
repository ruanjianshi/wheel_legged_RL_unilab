# 人形机器人IMU姿态补偿控制器

## 概述

这是一个基于IMU传感器的人形机器人姿态补偿控制系统。系统能够实时读取手柄输入和IMU姿态数据，通过PD控制算法自动补偿机器人的倾斜姿态，提供更稳定的运动控制。

## 核心功能

- **手柄输入处理**: 订阅 `/joy_input` 话题，获取遥控指令
- **IMU姿态感知**: 订阅 `/imu/data` 话题，实时获取机器人姿态
- **智能姿态补偿**: 基于pitch/roll角度的PD控制补偿
- **多级滤波处理**: 对输入信号和补偿信号分别滤波
- **实时数据发布**: 输出滤波后的控制指令和补偿数据

## 补偿机制详解

### 1. 数据采集与预处理

```
[手柄输入] → /joy_input → 提取速度向量[vx, vy, vrot]
[IMU传感器] → /imu/data → 姿态角度提取
```

#### IMU数据处理流程：
1. **姿态解算**: 从四元数提取欧拉角 (roll, pitch, yaw)
2. **数据存储**: 线程安全地保存角度数据

### 2. PD补偿算法

系统采用独立的PD控制器对X和Y方向进行姿态补偿：

#### X方向补偿 (基于Pitch角度)
```python
# 当机器人前后倾斜时进行补偿
compensation_x = pitch * kp_x + (pitch - pitch_last) * kd_x
```

#### Y方向补偿 (基于Roll角度)  
```python
# 当机器人左右倾斜时进行补偿
compensation_y = roll * kp_y + (roll - roll_last) * kd_y
```

#### 物理意义：
- **Pitch补偿**: 机器人前倾时自动减速，后倾时加速
- **Roll补偿**: 机器人左倾时向右补偿，右倾时向左补偿

### 3. 补偿值处理

#### 限幅处理：
```python
# 限制补偿值范围在[-1.0, 1.0]
if compensation > 1.0:  compensation = 1.0
if compensation < -1.0: compensation = -1.0
```

#### 死区处理：
```python
# 小于阈值的补偿值置零，避免微小抖动
if -0.05 < compensation < 0.05:  compensation = 0.0
```

### 4. 信号融合与输出

系统采用两级滤波融合机制：

#### 第一级：手柄输入滤波
```python
v_filtered = alpha * v_input + (1 - alpha) * v_filtered_last
```

#### 第二级：IMU补偿混合
```python
v_output = alpha_imu * v_compensation + (1 - alpha_imu) * v_filtered
```

## 参数配置

### 核心参数说明

| 参数类别 | 参数名 | 默认值 | 作用说明 |
|----------|--------|--------|----------|
| **输入滤波** | `alpha` | 0.5 | 手柄输入响应速度 |
| **补偿混合** | `alpha_imu` | 0.5 | IMU补偿强度 |

### PD控制器参数

#### X方向 (Pitch补偿)
- `pd_x/kp: 5.0` - 比例系数，控制补偿响应强度
- `pd_x/kd: 0.05` - 微分系数，提供阻尼减少超调

#### Y方向 (Roll补偿)  
- `pd_y/kp: 2.0` - 比例系数，通常小于X方向
- `pd_y/kd: 0.05` - 微分系数，保持稳定性

## 系统架构

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   手柄输入   │───▶│   一级滤波    │───▶│             │
└─────────────┘     └──────────────┘     │             │
                                         │   信号融合   │───▶ /joy输出
┌─────────────┐     ┌──────────────┐     │             │
│  IMU传感器   │───▶│  PD姿态补偿   │───▶│             │
└─────────────┘     └──────────────┘     └─────────────┘
       │                   │
       ▼                   ▼
┌─────────────┐     ┌─────────────────┐
│ 姿态角度提取 │     │/imu_compensation│
└─────────────┘     └─────────────────┘
```

## 数据流详解

### 输入数据流
1. **手柄数据**: `/joy_input` → 速度向量提取
2. **IMU原始数据**: `/imu/data` → 姿态四元数

### 处理流程
1. **IMU预处理**: 姿态解算 → 角度提取
2. **PD补偿计算**: 角度误差 → PD算法 → 补偿向量
3. **补偿后处理**: 限幅 → 死区 → 最终补偿值
4. **信号融合**: 手柄滤波 → IMU混合 → 输出生成

### 输出数据流
1. **控制输出**: `/joy` - 最终控制指令
2. **监控输出**: `/imu_compensation` - 实时补偿向量
3. **调试输出**: 终端显示X方向补偿值

## 使用指南

### 1. 基本启动
```bash
roslaunch robot_driver humanoid_driver.launch
```

### 2. 参数调优

#### 增强稳定性 (保守策略)
```yaml
alpha: 0.3        # 降低手柄响应速度
alpha_imu: 0.3    # 减弱补偿强度
pd_x/kp: 3.0      # 降低比例系数
```

#### 提高响应性 (激进策略)
```yaml
alpha: 0.7        # 提高手柄响应速度  
alpha_imu: 0.7    # 增强补偿强度
pd_x/kp: 8.0      # 提高比例系数
```

### 3. 实时监控
```bash
# 监控补偿效果
rostopic echo /imu_compensation

# 可视化补偿曲线
rqt_plot /imu_compensation/x /imu_compensation/y
```

## 调试与优化

### 补偿效果评估
- **观察终端输出**: X方向补偿值变化
- **检查补偿向量**: `/imu_compensation` 话题数据
- **验证姿态响应**: 倾斜机器人观察补偿反应

### 常见调优场景

#### 补偿过强 (振荡)
- 降低 `pd_x/kp` 和 `pd_y/kp` 
- 增加 `pd_x/kd` 和 `pd_y/kd`
- 降低 `alpha_imu`

#### 补偿不足 (反应迟钝)
- 提高 `pd_x/kp` 和 `pd_y/kp`
- 提高 `alpha_imu`

#### 噪声过大
- 增加 `pd_x/kd` 和 `pd_y/kd`

## 技术特点

- **实时性**: 50Hz控制频率，满足动态响应需求
- **稳定性**: 限幅死区处理，避免系统振荡
- **可调性**: 丰富的参数配置，适应不同场景
- **监控性**: 完整的数据输出，便于调试分析
- **安全性**: 线程安全的数据访问，补偿值限幅保护

## 文件结构

```
robot_driver/
├── src/controllers/
│   └── humanoid_controller.py     # 主控制器 (IMU补偿核心)
├── launch/
│   └── humanoid_driver.launch     # 启动文件
├── config/
│   └── params.yaml                # 参数配置 (补偿参数)
├── examples/
│   ├── remote_joy_receiver.py     # 远端接收示例
│   └── remote_joy_receiver.launch # 远端启动文件
└── README.md                      # 本文档
```

## 依赖要求

- ROS (Noetic推荐)
- Python 3
- tf.transformations
- numpy
- sensor_msgs
- geometry_msgs

---

